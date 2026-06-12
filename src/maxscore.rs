//! Exact top-k MaxScore query execution over BM25 (Turtle & Flood, 1995).
//!
//! WAND-family pivoting weakens as query terms multiply: with many cursors
//! the pivot prefix rarely clears the threshold, so little is skipped.
//! MaxScore instead splits the (max_score-ascending) term list at the
//! threshold into:
//!
//! - **non-essential** lists — the longest prefix whose summed upper bounds
//!   cannot beat the threshold alone. They are never iterated; they are only
//!   *probed* (seeked) for documents that already look competitive.
//! - **essential** lists — the rest. Only these drive document-at-a-time
//!   iteration: every candidate must contain at least one essential term,
//!   because a document with only non-essential terms is bounded below the
//!   threshold by construction.
//!
//! Scoring a candidate proceeds from the highest-bound non-essential list
//! downward, and aborts as soon as the partial score plus the remaining
//! lists' combined bound cannot reach the threshold. As the threshold rises,
//! the essential set shrinks toward the rarest terms.
//!
//! Like Block-Max WAND, every skip is justified by a safe upper bound, so
//! the top-k is exactly what an exhaustive BM25 scan would return (verified
//! against the naive oracle in tests). The engine uses MaxScore for queries
//! with many unique terms and Block-Max WAND otherwise.

use crate::block_max_wand::{Cursor, SearchHit, SearchStats, TopK};
use crate::bm25;
use crate::indexer::SearchableIndex;

/// Exact top-k MaxScore search. Duplicate terms are ignored, as in
/// [`crate::block_max_wand::search`].
pub fn search<I: SearchableIndex + ?Sized>(
    index: &I,
    terms: &[String],
    k: usize,
) -> (Vec<SearchHit>, SearchStats) {
    let mut unique_terms: Vec<&String> = Vec::with_capacity(terms.len());
    for term in terms {
        if !unique_terms.contains(&term) {
            unique_terms.push(term);
        }
    }

    let mut stats = SearchStats {
        num_docs_total: index.num_docs(),
        num_query_terms: unique_terms.len(),
        ..SearchStats::default()
    };

    if k == 0 {
        return (Vec::new(), stats);
    }

    let mut cursors: Vec<Cursor> = Vec::with_capacity(unique_terms.len());
    for term in &unique_terms {
        if let Some(postings) = index.term_postings(term) {
            if postings.df > 0 {
                cursors.push(Cursor::new(postings, &mut stats));
            }
        }
    }
    if cursors.is_empty() {
        return (Vec::new(), stats);
    }

    // Fixed order: ascending per-term upper bound. cum[i] bounds the best
    // possible combined contribution of lists 0..=i.
    cursors.sort_unstable_by(|a, b| {
        a.list_max_score()
            .partial_cmp(&b.list_max_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cum: Vec<f32> = cursors
        .iter()
        .scan(0.0f32, |acc, c| {
            *acc += c.list_max_score();
            Some(*acc)
        })
        .collect();

    let avg_doc_len = index.avg_doc_len();
    let n = cursors.len();
    let mut top_k = TopK::new(k);
    // First essential list: grows monotonically as the threshold rises.
    let mut essential = 0usize;

    loop {
        let threshold = top_k.threshold();
        while essential < n && cum[essential] <= threshold {
            essential += 1;
        }
        if essential >= n {
            // Even all lists combined cannot beat the threshold.
            break;
        }

        // Candidate: smallest current document among essential lists.
        let mut candidate = u32::MAX;
        for cursor in &cursors[essential..] {
            candidate = candidate.min(cursor.doc_id());
        }
        if candidate == u32::MAX {
            break; // essential lists exhausted
        }

        // Gather contributions from every cursor already positioned on the
        // candidate (essential or not), advancing them past it. No cursor
        // is ever left resting on an evaluated document.
        let doc_len = index.doc_len(candidate);
        let mut score = 0.0f32;
        for cursor in cursors.iter_mut() {
            if cursor.doc_id() == candidate {
                score += bm25::term_contribution(
                    cursor.idf(),
                    cursor.current_tf(),
                    doc_len,
                    avg_doc_len,
                );
                cursor.next(&mut stats);
            }
        }

        // Probe non-essential lists from the highest bound down, abandoning
        // as soon as even the remaining bounds cannot reach the threshold.
        for i in (0..essential).rev() {
            if score + cum[i] <= threshold {
                break;
            }
            let cursor = &mut cursors[i];
            if cursor.doc_id() < candidate {
                cursor.seek(candidate, &mut stats);
            }
            if cursor.doc_id() == candidate {
                score += bm25::term_contribution(
                    cursor.idf(),
                    cursor.current_tf(),
                    doc_len,
                    avg_doc_len,
                );
                cursor.next(&mut stats);
            }
        }

        stats.num_docs_scored += 1;
        if !index.is_deleted(candidate) {
            top_k.insert(candidate, score);
        }
    }

    (top_k.into_sorted_hits(), stats)
}
