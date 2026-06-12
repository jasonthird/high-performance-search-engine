//! Exact top-k Block-Max WAND query execution over BM25.
//!
//! The high-level idea:
//!
//! 1. Keep one cursor per query term, positioned in that term's posting list
//!    (sorted by doc_id). Cursors only ever move forward.
//! 2. Maintain a bounded min-heap of the best k results seen so far. Once the
//!    heap is full, its minimum score is the *threshold*: a document must
//!    score strictly above it to enter the top-k.
//! 3. WAND pivoting: sort cursors by their current doc_id and find the first
//!    prefix whose summed per-term upper bounds exceed the threshold. No
//!    document before the pivot document can possibly beat the threshold, so
//!    everything before it is skipped without scoring.
//! 4. Block-max refinement: per-term upper bounds are coarse. Before scoring
//!    the pivot, sum the *block* max scores of the blocks that contain the
//!    pivot document. If even that refined bound cannot beat the threshold,
//!    skip past the end of the shortest of those blocks — entire blocks of
//!    postings are skipped without decoding them.
//! 5. Only when a document survives both checks is it scored *exactly* with
//!    BM25. Because every skip is justified by a safe upper bound, the final
//!    top-k is identical to what a naive exhaustive BM25 scan would return.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use serde::Serialize;

use crate::bm25;
use crate::indexer::SearchableIndex;
use crate::postings::TermPostings;

/// Debug counters exposed by every search.
#[derive(Debug, Default, Clone, Serialize)]
pub struct SearchStats {
    pub num_docs_total: usize,
    pub num_query_terms: usize,
    pub num_postings_visited: usize,
    pub num_docs_scored: usize,
    pub num_blocks_visited: usize,
    pub num_blocks_skipped: usize,
}

/// A scored document (internal doc_id).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchHit {
    pub doc_id: u32,
    pub score: f32,
}

/// Entry in the top-k min-heap.
///
/// Ordering: by score ascending, then by doc_id *descending*. In a min-heap
/// this means that among equal scores the largest doc_id is evicted first,
/// so ties are resolved in favor of smaller doc_ids — the same tie-break as
/// sorting by (score desc, doc_id asc).
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    score: f32,
    doc_id: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Scores are always finite BM25 values, so partial_cmp cannot fail.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.doc_id.cmp(&self.doc_id))
    }
}

/// Bounded min-heap holding the current top-k candidates.
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Reverse<HeapEntry>>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// Current pruning threshold: the k-th best score once the heap is full,
    /// otherwise -inf (every positive score is competitive).
    pub fn threshold(&self) -> f32 {
        if self.k == 0 {
            return f32::INFINITY;
        }
        if self.heap.len() == self.k {
            self.heap
                .peek()
                .map(|e| e.0.score)
                .unwrap_or(f32::NEG_INFINITY)
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Insert a candidate if it strictly beats the threshold.
    pub fn insert(&mut self, doc_id: u32, score: f32) -> bool {
        if self.k == 0 || score <= self.threshold() {
            return false;
        }
        self.heap.push(Reverse(HeapEntry { score, doc_id }));
        if self.heap.len() > self.k {
            self.heap.pop();
        }
        true
    }

    /// Drain into hits sorted by (score desc, doc_id asc).
    pub fn into_sorted_hits(self) -> Vec<SearchHit> {
        let mut hits: Vec<SearchHit> = self
            .heap
            .into_iter()
            .map(|Reverse(e)| SearchHit {
                doc_id: e.doc_id,
                score: e.score,
            })
            .collect();
        hits.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        hits
    }
}

/// Forward-only cursor over one term's posting data.
///
/// Postings are pulled through [`TermPostings::decode_block`], so the cursor
/// works identically over in-memory postings and compressed, memory-mapped
/// blocks — and a block that is never entered is never decoded (nor paged in
/// from disk, in the mmap case).
pub(crate) struct Cursor<'a> {
    term: TermPostings<'a>,
    /// Term-level upper bound under the query's stats (cached).
    list_bound: f32,
    /// Current posting index (global across blocks).
    pos: usize,
    /// Index of the block containing `pos` (kept loosely in sync; see
    /// `align_block`).
    block: usize,
    /// Cached doc_id at `pos` (u32::MAX when exhausted).
    current_doc: u32,
    /// Lazily decoded doc_ids of `decoded_block`. Term frequencies are not
    /// decoded here — they are fetched individually only when scoring.
    decoded_docs: Vec<u32>,
    decoded_block: Option<usize>,
    /// Last posting position counted in the stats (avoid double-counting).
    counted_pos: Option<usize>,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(term: TermPostings<'a>, stats: &mut SearchStats) -> Self {
        let list_bound = term.list_bound();
        let mut cursor = Self {
            term,
            list_bound,
            pos: 0,
            block: 0,
            current_doc: u32::MAX,
            decoded_docs: Vec::new(),
            decoded_block: None,
            counted_pos: None,
        };
        cursor.load_current(stats);
        cursor
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        self.pos >= self.term.df
    }

    /// Current doc_id, or u32::MAX when exhausted (sorts last).
    pub(crate) fn doc_id(&self) -> u32 {
        self.current_doc
    }

    pub(crate) fn list_max_score(&self) -> f32 {
        self.list_bound
    }

    /// Bring `block` forward so it contains `pos`.
    fn align_block(&mut self) {
        while self.block < self.term.num_blocks() && self.term.block_end(self.block) <= self.pos {
            self.block += 1;
        }
    }

    /// Refresh the cached doc_id at `pos`, decoding its block if needed.
    fn load_current(&mut self, stats: &mut SearchStats) {
        if self.is_exhausted() {
            self.current_doc = u32::MAX;
            return;
        }
        self.align_block();
        if self.decoded_block != Some(self.block) {
            self.term.decode_docs(self.block, &mut self.decoded_docs);
            self.decoded_block = Some(self.block);
            stats.num_blocks_visited += 1;
        }
        self.current_doc = self.decoded_docs[self.pos - self.term.block_start(self.block)];
        if self.counted_pos != Some(self.pos) {
            self.counted_pos = Some(self.pos);
            stats.num_postings_visited += 1;
        }
    }

    /// Term frequency of the current posting. Only called when the document
    /// is actually scored; in the compressed source this is an O(1)
    /// random-access read, so tfs of skipped postings are never decoded.
    pub(crate) fn current_tf(&self) -> u32 {
        self.term
            .tf_at(self.block, self.pos - self.term.block_start(self.block))
    }

    pub(crate) fn idf(&self) -> f32 {
        self.term.idf
    }

    /// Move to the next posting (after the current one was scored).
    pub(crate) fn next(&mut self, stats: &mut SearchStats) {
        self.pos += 1;
        self.load_current(stats);
    }

    /// Advance to the first posting with doc_id >= target, skipping whole
    /// blocks via metadata wherever possible.
    pub(crate) fn seek(&mut self, target: u32, stats: &mut SearchStats) {
        if self.current_doc >= target {
            return; // also covers the exhausted case (u32::MAX)
        }
        self.align_block();
        // Skip entire blocks that end before the target: their postings are
        // never decoded.
        while self.block < self.term.num_blocks()
            && self.term.block_max_doc_ids[self.block] < target
        {
            self.block += 1;
            stats.num_blocks_skipped += 1;
        }
        if self.block >= self.term.num_blocks() {
            self.pos = self.term.df;
            self.current_doc = u32::MAX;
            return;
        }
        // Binary-search the landing position inside the target block (its
        // doc_ids are sorted and its max_doc_id >= target, so a position
        // exists). Decode the block first if this cursor hasn't yet.
        let start_idx = self.term.block_start(self.block);
        if self.decoded_block != Some(self.block) {
            self.term.decode_docs(self.block, &mut self.decoded_docs);
            self.decoded_block = Some(self.block);
            stats.num_blocks_visited += 1;
        }
        let from = self.pos.max(start_idx) - start_idx;
        let within = from + self.decoded_docs[from..].partition_point(|&d| d < target);
        self.pos = start_idx + within;
        self.load_current(stats);
    }

    /// Block-max "shallow" probe: the upper bound and boundary of the block
    /// that would contain the first posting with doc_id >= target. Reads only
    /// block metadata; does not move the cursor or decode anything. Returns
    /// None if no remaining doc reaches `target`.
    fn shallow_block_for(&self, target: u32) -> Option<(f32, u32)> {
        let mut b = self.block;
        while b < self.term.num_blocks() && self.term.block_max_doc_ids[b] < target {
            b += 1;
        }
        if b < self.term.num_blocks() {
            Some((self.term.block_bound(b), self.term.block_max_doc_ids[b]))
        } else {
            None
        }
    }
}

/// Exact top-k Block-Max WAND search over any [`SearchableIndex`].
///
/// `terms` must already be tokenized. Duplicates are ignored: each unique
/// term contributes once, so "pizza pizza" scores like "pizza".
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

    let avg_doc_len = index.avg_doc_len();
    let mut top_k = TopK::new(k);

    loop {
        // Cursors ordered by current doc_id; exhausted cursors sort last.
        cursors.sort_unstable_by_key(Cursor::doc_id);
        let threshold = top_k.threshold();

        // --- WAND pivot selection -------------------------------------
        // Find the first prefix of cursors whose summed term upper bounds
        // exceed the threshold. A document can only beat the threshold if it
        // contains at least the pivot term (or a later one): any document
        // before the pivot doc matches only a prefix of cursors whose total
        // upper bound is <= threshold.
        let mut pivot = None;
        let mut acc = 0.0f32;
        for (i, cursor) in cursors.iter().enumerate() {
            if cursor.is_exhausted() {
                break;
            }
            acc += cursor.list_max_score();
            if acc > threshold {
                pivot = Some(i);
                break;
            }
        }
        let Some(mut p) = pivot else {
            // Even using all remaining terms no document can beat the
            // threshold: the top-k is final.
            break;
        };
        let pivot_doc = cursors[p].doc_id();
        if pivot_doc == u32::MAX {
            break;
        }
        // Extend the pivot prefix over every cursor already positioned on
        // pivot_doc. Their contributions are part of pivot_doc's upper bound;
        // leaving them out would make the block-max skip below unsafe.
        while p + 1 < cursors.len() && cursors[p + 1].doc_id() == pivot_doc {
            p += 1;
        }

        // --- Block-max refinement --------------------------------------
        // Sum the *block-level* upper bounds of the blocks that would contain
        // pivot_doc. This is a much tighter (still safe) bound than the
        // per-list max used for pivoting. Also track the nearest block
        // boundary, which tells us how far we may safely jump if the bound
        // fails.
        let mut block_bound = 0.0f32;
        let mut min_block_boundary = u32::MAX;
        for cursor in &cursors[..=p] {
            if let Some((block_max, boundary)) = cursor.shallow_block_for(pivot_doc) {
                block_bound += block_max;
                min_block_boundary = min_block_boundary.min(boundary);
            }
        }

        if block_bound > threshold {
            if cursors[0].doc_id() == pivot_doc {
                // All cursors up to the pivot are aligned on pivot_doc:
                // score it exactly with BM25.
                let doc_len = index.doc_len(pivot_doc);
                let mut score = 0.0f32;
                for cursor in &mut cursors {
                    if cursor.doc_id() == pivot_doc {
                        score += bm25::term_contribution(
                            cursor.term.idf,
                            cursor.current_tf(),
                            doc_len,
                            avg_doc_len,
                        );
                        cursor.next(&mut stats);
                    }
                }
                stats.num_docs_scored += 1;
                if !index.is_deleted(pivot_doc) {
                    top_k.insert(pivot_doc, score);
                }
            } else {
                // Not aligned yet: move ONE lagging cursor up to the pivot —
                // the one with the largest upper bound, which tightens future
                // pivots fastest (the standard WAND heuristic). The others
                // stay lazy: if the pivot later jumps past them, their blocks
                // are never decoded at all. Documents in between match only
                // cursors[0..p], whose summed upper bounds are <= threshold,
                // so they are safely skipped.
                let mut chosen = 0; // cursors[0].doc_id() < pivot_doc here
                for (i, cursor) in cursors[..p].iter().enumerate().skip(1) {
                    if cursor.doc_id() < pivot_doc
                        && cursor.list_max_score() > cursors[chosen].list_max_score()
                    {
                        chosen = i;
                    }
                }
                cursors[chosen].seek(pivot_doc, &mut stats);
            }
        } else {
            // Even the block-level bound cannot beat the threshold, so no
            // document inside the current blocks can win. Jump past the
            // nearest block boundary (but not past the next cursor's current
            // document, which may add another term's contribution). As above,
            // move only the cursor with the largest upper bound; every cursor
            // in 0..=p has doc_id <= pivot_doc < next_doc, so all qualify.
            let mut next_doc = min_block_boundary.saturating_add(1);
            if let Some(next_cursor) = cursors.get(p + 1) {
                next_doc = next_doc.min(next_cursor.doc_id());
            }
            next_doc = next_doc.max(pivot_doc.saturating_add(1));
            let mut chosen = 0;
            for (i, cursor) in cursors[..=p].iter().enumerate().skip(1) {
                if cursor.list_max_score() > cursors[chosen].list_max_score() {
                    chosen = i;
                }
            }
            cursors[chosen].seek(next_doc, &mut stats);
        }
    }

    (top_k.into_sorted_hits(), stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_keeps_best_scores() {
        let mut top_k = TopK::new(3);
        for (doc_id, score) in [(0, 1.0), (1, 5.0), (2, 3.0), (3, 4.0), (4, 0.5)] {
            top_k.insert(doc_id, score);
        }
        let hits = top_k.into_sorted_hits();
        let ids: Vec<u32> = hits.iter().map(|h| h.doc_id).collect();
        assert_eq!(ids, vec![1, 3, 2]);
        assert_eq!(hits[0].score, 5.0);
    }

    #[test]
    fn top_k_threshold_tracks_kth_score() {
        let mut top_k = TopK::new(2);
        assert_eq!(top_k.threshold(), f32::NEG_INFINITY);
        top_k.insert(0, 2.0);
        assert_eq!(top_k.threshold(), f32::NEG_INFINITY);
        top_k.insert(1, 5.0);
        assert_eq!(top_k.threshold(), 2.0);
        top_k.insert(2, 3.0);
        assert_eq!(top_k.threshold(), 3.0);
        // Equal to threshold: rejected (must strictly beat it).
        assert!(!top_k.insert(3, 3.0));
    }

    #[test]
    fn top_k_breaks_ties_by_smaller_doc_id() {
        let mut top_k = TopK::new(2);
        top_k.insert(7, 1.0);
        top_k.insert(3, 1.0);
        top_k.insert(5, 2.0);
        let hits = top_k.into_sorted_hits();
        let ids: Vec<u32> = hits.iter().map(|h| h.doc_id).collect();
        // Doc 7 and 3 tie at 1.0; the larger doc_id (7) is evicted first.
        assert_eq!(ids, vec![5, 3]);
    }

    #[test]
    fn top_k_zero_returns_nothing() {
        let mut top_k = TopK::new(0);
        assert!(!top_k.insert(0, 10.0));
        assert!(top_k.into_sorted_hits().is_empty());
    }
}
