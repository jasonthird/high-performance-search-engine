//! Segmented index correctness:
//!
//! - searching across segments with global stats must match a naive oracle
//!   that scores live documents under the documented df semantics (df
//!   counts tombstoned docs until merge, as in Lucene);
//! - after a merge, the index must score identically to a from-scratch
//!   rebuild of the live documents (df fully clean).

use std::path::PathBuf;

use high_performance_search_engine::indexer::{build_index_weighted, InputDoc, SearchableIndex};
use high_performance_search_engine::postings::DEFAULT_BLOCK_SIZE;
use high_performance_search_engine::reorder::ReorderStrategy;
use high_performance_search_engine::searcher;
use high_performance_search_engine::segments::{SegmentedIndex, SegmentedWriter};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn generate_docs(num_docs: usize, vocab: usize, seed: u64, prefix: &str) -> Vec<InputDoc> {
    let mut rng = Lcg(seed);
    (0..num_docs)
        .map(|i| {
            let len = 5 + rng.below(40);
            let body = (0..len)
                .map(|_| format!("w{}", rng.below(vocab).min(rng.below(vocab))))
                .collect::<Vec<_>>()
                .join(" ");
            InputDoc {
                id: format!("{prefix}-{i}"),
                title: format!("title w{}", rng.below(vocab)),
                body,
            }
        })
        .collect()
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("high-performance-search-engine-seg-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

/// Score all live documents exhaustively under the segmented semantics:
/// global N/avg over live docs, df per term summed over raw segment
/// postings (including tombstoned docs).
fn oracle(
    batches: &[Vec<InputDoc>],
    deleted_ids: &[&str],
    query: &str,
    k: usize,
) -> Vec<(String, f32)> {
    use high_performance_search_engine::bm25;
    use high_performance_search_engine::tokenizer::Tokenizer;

    // Replicate per-segment indexes in memory.
    let segs: Vec<_> = batches
        .iter()
        .map(|docs| build_index_weighted(docs, true, 2, DEFAULT_BLOCK_SIZE, ReorderStrategy::None))
        .collect();
    let is_deleted = |id: &str| deleted_ids.contains(&id);

    let mut live_docs = 0u64;
    let mut live_len = 0u64;
    for (seg, docs) in segs.iter().zip(batches) {
        for (d, doc) in docs.iter().enumerate() {
            if !is_deleted(&doc.id) {
                live_docs += 1;
                live_len += seg.doc_len(d as u32) as u64;
            }
        }
    }
    let avg = live_len as f32 / live_docs as f32;

    let tokenizer = Tokenizer::new(true);
    let mut terms: Vec<String> = Vec::new();
    tokenizer.for_each_token(query, |t| {
        if !terms.iter().any(|x| x == t) {
            terms.push(t.to_owned());
        }
    });

    let mut scored: Vec<(String, f32, usize, u32)> = Vec::new();
    for (si, (seg, docs)) in segs.iter().zip(batches).enumerate() {
        let mut scores = vec![0.0f32; docs.len()];
        let mut matched = vec![false; docs.len()];
        for term in &terms {
            // Global df: raw postings across all segments.
            let df: usize = segs
                .iter()
                .map(|s| s.posting_list_for(term).map_or(0, |l| l.df()))
                .sum();
            if df == 0 {
                continue;
            }
            let idf = bm25::idf(live_docs as usize, df);
            if let Some(list) = seg.posting_list_for(term) {
                for p in &list.postings {
                    scores[p.doc_id as usize] +=
                        bm25::term_contribution(idf, p.tf, seg.doc_len(p.doc_id), avg);
                    matched[p.doc_id as usize] = true;
                }
            }
        }
        for (d, doc) in docs.iter().enumerate() {
            if matched[d] && !is_deleted(&doc.id) {
                scored.push((doc.id.clone(), scores[d], si, d as u32));
            }
        }
    }
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
    });
    scored.truncate(k);
    scored.into_iter().map(|(id, s, _, _)| (id, s)).collect()
}

fn assert_results_match(
    actual: &[searcher::SearchResult],
    expected: &[(String, f32)],
    context: &str,
) {
    const EPS: f32 = 1e-3;
    assert_eq!(actual.len(), expected.len(), "{context}: counts differ");
    let boundary = expected.last().map(|(_, s)| *s).unwrap_or(0.0);
    let near = |x: f32, y: f32| (x - y).abs() <= EPS * x.abs().max(1.0);
    for (rank, (a, (eid, escore))) in actual.iter().zip(expected).enumerate() {
        assert!(
            near(a.score, *escore),
            "{context}: rank {rank} score {} vs {escore}",
            a.score
        );
        assert!(
            a.id == *eid || near(a.score, boundary),
            "{context}: rank {rank} doc {} vs {eid} (not a boundary tie)",
            a.id
        );
    }
}

#[test]
fn segmented_search_matches_oracle_with_deletes() {
    let dir = temp_dir("oracle");
    let batches = vec![
        generate_docs(800, 50, 1, "a"),
        generate_docs(600, 50, 2, "b"),
        generate_docs(400, 50, 3, "c"),
    ];
    let deleted = ["a-13", "a-200", "b-5", "b-599", "c-0", "c-399"];

    let mut writer = SegmentedWriter::open_or_create(&dir, true, 2).unwrap();
    for batch in &batches {
        writer.add_documents(batch).unwrap();
    }
    for id in &deleted {
        assert!(writer.delete_document(id).unwrap(), "{id} should exist");
    }
    // Deleting twice reports not-found.
    assert!(!writer.delete_document("a-13").unwrap());

    let index = SegmentedIndex::open(&dir).unwrap();
    assert_eq!(index.num_segments(), 3);
    assert_eq!(index.num_docs_live(), 1800 - 6);

    let mut rng = Lcg(77);
    for q in 0..30 {
        let query = match q % 3 {
            0 => format!("w{}", rng.below(50)),
            1 => format!("w{} w{}", rng.below(50), rng.below(50)),
            _ => format!(
                "w{} w{} w{} w{} w{}",
                rng.below(50),
                rng.below(50),
                rng.below(50),
                rng.below(50),
                rng.below(50)
            ),
        };
        let outcome = index.search(&query, 10);
        let expected = oracle(&batches, &deleted, &query, 10);
        assert_results_match(&outcome.results, &expected, &format!("query={query:?}"));
        // Deleted docs never appear.
        for r in &outcome.results {
            assert!(
                !deleted.contains(&r.id.as_str()),
                "deleted doc {} returned",
                r.id
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn merge_equals_fresh_rebuild_of_live_docs() {
    let dir = temp_dir("merge");
    let batches = vec![
        generate_docs(500, 40, 10, "x"),
        generate_docs(700, 40, 11, "y"),
    ];
    let deleted = ["x-100", "x-499", "y-0", "y-350"];

    let mut writer = SegmentedWriter::open_or_create(&dir, true, 2).unwrap();
    for batch in &batches {
        writer.add_documents(batch).unwrap();
    }
    for id in &deleted {
        assert!(writer.delete_document(id).unwrap());
    }
    writer.merge_all().unwrap();

    let merged = SegmentedIndex::open(&dir).unwrap();
    assert_eq!(merged.num_segments(), 1);
    assert_eq!(merged.num_docs_live(), 1200 - 4);

    // Reference: a from-scratch single index over the live documents in the
    // same order.
    let live: Vec<InputDoc> = batches
        .iter()
        .flatten()
        .filter(|d| !deleted.contains(&d.id.as_str()))
        .cloned()
        .collect();
    let fresh = build_index_weighted(&live, true, 2, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);

    let mut rng = Lcg(5);
    for _ in 0..25 {
        let query = format!("w{} w{}", rng.below(40), rng.below(40));
        let merged_results = merged.search(&query, 10).results;
        let fresh_results = searcher::search(&fresh, &query, 10).results;
        assert_eq!(merged_results.len(), fresh_results.len(), "query={query:?}");
        for (m, f) in merged_results.iter().zip(&fresh_results) {
            assert_eq!(m.id, f.id, "query={query:?}");
            assert!(
                (m.score - f.score).abs() <= 1e-3 * m.score.abs().max(1.0),
                "query={query:?}: {} vs {}",
                m.score,
                f.score
            );
        }
    }

    // Update flow: replace a doc, verify the new content wins.
    let mut writer = SegmentedWriter::open_or_create(&dir, true, 2).unwrap();
    writer
        .update_document(InputDoc {
            id: "x-1".into(),
            title: "zzzuniqueterm in the title".into(),
            body: "zzzuniqueterm zzzuniqueterm body".into(),
        })
        .unwrap();
    let updated = SegmentedIndex::open(&dir).unwrap();
    let hits = updated.search("zzzuniqueterm", 5).results;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "x-1");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn upsert_detects_changes_by_content_hash() {
    let dir = temp_dir("upsert");
    let docs = generate_docs(200, 30, 21, "u");

    let mut writer = SegmentedWriter::open_or_create(&dir, true, 2).unwrap();
    let (added, updated, unchanged) = writer.upsert_documents(&docs).unwrap();
    assert_eq!((added, updated, unchanged), (200, 0, 0));

    // Re-feeding the identical corpus is a no-op.
    let (added, updated, unchanged) = writer.upsert_documents(&docs).unwrap();
    assert_eq!((added, updated, unchanged), (0, 0, 200));
    let index = SegmentedIndex::open(&dir).unwrap();
    assert_eq!(index.num_docs_live(), 200);

    // Change one document, add one new; the rest skip.
    let mut next = docs.clone();
    next[7].body = format!("{} zzzchanged", next[7].body);
    next.push(InputDoc {
        id: "u-new".into(),
        title: "brand new".into(),
        body: "zzzchanged entirely fresh".into(),
    });
    let (added, updated, unchanged) = writer.upsert_documents(&next).unwrap();
    assert_eq!((added, updated, unchanged), (1, 1, 199));

    let index = SegmentedIndex::open(&dir).unwrap();
    assert_eq!(index.num_docs_live(), 201);
    let hits = index.search("zzzchanged", 5).results;
    assert_eq!(hits.len(), 2);
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert!(ids.contains(&"u-7") && ids.contains(&"u-new"));

    // Content hashes survive a merge: re-feeding after compaction still
    // detects everything as unchanged.
    writer.merge_all().unwrap();
    let (added, updated, unchanged) = writer.upsert_documents(&next).unwrap();
    assert_eq!((added, updated, unchanged), (0, 0, 201));

    std::fs::remove_dir_all(&dir).ok();
}
