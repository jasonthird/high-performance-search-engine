//! Integration tests for the compressed, memory-mapped index format and
//! document reordering:
//!
//! - the mmap'd DiskIndex must return *identical* results to the in-memory
//!   index it was saved from (compression is lossless, the search path is
//!   shared),
//! - compression must actually shrink the postings,
//! - reordering must never change search results (it is a pure doc_id
//!   renumbering), and
//! - BP reordering must produce a smaller index than the natural order on a
//!   clusterable corpus.

use std::path::PathBuf;

use high_performance_search_engine::indexer::{build_index, Index, InputDoc, SearchableIndex};
use high_performance_search_engine::postings::{Posting, DEFAULT_BLOCK_SIZE};
use high_performance_search_engine::reorder::ReorderStrategy;
use high_performance_search_engine::searcher;
use high_performance_search_engine::storage::{load_index, save_index, DiskIndex};

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

/// Corpus with two interleaved "topics" using mostly disjoint vocabularies —
/// natural order is the worst case for gap compression, so reordering has
/// room to help.
fn clustered_docs(num_docs: usize, seed: u64) -> Vec<InputDoc> {
    let mut rng = Lcg(seed);
    (0..num_docs)
        .map(|i| {
            let topic = i % 2;
            let len = 10 + rng.below(40);
            let body = (0..len)
                .map(|_| format!("t{}w{}", topic, rng.below(80).min(rng.below(80))))
                .collect::<Vec<_>>()
                .join(" ");
            InputDoc {
                // Path-like ids: topic = directory.
                id: format!("/home/user/topic{}/file{:05}.txt", topic, i),
                title: format!("doc {i}"),
                body,
            }
        })
        .collect()
}

fn temp_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("high-performance-search-engine-{tag}-{}", std::process::id()))
}

fn save_load(index: &Index, tag: &str) -> (DiskIndex, u64, PathBuf) {
    let dir = temp_dir(tag);
    let size = save_index(index, &dir).unwrap();
    let disk = load_index(&dir).unwrap();
    (disk, size, dir)
}

/// (external_id, score) pairs — comparable across different doc_id orders.
fn results_by_external_id<I: SearchableIndex>(
    index: &I,
    query: &str,
    k: usize,
) -> Vec<(String, f32)> {
    searcher::search(index, query, k)
        .results
        .into_iter()
        .map(|r| (r.id, r.score))
        .collect()
}

/// Results must match pairwise on score. Document *sets* must match, except
/// for documents tied (within float tolerance) with the k-th score: ties are
/// broken by internal doc_id, which reordering legitimately changes, so a
/// tie group cut by the k boundary may contribute different members — and
/// fully included tie groups may appear in a different order.
fn assert_same_results(a: &[(String, f32)], b: &[(String, f32)], context: &str) {
    const EPS: f32 = 1e-3;
    assert_eq!(a.len(), b.len(), "{context}: result counts differ");
    let near = |x: f32, y: f32| (x - y).abs() <= EPS * x.abs().max(1.0);
    for (rank, ((_, score_a), (_, score_b))) in a.iter().zip(b).enumerate() {
        assert!(
            near(*score_a, *score_b),
            "{context}: rank {rank} score differs: {score_a} vs {score_b}"
        );
    }
    let boundary = a.last().map(|(_, s)| *s).unwrap_or(0.0);
    let ids_a: std::collections::HashSet<&String> = a.iter().map(|(id, _)| id).collect();
    let ids_b: std::collections::HashSet<&String> = b.iter().map(|(id, _)| id).collect();
    for (id, score) in a.iter().filter(|(id, _)| !ids_b.contains(id)) {
        assert!(
            near(*score, boundary),
            "{context}: doc {id} (score {score}) missing from other run and not a boundary tie"
        );
    }
    for (id, score) in b.iter().filter(|(id, _)| !ids_a.contains(id)) {
        assert!(
            near(*score, boundary),
            "{context}: doc {id} (score {score}) extra in other run and not a boundary tie"
        );
    }
}

const TEST_QUERIES: &[&str] = &[
    "t0w1 t0w5",
    "t1w2",
    "t0w0 t1w0 t0w3",
    "t1w10 t1w11 t1w12",
    "t0w70",
    "missing terms only",
];

#[test]
fn disk_index_returns_identical_results_to_memory_index() {
    let docs = clustered_docs(3000, 11);
    let index = build_index(&docs, true, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);
    let (disk, _, dir) = save_load(&index, "equiv");

    for query in TEST_QUERIES {
        for k in [1, 5, 20] {
            let mem = results_by_external_id(&index, query, k);
            let dsk = results_by_external_id(&disk, query, k);
            assert_same_results(&mem, &dsk, &format!("query={query:?} k={k}"));
        }
    }

    // The stats counters must agree too: the disk path skips exactly the
    // same blocks the in-memory path does.
    let mem_stats = searcher::search(&index, "t0w1 t0w5", 10).stats;
    let dsk_stats = searcher::search(&disk, "t0w1 t0w5", 10).stats;
    assert_eq!(mem_stats.num_docs_scored, dsk_stats.num_docs_scored);
    assert_eq!(
        mem_stats.num_postings_visited,
        dsk_stats.num_postings_visited
    );
    assert_eq!(mem_stats.num_blocks_skipped, dsk_stats.num_blocks_skipped);

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn compression_shrinks_postings() {
    let docs = clustered_docs(5000, 23);
    let index = build_index(&docs, true, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);
    let dir = temp_dir("size");
    save_index(&index, &dir).unwrap();

    let postings_bytes = std::fs::metadata(dir.join("postings.bin")).unwrap().len();
    let uncompressed = index.total_postings() * std::mem::size_of::<Posting>() as u64;
    assert!(
        postings_bytes * 2 < uncompressed,
        "expected < 50% of uncompressed size: {postings_bytes} vs {uncompressed}"
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn reordering_never_changes_search_results() {
    let docs = clustered_docs(2000, 37);
    let baseline = build_index(&docs, true, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);

    for strategy in [ReorderStrategy::Path, ReorderStrategy::Bp] {
        let reordered = build_index(&docs, true, DEFAULT_BLOCK_SIZE, strategy);
        assert_eq!(reordered.num_docs(), baseline.num_docs());
        assert_eq!(reordered.total_postings(), baseline.total_postings());
        for query in TEST_QUERIES {
            let expected = results_by_external_id(&baseline, query, 10);
            let actual = results_by_external_id(&reordered, query, 10);
            assert_same_results(&expected, &actual, &format!("{strategy:?} query={query:?}"));
        }
    }
}

#[test]
fn bp_reordering_shrinks_the_index() {
    let docs = clustered_docs(6000, 5);

    let mut sizes = std::collections::HashMap::new();
    for (tag, strategy) in [
        ("none", ReorderStrategy::None),
        ("path", ReorderStrategy::Path),
        ("bp", ReorderStrategy::Bp),
    ] {
        let index = build_index(&docs, true, DEFAULT_BLOCK_SIZE, strategy);
        let dir = temp_dir(&format!("reorder-{tag}"));
        save_index(&index, &dir).unwrap();
        let bytes = std::fs::metadata(dir.join("postings.bin")).unwrap().len();
        sizes.insert(tag, bytes);
        std::fs::remove_dir_all(dir).ok();
    }

    // The corpus interleaves two topics, so both path-sorting (ids encode the
    // topic directory) and BP should beat natural order. BP optimizes the
    // gap cost directly and should do at least as well as the heuristic.
    assert!(
        sizes["path"] < sizes["none"],
        "path reordering should shrink postings: {sizes:?}"
    );
    assert!(
        sizes["bp"] < sizes["none"],
        "bp reordering should shrink postings: {sizes:?}"
    );
}
