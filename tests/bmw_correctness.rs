//! Correctness tests for Block-Max WAND against a naive exhaustive BM25
//! reference implementation.
//!
//! The naive implementation lives only here, in test code. It is the
//! correctness oracle: for every query the production Block-Max WAND path
//! must return exactly the same top-k as scoring every matching document.

use std::cmp::Ordering;
use std::collections::HashSet;

use high_performance_search_engine::block_max_wand::{self, SearchHit};
use high_performance_search_engine::bm25;
use high_performance_search_engine::indexer::{build_index, Index, InputDoc, SearchableIndex};
use high_performance_search_engine::maxscore;
use high_performance_search_engine::postings::DEFAULT_BLOCK_SIZE;
use high_performance_search_engine::reorder::ReorderStrategy;

// ---------------------------------------------------------------------------
// Naive BM25 oracle (test-only)
// ---------------------------------------------------------------------------

/// Exhaustively score every document that matches at least one query term,
/// then return the top-k by (score desc, doc_id asc).
fn naive_bm25_top_k(index: &Index, terms: &[String], k: usize) -> Vec<(u32, f32)> {
    let mut scores = vec![0.0f32; index.num_docs()];
    let mut matched = vec![false; index.num_docs()];

    let mut seen = HashSet::new();
    for term in terms {
        if !seen.insert(term.as_str()) {
            continue; // deduplicate, same as the production searcher
        }
        let Some(list) = index.posting_list_for(term) else {
            continue;
        };
        for posting in &list.postings {
            let doc_len = index.doc_len(posting.doc_id);
            scores[posting.doc_id as usize] +=
                bm25::term_contribution(list.idf, posting.tf, doc_len, index.avg_doc_len());
            matched[posting.doc_id as usize] = true;
        }
    }

    let mut hits: Vec<(u32, f32)> = (0..index.num_docs() as u32)
        .filter(|&d| matched[d as usize])
        .map(|d| (d, scores[d as usize]))
        .collect();
    hits.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    hits.truncate(k);
    hits
}

/// Assert BMW and the oracle agree. Scores must match pairwise; document sets
/// must match, except that documents tied (within float tolerance) with the
/// k-th score may legitimately differ between implementations.
fn assert_top_k_equivalent(bmw: &[SearchHit], naive: &[(u32, f32)], context: &str) {
    const EPS: f32 = 1e-3;
    assert_eq!(
        bmw.len(),
        naive.len(),
        "{context}: result counts differ (bmw={}, naive={})",
        bmw.len(),
        naive.len()
    );
    for (rank, (hit, &(naive_doc, naive_score))) in bmw.iter().zip(naive).enumerate() {
        let tolerance = EPS * naive_score.abs().max(1.0);
        assert!(
            (hit.score - naive_score).abs() <= tolerance,
            "{context}: rank {rank} score mismatch: bmw={} ({}), naive={} ({})",
            hit.score,
            hit.doc_id,
            naive_score,
            naive_doc
        );
    }
    let boundary = naive.last().map(|&(_, s)| s).unwrap_or(0.0);
    let bmw_docs: HashSet<u32> = bmw.iter().map(|h| h.doc_id).collect();
    let naive_docs: HashSet<u32> = naive.iter().map(|&(d, _)| d).collect();
    for hit in bmw {
        if !naive_docs.contains(&hit.doc_id) {
            assert!(
                (hit.score - boundary).abs() <= EPS * boundary.abs().max(1.0),
                "{context}: bmw returned doc {} (score {}) missing from naive and not a boundary tie",
                hit.doc_id,
                hit.score
            );
        }
    }
    for &(doc, score) in naive {
        if !bmw_docs.contains(&doc) {
            assert!(
                (score - boundary).abs() <= EPS * boundary.abs().max(1.0),
                "{context}: naive returned doc {doc} (score {score}) missing from bmw and not a boundary tie"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic pseudo-random corpus generation (no rand crate needed)
// ---------------------------------------------------------------------------

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

/// Generate documents over a small vocabulary with a skewed term
/// distribution (low word indices are much more frequent).
fn generate_docs(num_docs: usize, vocab: usize, seed: u64) -> Vec<InputDoc> {
    let mut rng = Lcg(seed);
    (0..num_docs)
        .map(|i| {
            let len = 5 + rng.below(60);
            let body = (0..len)
                .map(|_| {
                    let word = rng.below(vocab).min(rng.below(vocab));
                    format!("w{word}")
                })
                .collect::<Vec<_>>()
                .join(" ");
            InputDoc {
                id: format!("doc-{i}"),
                title: String::new(),
                body,
            }
        })
        .collect()
}

fn terms(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| w.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn bmw_matches_naive_on_handcrafted_corpus() {
    let docs = vec![
        ("d0", "cheap pizza pizza montreal"),
        ("d1", "expensive pizza toronto"),
        ("d2", "cheap tacos montreal montreal"),
        ("d3", "pizza pizza pizza"),
        ("d4", "montreal weather report"),
        ("d5", "cheap cheap cheap flights"),
    ];
    let docs: Vec<InputDoc> = docs
        .into_iter()
        .map(|(id, body)| InputDoc {
            id: id.into(),
            title: String::new(),
            body: body.into(),
        })
        .collect();
    let index = build_index(&docs, false, 2, ReorderStrategy::None); // tiny blocks to exercise block logic

    for query in [
        vec!["pizza"],
        vec!["cheap", "pizza"],
        vec!["cheap", "pizza", "montreal"],
        vec!["montreal"],
        vec!["unknown"],
        vec![],
    ] {
        for k in [1, 2, 3, 10] {
            let query_terms = terms(&query);
            let (bmw, _) = block_max_wand::search(&index, &query_terms, k);
            let naive = naive_bm25_top_k(&index, &query_terms, k);
            assert_top_k_equivalent(&bmw, &naive, &format!("query={query:?} k={k}"));
        }
    }
}

#[test]
fn bmw_matches_naive_on_random_corpora() {
    for seed in [1u64, 42, 2026] {
        for num_docs in [200usize, 1500] {
            let docs = generate_docs(num_docs, 50, seed);
            // Small blocks so even the small corpus has many blocks per list.
            let index = build_index(&docs, true, 16, ReorderStrategy::None);

            let mut rng = Lcg(seed ^ 0xdeadbeef);
            for _ in 0..25 {
                let num_terms = 1 + rng.below(4);
                let query_terms: Vec<String> = (0..num_terms)
                    .map(|_| format!("w{}", rng.below(50)))
                    .collect();
                for k in [1usize, 5, 10] {
                    let (bmw, stats) = block_max_wand::search(&index, &query_terms, k);
                    let naive = naive_bm25_top_k(&index, &query_terms, k);
                    let (ms, _) = maxscore::search(&index, &query_terms, k);
                    assert_top_k_equivalent(
                        &ms,
                        &naive,
                        &format!(
                            "maxscore seed={seed} docs={num_docs} query={query_terms:?} k={k}"
                        ),
                    );
                    assert_top_k_equivalent(
                        &bmw,
                        &naive,
                        &format!("seed={seed} docs={num_docs} query={query_terms:?} k={k}"),
                    );
                    assert_eq!(stats.num_docs_total, num_docs);
                }
            }
        }
    }
}

#[test]
fn bmw_matches_naive_with_default_block_size() {
    let docs = generate_docs(3000, 40, 7);
    let index = build_index(&docs, true, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);
    let mut rng = Lcg(99);
    for _ in 0..20 {
        let query_terms: Vec<String> = (0..1 + rng.below(3))
            .map(|_| format!("w{}", rng.below(40)))
            .collect();
        let (bmw, _) = block_max_wand::search(&index, &query_terms, 10);
        let naive = naive_bm25_top_k(&index, &query_terms, 10);
        assert_top_k_equivalent(&bmw, &naive, &format!("query={query_terms:?}"));
    }
}

/// Corpus engineered so that a rare, high-scoring term fills the heap with
/// scores a common term alone can never reach: WAND must skip blocks of the
/// common term's posting list.
fn skewed_index(num_docs: usize) -> Vec<InputDoc> {
    (0..num_docs)
        .map(|i| {
            // Every doc contains "common" once, padded with unique-ish filler.
            // Every 250th doc also contains "rare" several times in a short doc,
            // making rare+common docs score far above common-only docs.
            let body = if i % 250 == 0 {
                "rare rare rare rare common".to_string()
            } else {
                format!(
                    "common filler{} filler{} padding padding padding",
                    i % 17,
                    i % 31
                )
            };
            InputDoc {
                id: format!("doc-{i}"),
                title: String::new(),
                body,
            }
        })
        .collect()
}

#[test]
fn bmw_skips_blocks_when_threshold_is_high() {
    let docs = skewed_index(20_000);
    let index = build_index(&docs, true, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);
    let query_terms = terms(&["common", "rare"]);

    let (bmw, stats) = block_max_wand::search(&index, &query_terms, 10);
    let naive = naive_bm25_top_k(&index, &query_terms, 10);
    assert_top_k_equivalent(&bmw, &naive, "skewed corpus");

    // The "common" list has 20k postings (~156 blocks of 128). Once the heap
    // is full of rare+common docs, whole blocks of "common" must be skipped.
    assert!(
        stats.num_blocks_skipped > 0,
        "expected block skipping, got stats {stats:?}"
    );
    assert!(
        stats.num_docs_scored < stats.num_docs_total / 2,
        "expected far fewer docs scored than corpus size, got {stats:?}"
    );
    assert!(
        stats.num_postings_visited < 2 * stats.num_docs_total,
        "visited suspiciously many postings: {stats:?}"
    );
}

#[test]
fn selective_queries_do_not_scan_all_documents() {
    let docs = skewed_index(10_000);
    let index = build_index(&docs, true, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);

    // Single rare term: only its own postings can ever be touched. There are
    // only 40 "rare" docs in 10k, and since they all tie on score, BMW may
    // stop scoring as soon as the heap threshold equals that score.
    let (hits, stats) = block_max_wand::search(&index, &terms(&["rare"]), 10);
    assert_eq!(hits.len(), 10);
    assert!(stats.num_docs_scored <= 10_000 / 250); // at most every 250th doc
    assert!(stats.num_docs_scored < stats.num_docs_total);
    assert!(stats.num_postings_visited < stats.num_docs_total);
}

#[test]
fn k_larger_than_matches_returns_all_matches() {
    let docs = generate_docs(100, 30, 5);
    let index = build_index(&docs, true, 8, ReorderStrategy::None);
    let query_terms = terms(&["w29"]); // rarest word in the skewed distribution
    let (bmw, _) = block_max_wand::search(&index, &query_terms, 1000);
    let naive = naive_bm25_top_k(&index, &query_terms, 1000);
    assert_eq!(bmw.len(), naive.len());
    assert_top_k_equivalent(&bmw, &naive, "k larger than matches");
}

#[test]
fn maxscore_matches_naive_on_long_queries() {
    for seed in [3u64, 77] {
        let docs = generate_docs(2500, 40, seed);
        let index = build_index(&docs, true, 32, ReorderStrategy::None);
        let mut rng = Lcg(seed ^ 0xabc);
        for _ in 0..15 {
            let num_terms = 5 + rng.below(6); // 5..10 terms
            let query_terms: Vec<String> = (0..num_terms)
                .map(|_| format!("w{}", rng.below(40)))
                .collect();
            for k in [1usize, 10] {
                let (ms, stats) = maxscore::search(&index, &query_terms, k);
                let naive = naive_bm25_top_k(&index, &query_terms, k);
                assert_top_k_equivalent(
                    &ms,
                    &naive,
                    &format!("seed={seed} query={query_terms:?} k={k}"),
                );
                assert!(stats.num_docs_scored <= stats.num_docs_total);
            }
        }
    }
}
