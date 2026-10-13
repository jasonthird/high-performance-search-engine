//! Scoped retrieval (the MCP `path_glob`): the scope is applied inside the
//! evaluators, so a scoped top-k is the exact top-k among in-scope
//! documents — never a global top-k filtered afterwards — and hybrid score
//! normalization sees in-scope candidates only.

use std::path::PathBuf;

use high_performance_search_engine::indexer::{build_index, InputDoc};
use high_performance_search_engine::reorder::ReorderStrategy;
use high_performance_search_engine::searcher::AnyIndex;
use high_performance_search_engine::segments::{SegmentedIndex, SegmentedWriter};
use high_performance_search_engine::storage;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "high-performance-search-engine-scope-{tag}-{}",
        std::process::id()
    ));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `needle` occurs in every document; the `tail/` files repeat it, so an
/// unscoped top-k is all tails and the `src/` hits sit below any cutoff.
fn corpus() -> Vec<Vec<InputDoc>> {
    let mut batches = Vec::new();
    for b in 0..3 {
        let mut batch = Vec::new();
        for i in 0..40 {
            let n = b * 40 + i;
            let (dir, reps) = if i % 4 == 0 { ("src", 1) } else { ("tail", 6) };
            batch.push(InputDoc {
                id: format!("{dir}/file{n:03}.rs:{}-{}", n + 1, n + 3),
                title: format!("{dir}/file{n:03}.rs::item{n}"),
                body: format!("{} filler{n} more words here", "needle ".repeat(reps)),
            });
        }
        batches.push(batch);
    }
    batches
}

fn in_src(path: &str) -> bool {
    high_performance_search_engine::repo::glob_match("src/**", path)
}

fn id_in_src(id: &str) -> bool {
    high_performance_search_engine::repo::parse_id(id).is_some_and(|(path, _, _)| in_src(path))
}

#[test]
fn scoped_lexical_top_k_equals_the_exhaustive_in_scope_ranking() {
    let dir = temp_dir("lexical");
    let batches = corpus();
    {
        let mut writer = SegmentedWriter::open_or_create_ex(&dir, true, 2, true).unwrap();
        for batch in &batches {
            writer.add_documents(batch).unwrap();
        }
        // A tombstoned in-scope doc must not count or appear.
        assert!(writer.delete_document("src/file004.rs:5-7").unwrap());
    }
    let index = AnyIndex::open(&dir).unwrap();
    let scope = index.scope(&in_src);
    assert_eq!(scope.matched(), 30 - 1);

    // Unscoped: every top hit is a tail; nothing from src/ within k=10.
    let global = index.search("needle", 10);
    assert!(global.results.iter().all(|r| r.id.starts_with("tail/")));

    // Scoped: exactly the in-scope prefix of the exhaustive ranking. Hiding
    // documents changes no score (global statistics are unchanged), so the
    // oracle is the full ranking restricted to the scope.
    let exhaustive = index.search("needle", 1000);
    let oracle: Vec<&str> = exhaustive
        .results
        .iter()
        .map(|r| r.id.as_str())
        .filter(|id| id_in_src(id))
        .take(10)
        .collect();
    let scoped = index.search_scoped("needle", 10, Some(&scope));
    let got: Vec<&str> = scoped.results.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(got, oracle);
    assert_eq!(got.len(), 10);
    assert!(!got.contains(&"src/file004.rs:5-7"), "tombstoned doc hidden");
    for r in &scoped.results {
        let plain = exhaustive.results.iter().find(|x| x.id == r.id).unwrap();
        assert_eq!(r.score, plain.score, "scoping must not change scores");
    }

    // Multi-term queries take the MaxScore path; same contract.
    let long = "needle filler001 filler005 filler009 more words";
    let scoped = index.search_scoped(long, 5, Some(&scope));
    let oracle: Vec<String> = index
        .search(long, 1000)
        .results
        .into_iter()
        .map(|r| r.id)
        .filter(|id| id_in_src(id))
        .take(5)
        .collect();
    assert_eq!(scoped.results.iter().map(|r| r.id.clone()).collect::<Vec<_>>(), oracle);

    // An empty scope yields nothing, not a fallback to the global ranking.
    let nothing = index.scope(&|_| false);
    assert_eq!(nothing.matched(), 0);
    assert!(index.search_scoped("needle", 10, Some(&nothing)).results.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn single_layout_scopes_too() {
    let dir = temp_dir("single");
    let docs: Vec<InputDoc> = corpus().into_iter().flatten().collect();
    let index = build_index(&docs, true, 128, ReorderStrategy::None);
    storage::save_index(&index, &dir).unwrap();
    let any = AnyIndex::open(&dir).unwrap();
    let scope = any.scope(&in_src);
    assert_eq!(scope.matched(), 30);
    let scoped = any.search_scoped("needle", 50, Some(&scope));
    assert_eq!(scoped.results.len(), 30);
    assert!(scoped.results.iter().all(|r| r.id.starts_with("src/")));
    let oracle: Vec<String> = any
        .search("needle", 1000)
        .results
        .into_iter()
        .map(|r| r.id)
        .filter(|id| id_in_src(id))
        .collect();
    assert_eq!(scoped.results.iter().map(|r| r.id.clone()).collect::<Vec<_>>(), oracle);
    std::fs::remove_dir_all(&dir).ok();
}

/// Sparse, dense and empty scopes against the exhaustive oracle, on both
/// evaluators, with tombstones interleaved: the leap over hidden documents
/// must never skip an admitted one.
#[test]
fn scoped_search_matches_the_oracle_for_random_scopes() {
    let dir = temp_dir("oracle");
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let vocab = ["alpha", "beta", "gamma", "delta", "needle", "omega", "sigma", "tau"];
    let mut docs = Vec::new();
    for n in 0..3000u64 {
        let len = 3 + (next() % 12) as usize;
        let body: Vec<&str> = (0..len).map(|_| vocab[(next() % vocab.len() as u64) as usize]).collect();
        let file = n / 7; // seven consecutive chunks per file
        docs.push(InputDoc {
            id: format!("d{}/f{file:04}.rs:{}-{}", file % 3, n * 3 + 1, n * 3 + 3),
            title: format!("d{}/f{file:04}.rs::c{n}", file % 3),
            body: body.join(" "),
        });
    }
    {
        let mut writer = SegmentedWriter::open_or_create_ex(&dir, true, 2, true).unwrap();
        for batch in docs.chunks(1000) {
            writer.add_documents(batch).unwrap();
        }
        for n in (0..3000u64).step_by(11) {
            let id = &docs[n as usize].id;
            assert!(writer.delete_document(id).unwrap());
        }
    }
    let deleted: std::collections::HashSet<&str> =
        (0..3000u64).step_by(11).map(|n| docs[n as usize].id.as_str()).collect();
    let index = AnyIndex::open(&dir).unwrap();
    let queries = [
        "needle",                              // one term: WAND
        "alpha omega",                         // two terms: WAND
        "alpha beta gamma delta needle omega", // six terms: MaxScore
    ];
    type Accept = Box<dyn Fn(&str) -> bool + Sync>;
    let scopes: Vec<(&str, Accept)> = vec![
        ("two files in 429", Box::new(|p: &str| p.ends_with("f0400.rs") || p.ends_with("f0007.rs"))),
        ("a third of the files", Box::new(|p: &str| p.starts_with("d1/"))),
        ("everything", Box::new(|_: &str| true)),
        ("nothing", Box::new(|_: &str| false)),
        ("every 50th file", Box::new(|p: &str| p[4..8].parse::<u32>().unwrap() % 50 == 0)),
    ];
    for (label, accept) in &scopes {
        let scope = index.scope(accept.as_ref());
        for q in queries {
            for k in [1usize, 5, 40] {
                let got: Vec<String> = index
                    .search_scoped(q, k, Some(&scope))
                    .results
                    .into_iter()
                    .map(|r| r.id)
                    .collect();
                let oracle: Vec<String> = index
                    .search(q, 10_000)
                    .results
                    .into_iter()
                    .map(|r| r.id)
                    .filter(|id| {
                        let path = high_performance_search_engine::repo::parse_id(id).unwrap().0;
                        accept(path) && !deleted.contains(id.as_str())
                    })
                    .take(k)
                    .collect();
                assert_eq!(got, oracle, "scope {label:?}, query {q:?}, k={k}");
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn spelling_correction_still_applies_inside_a_scope() {
    let dir = temp_dir("spell");
    {
        let mut writer = SegmentedWriter::open_or_create_ex(&dir, true, 2, true).unwrap();
        writer.add_documents(&corpus().concat()).unwrap();
    }
    let index = AnyIndex::open(&dir).unwrap();
    let scope = index.scope(&in_src);
    let outcome = index.search_scoped("neddle", 3, Some(&scope));
    assert_eq!(outcome.corrected.as_deref(), Some("needle"));
    assert_eq!(outcome.results.len(), 3);
    assert!(outcome.results.iter().all(|r| r.id.starts_with("src/")));
    // The raw segmented API exposes the same scope.
    let seg = SegmentedIndex::open(&dir).unwrap();
    let (hits, _) = seg.search_hits_raw_scoped("needle", 5, Some(&scope));
    assert_eq!(hits.len(), 5);
    assert!(hits.iter().all(|&(si, doc, _)| scope.contains(si, doc)));
    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(feature = "semantic")]
mod hybrid {
    use super::*;
    use high_performance_search_engine::embeddings::{l2_normalize, write_f16};
    use high_performance_search_engine::query::{
        self, FusionArg, PqMode, QueryEncoder, RankMode, SearchOpts,
    };

    /// A query encoder that returns one fixed vector: the test controls
    /// the geometry, no model is loaded.
    struct Fixed(Vec<f32>);

    impl QueryEncoder for Fixed {
        fn embed_query(&self, _query: &str) -> anyhow::Result<Vec<f32>> {
            Ok(self.0.clone())
        }
    }

    fn unit(v: [f32; 4]) -> Vec<f32> {
        let mut v = v.to_vec();
        l2_normalize(&mut v);
        v
    }

    fn opts(mode: RankMode) -> SearchOpts {
        SearchOpts {
            mode,
            semantic_candidates: 20,
            fusion: FusionArg::Weighted,
            alpha: 0.15,
            rrf_k: 60.0,
            nprobe: 0,
            pq: PqMode::Off,
            ann_ef: 128,
            exact_vectors: true,
        }
    }

    /// Three documents over two segments:
    /// - `src/a.rs`   in scope, the only in-scope lexical match, orthogonal to the query;
    /// - `tail/b.rs`  out of scope, strongest on both signals;
    /// - `src/c.rs`   in scope, no lexical match, cosine 0.6.
    fn build(dir: &std::path::Path) -> AnyIndex {
        let docs = [
            InputDoc { id: "src/a.rs:1-2".into(), title: "src/a.rs::a".into(), body: "needle alpha".into() },
            InputDoc { id: "tail/b.rs:1-2".into(), title: "tail/b.rs::b".into(), body: "needle needle needle beta".into() },
            InputDoc { id: "src/c.rs:1-2".into(), title: "src/c.rs::c".into(), body: "gamma only".into() },
        ];
        let vectors = [unit([0.0, 1.0, 0.0, 0.0]), unit([1.0, 0.0, 0.0, 0.0]), unit([0.6, 0.8, 0.0, 0.0])];
        let mut writer = SegmentedWriter::open_or_create_ex(dir, true, 2, true).unwrap();
        let first = writer.add_documents(&docs[..2]).unwrap();
        let second = writer.add_documents(&docs[2..]).unwrap();
        drop(writer);
        write_f16(&dir.join(first), 4, &vectors[..2]).unwrap();
        write_f16(&dir.join(second), 4, &vectors[2..]).unwrap();
        AnyIndex::open(dir).unwrap()
    }

    #[test]
    fn hybrid_scope_restricts_both_signals_and_normalizes_within_scope() {
        let dir = temp_dir("hybrid");
        let index = build(&dir);
        let encoder = Fixed(unit([1.0, 0.0, 0.0, 0.0]));
        let scope = index.scope(&in_src);
        assert_eq!(scope.matched(), 2);

        let global = query::run_ranked_with(&index, &encoder, "needle", 3, &opts(RankMode::Hybrid)).unwrap();
        assert_eq!(global.results[0].id, "tail/b.rs:1-2", "unscoped, the tail wins both signals");

        let scoped = query::run_ranked_scoped(&index, &encoder, "needle", 3, &opts(RankMode::Hybrid), Some(&scope)).unwrap();
        let ids: Vec<&str> = scoped.results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["src/c.rs:1-2", "src/a.rs:1-2"], "exactly the in-scope documents, best first");
        // `a` is the only lexical match in scope, so its BM25 normalizes to
        // 1.0 (score alpha). Normalized against the hidden tail it would
        // collapse to 0.0 — the post-filter behaviour this replaces.
        let a = &scoped.results[1];
        assert!(a.bm25 > 0.0 && a.semantic.abs() < 1e-3, "{a:?}", a = (a.bm25, a.semantic));
        assert!((a.score - 0.15).abs() < 1e-3, "score {}", a.score);
        let c = &scoped.results[0];
        assert!((c.score - 0.85 * 0.6).abs() < 1e-2, "score {}", c.score);

        for mode in [RankMode::Semantic, RankMode::Rerank] {
            let run = query::run_ranked_scoped(&index, &encoder, "needle", 3, &opts(mode), Some(&scope)).unwrap();
            assert!(!run.results.is_empty(), "{mode:?}");
            assert!(run.results.iter().all(|r| r.id.starts_with("src/")), "{mode:?}: {:?}",
                run.results.iter().map(|r| &r.id).collect::<Vec<_>>());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn single_layout_hybrid_scope() {
        let dir = temp_dir("hybrid-single");
        let docs = [
            InputDoc { id: "src/a.rs:1-2".into(), title: "src/a.rs::a".into(), body: "needle alpha".into() },
            InputDoc { id: "tail/b.rs:1-2".into(), title: "tail/b.rs::b".into(), body: "needle needle needle beta".into() },
            InputDoc { id: "src/c.rs:1-2".into(), title: "src/c.rs::c".into(), body: "gamma only".into() },
        ];
        let index = build_index(&docs, true, 128, ReorderStrategy::None);
        storage::save_index(&index, &dir).unwrap();
        let vectors = [unit([0.0, 1.0, 0.0, 0.0]), unit([1.0, 0.0, 0.0, 0.0]), unit([0.6, 0.8, 0.0, 0.0])];
        write_f16(&dir, 4, &vectors).unwrap();
        let any = AnyIndex::open(&dir).unwrap();
        let encoder = Fixed(unit([1.0, 0.0, 0.0, 0.0]));
        let scope = any.scope(&in_src);
        for mode in [RankMode::Hybrid, RankMode::Semantic, RankMode::Rerank] {
            let run = query::run_ranked_scoped(&any, &encoder, "needle", 3, &opts(mode), Some(&scope)).unwrap();
            assert!(!run.results.is_empty(), "{mode:?}");
            assert!(run.results.iter().all(|r| r.id.starts_with("src/")), "{mode:?}");
        }
        let hybrid = query::run_ranked_scoped(&any, &encoder, "needle", 3, &opts(RankMode::Hybrid), Some(&scope)).unwrap();
        assert_eq!(hybrid.results[0].id, "src/c.rs:1-2");
        assert!((hybrid.results[1].score - 0.15).abs() < 1e-3);
        std::fs::remove_dir_all(&dir).ok();
    }
}
