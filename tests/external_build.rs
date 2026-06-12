//! The external (sharded, spill-to-disk) builder must produce an index that
//! answers every query identically to the in-memory builder's output.

use std::io::Write;
use std::path::PathBuf;

use high_performance_search_engine::external::build_index_external;
use high_performance_search_engine::indexer::{build_index_weighted, InputDoc, SearchableIndex};
use high_performance_search_engine::postings::DEFAULT_BLOCK_SIZE;
use high_performance_search_engine::reorder::ReorderStrategy;
use high_performance_search_engine::searcher;
use high_performance_search_engine::storage::{load_index, save_index};

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

fn generate_docs(num_docs: usize, vocab: usize, seed: u64) -> Vec<InputDoc> {
    let mut rng = Lcg(seed);
    (0..num_docs)
        .map(|i| {
            let len = 5 + rng.below(60);
            let body = (0..len)
                .map(|_| format!("w{}", rng.below(vocab).min(rng.below(vocab))))
                .collect::<Vec<_>>()
                .join(" ");
            InputDoc {
                id: format!("doc-{i}"),
                title: format!("title w{}", rng.below(vocab)),
                body,
            }
        })
        .collect()
}

fn temp_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("high-performance-search-engine-ext-{tag}-{}", std::process::id()))
}

#[test]
fn external_build_matches_in_memory_build() {
    let docs = generate_docs(18000, 60, 99);

    // In-memory reference.
    let mem_dir = temp_dir("mem");
    let index = build_index_weighted(&docs, true, 2, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);
    save_index(&index, &mem_dir).unwrap();
    let mem = load_index(&mem_dir).unwrap();

    // External build from a JSONL stream, with a spill budget small enough
    // to force many shards.
    let mut jsonl = Vec::new();
    for doc in &docs {
        writeln!(
            jsonl,
            r#"{{"id":"{}","title":"{}","body":"{}"}}"#,
            doc.id, doc.title, doc.body
        )
        .unwrap();
    }
    let ext_dir = temp_dir("ext");
    let stats = build_index_external(
        &mut jsonl.as_slice(),
        &ext_dir,
        true,
        2,
        DEFAULT_BLOCK_SIZE,
        10_000, // tiny budget: many spills
    )
    .unwrap();
    assert!(stats.num_shards > 3, "expected several shards");
    let ext = load_index(&ext_dir).unwrap();

    assert_eq!(ext.num_docs(), mem.num_docs());
    assert_eq!(ext.num_terms(), mem.num_terms());
    assert_eq!(ext.total_postings(), mem.total_postings());
    assert_eq!(stats.num_postings, mem.total_postings());
    assert!((ext.avg_doc_len() - mem.avg_doc_len()).abs() < 1e-6);

    let mut rng = Lcg(7);
    for q in 0..40 {
        let query = match q % 4 {
            0 => format!("w{}", rng.below(60)),
            1 => format!("w{} w{}", rng.below(60), rng.below(60)),
            2 => format!("w{} w{} w{}", rng.below(60), rng.below(60), rng.below(60)),
            _ => "missing terms only".to_string(),
        };
        let a = searcher::search(&mem, &query, 10);
        let b = searcher::search(&ext, &query, 10);
        assert_eq!(a.results.len(), b.results.len(), "query {query:?}");
        for (x, y) in a.results.iter().zip(&b.results) {
            assert_eq!(x.id, y.id, "query {query:?}");
            assert!((x.score - y.score).abs() < 1e-4, "query {query:?}");
        }
        assert_eq!(
            a.stats.num_docs_scored, b.stats.num_docs_scored,
            "query {query:?}"
        );
    }

    // Doc summaries resolve identically through the streamed doc store.
    assert_eq!(ext.doc_summary(0).id, mem.doc_summary(0).id);
    assert_eq!(ext.doc_summary(17999).title, mem.doc_summary(17999).title);

    std::fs::remove_dir_all(mem_dir).ok();
    std::fs::remove_dir_all(ext_dir).ok();
}
