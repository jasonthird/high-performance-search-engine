//! Throughput of the segmented brute-force semantic scan, before/after
//! range-level parallelism. Synthetic vectors: scan cost does not depend
//! on their values.

use std::time::Instant;

use high_performance_search_engine::embeddings::{self, EmbeddingStore};
use high_performance_search_engine::hybrid::{segmented_semantic_pool, SegmentStores};

fn main() -> anyhow::Result<()> {
    let dim = 768usize;
    let n = 100_000usize;
    let dir = std::env::temp_dir().join("hips-seg-scan-bench");
    std::fs::create_dir_all(&dir)?;

    let mut state = 0x12345678u64;
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect();
        embeddings::l2_normalize(&mut v);
        vectors.push(v);
    }
    embeddings::write_f16(&dir, dim as u32, &vectors)?;
    let store = EmbeddingStore::open(&dir)?;
    let stores = SegmentStores {
        stores: vec![Some(store)],
    };
    let query = vectors[0].clone();
    let live = |_si: usize, _doc: u32| true;

    // Warm the page cache.
    let _ = segmented_semantic_pool(&stores, &live, &query, 200);
    let runs = 20;
    let t = Instant::now();
    for _ in 0..runs {
        let hits = segmented_semantic_pool(&stores, &live, &query, 200);
        std::hint::black_box(&hits);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
    println!(
        "{n} docs, one segment: {ms:.2} ms/scan  ({:.2} M docs/s)",
        n as f64 / ms / 1000.0
    );
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
