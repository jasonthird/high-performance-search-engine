//! Size sweep for the exact segmented semantic scan. Synthetic normalized
//! vectors measure scan throughput, not real-query relevance or ANN recall.
//! Run: cargo run --release --example seg_scan_bench -- 10000,100000,1000000 10

use std::time::Instant;

use high_performance_search_engine::embeddings::{self, EmbeddingStore};
use high_performance_search_engine::hybrid::{segmented_semantic_pool, SegmentStores};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let sizes = args.next().unwrap_or_else(|| "10000,100000".into());
    let runs: usize = args.next().unwrap_or_else(|| "10".into()).parse()?;
    anyhow::ensure!(runs > 0, "runs must be positive");
    println!("docs,vector_mib,pool,scored_vectors,p50_ms,p95_ms,million_docs_per_s");
    for size in sizes.split(',') {
        let n: usize = size.parse()?;
        anyhow::ensure!(n > 0 && n <= u32::MAX as usize, "invalid document count");
        bench(n, runs)?;
    }
    Ok(())
}

fn bench(n: usize, runs: usize) -> anyhow::Result<()> {
    let dim = 768usize;
    let dir = tempfile::tempdir()?;

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
    embeddings::write_f16(dir.path(), dim as u32, &vectors)?;
    let query = vectors[0].clone();
    drop(vectors);
    let store = EmbeddingStore::open(dir.path())?;
    let vector_mib = store.size_bytes() as f64 / (1024.0 * 1024.0);
    let stores = SegmentStores {
        stores: vec![Some(store)],
    };
    let live = |_si: usize, _doc: u32| true;

    // Warm the page cache.
    let _ = segmented_semantic_pool(&stores, &live, &query, 200);
    let mut elapsed = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let hits = segmented_semantic_pool(&stores, &live, &query, 200);
        std::hint::black_box(&hits);
        elapsed.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    elapsed.sort_by(f64::total_cmp);
    let p50 = elapsed[(runs - 1) / 2];
    let p95 = elapsed[(runs * 95).div_ceil(100) - 1];
    println!(
        "{n},{vector_mib:.2},200,{n},{p50:.3},{p95:.3},{:.3}",
        n as f64 / p50 / 1000.0
    );
    Ok(())
}
