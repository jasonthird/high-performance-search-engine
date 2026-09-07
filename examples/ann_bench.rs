//! HNSW vs exact scan, with held-out vectors and measured dot-product counts.
//! ann_bench <embedding-directory|synthetic> [sizes] [queries] [pool]
//! Existing source indexes are read-only; all benchmark artifacts are temporary.
use high_performance_search_engine::{
    embeddings::{self, EmbeddingStore},
    hnsw::HnswIndex,
    hybrid::{segmented_semantic_pool, segmented_semantic_pool_with_options, SegmentStores},
};
use std::{collections::HashSet, path::Path, time::Instant};

fn random(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 40) as f32 / 8388608.0 - 1.0
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let source = args.next().unwrap_or_else(|| "synthetic".into());
    let sizes = args.next().unwrap_or_else(|| "10000,100000".into());
    let queries: usize = args.next().unwrap_or_else(|| "24".into()).parse()?;
    let pool: usize = args.next().unwrap_or_else(|| "10".into()).parse()?;
    anyhow::ensure!(queries > 0 && pool > 0, "queries and pool must be positive");
    let source = if source == "synthetic" {
        None
    } else {
        Some(EmbeddingStore::open(Path::new(&source))?)
    };
    println!("docs,dim,pool,ef,build_s,graph_mib,exact_p50_ms,ann_p50_ms,ann_p95_ms,recall_at_10,recall_at_pool,mean_dot_products,dot_products_per_doc,fallbacks");
    for size in sizes.split(',') {
        bench(source.as_ref(), size.parse()?, queries, pool)?;
    }
    Ok(())
}

fn bench(source: Option<&EmbeddingStore>, n: usize, nq: usize, pool: usize) -> anyhow::Result<()> {
    anyhow::ensure!(n >= pool && n <= u32::MAX as usize, "invalid corpus size");
    let dim = source.map_or(768, |s| s.dim() as usize);
    let mut training = Vec::with_capacity(n);
    let mut queries = Vec::with_capacity(nq);
    if let Some(source) = source {
        anyhow::ensure!(
            source.num_docs() as usize >= n + nq,
            "source has too few rows"
        );
        let held_out: HashSet<_> = (0..nq).map(|i| i * (n + nq) / nq).collect();
        for id in 0..(n + nq) {
            let mut v = vec![0.0; dim];
            source.copy_f32(id as u32, &mut v);
            if held_out.contains(&id) {
                queries.push(v);
            } else {
                training.push(v);
            }
        }
    } else {
        // Clustered synthetic embeddings; useful for scaling, not a claim about
        // real CodeRankEmbed recall. Queries are independent samples, not rows.
        let mut state = 19u64;
        let centers: Vec<Vec<_>> = (0..128)
            .map(|_| (0..dim).map(|_| random(&mut state)).collect())
            .collect();
        for i in 0..(n + nq) {
            let center = &centers[i % centers.len()];
            let mut v: Vec<_> = center
                .iter()
                .map(|&x| x + 0.25 * random(&mut state))
                .collect();
            embeddings::l2_normalize(&mut v);
            if i < n {
                training.push(v);
            } else {
                queries.push(v);
            }
        }
    }
    let dir = tempfile::tempdir()?;
    embeddings::write_f16(dir.path(), dim as u32, &training)?;
    drop(training);
    let store = EmbeddingStore::open(dir.path())?;
    let t = Instant::now();
    let graph = HnswIndex::build(&store)?;
    graph.save(dir.path())?;
    let build_s = t.elapsed().as_secs_f64();
    drop(graph);
    let graph_mib = std::fs::metadata(dir.path().join(high_performance_search_engine::hnsw::FILE))?
        .len() as f64
        / 1048576.0;
    let stores = SegmentStores {
        stores: vec![Some(store)],
    };
    let live = |_, _| true;
    let mut truth = Vec::new();
    let mut exact_ms = Vec::new();
    segmented_semantic_pool(&stores, &live, &queries[0], pool);
    for q in &queries {
        let t = Instant::now();
        let hits = segmented_semantic_pool(&stores, &live, q, pool);
        exact_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        truth.push(hits.into_iter().map(|h| h.doc_id).collect::<Vec<_>>());
    }
    exact_ms.sort_by(f64::total_cmp);
    // Exclude graph loading from warm-query timing, as with the exact mmap scan.
    stores.stores[0]
        .as_ref()
        .unwrap()
        .hnsw()
        .expect("fresh valid graph");
    for ef in [64, 128, 256, 512] {
        let mut latency = Vec::new();
        let (mut recall10, mut recall_pool, mut work, mut fallback) = (0.0, 0.0, 0, 0);
        for (q, expected) in queries.iter().zip(&truth) {
            let t = Instant::now();
            let (hits, stats) =
                segmented_semantic_pool_with_options(&stores, &live, q, pool, ef, false);
            latency.push(t.elapsed().as_secs_f64() * 1000.0);
            let wanted: HashSet<_> = expected.iter().copied().collect();
            let top = pool.min(10);
            let wanted10: HashSet<_> = expected[..top].iter().copied().collect();
            recall10 += hits
                .iter()
                .take(top)
                .filter(|h| wanted10.contains(&h.doc_id))
                .count() as f64
                / top as f64;
            recall_pool +=
                hits.iter().filter(|h| wanted.contains(&h.doc_id)).count() as f64 / pool as f64;
            work += stats.distance_computations;
            fallback += stats.graph_fallbacks;
        }
        latency.sort_by(f64::total_cmp);
        println!("{n},{dim},{pool},{ef},{build_s:.3},{graph_mib:.3},{:.3},{:.3},{:.3},{:.4},{:.4},{:.1},{:.4},{fallback}",
            exact_ms[(nq - 1) / 2], latency[(nq - 1) / 2], latency[(nq * 95).div_ceil(100) - 1],
            recall10 / nq as f64, recall_pool / nq as f64, work as f64 / nq as f64, work as f64 / nq as f64 / n as f64);
    }
    Ok(())
}
