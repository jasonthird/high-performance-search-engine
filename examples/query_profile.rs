//! Profile uncached and cached ranked searches separately on a read-only index.
//! query_profile INDEX QUERIES [OUTPUT_JSON]
#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("requires --features semantic");
}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use high_performance_search_engine::{
        embedder::Embedder,
        query::{run_ranked_with, FusionArg, PqMode, RankMode, SearchOpts},
        searcher::AnyIndex,
    };
    use std::{path::Path, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!(args.len() >= 2, "query_profile INDEX QUERIES [OUTPUT_JSON]");
    let t = Instant::now();
    let index = AnyIndex::open(Path::new(&args[0]))?;
    let index_load_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let encoder = Embedder::load()?;
    let encoder_load_ms = t.elapsed().as_secs_f64() * 1000.0;
    let queries = std::fs::read_to_string(&args[1])?;
    let opts = SearchOpts {
        mode: RankMode::Hybrid,
        semantic_candidates: 200,
        fusion: FusionArg::Weighted,
        alpha: 0.15,
        rrf_k: 60.0,
        nprobe: 0,
        pq: PqMode::Off,
        ann_ef: 128,
        exact_vectors: false,
    };
    let t = Instant::now();
    // Warm both short and maximum-length shapes plus index validation using
    // queries outside the measured list. Measured pass 0 still misses the LRU.
    run_ranked_with(&index, &encoder, "profiling warmup", 10, &opts)?;
    run_ranked_with(&index, &encoder, &"profiling warmup ".repeat(40), 10, &opts)?;
    let warmup_ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut rows = Vec::new();
    for pass in 0..2 {
        for query in queries.lines().filter(|q| !q.trim().is_empty()) {
            let cached = encoder.query_cached(query);
            let run = run_ranked_with(&index, &encoder, query, 10, &opts)?;
            rows.push(
                serde_json::json!({"pass": pass, "query": query, "cached": cached,
                    "bm25_ms": run.bm25_ms, "embed_ms": run.embed_ms,
                    "score_ms": run.score_ms, "total_ms": run.total_ms,
                    "hits": run.results.iter().map(|h| &h.id).collect::<Vec<_>>(),
                    "vector": encoder.embed_query(query)?,
                }),
            );
        }
    }
    let report = serde_json::json!({"index_load_ms": index_load_ms,
        "encoder_load_ms": encoder_load_ms, "warmup_ms": warmup_ms,
        "encoder": encoder.diagnostics(), "queries": rows});
    if let Some(output) = args.get(2) {
        std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    }
    println!("index_load_ms={index_load_ms:.3} encoder_load_ms={encoder_load_ms:.3}");
    for pass in 0..2 {
        let rows: Vec<_> = rows.iter().filter(|r| r["pass"] == pass).collect();
        anyhow::ensure!(!rows.is_empty(), "empty query list");
        let mean = |key: &str| {
            rows.iter().map(|r| r[key].as_f64().unwrap()).sum::<f64>() / rows.len() as f64
        };
        println!(
            "pass={pass} queries={} bm25_ms={:.3} embed_ms={:.3} score_ms={:.3} total_ms={:.3}",
            rows.len(),
            mean("bm25_ms"),
            mean("embed_ms"),
            mean("score_ms"),
            mean("total_ms")
        );
    }
    Ok(())
}
