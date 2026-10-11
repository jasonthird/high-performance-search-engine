//! Validate the normal encoder's persistent cache on complete benchmark queries.
#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("requires --features semantic");
    std::process::exit(1);
}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use clap::Parser;
    use high_performance_search_engine::embedder::{Embedder, QUERY_PREFIX};
    use serde_json::{json, Value};
    use std::{collections::BTreeMap, io::BufRead, path::PathBuf, time::Instant};
    #[derive(Parser)]
    struct Args {
        #[arg(long, default_value = "benchmarks/gemma/queries.jsonl")]
        queries: PathBuf,
        #[arg(long)]
        out: PathBuf,
    }
    let args = Args::parse();
    let encoder = Embedder::load()?;
    let mut rows = Vec::new();
    let mut vectors = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args.queries)?).lines() {
        let item: Value = serde_json::from_str(&line?)?;
        let query: String = item["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing query"))?
            .chars()
            .take(8000)
            .collect();
        // Bypass both cache levels to obtain the same prefixed, contextual
        // full-model prediction as an uncached search (64-token cap).
        let prefixed = format!("{QUERY_PREFIX}{query}");
        let t = Instant::now();
        let full = encoder.embed_docs(&[&prefixed])?.pop().unwrap();
        let full_ms = t.elapsed().as_secs_f64() * 1000.0;
        let cached = encoder.query_cached(&query);
        let before = encoder.diagnostics()["query_model_calls"].as_u64().unwrap();
        let t = Instant::now();
        let actual = encoder.embed_query(&query)?;
        let first_ms = t.elapsed().as_secs_f64() * 1000.0;
        anyhow::ensure!(
            actual == full,
            "cache changed full-model vector for {}",
            item["id"]
        );
        let after = encoder.diagnostics()["query_model_calls"].as_u64().unwrap();
        anyhow::ensure!(
            after - before == u64::from(!cached),
            "incorrect miss inference count"
        );
        let t = Instant::now();
        let repeat = encoder.embed_query(&query)?;
        let repeat_ms = t.elapsed().as_secs_f64() * 1000.0;
        anyhow::ensure!(
            repeat == full && encoder.diagnostics()["query_model_calls"].as_u64().unwrap() == after,
            "repeat called model or changed vector"
        );
        rows.push(json!({"id":item["id"],"mode":item["mode"],"query":query,"initial_cache_hit":cached,
            "full_encode_ms":full_ms,"first_encode_ms":first_ms,"repeat_encode_ms":repeat_ms,"full_vector_equal":true}));
        vectors.push(full);
    }
    let initial_diagnostics = encoder.diagnostics();
    high_performance_search_engine::flush_query_cache_writes();
    drop(encoder);
    // A fresh encoder starts with an empty query LRU; entries must come from disk.
    let restarted = Embedder::load()?;
    for (row, expected) in rows.iter_mut().zip(&vectors) {
        let query = row["query"].as_str().unwrap();
        anyhow::ensure!(
            restarted.query_cached(query),
            "persistent cache missing query"
        );
        let t = Instant::now();
        let actual = restarted.embed_query(query)?;
        let reload_ms = t.elapsed().as_secs_f64() * 1000.0;
        anyhow::ensure!(actual == *expected, "JSON round trip changed vector");
        row["reload_encode_ms"] = json!(reload_ms);
    }
    anyhow::ensure!(
        restarted.diagnostics()["query_model_calls"] == 0,
        "reload ran query inference"
    );
    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        if v.len() % 2 == 0 {
            (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
        } else {
            v[v.len() / 2]
        }
    }
    let mut summary = BTreeMap::new();
    for mode in ["agent", "idq", "issue"] {
        let subset: Vec<_> = rows.iter().filter(|r| r["mode"] == mode).collect();
        if subset.is_empty() {
            continue;
        }
        let metric =
            |field: &str| median(subset.iter().map(|r| r[field].as_f64().unwrap()).collect());
        summary.insert(mode,json!({"queries":subset.len(),"initial_cache_hits":subset.iter().filter(|r|r["initial_cache_hit"]==true).count(),
            "full_encode_ms":metric("full_encode_ms"),"first_encode_ms":metric("first_encode_ms"),
            "repeat_encode_ms":metric("repeat_encode_ms"),"reload_encode_ms":metric("reload_encode_ms")}));
    }
    let report = json!({"queries":rows.len(),"source_queries":args.queries,"encoder":initial_diagnostics,
        "reloaded_encoder":restarted.diagnostics(),"unchanged_vectors":rows.len(),"summary":summary,"results":rows});
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    serde_json::to_writer_pretty(
        std::io::BufWriter::new(std::fs::File::create(&args.out)?),
        &report,
    )?;
    println!("{}", serde_json::to_string_pretty(&report["summary"])?);
    Ok(())
}
