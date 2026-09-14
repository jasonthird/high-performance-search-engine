//! Encode a sample of a repository's chunks plus a fixed query set and
//! write the vectors and timings to JSON: the tool behind backend
//! comparisons (fidelity against a reference dump, throughput, batch-1
//! query latency).
//!
//! ```sh
//! cargo run --release --features semantic --example embed_dump -- <repo> [n] [out.json]
//! HPS_EMBED_DEVICE=cpu cargo run --release --features semantic --example embed_dump -- . 256 cpu.json
//! ```
//!
//! Two dumps of the same repo and `n` are directly comparable: the chunk
//! sample is a fixed stride over the sorted walk, the queries are constant.
#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("build with --features semantic");
    std::process::exit(1);
}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use high_performance_search_engine::embedder::{EmbedUse, Embedder};
    use high_performance_search_engine::repo;
    use std::{path::Path, time::Instant};

    let mut args = std::env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".into());
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let out = args.next().unwrap_or_else(|| "embed-dump.json".into());
    let chunks = repo::collect_chunks(Path::new(&root))?;
    let stride = (chunks.len() / n).max(1);
    let texts: Vec<String> = chunks
        .iter()
        .step_by(stride)
        .take(n)
        .map(|c| format!("{}\n{}", c.title(), c.body))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let queries = [
        "how does the watcher lease get released when the agent session ends",
        "block max scores allow safe skipping",
        "download a grammar and verify its sha256 checksum",
        "tokenize text into lowercase alphanumeric terms",
        "merge segments and drop tombstones",
        "L2 normalize the CLS token embedding",
        "spawn the detached watcher process with setsid",
        "return an actionable error when the disk is full",
    ];

    let t = Instant::now();
    let index = Embedder::load_for(EmbedUse::Index)?;
    let load_index_s = t.elapsed().as_secs_f64();
    let doc_tokens: usize = refs.iter().map(|r| index.tokenize(r).len()).sum();
    let batch = index.effective_batch();
    let _ = index.embed_docs_with(&refs[..refs.len().min(8)], batch, true)?; // warm-up
    let mut best = f64::INFINITY;
    let mut doc_vecs = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        doc_vecs = index.embed_docs_with(&refs, batch, true)?;
        best = best.min(t.elapsed().as_secs_f64());
    }

    let t = Instant::now();
    let query = Embedder::load_for(EmbedUse::Query)?;
    let load_query_s = t.elapsed().as_secs_f64();
    let _ = query.embed_query("warm up query")?;
    let mut q_ms = Vec::new();
    let mut q_vecs = Vec::new();
    for q in queries {
        let t = Instant::now();
        q_vecs.push(query.embed_query(q)?);
        q_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mut sorted = q_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let json = serde_json::json!({
        "backend": index.diagnostics(),
        "docs": texts.iter().zip(&doc_vecs).map(|(t, v)| serde_json::json!({"text": t, "vec": v})).collect::<Vec<_>>(),
        "queries": queries.iter().zip(&q_vecs).map(|(t, v)| serde_json::json!({"text": t, "vec": v})).collect::<Vec<_>>(),
        "timing": {
            "load_index_model_s": load_index_s,
            "load_query_model_s": load_query_s,
            "docs": refs.len(), "doc_tokens": doc_tokens, "batch": batch,
            "docs_s_best_of_3": best, "docs_tokens_per_s": doc_tokens as f64 / best,
            "query_ms_each": q_ms, "query_ms_median": sorted[sorted.len() / 2],
        }
    });
    std::fs::write(&out, serde_json::to_string(&json)?)?;
    println!("{}", serde_json::to_string_pretty(&json["timing"])?);
    Ok(())
}
