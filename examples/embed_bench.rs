//! How fast can CodeRankEmbed encode a real corpus, and what governs it?
//!
//! Indexing throughput is dominated by this encoder, so the two knobs that
//! matter are measured directly on real chunks:
//!
//! - **Batch size.** Metal is kernel-launch bound on short BERT sequences,
//!   so tiny batches waste most of the GPU.
//! - **Length bucketing.** Padding is `BatchLongest`; with chunks in file
//!   order a single long chunk inflates everything batched with it.
//!
//! ```sh
//! cargo run --release --features semantic --example embed_bench -- <repo> [n]
//! ```

#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("build with --features semantic");
    std::process::exit(1);
}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use std::path::Path;
    use std::time::Instant;

    use high_performance_search_engine::embedder::{EmbedUse, Embedder};
    use high_performance_search_engine::repo;

    let mut args = std::env::args().skip(1);
    let root = args.next().expect("usage: embed_bench <repo> [n]");
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);

    let chunks = repo::collect_chunks(Path::new(&root))?;
    // Sample across the corpus rather than taking a prefix, so the length
    // distribution matches the whole tree.
    let stride = (chunks.len() / n).max(1);
    let texts: Vec<String> = chunks
        .iter()
        .step_by(stride)
        .take(n)
        .map(|c| format!("{}\n{}", c.title(), c.body))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let mut lens: Vec<usize> = texts.iter().map(|t| t.len()).collect();
    lens.sort_unstable();
    println!(
        "\n{} chunks sampled from {} ({} total). bytes: p50 {}, p90 {}, max {}\n",
        refs.len(),
        root,
        chunks.len(),
        lens[lens.len() / 2],
        lens[lens.len() * 9 / 10],
        lens[lens.len() - 1]
    );

    // Must be the *indexing* encoder: the query encoder truncates at 64
    // tokens, which would silently measure a much smaller problem.
    let embedder = Embedder::load_for(EmbedUse::Index)?;
    println!(
        "seq cap: {} tokens (HPS_EMBED_MAX_SEQ)\n",
        std::env::var("HPS_EMBED_MAX_SEQ").unwrap_or_else(|_| "512".into())
    );
    // Warm up Metal pipelines outside the measurement.
    let _ = embedder.embed_docs_with(&refs[..refs.len().min(8)], 8, true)?;

    // Configurations are interleaved and each is measured several times,
    // taking the best. A single sequential sweep is confounded by thermal
    // drift: later configurations run on a hotter GPU and look slower
    // regardless of their merits.
    let configs: Vec<(usize, bool)> = [4usize, 8, 16, 32, 64, 128]
        .iter()
        .flat_map(|&b| [(b, false), (b, true)])
        .collect();
    let rounds = 3;
    let mut best = vec![f64::INFINITY; configs.len()];
    for round in 0..rounds {
        for (i, &(batch, bucket)) in configs.iter().enumerate() {
            let t = Instant::now();
            let out = embedder.embed_docs_with(&refs, batch, bucket)?;
            let secs = t.elapsed().as_secs_f64();
            assert_eq!(out.len(), refs.len());
            best[i] = best[i].min(secs);
        }
        eprintln!("  round {}/{rounds} done", round + 1);
    }

    let slowest = best.iter().cloned().fold(0.0f64, f64::max);
    println!(
        "{:>7} {:>10} {:>12} {:>12} {:>9}",
        "batch", "bucketed", "best s", "ms/chunk", "speedup"
    );
    for (i, &(batch, bucket)) in configs.iter().enumerate() {
        println!(
            "{batch:>7} {:>10} {:>12.2} {:>12.1} {:>8.2}x",
            if bucket { "yes" } else { "no" },
            best[i],
            best[i] * 1000.0 / refs.len() as f64,
            slowest / best[i]
        );
    }
    println!();
    Ok(())
}
