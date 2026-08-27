//! What does product quantization actually cost and save?
//!
//! PQ replaces each document vector with 16 centroid ids, so scoring becomes
//! table lookups instead of 768 multiply-adds. That trade only pays off once
//! scanning full vectors is slower than the approximation costs in recall.
//! This measures both halves against exact scoring, at whatever corpus size
//! the given index happens to be.
//!
//! Two sources of error are separated, because they are easy to conflate:
//!
//! - **IVF pruning**: only `nprobe` of `k` clusters are opened at all.
//! - **PQ approximation**: documents that *were* opened get approximate
//!   scores.
//!
//! Ground truth is `--no-pq` with every cluster probed.
//!
//! ```sh
//! cargo run --release --features semantic --example pq_tradeoff -- <index> <queries.txt>
//! ```

use std::collections::HashSet;
use std::path::Path;

use high_performance_search_engine::cli::{FusionArg, PqMode, RankMode, SearchOpts};
use high_performance_search_engine::searcher::AnyIndex;

#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("build with --features semantic");
    std::process::exit(1);
}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use high_performance_search_engine::cli::run_ranked_with;
    use high_performance_search_engine::embedder::Embedder;

    let mut args = std::env::args().skip(1);
    let index_dir = args.next().expect("usage: pq_tradeoff <index> <queries>");
    let queries_path = args.next().expect("usage: pq_tradeoff <index> <queries>");
    let top_k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let index = AnyIndex::open(Path::new(&index_dir))?;
    let queries: Vec<String> = std::fs::read_to_string(&queries_path)?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    let clusters = index.ivf().map(|i| i.num_clusters() as usize).unwrap_or(0);
    let auto_nprobe = index.ivf().map(|i| i.auto_nprobe()).unwrap_or(0);
    let n = index.num_docs();
    let embedder = Embedder::load()?;

    let base = SearchOpts {
        mode: RankMode::Hybrid,
        semantic_candidates: 200,
        fusion: FusionArg::Rrf,
        alpha: 0.5,
        rrf_k: 60.0,
        nprobe: 0,
        pq: PqMode::Auto,
    };

    println!(
        "\n{n} chunks, {clusters} IVF clusters, auto nprobe {auto_nprobe} \
         ({:.1}% of clusters), {} queries, top-{top_k}",
        100.0 * auto_nprobe as f64 / clusters.max(1) as f64,
        queries.len()
    );

    for mode in [RankMode::Semantic, RankMode::Hybrid] {
        // Ground truth: no approximation of any kind.
        let truth_opts = SearchOpts {
            mode,
            nprobe: clusters,
            pq: PqMode::Off,
            ..base
        };
        let configs = [
            ("IVF only (exact scores)", SearchOpts { mode, nprobe: 0, pq: PqMode::Off, ..base }),
            ("IVF + PQ (approx scores)", SearchOpts { mode, nprobe: 0, pq: PqMode::Force, ..base }),
        ];

        // Warm up so Metal pipelines and page cache are not on the clock.
        for q in queries.iter().take(3) {
            let _ = run_ranked_with(&index, &embedder, q, top_k, &truth_opts)?;
        }

        let mut truth: Vec<HashSet<String>> = Vec::new();
        let mut truth_score = 0.0f64;
        let mut truth_total = 0.0f64;
        for q in &queries {
            let run = run_ranked_with(&index, &embedder, q, top_k, &truth_opts)?;
            truth_score += run.score_ms;
            truth_total += run.total_ms;
            truth.push(run.results.into_iter().map(|r| r.id).collect());
        }
        let qn = queries.len() as f64;

        println!("\n  mode = {mode:?}");
        println!(
            "    {:<26} {:>9} {:>10} {:>10}",
            "config", "recall@10", "score ms", "total ms"
        );
        println!(
            "    {:<26} {:>9.3} {:>10.3} {:>10.3}",
            "exact (all clusters)",
            1.0,
            truth_score / qn,
            truth_total / qn
        );
        for (label, opts) in &configs {
            let mut recall = 0.0f64;
            let mut score = 0.0f64;
            let mut total = 0.0f64;
            for (i, q) in queries.iter().enumerate() {
                let run = run_ranked_with(&index, &embedder, q, top_k, opts)?;
                score += run.score_ms;
                total += run.total_ms;
                let got: HashSet<String> = run.results.into_iter().map(|r| r.id).collect();
                let hits = got.intersection(&truth[i]).count();
                recall += hits as f64 / truth[i].len().max(1) as f64;
            }
            println!(
                "    {label:<26} {:>9.3} {:>10.3} {:>10.3}",
                recall / qn,
                score / qn,
                total / qn
            );
        }
    }
    println!();
    Ok(())
}
