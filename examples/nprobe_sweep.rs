//! Recall-vs-nprobe curve for IVF semantic retrieval.
//!
//! `auto_nprobe` was a guess (`clamp(k/4, 1, 8)`); at 31k chunks that
//! probes 3% of clusters and loses a third of semantic recall. This sweeps
//! nprobe against ground truth (exact scores, every cluster) so the default
//! can be chosen from a curve instead.

#[cfg(not(feature = "semantic"))]
fn main() {}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use std::collections::HashSet;
    use std::path::Path;

    use high_performance_search_engine::query::{FusionArg, PqMode, RankMode, SearchOpts};
    use high_performance_search_engine::query::run_ranked_with;
    use high_performance_search_engine::embedder::Embedder;
    use high_performance_search_engine::searcher::AnyIndex;

    let mut args = std::env::args().skip(1);
    let index_dir = args.next().expect("usage: nprobe_sweep <index> <queries>");
    let queries_path = args.next().expect("usage: nprobe_sweep <index> <queries>");
    let top_k = 10usize;

    let index = AnyIndex::open(Path::new(&index_dir))?;
    let clusters = index.ivf().map(|i| i.num_clusters() as usize).unwrap_or(0);
    anyhow::ensure!(clusters > 0, "index has no ivf.bin");
    let queries: Vec<String> = std::fs::read_to_string(&queries_path)?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    let embedder = Embedder::load()?;

    let base = SearchOpts {
        mode: RankMode::Semantic,
        semantic_candidates: 200,
        fusion: FusionArg::Rrf,
        alpha: 0.5,
        rrf_k: 60.0,
        nprobe: clusters,
        pq: PqMode::Off,
    };
    // Ground truth once per query.
    let mut truth: Vec<HashSet<String>> = Vec::new();
    for q in &queries {
        let run = run_ranked_with(&index, &embedder, q, top_k, &base)?;
        truth.push(run.results.into_iter().map(|r| r.id).collect());
    }

    println!(
        "\n{} chunks, {clusters} clusters, {} queries, semantic mode, exact scores\n",
        index.num_docs(),
        queries.len()
    );
    println!("{:>7} {:>10} {:>10} {:>10}", "nprobe", "% probed", "recall@10", "score ms");
    for nprobe in [1usize, 2, 4, 8, 16, 32, 64, 128, clusters] {
        let opts = SearchOpts { nprobe, ..base };
        let mut recall = 0.0f64;
        let mut score_ms = 0.0f64;
        for (i, q) in queries.iter().enumerate() {
            let run = run_ranked_with(&index, &embedder, q, top_k, &opts)?;
            score_ms += run.score_ms;
            let got: HashSet<String> = run.results.into_iter().map(|r| r.id).collect();
            recall += got.intersection(&truth[i]).count() as f64 / truth[i].len().max(1) as f64;
        }
        let n = queries.len() as f64;
        println!(
            "{nprobe:>7} {:>9.1}% {:>10.3} {:>10.3}",
            100.0 * nprobe as f64 / clusters as f64,
            recall / n,
            score_ms / n
        );
    }
    println!();
    Ok(())
}
