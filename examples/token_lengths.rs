//! Token-length distribution of a repo's chunks under the CodeRankEmbed
//! tokenizer — vocabulary only, no model weights, so it can run while the
//! GPU is busy encoding.
//!
//! This predicts what a shorter sequence cap costs (chunks truncated) and
//! saves (padded tokens not computed) before spending any GPU time on it.
//!
//! ```sh
//! cargo run --release --features semantic --example token_lengths -- <repo> [model.gguf]
//! ```
#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("build with --features semantic");
}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use std::path::{Path, PathBuf};

    use high_performance_search_engine::{embedder, ggml, repo, wordpiece::WordPiece};

    let root = std::env::args().nth(1).expect("usage: token_lengths <repo> [model.gguf]");
    let model = match std::env::args().nth(2) {
        Some(p) => PathBuf::from(p),
        None => embedder::model_path()?,
    };
    let vocab = ggml::read_vocab(&model)?;
    let tokenizer = WordPiece::from_gguf(&vocab.tokens, &vocab.token_types, vocab.unk, vocab.cls, vocab.sep);
    let chunks = repo::collect_chunks(Path::new(&root))?;
    let texts: Vec<String> = chunks
        .iter()
        .map(|c| format!("{}\n{}", c.title(), c.body))
        .collect();

    // Keep file order too: padded cost depends on batching order.
    let mut lens: Vec<usize> = texts.iter().map(|t| tokenizer.encode(t, usize::MAX).len()).collect();
    let file_order = lens.clone();
    lens.sort_unstable();
    let n = lens.len();
    let pct = |p: usize| lens[(n * p / 100).min(n - 1)];
    let over = |cap: usize| lens.iter().filter(|&&l| l > cap).count();
    let total: usize = lens.iter().sum();
    let capped_total = |cap: usize| -> usize { lens.iter().map(|&l| l.min(cap)).sum() };

    println!("\n{n} chunks tokenized (CodeRankEmbed tokenizer)");
    println!(
        "tokens: p50 {}, p90 {}, p95 {}, p99 {}, max {}, mean {:.1}",
        pct(50),
        pct(90),
        pct(95),
        pct(99),
        lens[n - 1],
        total as f64 / n as f64
    );
    println!("\n{:>6} {:>10} {:>12} {:>14}", "cap", "truncated", "tokens kept", "of untruncated");
    for cap in [128usize, 192, 256, 384, 512, 1024, 2048] {
        println!(
            "{cap:>6} {:>10} {:>12} {:>13.1}%",
            over(cap),
            capped_total(cap),
            100.0 * capped_total(cap) as f64 / total as f64
        );
    }

    // Padded cost of batches of 32 in file order versus sorted by length:
    // what bucketing buys at each cap.
    let padded = |order: &[usize], cap: usize| -> usize {
        order
            .chunks(32)
            .map(|b| b.iter().map(|&l| l.min(cap)).max().unwrap_or(0) * b.len())
            .sum()
    };
    println!("\n{:>6} {:>14} {:>14} {:>8}", "cap", "padded (file)", "padded (sort)", "saving");
    for cap in [256usize, 512, 2048] {
        let f = padded(&file_order, cap);
        let s = padded(&lens, cap);
        println!("{cap:>6} {f:>14} {s:>14} {:>7.1}%", 100.0 * (1.0 - s as f64 / f as f64));
    }
    Ok(())
}
