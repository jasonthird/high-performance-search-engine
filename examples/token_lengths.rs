//! Token-length distribution of a repo's chunks under the CodeRankEmbed
//! tokenizer — CPU-only, so it can run while the GPU is busy encoding.
//!
//! This predicts what a shorter sequence cap costs (chunks truncated) and
//! saves (padded tokens not computed) before spending any GPU time on it.
//!
//! ```sh
//! cargo run --release --features semantic --example token_lengths -- <repo>
//! ```

#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("build with --features semantic");
}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use std::path::Path;

    use high_performance_search_engine::repo;
    use tokenizers::Tokenizer;

    let root = std::env::args().nth(1).expect("usage: token_lengths <repo>");
    let tok_path = std::env::args().nth(2).unwrap_or_else(|| {
        format!(
            "{}/.cache/huggingface/hub/models--nomic-ai--CodeRankEmbed/snapshots/3c4b60807d71f79b43f3c4363786d9493691f8b1/tokenizer.json",
            std::env::var("HOME").unwrap()
        )
    });
    let mut tokenizer =
        Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    // The shipped tokenizer.json pads and truncates to 512, which would make
    // every length read as exactly 512; strip both to see true lengths.
    let _ = tokenizer.with_padding(None);
    let _ = tokenizer.with_truncation(None);
    let chunks = repo::collect_chunks(Path::new(&root))?;
    let texts: Vec<String> = chunks
        .iter()
        .map(|c| format!("{}\n{}", c.title(), c.body))
        .collect();

    // Keep file order too: padded cost depends on batching order.
    let mut lens: Vec<usize> = Vec::with_capacity(texts.len());
    for batch in texts.chunks(512) {
        let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
        let encs = tokenizer
            .encode_batch(refs, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        lens.extend(encs.iter().map(|e| e.len()));
    }
    let file_order = lens.clone();
    lens.sort_unstable();
    let n = lens.len();
    let pct = |p: usize| lens[(n * p / 100).min(n - 1)];
    let over = |cap: usize| lens.iter().filter(|&&l| l > cap).count();
    let total: usize = lens.iter().sum();
    let capped_total = |cap: usize| -> usize { lens.iter().map(|&l| l.min(cap)).sum() };

    println!("\n{n} chunks tokenized (CodeRankEmbed tokenizer)");
    println!(
        "tokens: p50 {}  p75 {}  p90 {}  p95 {}  p99 {}  max {}",
        pct(50), pct(75), pct(90), pct(95), pct(99), lens[n - 1]
    );
    println!("\n{:>6} {:>12} {:>14} {:>16}", "cap", "truncated", "% truncated", "token work vs 512");
    for cap in [512usize, 384, 256, 192, 128] {
        let t = over(cap);
        println!(
            "{cap:>6} {t:>12} {:>13.1}% {:>15.1}%",
            100.0 * t as f64 / n as f64,
            100.0 * capped_total(cap) as f64 / capped_total(512).max(1) as f64
        );
    }
    let _ = total;

    // Padded cost model: a batch computes batch_size x max(len) tokens, so
    // padding waste depends entirely on how similar batch members are.
    let padded = |order: &[usize], batch: usize, cap: usize| -> usize {
        order
            .chunks(batch)
            .map(|b| b.len() * b.iter().map(|&l| l.min(cap)).max().unwrap_or(0))
            .sum()
    };
    let real: usize = lens.iter().map(|&l| l.min(512)).sum();
    println!("padded token work at cap 512 (100% = tokens actually needed):");
    for (label, order) in [("file order", &file_order), ("sorted (bucketed)", &lens)] {
        for batch in [4usize, 16, 64] {
            println!(
                "  {label:<18} batch {batch:>3}: {:>6.0}%",
                100.0 * padded(order, batch, 512) as f64 / real as f64
            );
        }
    }
    println!();
    Ok(())
}
