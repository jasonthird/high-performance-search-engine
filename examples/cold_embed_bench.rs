//! Compare the former byte-sorted serial scheduler with token-aware pipelining.
//! Same model, batch size, truncation and input rows; no embedding cache.
#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("requires --features semantic");
}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use high_performance_search_engine::{
        embedder::{EmbedUse, Embedder},
        repo,
    };
    use std::{path::Path, time::Instant};
    let mut args = std::env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".into());
    let n: usize = args.next().unwrap_or_else(|| "128".into()).parse()?;
    anyhow::ensure!(n > 0, "sample size must be positive");
    let chunks = repo::collect_chunks(Path::new(&root))?;
    let texts: Vec<_> = chunks
        .iter()
        .step_by((chunks.len() / n).max(1))
        .take(n)
        .map(|c| format!("{}\n{}", c.title(), c.body))
        .collect();
    anyhow::ensure!(!texts.is_empty(), "empty corpus");
    let refs: Vec<_> = texts.iter().map(String::as_str).collect();
    let start = Instant::now();
    let encoder = Embedder::load_for(EmbedUse::Index)?;
    let batch = encoder.effective_batch();
    println!(
        "model_load_s={:.3} rows={} batch={batch}",
        start.elapsed().as_secs_f64(),
        refs.len()
    );
    // Warm every shape used by the sample, excluding plan compilation below.
    let warmup = Instant::now();
    encoder.embed_docs(&refs)?;
    println!("shape_warmup_s={:.3}", warmup.elapsed().as_secs_f64());
    let mut legacy_times = Vec::new();
    let mut pipeline_times = Vec::new();
    let mut min_cosine = 1.0f32;
    for round in 0..3 {
        let mut results = [Vec::new(), Vec::new()];
        for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let start = Instant::now();
            let output = if mode == 0 {
                let mut output = vec![Vec::new(); refs.len()];
                for (window, rows) in refs.chunks(256).enumerate() {
                    let mut order: Vec<_> = (0..rows.len()).collect();
                    order.sort_by_key(|&i| rows[i].len());
                    for group in order.chunks(batch) {
                        let input: Vec<_> = group.iter().map(|&i| rows[i]).collect();
                        for (&i, vector) in group
                            .iter()
                            .zip(encoder.embed_docs_with(&input, batch, false)?)
                        {
                            output[window * 256 + i] = vector;
                        }
                    }
                }
                output
            } else {
                encoder.embed_docs(&refs)?
            };
            let seconds = start.elapsed().as_secs_f64();
            if mode == 0 {
                legacy_times.push(seconds);
            } else {
                pipeline_times.push(seconds);
            }
            results[mode] = output;
        }
        for (row, (a, b)) in results[0].iter().zip(&results[1]).enumerate() {
            anyhow::ensure!(a.len() == 768 && b.len() == 768 && a.iter().chain(b).all(|x| x.is_finite()), "invalid embedding");
            let cosine: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            if cosine < 0.995 && round == 0 {
                let best = results[0].iter().enumerate().map(|(i,v)| (i, v.iter().zip(b).map(|(x,y)| x*y).sum::<f32>()))
                    .max_by(|a,b| a.1.total_cmp(&b.1)).unwrap();
                eprintln!("mismatch row={row} cosine={cosine:.6} bytes={} nearest={best:?} norms=({:.3},{:.3})", texts[row].len(), a.iter().map(|x|x*x).sum::<f32>(), b.iter().map(|x|x*x).sum::<f32>());
            }
            min_cosine = min_cosine.min(cosine);
        }
        eprintln!("round {}: legacy {:.3}s pipeline {:.3}s", round + 1, legacy_times[round], pipeline_times[round]);
    }
    legacy_times.sort_by(f64::total_cmp);
    pipeline_times.sort_by(f64::total_cmp);
    println!(
        "legacy_median_s={:.3} pipeline_median_s={:.3} speedup={:.3} min_cosine={min_cosine:.6}",
        legacy_times[1],
        pipeline_times[1],
        legacy_times[1] / pipeline_times[1]
    );
    anyhow::ensure!(
        min_cosine > 0.995,
        "scheduler changed embeddings beyond FP16 tolerance"
    );
    Ok(())
}
