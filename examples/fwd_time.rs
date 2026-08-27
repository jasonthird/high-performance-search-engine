//! Direct forward-pass timing at fixed shapes, for before/after comparison
//! of attention changes. Prints ms per batch at the same (seq, batch)
//! points profiled earlier.

#[cfg(not(feature = "semantic"))]
fn main() {}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;

    use high_performance_search_engine::embedder::{EmbedUse, Embedder};

    let embedder = Embedder::load_for(EmbedUse::Index)?;
    // Synthetic texts sized to tokenize near the target lengths.
    // "a " tokenizes to one token per repeat, so seq is controlled exactly.
    let word = "a ";
    for (label, reps_words, batch) in [
        ("seq ~14, batch 1", 12usize, 1usize),
        ("seq ~92, batch 4", 90, 4),
        ("seq ~200, batch 4", 198, 4),
        ("seq ~510, batch 4", 508, 4),
    ] {
        let text = word.repeat(reps_words);
        let texts: Vec<&str> = std::iter::repeat(text.as_str()).take(batch).collect();
        // Warm.
        let _ = embedder.embed_docs_with(&texts, batch, false)?;
        let runs = 5;
        let t = Instant::now();
        for _ in 0..runs {
            let _ = embedder.embed_docs_with(&texts, batch, false)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        println!("{label:<20} batch {batch}: {ms:8.1} ms/batch  {:6.1} ms/chunk", ms / batch as f64);
    }
    Ok(())
}
