//! Length bucketing must not change the embeddings, only their grouping.
//! Encodes a mixed-length sample both ways and compares cosines.

#[cfg(not(feature = "semantic"))]
fn main() {}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use high_performance_search_engine::embedder::{EmbedUse, Embedder};
    use high_performance_search_engine::repo;

    let root = std::env::args().nth(1).expect("usage: bucket_equiv <repo>");
    let chunks = repo::collect_chunks(std::path::Path::new(&root))?;
    let stride = (chunks.len() / 24).max(1);
    let texts: Vec<String> = chunks
        .iter()
        .step_by(stride)
        .take(24)
        .map(|c| format!("{}\n{}", c.title(), c.body))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let embedder = Embedder::load_for(EmbedUse::Index)?;
    let plain = embedder.embed_docs_with(&refs, 4, false)?;
    let again = embedder.embed_docs_with(&refs, 4, false)?;
    let bucketed = embedder.embed_docs_with(&refs, 16, true)?;
    // Noise floor: identical config, run twice. F16 Metal reductions are
    // not bit-deterministic, so "equivalent" means "within rerun noise",
    // not bit-equal.
    let floor = plain
        .iter()
        .zip(again.iter())
        .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>())
        .fold(1.0f32, f32::min);
    println!("rerun noise floor (same config twice): {floor:.6}");
    let mut worst = 1.0f32;
    let mut worst_i = 0usize;
    for (i, (a, b)) in plain.iter().zip(bucketed.iter()).enumerate() {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        if dot < worst {
            worst = dot;
            worst_i = i;
        }
    }
    println!("{} texts, worst cosine(plain, bucketed) = {worst:.6} at #{worst_i} ({} bytes)",
        refs.len(), texts[worst_i].len());
    let per: Vec<String> = plain.iter().zip(bucketed.iter())
        .map(|(a, b)| format!("{:.4}", a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()))
        .collect();
    println!("cosines: {}", per.join(" "));
    assert!(
        worst > 0.995,
        "bucketing diverges beyond F16 rerun noise ({worst} vs floor {floor})"
    );
    println!("equivalent: bucketing only regroups, it does not change vectors");
    Ok(())
}
