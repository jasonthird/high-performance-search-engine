//! The fused-SDPA Metal path must produce the same embeddings as the naive
//! attention (still used on CPU). Cross-device cosine, batch>1 so the
//! padding mask is exercised.

#[cfg(not(feature = "semantic"))]
fn main() {}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use high_performance_search_engine::embedder::{EmbedUse, Embedder};
    use high_performance_search_engine::repo;

    let root = std::env::args().nth(1).expect("usage: sdpa_equiv <repo>");
    let chunks = repo::collect_chunks(std::path::Path::new(&root))?;
    let stride = (chunks.len() / 16).max(1);
    let texts: Vec<String> = chunks
        .iter()
        .step_by(stride)
        .take(16)
        .map(|c| format!("{}\n{}", c.title(), c.body))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    std::env::set_var("HPS_EMBED_DEVICE", "metal");
    let metal = Embedder::load_for(EmbedUse::Index)?.embed_docs_with(&refs, 4, false)?;
    std::env::set_var("HPS_EMBED_DEVICE", "cpu");
    let cpu = Embedder::load_for(EmbedUse::Index)?.embed_docs_with(&refs, 4, false)?;

    let mut worst = 1.0f32;
    for (a, b) in metal.iter().zip(cpu.iter()) {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        worst = worst.min(dot);
    }
    println!("worst cosine(metal fused sdpa, cpu naive) = {worst:.6}");
    // Metal runs F16, CPU runs F32: expect small drift, not divergence.
    assert!(worst > 0.99, "fused SDPA diverges from naive attention");
    println!("fused SDPA matches naive attention across devices");
    Ok(())
}
