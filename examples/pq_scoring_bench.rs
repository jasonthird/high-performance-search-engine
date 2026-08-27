//! Where is the crossover at which PQ scoring beats exact FP16 scoring?
//!
//! Recall is measured elsewhere (`pq_tradeoff`); this isolates *speed*.
//! Scoring cost per document does not depend on codebook quality, only on
//! layout and arithmetic, so this builds sidecars with synthetic codebooks
//! and codes rather than paying for k-means.
//!
//! What it compares, per candidate document:
//! - FP16: 768 multiply-adds against the stored row.
//! - PQ/ADC: one table build per query, then M=16 lookups and a sum.
//!
//! The table build is a fixed per-query cost (M x 256 dot products of
//! 48 dims = ~196k FLOPs), so PQ only wins once enough documents are scored
//! to amortize it.
//!
//! ```sh
//! cargo run --release --example pq_scoring_bench
//! ```

use std::hint::black_box;
use std::time::Instant;

use high_performance_search_engine::embeddings::{self, EmbeddingStore, CODERANK_DIM};
use high_performance_search_engine::pq;

/// Deterministic pseudo-random unit vectors; xorshift keeps this dependency
/// free and repeatable.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

fn main() -> anyhow::Result<()> {
    let dim = CODERANK_DIM;
    let m = 16;
    let sub = dim / m;
    let dir = std::env::temp_dir().join("hpse-pq-scoring-bench");
    std::fs::create_dir_all(&dir)?;

    println!(
        "\nscoring {dim}-d vectors: exact FP16 dot vs PQ/ADC (M={m}, 256 codes/subspace)\n"
    );
    // The FP16 path decodes each half via a software bit-twiddle, 768 times
    // per document. An f32 dot over the same values in memory is the ceiling
    // that path could reach if decoding were free, and separates "PQ is
    // fast" from "the exact path is slow".
    println!(
        "{:>10} {:>11} {:>11} {:>11} {:>9} {:>9}",
        "candidates", "FP16 us", "f32 us", "PQ us", "PQ/FP16", "PQ/f32"
    );

    let mut rng = Rng(0x2545F4914F6CDD1D);
    for &n in &[100usize, 1_000, 10_000, 50_000, 200_000] {
        // Document vectors.
        let mut vectors = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            embeddings::l2_normalize(&mut v);
            vectors.push(v);
        }
        embeddings::write_f16(&dir, dim as u32, &vectors)?;

        // Synthetic codebooks and codes: scoring cost is independent of how
        // good the quantizer is, so training here would only cost time.
        let codebooks: Vec<f32> = (0..m * 256 * sub).map(|_| rng.next_f32()).collect();
        let codes: Vec<u8> = (0..n * m).map(|_| (rng.next_f32().abs() * 255.0) as u8).collect();
        pq::write_with_codebooks(&dir, &codebooks, &codes, n, m, sub, 256)?;

        let store = EmbeddingStore::open(&dir)?;
        let pqi = pq::PqIndex::open(&dir)?;
        let mut query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
        embeddings::l2_normalize(&mut query);

        // Enough repeats that even the smallest n is measured above timer noise.
        let reps = (2_000_000 / n).max(3);

        let t = Instant::now();
        for _ in 0..reps {
            let mut acc = 0.0f32;
            for doc in 0..n as u32 {
                acc += store.cosine(doc, &query);
            }
            black_box(acc);
        }
        let fp16_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        let t = Instant::now();
        for _ in 0..reps {
            // The table build is per query, so it belongs inside the loop.
            let tables = pqi.prepare(&query);
            let mut acc = 0.0f32;
            for doc in 0..n as u32 {
                acc += pqi.adc(&tables, doc);
            }
            black_box(acc);
        }
        let pq_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        // Ceiling: same arithmetic, no per-element decode.
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        let t = Instant::now();
        for _ in 0..reps {
            let mut acc = 0.0f32;
            for doc in 0..n {
                let row = &flat[doc * dim..(doc + 1) * dim];
                let mut d = 0.0f32;
                for i in 0..dim {
                    d += row[i] * query[i];
                }
                acc += d;
            }
            black_box(acc);
        }
        let f32_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        println!(
            "{n:>10} {fp16_us:>11.1} {f32_us:>11.1} {pq_us:>11.1} {:>8.2}x {:>8.2}x",
            fp16_us / pq_us,
            f32_us / pq_us
        );
    }
    std::fs::remove_dir_all(&dir).ok();
    println!();
    Ok(())
}
