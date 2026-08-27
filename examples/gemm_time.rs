//! Bare Metal GEMM throughput at NomicBert's shapes, to test whether the
//! encoder is GEMM-bound and how far Candle's kernels sit from the chip.

#[cfg(not(feature = "semantic"))]
fn main() {}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;

    use candle_core::{DType, Device, Tensor};

    let dev = Device::new_metal(0)?;
    for (label, m, k, n) in [
        // batch*seq x hidden x ffn — the fc11/fc12 shape at seq 512, batch 4
        ("ffn up  (2048x768x3072)", 2048usize, 768usize, 3072usize),
        ("ffn down(2048x3072x768)", 2048, 3072, 768),
        ("qkv     (2048x768x2304)", 2048, 768, 2304),
        ("square  (2048x2048x2048)", 2048, 2048, 2048),
    ] {
        for dtype in [DType::F16, DType::F32] {
            let a = Tensor::randn(0f32, 1f32, (m, k), &dev)?.to_dtype(dtype)?;
            let b = Tensor::randn(0f32, 1f32, (k, n), &dev)?.to_dtype(dtype)?;
            // Warm.
            let _ = a.matmul(&b)?.sum_all()?.to_scalar::<f32>().unwrap_or(0.0);
            let runs = 20;
            let t = Instant::now();
            for _ in 0..runs {
                let c = a.matmul(&b)?;
                std::hint::black_box(&c);
            }
            // Force completion before stopping the clock.
            let _ = a.matmul(&b)?.sum_all()?.to_scalar::<f32>().unwrap_or(0.0);
            let secs = t.elapsed().as_secs_f64() / runs as f64;
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!(
                "{label} {dtype:?}: {:8.2} ms  {:6.2} TFLOP/s",
                secs * 1000.0,
                flops / secs / 1e12
            );
        }
    }
    Ok(())
}
