# ANE prototype: CodeRankEmbed via CoreML

Measured 2026-08-29 on an M3 / 10-core GPU / 24 GB, batch 8 x seq 256, F16:

| runtime                    | tok/s  | chunks/s | 810-chunk corpus |
|----------------------------|--------|----------|------------------|
| candle + Metal (shipping)  |  6,300 |      28  | 28.6s            |
| CoreML, CPU_AND_GPU        | 11,125 |      44  | ~18s             |
| CoreML, CPU_AND_NE (ANE)   | 25,406 |      99  | ~8s              |

- 4.0x over the shipping encoder; fidelity vs fp32 torch reference:
  mean cosine 0.99968, min 0.99963 over real source chunks (safer than
  our own f16 sidecar rounding).
- CoreML-on-GPU alone is 1.75x candle, quantifying candle's Metal
  kernel overhead as roughly half the gap.
- ANE is also the low-power engine: sustained indexing should throttle
  far less than the GPU path.

Environment quirks (all handled in ane_convert.py):
- transformers 5.x + the nomic remote code needs safe_serialization=True;
- torch.jit.trace needs check_trace=False (rope cache branch) and a
  warmup call first; torch.export is NOT usable (unsupported alias op);
- coremltools 9.0 needs numpy<2 (int(np.array([x])) removal) and
  torch<=2.7.

Productionizing needs: static-shape buckets (e.g. 64/128/256/512
mlpackages or enumerated shapes), CoreML invocation from Rust (objc2
CoreML bindings or a sidecar process), and a recall eval on the
switched vectors. Until then this stays a measured prototype.

Run: uv venv; uv pip install 'torch==2.7.0' transformers coremltools 'numpy<2' einops
     python ane_convert.py 8 256 && python ane_bench.py && python ane_fidelity.py

**Update 2026-08-30**: shipped as `src/coreml.rs`. The encoder auto-uses
compiled models from `<cache>/coreml/` when present (this directory's
scripts produce them); `HIPS_ENCODER=candle` forces the fallback.
Measured e2e: 810-chunk cold index 28.6s -> 13.0s (2.2x; first-ever run
adds one-time ANE plan compilation the OS then caches), hybrid query
0.10s wall including model load.
