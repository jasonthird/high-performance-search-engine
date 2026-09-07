# CodeRankEmbed CoreML conversion and benchmarks

CoreML/Apple Neural Engine support is implemented in
[`src/coreml.rs`](../../src/coreml.rs) and selected by
[`src/embedder.rs`](../../src/embedder.rs) in macOS builds with
`--features semantic`. The directory keeps its historical `ane-prototype`
name; its scripts now provide local model conversion and benchmark tools.

The runtime detects compiled models under `<csearch cache>/coreml/`.
The cache root is `CSEARCH_CACHE_DIR`, otherwise
`$XDG_CACHE_HOME/csearch`, otherwise `~/.cache/csearch`.
Install batch-8 document models at sequence 64/128/256/512 and one batch-1
query model at sequence 64. Shapes load on first use; document batches use
the smallest fitting shape. A query-only installation cannot encode
documents. `HIPS_ENCODER=candle` forces the portable backend.

Compiled CoreML models are not downloaded or converted by hips. The runtime
still obtains the original Hugging Face model files (configuration,
tokenizer, and weights paths); these have their own cache and may download
on first use. Python is needed for conversion, not for hips queries.

## Build and install the model family

Run on macOS with the Xcode `coremlcompiler` tool available. The conversion
script targets macOS 14. The environment below reflects the recorded
conversion setup; these are reproduction pins, not a claim about the
latest supported Python packages.

```sh
cd scripts/ane-prototype
uv venv
uv pip install 'torch==2.7.0' transformers 'coremltools==9.0' 'numpy<2' einops

coreml_output_dir="${CSEARCH_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/csearch}/coreml"
mkdir -p "$coreml_output_dir"
for seq in 64 128 256 512; do
  uv run python ane_convert.py 8 "$seq"
  xcrun coremlcompiler compile "coderank_b8_s$seq.mlpackage" "$coreml_output_dir"
done
uv run python ane_convert.py 1 64
xcrun coremlcompiler compile coderank_b1_s64.mlpackage "$coreml_output_dir"
```

The conversion script emits `.mlpackage` files in the working directory;
the compiler produces the `coderank_b{B}_s{S}.mlmodelc` directories that
the Rust loader recognizes. Models can occupy substantial disk space:
this is a separate optional model family, unrelated to the grammar cache.
Regenerate the family together if changing weights or conversion settings.

## Historical measurements

Measured 2026-08-29 on an M3 / 10-core GPU / 24 GB, batch 8 × sequence 256,
F16. These figures compare the encoder implementations at that time, before
later Candle changes; they are not a new benchmark of the current fork.

| Runtime | Tokens/s | Chunks/s | 810-chunk corpus |
|---|---:|---:|---:|
| Candle + Metal (then-current baseline) | 6,300 | 28 | 28.6 s |
| CoreML, CPU_AND_GPU | 11,125 | 44 | ~18 s |
| CoreML, CPU_AND_NE (ANE) | 25,406 | 99 | ~8 s |

The prototype's ANE throughput was 4.0× the measured Candle baseline.
Against FP32 PyTorch on sampled source chunks, mean cosine was 0.99968 and
minimum cosine 0.99963. This is an embedding-fidelity check, not a
repository-wide recall evaluation.

The 2026-08-30 Rust integration measured an 810-chunk cold index at
13.0 s versus 28.6 s with Candle, and a hybrid query at 0.10 s wall time
including model loading. The first use of each shape also pays ANE plan
compilation, which the OS caches.

## Reproduce the micro-benchmarks

After producing `coderank_b8_s256.mlpackage`:

```sh
uv run python ane_bench.py
uv run python ane_fidelity.py
```

`ane_bench.py` compares CoreML compute-unit choices with synthetic token
inputs. `ane_fidelity.py` compares against FP32 PyTorch on source chunks;
it currently hard-codes the author's checkout path. Adjust that source
glob to your checkout before running it. Neither script is a full search
recall benchmark; use the project's `eval-gen` / `eval-code` workflow
for retrieval quality.

Conversion workarounds in `ane_convert.py` include
`safe_serialization=True` for the Nomic remote model code, a warmup before
`torch.jit.trace`, and `check_trace=False` for the RoPE cache branch.
The recorded setup used NumPy below 2 and Torch 2.7.0 with coremltools 9.0.
