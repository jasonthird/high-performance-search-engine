# CodeRankEmbed CoreML conversion and benchmarks

CoreML/Apple Neural Engine support is implemented in
[`src/coreml.rs`](../../src/coreml.rs) and selected by
[`src/embedder.rs`](../../src/embedder.rs) in macOS builds with
`--features semantic`. The directory keeps its historical `ane-prototype`
name; its scripts now provide local model conversion and benchmark tools.

The runtime detects compiled models under `<csearch cache>/coreml/`.
The cache root is `CSEARCH_CACHE_DIR`, otherwise
`$XDG_CACHE_HOME/csearch`, otherwise `~/.cache/csearch`.
Install validated batch-8 document models at sequence 64/128/256/512 and one batch-1
query model at sequence 64. Shapes load on first use; document batches use
the smallest fitting shape. A query-only installation cannot encode
documents. The runtime falls back to ggml if the installed shapes cannot
cover the configured token cap. `HIPS_ENCODER=ggml` forces the portable backend.
Document shapes are checked for batch-permutation equivalence on first use.
Failed shapes are skipped for larger fitting shapes; if none pass, use ggml
and regenerate the family. This check caught the locally installed batch-8/64
artifact during the September 8 scheduler benchmark; installed files were not
changed. Validation adds cold-start work and is a smoke test, not a full
comparison against the original weights.
The profiling follow-up reproduced the short-shape failure. The export
wrapper now uses a finite additive attention mask: the original `-FLT_MAX`
overflows to `-inf` in FP16 and can produce NaNs. This fixes the CPU/GPU
failure. A second failure was isolated to the packed value tensor in the
attention calculation on CPU_AND_NE. Short batched exports now compute the
value projection independently with the same weights, avoiding that path.
The rebuilt batch-8/64 model passes the nine-case ANE fidelity check (minimum
FP32 cosine 0.999480; row-permutation cosine 0.9999998) and its compiled
artifact is about twice as fast as the 128-token fallback in a paired warm
benchmark on this M3. A rebuilt batch-8/96 package also passes (minimum
FP32 cosine 0.999579). Existing installed models remain unchanged.
Do not install a regenerated shape merely because conversion succeeds.
Document scheduling now uses actual token lengths and a bounded CPU-prefetch
queue. Recheck the historical timings below with
`cargo run --release --features semantic --example cold_embed_bench -- . 128`
from the repository root; use `HIPS_ENCODER=ggml` for the other backend.

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
  uv run python ane_convert.py 8 "$seq" &&
  uv run python ane_validate.py "coderank_b8_s$seq.mlpackage" --batch 8 --seq "$seq" --root ../.. &&
  xcrun coremlcompiler compile "coderank_b8_s$seq.mlpackage" "$coreml_output_dir"
done
uv run python ane_convert.py 1 64 &&
uv run python ane_validate.py coderank_b1_s64.mlpackage --batch 1 --seq 64 --root ../.. &&
xcrun coremlcompiler compile coderank_b1_s64.mlpackage "$coreml_output_dir"
```

The conversion script emits `.mlpackage` files in the working directory;
the compiler produces the `coderank_b{B}_s{S}.mlmodelc` directories that
the Rust loader recognizes. Models can occupy substantial disk space:
this is a separate optional model family, unrelated to the grammar cache.
Regenerate the family together if changing weights or conversion settings.

Validate each package **before** compiling or installing it. The validator
compares nine full, mixed-length and partial batches against untouched FP32
CodeRankEmbed, and reverses row order to check batch identity:

```sh
uv run python ane_validate.py coderank_b8_s128.mlpackage --batch 8 --seq 128 --root ../..
```

The default compute units are `CPU_AND_NE`; `--compute-units CPU_ONLY` and
`--compute-units CPU_AND_GPU` isolate device-specific failures. Passing on
one device does not validate another. The validator also accepts `.mlmodelc`
directories. Conversion itself now checks the finite-mask wrapper and its
TorchScript trace against the original FP32 model before exporting. For
batched shapes shorter than 128 tokens, that includes the independent value
projection workaround. The reference weights/model are not rewritten.

The Rust loader supports multiple batch-1 query shapes, selecting the smallest
that fits without lowering the configured 64-token cap. An optional validated
`coderank_b1_s32.mlmodelc` can coexist with `coderank_b1_s64.mlmodelc`.
The measured latency difference on this M3 was small; the profiling work did
not install extra query models or replace existing document models. See
[`docs/PERFORMANCE_PROFILE.md`](../../docs/PERFORMANCE_PROFILE.md).

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
