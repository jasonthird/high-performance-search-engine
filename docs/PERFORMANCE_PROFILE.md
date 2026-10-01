# Search and embedding profile

Profiled on an Apple M3 MacBook Air, 10-core GPU, 24 GB unified memory.
The starting working tree was committed as `910af1a` before compilation.
All timings below are local release-build measurements, not portable targets.

## Changes retained

- Cached hybrid queries no longer create a BM25 helper thread. Uncached
  hybrid queries still overlap lexical work with model inference. Segmented
  semantic-only queries also avoid spawning an empty lexical task.
- GPU bisection compiles with `--features semantic,gpu` without adding unsafe
  thread-sharing implementations. Siblings use local async futures, submitting
  GPU work before awaiting completion. A private executor polls completion
  every 50 microseconds; it neither moves Metal handles to Rayon workers nor
  requires a nested Tokio runtime. CPU leaves run in Rayon after the GPU
  levels finish, so a long CPU leaf cannot stall submission of other GPU work.
  A cancellation guard waits before page-backed resources can be released.
  Parent scratch buffers are released before child allocation. Four existing
  unsafe Send/Sync declarations were removed; no unsafe blocks were added.
- The scoped tokenizer producer borrows the immutable tokenizer instead of
  cloning its vocabulary/caches on every call. This removes an allocation
  cost, but the document benchmark did not show a material throughput gain.
- CoreML export uses a finite attention mask and validates the wrapper and
  TorchScript trace against the untouched FP32 model. A separate validator
  checks converted-model fidelity, masks, partial batches and row identity.
- Query encoding can select the smallest fitting installed batch-1 CoreML
  shape. The configured query token cap is unchanged. No extra model was
  installed and no existing model or embedding index was replaced.

## Cached search: measured improvement

Five alternating before/after runs used the same existing 64-token CoreML
query model and repository index, with 21 distinct queries followed by their
cached repetitions. The table gives the median of each run's mean latency:

| Query state | Before | After |
|---|---:|---:|
| Uncached hybrid search | 4.199 ms | 4.192 ms |
| Cached hybrid search | 0.272 ms | 0.241 ms |

Cached latency fell about 11%, or 31 microseconds. All recorded vectors,
result IDs/ranks and cache-hit flags matched across the ten runs. The
uncached path is effectively unchanged. This measures the engine facade,
not MCP transport, file watching, process launch or first model compilation.
Timing varies with CPU scheduling and thermal state; the change primarily
removes thread creation/join overhead rather than making vector math faster.

## Exact vector scoring: the tested GPU path was slower

The archived `docs/benchmarks/accelerator_bench.rs` uses safe Candle tensor APIs to compare
resident FP32 Metal matrix-vector multiplication with the existing CPU SIMD
scan over stored FP16 vectors. It includes query upload, score readback and
top-200 selection; resident tensor preparation is reported separately.
Queries remain FP32, avoiding a change to query precision merely to speed
the GPU path. Configurations alternate for 16 rounds per corpus size.

| Vectors | CPU median | Metal median | Metal preparation |
|---:|---:|---:|---:|
| 1,500 | 0.158 ms | 0.432 ms | 7.99 ms |
| 10,000 | 0.454 ms | 0.878 ms | 5.37 ms |
| 100,000 | 2.808 ms | 4.493 ms | 110.66 ms |
| 500,000 | 12.418 ms | 22.775 ms | 1,182.45 ms |

All measured top-200 sets matched; maximum score error was below 4.5e-8.
The extra resident FP32 matrix also doubles the vector bytes compared with
the FP16 sidecar. This GPU experiment remains a benchmark, not a search
default. This source predates the migration from Candle to ggml and is kept
as historical benchmark source, outside Cargo's runnable examples. Re-running
it requires a separate Candle-enabled harness; the current crate does not
depend on Candle. It does not rule out a future fused FP16-storage/FP32-accumulation
Metal kernel or GPU batching of many simultaneous queries.

A separate single-range CPU scan change showed no consistent improvement
in four alternating before/after sweeps and was removed. HNSW remains the
existing route for large segments; these exact-scan timings do not measure
graph retrieval or end-to-end search.

## Document inference and model fidelity

The fixed sample contains 128 chunks from an archive of `910af1a`. Content
caches are bypassed; model loading and shape warmup are excluded. The current
CoreML scheduler takes about 1.5 seconds for the sample. Before/after trials
of tokenizer borrowing were effectively tied (roughly 1.49–1.52 seconds).
These are sampled chunks, not an end-to-end cold repository build.

The per-prediction profile of the installed document family was:

| CoreML shape | Warm prediction median |
|---|---:|
| Batch 8, sequence 128 | 33.9 ms |
| Batch 8, sequence 256 | 82.3 ms |
| Batch 8, sequence 512 | 224.6 ms |

The installed batch-8/64 artifact fails the existing permutation check, so
short batches use the 128-token shape. Rebuilding batch-8/64 and batch-8/96
with the original converter reproduced severe fidelity failures.

The generic HF mask computes `(1 - mask) * -FLT_MAX`. FP16 conversion turns
the constant into negative infinity; unmasked positions then multiply zero
by infinity. A finite `-10000` additive mask fixes the tested CPU/GPU outputs
without changing the sampled FP32 reference outputs beyond the asserted
tolerance. The resulting batch-8/64 model matched FP32 with minimum cosine
0.999826 on CPU and 0.999995 on CoreML GPU, but **still failed on ANE**.
Thus the overflow is a demonstrated bug, not a complete explanation or fix
for the ANE shape problem. The valid GPU path took about 46 ms per batch,
slower than the existing 128-token CPU+ANE configuration at about 34 ms.
No rejected replacement was installed. The installed 128-token artifact
passed the stronger sampled check with minimum reference cosine 0.999593.

### Short-batch ANE investigation (September 9)

Layer-by-layer exports with the finite mask localize the remaining batch-8/64
failure to the first attention block. With CPU_AND_NE, rows 0/1/4/5 remain
correct while rows 2/3/6/7 diverge. The following probes compare the entire
intermediate tensors against FP32 using identical inputs:

| Export endpoint | Minimum per-row cosine |
|---|---:|
| Embedding stem | 0.9999997 |
| Packed QKV projection | 0.9999997 |
| Rotary embeddings | 0.9999997 |
| Query × key scores | 0.9999987 |
| Masked softmax | 0.9999978 |
| Softmax × value context | 0.0193 |
| Context with independent value projection | 0.9999982 |

Exporting the full hidden sequence or removing CLS normalization does not
fix the failure. Python receives an owned, contiguous output array, so this
reproduction does not depend on the Rust output-copy implementation. The
second matrix multiplication passes in isolation when its inputs come from
materialized FP32 reference tensors. Transposing the multiplication,
splitting it by batch row, or padding its reduction dimension does not fix
the composed attention graph.

Computing V independently with the same last third of the QKV projection
weights fixes the isolated attention context. This points to the packed
value-view execution path in the composed CoreML graph; it does not establish
the precise compiler/driver defect. MLComputePlan assigns both attention
matrix multiplications, the projection, and softmax to the Neural Engine in
the failing graph. The passing isolated multiplication is also assigned to
the Neural Engine. These are planned devices, not execution-trace samples.

The full batch-8/64 model with the independent value projection passes all
nine source-text fidelity cases: minimum FP32 cosine **0.999480**, minimum
row-permutation cosine **0.9999998**. The exporter checks its rewritten model
and trace against an untouched FP32 model before CoreML conversion. The
workaround applies to batched exports shorter than 128 tokens, retains the
original weights, and adds no unsafe Rust code.
The compiled `.mlmodelc` artifact independently passes the same nine cases
with the same minimum cosines.
The rebuilt batch-8/96 package also passes all nine cases, with minimum
FP32 cosine **0.999579** and row-permutation cosine **0.9999998**. This export
also exercises the full-length, mixed-length, and partial-batch FP32/trace
assertions in the final converter.

A compiled candidate and the installed 128-token model were compared with
identical mixed-length token rows (padded to each shape), five warmup calls
per model, then 60 rounds alternating their order. Median warm prediction
was **31.7 ms versus 64.9 ms** (about 2.05× faster); p90 was 54.3 versus
111.1 ms. Both matched the synthetic-input FP32 reference above 0.99914.
Absolute latency varied substantially from the earlier profile; compare
these paired measurements with each other, not with earlier isolated runs.
This is inference latency, not an end-to-end indexing speedup.

The corrected model's compute plan assigns all 24 attention matrix
multiplications, 72 linear operations, 12 softmax operations, and 24 encoder
layer normalizations to the Neural Engine. Input embedding lookup and its
initial layer normalization remain assigned to CPU. Installed models and
existing embedding indexes were not replaced. The candidate package and
compiled model are under `/tmp/hips-ane-investigation/coderank_b8_s64.*`.

An experimental finite-mask batch-1/32 query model passed the sampled ANE
configuration check (minimum reference cosine 0.999612). Three alternating
query-family comparisons showed only a small difference: median run-average
uncached search latency 4.24 versus 4.11 ms, with identical measured vectors
and top-10 results across 21 queries. This is a marginal result on a small
sample, not sufficient reason to install another model by default.

CoreML was configured for CPU and Neural Engine during these runs. Its
configuration does not identify the execution device of each operation;
the runtime still reports `execution_device: not_observed`.

## Reproduce

Run hardware benchmarks individually, with other builds/tests stopped.
They require macOS accelerator access; a CPU fallback is not an ANE timing.

```sh
cargo build --release --features semantic,gpu --bin hips \
  --example query_profile --example cold_embed_bench
cargo run --release --features semantic --example query_profile -- INDEX QUERIES OUTPUT_JSON
cargo run --release --features semantic \
  --example cold_embed_bench -- FIXED_CORPUS 128
python scripts/ane-prototype/ane_validate.py MODEL --batch 8 --seq 128 --root FIXED_CORPUS
```

`query_profile` separates index/model load and shape warmup from the measured
queries, records actual cache-hit flags and per-stage times, and saves vectors
and result IDs for before/after comparisons. Use unique query lines to measure
uncached inference, followed by the same lines for cache hits. Compiler and
model dependencies for the conversion experiments were Torch 2.7.0,
coremltools 9.0, NumPy 1.26.4, and Transformers 5.16.1.

The release test suite requires prebuilt grammars and loopback access:

```sh
HIPS_GRAMMAR_DIR="$PWD/grammars/target/release" HIPS_GRAMMAR_OFFLINE=1 \
  HIPS_REQUIRE_METAL_TESTS=1 HIPS_NO_LOG=1 \
  cargo test --release --features semantic,gpu --lib --tests
```

The GPU regression asserts both permutation/cost correctness and at least
two simultaneously outstanding sibling commands, with no commands remaining
when the operation returns. Model-dependent ignored integration tests are
separate from the explicitly executed CoreML fidelity and query benchmarks.
The full release run passed 189 tests, with zero failures and four ignored
model-dependent tests.
