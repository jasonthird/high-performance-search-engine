# CPU encoder quantization

CodeRankEmbed GGUF weights can be converted from F16 to Q8_0 or Q4_0
for a smaller offline artifact. This changes model weights, not the FP16
document-vector storage format. Reduced size does not guarantee lower
latency or unchanged retrieval quality; measure both on the target CPU.

Build without GPU or tree-sitter dependencies for a portable CPU experiment:

```sh
git submodule update --init vendor/ggml
HIPS_GGML_GPU=0 GGML_NATIVE=OFF cargo build --release --locked --no-default-features --features semantic
python3 -m venv .venv-quantize
.venv-quantize/bin/pip install gguf==0.19.0 numpy
.venv-quantize/bin/python scripts/quantize_encoder.py CodeRankEmbed-f16.gguf CodeRankEmbed-q8_0.gguf
.venv-quantize/bin/python scripts/quantize_encoder.py CodeRankEmbed-f16.gguf CodeRankEmbed-q4_0.gguf --type Q4_0
```

The converter preserves metadata and tensor shapes, quantizes eligible
two-dimensional weight matrices, and retains normalization vectors and
biases at their original precision. Start from the original F16 model.
The engine casts the token-type embedding row to F32 before addition;
without this, a quantized token-type matrix aborts in ggml's CPU add.

Run with a cached model and a **fresh index/cache per precision and sequence
length**. Changing the encoder artifact or truncation while reusing cached
document embeddings can mix incompatible outputs.

```sh
export HIPS_ENCODER=ggml HPS_EMBED_DEVICE=cpu HIPS_MODEL_OFFLINE=1
export HIPS_GRAMMAR_OFFLINE=1 HPS_EMBED_THREADS=2
export HIPS_MODEL_PATH=/absolute/path/CodeRankEmbed-q8_0.gguf
export CSEARCH_CACHE_DIR=/tmp/hips-q8-seq128-cache
export HPS_EMBED_MAX_SEQ=128
target/release/hips -v index-repo --root /path/to/repo --index /tmp/hips-q8-seq128-index
target/release/hips -v search --index /tmp/hips-q8-seq128-index --mode semantic --query 'parse content type headers'
```

CPU thread settings are not a hard memory/CPU sandbox; use container limits
when benchmarking a restricted environment. The smaller sequence cap is
an independent quality/speed tradeoff from weight quantization.

Regression check without a downloaded model:

```sh
HIPS_GGML_GPU=0 cargo test --release --no-default-features --features semantic --lib quantized_type_embedding_addition_runs_on_cpu
```

For full encoder checks, set `HIPS_MODEL_PATH` to the quantized artifact and
run the model-backed ggml tests. Test each target backend separately;
CPU verification does not establish GPU backend compatibility.
