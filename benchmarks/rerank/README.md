# Second-stage reranker experiment

Today `--mode rerank` scores BM25 candidates by cosine against the stored
CodeRankEmbed document vectors. That is the same signal `--mode hybrid`
already retrieves with, so it cannot add information; it was never measured.
This benchmark measures it, and compares small cross-encoders that read the
query and each candidate chunk together.

It reuses the [gemma4agentcomp setup](../gemma/README.md): 129 task snapshots,
code-only 80-line windows with a 60-line stride, issue/idq/agent queries
bounded to 8,000 characters, and file-level R@k for the first reference file.
Each variant is cut to 20 chunks before duplicate paths collapse.

The first stage is unchanged hips: BM25, exact semantic, and weighted hybrid
(alpha 0.15, 200 semantic candidates). The worker also returns a 50-chunk
hybrid pool and a BM25 pool. Each cross-encoder scores the top 50 hybrid
chunks and the top 20 BM25 chunks, with a 512-token pair limit. Scores are
stored, so the report evaluates:

- `hybrid top20` / `hybrid top50`: reranker order over that many chunks.
- `... rrf`: reciprocal rank fusion (k = 60) of hybrid and reranker ranks.
- `bm25 top20`: reranker order over BM25's top 20.

Models are limited to under 300M parameters; LLM rerankers (Qwen3-Reranker,
jina-reranker-v3) are out of scope. Latency is PyTorch on an Apple M3, each
model on its fastest measured device (DeBERTa models on the CPU, the rest on
MPS in fp16). It shows relative cost, not a ggml implementation's speed.
For scale, a CodeRankEmbed query encode takes about 7 ms on the same machine.

## Reproduce

```sh
cargo build --release --offline --features semantic --example static_query

HIPS_MODEL_OFFLINE=1 HIPS_ENCODER=ggml HPS_EMBED_DEVICE=metal \
  uv run --python 3.12 scripts/bench_rerank.py \
  --snapshots "$HOME/gitProjects/gemma4agentcomp/data/competition/snapshots" \
  --words data/static-query/words-768.json \
  --out target/rerank-bench

python3 scripts/report_rerank.py --results target/rerank-bench
```

The worker loads the static word table only because it shares the static
query benchmark's binary; static variants are not reported here. `--limit 4`
runs a balanced smoke test, and completed tasks resume when inputs match.

## Findings (2026-10-12, full 129-task run)

See [results.md](results.md). Significance is a two-sided sign test over
queries whose reciprocal file rank changed versus `hybrid_full`.

- The cosine `rerank` mode does no harm; it is redundant. Its MRR is within
  ±0.015 of hybrid on every query type, and it lowers issue R@10 from 0.791 to 0.752.
- Only one result is both significant and useful: jina-v2-base fused with
  hybrid by RRF on issue queries. MRR goes from 0.488 to 0.530 (37 better, 17 worse,
  p = 0.009), at about 1.5 s per query for 20 chunks. That model is the largest
  tested (278M) and is licensed CC-BY-NC.
- On agent queries (n = 37) no reranker beats hybrid. The best result is
  jina-v1-turbo over the top 50: MRR 0.600 to 0.602.
- On identifier queries, BM25 is already stronger than hybrid (MRR 0.390 vs 0.372). Reranking BM25's
  top 20 with jina-v1-turbo gives 0.443 (45 better, 21 worse, p = 0.004).
- Reranking BM25 candidates for issue queries is significantly worse than
  hybrid for every model, by 0.05 to 0.11 MRR.
- Replacing hybrid's order outright is less reliable than RRF fusion. Going from 50
  to 20 candidates rarely costs quality and cuts latency by about 2.5x.
- No reranker improves R@10 meaningfully. They reorder the files hybrid
  already found, and cannot recover files the first stage missed.
