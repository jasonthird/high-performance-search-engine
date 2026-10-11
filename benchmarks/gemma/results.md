# 768-dimensional static query results

Completed 129 task snapshots. Each static/full pair scores identical full-model document vectors.

Word table: `data/static-query/words-768.json`, 67.3 MB.

| Queries | Encoder / ranking | N | R@1 | R@10 | MRR | Encode ms (median, uncached) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| agent | bm25 | 37 | 0.405 | 0.757 | 0.536 | — |
| agent | semantic_full | 37 | 0.378 | 0.757 | 0.525 | 6.8301 |
| agent | semantic_static | 37 | 0.297 | 0.703 | 0.408 | 0.0103 |
| agent | hybrid_full | 37 | 0.486 | 0.838 | 0.600 | 6.8301 |
| agent | hybrid_static | 37 | 0.432 | 0.838 | 0.562 | 0.0103 |
| idq | bm25 | 129 | 0.256 | 0.682 | 0.390 | — |
| idq | semantic_full | 129 | 0.217 | 0.527 | 0.309 | 7.2348 |
| idq | semantic_static | 129 | 0.178 | 0.512 | 0.285 | 0.0157 |
| idq | hybrid_full | 129 | 0.256 | 0.620 | 0.372 | 7.2348 |
| idq | hybrid_static | 129 | 0.302 | 0.659 | 0.420 | 0.0157 |
| issue | bm25 | 129 | 0.209 | 0.550 | 0.330 | — |
| issue | semantic_full | 129 | 0.287 | 0.721 | 0.420 | 10.3247 |
| issue | semantic_static | 129 | 0.062 | 0.581 | 0.225 | 0.2392 |
| issue | hybrid_full | 129 | 0.341 | 0.791 | 0.488 | 10.3247 |
| issue | hybrid_static | 129 | 0.194 | 0.636 | 0.318 | 0.2392 |

Variant errors: 2. Median vocabulary coverage: 80.0%.

Backend: `ggml`, device `MTL0 (Apple M3)`.
Word-table load: 1031.8 ms; both full encoders loaded in 344.7 ms.

Latency measures a resident process. It excludes full-query cache hits and does not promise lower cold-start latency.
The static table is an approximation; JSON lookups do not reproduce contextual model predictions.

See [methodology and commands](README.md). Full precision metrics, per-repository results and input hashes are in the adjacent JSON.

Raw outputs: `target/static-query-bench-gpu/raw.jsonl` (generated, ignored by Git).

## Queries requiring fallback

- `rich_3105:agent`: `3104` — no known terms in static query; use full CodeRankEmbed or BM25.

These count as misses in the measured static variants. The search command supports `--fallback-full`.

## Paired hybrid top-10 outcomes

| Queries | Both succeed | Full only | Static only | Neither |
| --- | ---: | ---: | ---: | ---: |
| agent | 30 | 1 | 1 | 5 |
| idq | 70 | 10 | 15 | 34 |
| issue | 76 | 26 | 6 | 21 |

Equal aggregate success rates can hide different query outcomes; compare ranking quality as well.
