# Persistent word lookup results

Completed 129 task snapshots. Each static/full pair scores identical full-model document vectors.

Word table: `data/static-query/words-768.json`, 67.3 MB.

| Queries | Encoder / ranking | N | R@1 | R@10 | MRR | Encode ms (median; full cache hits excluded) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| agent | bm25 | 37 | 0.405 | 0.757 | 0.536 | — |
| agent | semantic_full | 37 | 0.378 | 0.757 | 0.525 | 6.8408 |
| agent | semantic_static | 37 | 0.297 | 0.703 | 0.408 | 0.0103 |
| agent | hybrid_full | 37 | 0.486 | 0.838 | 0.600 | 6.8408 |
| agent | hybrid_static | 37 | 0.432 | 0.838 | 0.562 | 0.0103 |
| agent | semantic_dynamic | 37 | 0.405 | 0.730 | 0.495 | 0.0100 |
| agent | hybrid_dynamic | 37 | 0.514 | 0.838 | 0.611 | 0.0100 |
| idq | bm25 | 129 | 0.256 | 0.682 | 0.390 | — |
| idq | semantic_full | 129 | 0.217 | 0.527 | 0.309 | 7.6168 |
| idq | semantic_static | 129 | 0.178 | 0.512 | 0.285 | 0.0162 |
| idq | hybrid_full | 129 | 0.256 | 0.620 | 0.372 | 7.6168 |
| idq | hybrid_static | 129 | 0.302 | 0.659 | 0.420 | 0.0162 |
| idq | semantic_dynamic | 129 | 0.209 | 0.527 | 0.314 | 0.0116 |
| idq | hybrid_dynamic | 129 | 0.302 | 0.651 | 0.410 | 0.0116 |
| issue | bm25 | 129 | 0.209 | 0.550 | 0.330 | — |
| issue | semantic_full | 129 | 0.287 | 0.721 | 0.420 | 16.5332 |
| issue | semantic_static | 129 | 0.062 | 0.581 | 0.225 | 0.1215 |
| issue | hybrid_full | 129 | 0.341 | 0.791 | 0.488 | 16.5332 |
| issue | hybrid_static | 129 | 0.194 | 0.636 | 0.318 | 0.1215 |
| issue | semantic_dynamic | 129 | 0.070 | 0.512 | 0.208 | 20.5369 |
| issue | hybrid_dynamic | 129 | 0.225 | 0.597 | 0.335 | 20.5369 |

Variant errors: 2. Median vocabulary coverage: 80.0%.

Backend: `ggml`, device `MTL0 (Apple M3)`.
Word-table load: 530.5 ms; both full encoders loaded in 339.7 ms.

Latency measures a resident process. It excludes full-query cache hits and does not promise lower cold-start latency.
The static table is an approximation; JSON lookups do not reproduce contextual model predictions.

See [methodology and commands](README.md). Full precision metrics, per-repository results and input hashes are in the adjacent JSON.

Raw outputs: `target/growing-query-bench-buffered/raw.jsonl` (generated, ignored by Git).

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

## Persistent word lookup

Added 861 distinct words across 130/295 first-pass queries; sidecar 1.66 MB.
All 295 immediate repeats reproduced the same vector with zero model calls. Median repeat encoding: 0.0089 ms.
Median first-pass encoding for queries with misses: 20.577 ms, including atomic JSON persistence.

Misses are encoded as individual words using the base table’s prefix setting, then quantized to the same 768-dimensional format. The full query is still an approximation.
This sidecar assumes a single writer. Document vectors are reused from the original experiment; query timing comparisons are measured again in this run.
Growing first-pass timings include existing word-cache hits from earlier benchmark queries. Issue queries are submitted before identifier and agent queries for each task.

| Queries | With word misses | First-pass encode ms | Repeated encode ms |
| --- | ---: | ---: | ---: |
| agent | 4/37 | 0.0100 | 0.0039 |
| idq | 5/129 | 0.0116 | 0.0041 |
| issue | 121/129 | 20.5369 | 0.0388 |

| Queries | Full and lookup succeed | Full only | Lookup only | Neither |
| --- | ---: | ---: | ---: | ---: |
| agent | 30 | 1 | 1 | 5 |
| idq | 71 | 9 | 13 | 36 |
| issue | 72 | 30 | 5 | 22 |
