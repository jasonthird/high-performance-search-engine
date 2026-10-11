# Persistent full-query cache results

These measurements used synchronous saves. See the [current asynchronous implementation](query-cache-async-results.md) for response-path timings.

The ordinary query encoder checks its memory LRU, then a shared JSON disk cache, then CodeRankEmbed. A miss computes the full prefixed query vector and persists it. Keys match complete query text; indexing is independent.

All 295 benchmark inputs were new exact queries, so all 295 first-pass lookups missed. All 129 long issues called the full model. All 295 immediate repeats and all 295 lookups in a fresh encoder used cached vectors with zero query inference calls. Every vector matched an uncached full-model prediction exactly, including after JSON round trip.

| Queries | N | Full model ms | Initial miss + save ms | Memory hit ms | Disk hit after encoder reload ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| agent | 37 | 9.617 | 12.286 | 0.0026 | 0.044 |
| idq | 129 | 10.861 | 13.532 | 0.0028 | 0.045 |
| issue | 129 | 13.723 | 17.765 | 0.0031 | 0.048 |

Stored 295 JSON entries, 2.96 MB total. Each holds the original 768-dimensional float32 query vector; no word averaging or int8 quantization.

Times are medians for the resident encoder. Encoder loading, weight fingerprinting and mandatory inference warmup are excluded. The encoder still loads and warms up on startup; a cache hit skips inference for the requested query.

This validates query-vector equality rather than rerunning labeled document retrieval. The identical vectors preserve the full encoder’s scores for the same document index and query; original full-model retrieval metrics are recorded in [the retrieval benchmark](results.md).

Four cache unit tests passed (exact keys, persistence, corrupt/mismatched entries, concurrent writes and model fingerprints). Native model tests passed for complete-issue misses, repeats, reload and query/index isolation, and against the reference model vectors. Two separate CLI search processes returned identical results with one query entry saved. A native MCP test confirmed cache hits after a server restart and after switching repository indexes, with zero query inference calls. Indexing documents did not populate query entries. Existing runtime checks passed (12 tests; four environment-specific tests remained ignored), followed by the new MCP integration test.

Raw per-query timings are in `target/exact-query-cache-final/report.json` (generated and ignored by Git). Artifact hashes and full-precision summaries are in [query-cache-results.json](query-cache-results.json).

Reproduce with an empty cache directory:

```sh
cargo build --release --offline --features semantic --example query_cache
HIPS_MODEL_OFFLINE=1 HIPS_ENCODER=ggml HPS_EMBED_DEVICE=metal \
  HIPS_QUERY_CACHE_DIR="$PWD/target/exact-query-cache-final/cache" \
  target/release/examples/query_cache --out target/exact-query-cache-final/report.json
```
