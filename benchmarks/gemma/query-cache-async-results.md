# Asynchronous query-cache writes

Search caches each new vector in memory immediately and queues its JSON persistence on one background worker. Serialization, file creation, synchronization and atomic replacement run on that worker. The response path uses a nonblocking bounded queue and never waits for the save.

All 295 complete queries missed on the first pass. Their vectors, immediate repeats and persisted reloads matched the uncached full-model predictions exactly. All 295 entries persisted. The benchmark explicitly drains writes before testing reload, outside the query timing.

| Queries | N | Model-only ms | Miss + enqueue ms | Memory hit ms | Reloaded disk hit ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| agent | 37 | 7.649 | 7.506 | 0.0017 | 0.051 |
| idq | 129 | 8.263 | 7.786 | 0.0015 | 0.050 |
| issue | 129 | 10.900 | 10.660 | 0.0018 | 0.054 |

These are resident-encoder medians. Small differences between model-only and miss timings reflect measurement variation; the disk save no longer sits on the response path. Startup loading and warmup remain outside these timings.

Six cache unit tests passed, including an intentionally stalled writer, queue saturation, write failure recovery, concurrent atomic writes and invalid entries. The native MCP integration test passed across server restarts and different repository indexes.

The queue holds up to 64 pending saves. If full, the disk save is skipped while the memory entry remains usable. Normal CLI/MCP shutdown flushes client output before draining queued writes; long-lived clients do not drain per request. Abrupt termination can lose pending cache entries.

Reproduce with a fresh cache directory:

```sh
cargo build --release --offline --features semantic --example query_cache
HIPS_MODEL_OFFLINE=1 HIPS_ENCODER=ggml HPS_EMBED_DEVICE=metal \
  HIPS_QUERY_CACHE_DIR="$PWD/target/query-cache-async/cache" \
  target/release/examples/query_cache --out target/query-cache-async/report.json
```

[Full precision metrics](query-cache-async-results.json). [Earlier synchronous-save measurements](query-cache-results.md).
