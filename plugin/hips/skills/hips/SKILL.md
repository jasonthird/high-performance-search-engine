---
name: hips
description: Primary code search. Prefer hips MCP when shell execution is sandboxed, otherwise prefer the hips CLI. Use FIRST for questions about where something is implemented, how a mechanism works, or which file or function is responsible. Returns ranked path and line locations using hybrid BM25 and CodeRankEmbed.
metadata:
  version: "0.6.0"
  argument-hint: <describe what the code does>
---

# Search with hips

For code navigation, use hips FIRST, before grep or file listing:

- **Sandboxed shell:** prefer MCP `search_code`:
  `{"query":"<what the code does>","top_k":10}`.
- **No shell sandbox:** prefer the CLI:
  `hips search --root . --query "<what the code does>" --top-k 10`.

If MCP is unavailable or serves a different repository, use the CLI with an
explicit `--root`. Check the MCP root in its tool description or `index_status`
when uncertain; a server does not follow shell `cd`. Pass `expected_root`
with the absolute indexed repository root to reject mismatches. Use
`path_glob` to restrict MCP results to a subtree.

Open the top 1–3 returned `path:startLine-endLine` locations. Refine a query
if needed; use grep for a known exact literal or when hips returns no relevant
hits. Describe behavior rather than supplying a bag of keywords.

MCP `verbose` and `include_snippet` default to `false`. Request
`include_snippet: true` only when needed: up to 4 lines for every returned hit.

## Debugging MCP

Use `search_code` with `verbose: true` on a representative query. The normal
results stay readable; an additional text block and `structuredContent.diagnostics`
report the server PID, root, index, actual encoder backend, CoreML settings,
fallback reasons, finite hit scores, timing, and query-cache reuse.

Use `index_status` with `verbose: true` to inspect the resident encoder and
last search without loading a model. `index_has_embeddings` means vectors
exist; `encoder.state: not_loaded` is not proof that CoreML works. Run a
semantic or hybrid search first. Repeating a successful query should report
`query_cache_hit: true` in the same process.

`backend: coreml` plus a successful semantic search confirms CoreML inference.
Its `compute_units` are allowed devices; `execution_device: not_observed`
means hips has not measured whether individual operations ran on CPU or ANE.
`backend: candle` and `candle_device` identify the alternative runtime. A
CoreML-to-Candle retry retains semantic search and is disclosed in the normal
response. A failed hybrid/semantic call has `isError: true`; do not present it
as a successful search. Report an explicit lexical recovery if one is used.

After upgrading hips, reconnect/restart the MCP server to load the new binary
and refresh the tool schema. Older servers may not accept `verbose` or
`expected_root`; check `tools/list` or reconnect before using these fields.

## CLI diagnostics and scope

```sh
hips search --root /absolute/path/to/repository --query "<what the code does>" --top-k 10
```

Use `-v` for runtime diagnostics and `--json` for JSONL results. A `using
lexical BM25` note means semantic retrieval was unavailable. A Candle retry
can still provide hybrid results. Cache access alone does not grant macOS
GPU, ANE, or IOSurface access; see the sibling `hips-install` skill for setup.

The CLI can reuse the nearest indexed ancestor and rebase paths; this searches
the entire ancestor, so `../` hits can be valid. MCP reports paths relative
to its pinned root. Both honor `.gitignore` and skip common build directories.

Indexes are maintained automatically in the MCP and plugin workflows. Do not
run `index-repo` unless search reports no index or requests vector repair. Use `index_status` (MCP) or
`hips status --root <repository>` (CLI) to inspect freshness; use MCP `reindex`
if watching is unavailable. For installation or updates, use `hips-install`.
`HIPS_NO_LOG=1` disables the usage log under `~/.cache/csearch/usage.jsonl`.

MCP checks for invalid or missing live document embeddings before semantic
search and rebuilds affected files, even if their text is unchanged. Debug
status reports `invalid_live_embeddings`; search reports
`repaired_embeddings`. A failed repair is an error, not a successful hybrid
result. For CLI repair, run `hips index-repo --root <repository>` and retry.
