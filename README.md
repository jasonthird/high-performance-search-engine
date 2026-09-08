# hips — a small search engine for agents

`hips` is a small Rust search engine that coding agents can use:
point it at a repository (or any pile of documents and PDFs) and it gives
Claude Code, Codex, OpenCode and friends a search tool that answers "where
is this implemented?" with ranked `path:line` locations, keeps its index
fresh in the background while they edit, and costs them a few lines of
context per query. It ships as a CLI, an MCP server, and a Claude Code
plugin, and needs no service, database, or Python at query time.

Underneath is a **full-text search engine written in Rust from scratch** —
no Tantivy, Lucene, or any other search-engine crate. It builds an inverted
index and answers top-k queries with **BM25 scoring executed by exact
Block-Max WAND**, over **compressed (delta + bit-packed), memory-mapped
posting lists**, with optional **document reordering (recursive graph
bisection)** for better compression; it scales from a repository to all of
English Wikipedia on a laptop. The code-search layer walks a repository into
declaration-sized chunks that keep their line numbers and adds
**CodeRankEmbed vectors** (Candle, or CoreML on the Apple Neural Engine)
for hybrid lexical + semantic retrieval.

```
JSONL docs ──index──▶ inverted index + block metadata ──search──▶ exact BM25 top-k
                      (compressed, mmap'd, immutable)            (Block-Max WAND)

source tree ──index-repo──▶ segments: postings + FP16 vectors ──search──▶ path:line hits
             (gitignore-aware, chunked by declaration)   (BM25 + CodeRankEmbed)
```

![hips: hybrid code search over a repository, kept fresh by the background watcher — a file written moments earlier is already a hit, and a verbose query shows the 14 ms hybrid retrieval breakdown](demo.gif)

The theory and original papers behind every algorithm used here are
documented in [docs/THEORY.md](docs/THEORY.md).

## Status

Small by design: one executable, separately loadable grammar libraries, and
index/model caches. Maintainers can package no grammars, a chosen subset, or
all of them; omitted languages download when first indexed. The watcher runs
only when requested directly or by an agent session. It is in daily use as
the code-search tool this repository itself is developed with, and every
benchmark in this README was taken on the author's laptop. Architecture and
usage were reviewed against the working tree on 2026-09-07; historical
benchmark numbers are retained with their original corpus and configuration.
It is not a hosted search service — no clustering, no replication — and the engine
half doubles as a readable, tested walk through how modern lexical search
works end to end: indexing, compression, memory-mapped storage, exact
dynamic pruning, batch updates, benchmarking, and a small HTTP API.

## Requirements

- A stable Rust toolchain and a C/C++ build toolchain for source builds,
  edition 2021. The file-lock APIs require at least Rust 1.89; this is not
  a tested minimum version for every locked dependency.
- `--features semantic` for hybrid code search: pulls in Candle and the
  tokenizers crate and downloads the ~550 MB CodeRankEmbed weights on first
  use (into the Hugging Face cache). Off by default so the lexical engine
  stays dependency-light.
- macOS with Metal only for the optional `gpu` feature (`--reorder bp-gpu`)
  and for the CoreML/Neural Engine encoder backend; the default build is
  CPU-only and portable, and on Linux the encoder runs through Candle.
- Grammar downloads target macOS and glibc Linux on arm64/x86_64. A user's
  machine needs no compiler when using prebuilt libraries. See
  [grammar packaging](grammars/README.md) for native builds and offline installs.
- Python 3 for the corpus helper scripts in `scripts/`, and a Python
  environment with coremltools to produce the CoreML models
  (`scripts/ane-prototype/README.md`).

## Quick start

Code search over a repository (the common case):

```sh
cargo install --path . --features semantic     # installs `hips`
hips index-repo --root .                        # once; later runs are incremental
hips search --root . --query "where do we validate auth tokens"
hips status --root .                            # what is indexed, watcher, sessions
```

The grammar loader pins the `grammars-v1` release. Those assets must be
published before unbundled first-use downloads can succeed. For a checkout
before that release, or for offline grammar use:

```sh
cargo build --manifest-path grammars/Cargo.toml --release --locked
export HIPS_GRAMMAR_DIR="$PWD/grammars/target/release"
export HIPS_GRAMMAR_OFFLINE=1
```

Grammar offline mode does not disable encoder downloads. Use `--lexical`
when no encoder is available. Missing grammars warn and use heuristic chunks.
See [tests](#run-the-tests) for the full development setup.

The lexical engine on a JSONL corpus:

```sh
cargo run --release -- index --input data/sample_docs.jsonl --out ./index
cargo run --release -- search --index ./index --query "cheap pizza montreal" --top-k 5
```

Run it as an HTTP service:

```sh
cargo run --release -- serve --index ./index --addr 127.0.0.1:8080
curl 'http://127.0.0.1:8080/search?q=cheap+pizza&k=5'
curl 'http://127.0.0.1:8080/stats'
```

For repeated manual queries without reloading the index:

```sh
cargo run --release -- repl --index ./index --top-k 10
```

## Project layout

- `src/cli.rs` - command routing: `index`, `search`, `repl`, `serve`,
  `bench`, `add`, `delete`, `merge`, `migrate` for the engine; `index-repo`,
  `mcp`, `watch`, `session`, `status`, `eval-gen`, `eval-code`, `embed` for
  code search. `src/query.rs` is the ranking facade the CLI, REPL, MCP
  server and evals share.
- `src/tokenizer.rs`, `src/indexer.rs`, `src/postings.rs`,
  `src/compress.rs` - tokenization (plain and code-aware), inverted-index
  construction, block metadata, and bit-packed postings.
- `src/searcher.rs`, `src/block_max_wand.rs`, `src/maxscore.rs`,
  `src/bm25.rs`, `src/spell.rs` - exact BM25 top-k query execution and
  query spelling correction.
- `src/storage.rs`, `src/segments.rs`, `src/external.rs`, `src/migrate.rs` -
  on-disk format, memory mapping, segmented updates, sharded external
  builds, and format migration.
- `src/reorder.rs`, `src/reorder/gpu.rs` - document reordering (BP on CPU,
  zero-copy Metal behind `--features gpu`).
- `src/api.rs` - Axum HTTP API.
- `src/embedder.rs`, `src/coreml.rs`, `src/embeddings.rs`, `src/embcache.rs`,
  `src/ivf.rs`, `src/pq.rs`, `src/hybrid.rs` - CodeRankEmbed inference
  (Candle; CoreML/ANE on macOS), FP16 vector storage and scoring, the
  content-keyed embedding cache, IVF/PQ, and score fusion.
- `src/repo.rs`, `src/treesit.rs`, `src/grammar.rs`, `src/codeindex.rs`,
  `tree-sitters/` - repository walking, declaration chunking across 41 grammar
  variants, lazy library loading, embedded definition queries, heuristic
  fallback, PDF pages, and incremental segmented rebuilds.
- `grammars/` - independent Cargo workspace of native grammar libraries;
  `.github/workflows/grammars.yml` builds/tests four platforms and publishes
  individual libraries and checksums on a grammar release tag.
- `src/watch.rs`, `src/daemon.rs`, `src/mcp.rs`, `src/usagelog.rs` -
  filesystem watching, the session-leased background watcher, the MCP
  server over stdio, and the search usage log.
- `src/eval.rs` - CodeSearchNet-style eval generation and recall metrics.
- `skills/hips/`, `plugin/hips/`, `.claude-plugin/` - the agent skill, the
  Claude Code plugin (skill + hooks + MCP registration), and the plugin
  marketplace manifest.
- `scripts/` - corpus converters (CirrusSearch dumps, directory crawls) and
  the CoreML conversion of CodeRankEmbed (`scripts/ane-prototype/`).
- `examples/` - micro-benchmarks and equivalence checks behind the numbers
  in this README (`pq_scoring_bench`, `bucket_equiv`, `embed_bench`, ...).
- `docs/THEORY.md` - algorithm notes and paper references.
- `tests/` - Block-Max WAND vs naive-BM25 oracles, persistence and
  reordering, external builds, segmented indexes, the repo lifecycle, the
  MCP transport, and the background watcher.

## What this engine does

- **Indexes** JSONL documents (`{"id", "title", "body"}`), tokenizing title +
  body as one searchable field, in parallel with rayon.
- **Stores** an inverted index: term dictionary (`term -> u32 term_id`),
  posting lists sorted by `doc_id` (each posting is `{doc_id: u32, tf: u32}`),
  document metadata and lengths, and per-block metadata for skipping.
- **Searches** with BM25 (`k1 = 1.2`, `b = 0.75`) using exact top-k dynamic
  pruning: Block-Max WAND for short queries, MaxScore (Turtle & Flood 1995)
  for queries of 5+ unique terms, where WAND's pivot prefix rarely clears
  the threshold. Both lexical evaluators are exact; a naive BM25 scorer
  exists only in tests as their oracle. Optional hybrid/vector retrieval
  has separate candidate-selection and fusion semantics.
- **Serves** concurrent queries over HTTP from a read-only, `Arc`-shared index.

### "Sublinear" in the practical retrieval sense

The lexical path is sublinear in the practical retrieval sense because it does not
scan every document. It retrieves candidates from inverted indexes (only
documents containing at least one query term can ever be touched) and skips
non-competitive blocks of postings using Block-Max WAND. Worst-case queries
(e.g. every query term appears in every document, or k is huge) may still
touch many postings, but normal top-k queries evaluate far fewer documents
than the full corpus — the debug counters in every response let you verify
this.

## How BM25 works

For a query Q and document D:

```
score(D, Q) = Σ over query terms q of:
    idf(q) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))

idf(q) = ln(1 + (N - df + 0.5) / (df + 0.5))
```

- `tf` — how often the term occurs in the document (more is better, with
  diminishing returns: the contribution saturates at `idf * (k1 + 1)`).
- `idf` — rare terms are worth more than common ones (`N` = total docs,
  `df` = docs containing the term). This formulation is always positive.
- The denominator normalizes by document length: a term occurrence in a short
  document is stronger evidence than in a very long one. `b` controls how
  much length matters; `k1` controls tf saturation.

## How an inverted index avoids scanning every document

Instead of storing documents and scanning them per query, the index is
inverted: for each *term*, it stores the sorted list of documents containing
it. A query only ever reads the posting lists of its own terms. If "pizza"
appears in 5,000 of 100,000 documents, a query for "pizza" considers at most
5,000 candidates — the other 95,000 documents are never touched. Block-Max
WAND then skips most of those candidates too.

## How Block-Max WAND works

Posting lists are split into fixed-size blocks (128 postings by default).
Each block stores `max_doc_id` and the impact pair `(max_tf, min_doc_len)`.
BM25 increases with tf and decreases with document length, so the pair gives
an upper bound under the current idf and average length. Computing bounds
at query time keeps skipping safe when segmented updates change statistics.

Query execution keeps one forward-only cursor per query term and a bounded
min-heap of the best k results. Once the heap holds k results, its minimum
score is the **threshold**: a document must score strictly above it to matter.
Each iteration:

1. **Pivot selection (WAND):** sort cursors by current `doc_id` and find the
   shortest prefix whose summed *per-term* upper bounds exceed the threshold.
   The first document of the last cursor in that prefix is the *pivot*. Any
   document before the pivot matches only a subset of terms whose combined
   best-case score is ≤ threshold, so it is skipped without scoring.
2. **Block-max refinement:** the per-term bound is coarse (one number for the
   whole list). So before scoring the pivot, sum the `block_max_score` of the
   blocks that contain the pivot for each prefix cursor — a much tighter
   bound. If even this cannot beat the threshold, jump all prefix cursors
   past the nearest block boundary: **whole blocks are skipped without
   decoding a single posting**.
3. **Exact scoring:** only documents that survive both checks are scored with
   full BM25, and only they may enter the heap (raising the threshold and
   making future skips more aggressive).

### Why block max scores allow safe skipping

The impact-derived `block_max_score` bounds every posting in the block.
The maximum tf and minimum length can belong to different documents, so the
bound may be larger than every actual score; it must never be smaller. Summing these per-term
bounds gives a number the document's real score can never exceed. If that
bound is ≤ the current k-th best score, the document cannot enter the top-k —
skipping it cannot change the result.

### Why results are still exact, not approximate

Block-Max WAND never *estimates* a score. Every skip is justified by a proven
upper bound: documents are only skipped when they provably cannot beat the
current k-th result, and every returned document was scored with the full,
exact BM25 formula. The output is therefore identical to exhaustively scoring
every matching document (verified in the test suite against a naive BM25
oracle on handcrafted and randomized corpora). Ties use ascending internal
doc_id; reordering can therefore change which equally scored hits fit at the
k-th boundary.

## Usage

### Build the index

```sh
cargo run --release -- index --input ./data/sample_docs.jsonl --out ./index
# optional: document reordering for better compression
cargo run --release -- index --input ./data/docs.jsonl --out ./index --reorder bp
```

Input is JSONL, one document per line:

```json
{"id": "doc-123", "title": "Some title", "body": "Some body text"}
```

Indexing is multithreaded: documents are parsed and tokenized in parallel,
partial inverted indexes are built per worker chunk, then merged; final
posting lists are sorted by `doc_id` and block metadata is computed last.

`--code` enables a code-oriented tokenizer: each identifier is stored as
a whole (so `getUserByOrganizationId` still matches exactly) and also
split on camelCase, snake_case, SCREAMING_SNAKE_CASE, and dotted paths.

`--title-weight N` (default 2) applies BM25F-lite field weighting: each
title occurrence of a term counts as N occurrences in the folded tf, so
title matches outrank otherwise-equal body matches. Set 1 to disable.

`--reorder` controls doc_id assignment: `none` (input order), `path` (sort
by external id — clusters file paths/URLs), or `bp` (recursive graph
bisection, minimizes the estimated compressed size; slower to build).
Reordering preserves scores; renumbering can change tie order among equal scores.

There is also an experimental `bp-gpu` strategy (build with
`--features gpu`, macOS only) that runs BP's gain
computation as a hand-written Metal kernel via objc2-metal with **true
zero-copy buffers**: every array the GPU touches is page-aligned host
memory wrapped with `newBufferWithBytesNoCopy`, so nothing is ever
uploaded or read back — the CPU patches degree counters in place and
sorts gains straight out of shared pages. Measured: the reorder phase
runs ~25% faster than the rayon CPU path (best run 36%) — see
docs/THEORY.md for the progression from the 14x-slower Burn port through
CubeCL to this.

There is also a crawler script to build a corpus from a directory tree of
text files, and a converter for Wikimedia CirrusSearch dumps (plain-text
Wikipedia, no wikitext parsing needed):

```sh
python3 scripts/crawl_to_jsonl.py ~ data/home_docs.jsonl
# Simple English Wikipedia, complete (~278k articles):
curl -O https://dumps.wikimedia.org/other/cirrussearch/20251229/simplewiki-20251229-cirrussearch-content.json.gz
python3 scripts/cirrus_to_jsonl.py simplewiki-20251229-cirrussearch-content.json.gz data/simplewiki.jsonl
# English Wikipedia, streamed with a document cap (the full 43GB dump
# never touches disk; curl stops when the cap is reached):
curl -s https://dumps.wikimedia.org/other/cirrussearch/20251229/enwiki-20251229-cirrussearch-content.json.gz \
  | gunzip -c | python3 scripts/cirrus_to_jsonl.py - data/enwiki-1m.jsonl 1000000
```

Simple English Wikipedia (278k articles, 30M postings) indexes in ~20s
and serves "theory of relativity" -> the *Theory of relativity* article
at rank 1 in 0.6ms, scoring 0.3% of the corpus.

### On-disk format: compressed + memory-mapped

A single lexical index (and each immutable lexical segment) has three core files:

- `meta.bin` — the slim RAM-resident core: document lengths, the sorted
  term dictionary (one concatenated string, binary searched), per-term
  document frequencies, and flattened block skip-tables. Everything
  derivable is recomputed at load instead of stored (block ranges from the
  fixed block size, idf from df, region offsets as a prefix sum).
- `postings.bin` — posting lists as delta-encoded, bit-packed blocks
  (gaps between sorted doc_ids packed at the smallest width that fits the
  block; ~4.5x smaller than raw postings in practice). This file is
  **memory-mapped**: the OS pages data in lazily, so blocks that Block-Max
  WAND skips are never decoded *and never read from disk*.
- `docs.bin` — the document store (id, title, snippet per doc), also
  memory-mapped: scoring never touches it, and only the top-k hits of a
  query are ever resolved, so it costs almost no memory.

Indexing streams the input in chunks (raw text never accumulates), interns
terms to dense ids on the fly, and builds posting lists with a single
ordered scatter pass — no hash-map merging, and the lists come out sorted.

For corpora whose postings don't fit in memory, `--external` switches to a
sharded spill-to-disk build: posting triples spill to sorted shard files at
a ~2.3GB budget and a k-way merge writes compressed blocks directly into
the final index. Peak memory is independent of corpus size; `--input -`
reads JSONL from stdin so a corpus can be indexed straight off the network
without ever being stored. (Document reordering is unavailable in this
mode.)

There is also an HTTP load generator for throughput measurement:

```sh
cargo run --release --example http_load -- 127.0.0.1:8080 32 10
```

### Search from the CLI

```sh
cargo run --release -- search --index ./index --query "cheap pizza montreal" --top-k 10
```

### Run the HTTP server

```sh
cargo run --release -- serve --index ./index --addr 127.0.0.1:8080
```

The index is loaded once, shared read-only via `Arc`, and queried
concurrently; nothing mutates it during searches. This HTTP endpoint runs
lexical BM25 search over either layout; it does not expose the CLI/MCP
hybrid modes.

```
GET /search?q=cheap+pizza&k=10
```

```json
{
  "query": "cheap pizza",
  "took_ms": 0.05,
  "num_docs_total": 100000,
  "num_query_terms": 2,
  "num_postings_visited": 1204,
  "num_docs_scored": 842,
  "num_blocks_visited": 40,
  "num_blocks_skipped": 87,
  "results": [
    {"id": "doc-123", "score": 4.23, "title": "Cheap pizza in Montreal"}
  ]
}
```

```
GET /stats
```

```json
{"num_docs": 100000, "num_terms": 50000, "avg_doc_len": 132.5, "index_size_bytes": 12345678}
```

### Run the benchmark

```sh
cargo run --release -- bench --index ./index --queries ./data/queries.txt --top-k 10
```

Reports query count, average / p50 / p95 latency, average postings visited,
docs scored, blocks visited and skipped — plus two ratios that prove the
engine is not scanning the corpus:

- **avg docs scored / total docs** — fraction of the corpus actually scored.
- **avg postings visited / query-term postings** — fraction of the query
  terms' own posting lists actually decoded (the rest was skipped).

Measured on **all of English Wikipedia** (7,110,635 articles, 22.2M terms,
1.85B postings; 8-core M3 MacBook Air 13-inch, 24GB RAM):

| Metric | Value |
|---|---|
| Index size | 6.50 GB (3.52 B/posting incl. doc store, ids, content hashes) |
| Build from the local 43GB dump | **14m09s** end-to-end (gunzip-dominated); 38s shard merge; 5.5GB peak RSS |
| Query latency (500 title-derived queries, warm) | p50 0.58 ms, avg 2.4 ms, p95 11 ms, p99 26 ms |
| Corpus scored per query | 0.58% of 7.1M documents |
| HTTP throughput (8 conns, same query mix) | ~1,850 req/s, p50 2.9 ms, p99 20 ms, 0 errors |
| Search process RSS | **14 MB** against the 6.5 GB index |

And on a 108k-document corpus (a crawled home directory, 679 MB of text,
19.2M postings):

| Metric | Value |
|---|---|
| Query latency (CLI, single) | 0.11 ms avg, 0.07 ms p50 |
| HTTP throughput (8 conns) | ~31,000 req/s, p50 0.21 ms, zero errors |
| Corpus scored per query | ~1% of documents |
| Index build (no reorder) | 3.6 s, 860 MB peak RSS |
| Index size | 104 MB total (32 MB postings = 1.67 B/posting, 25 MB metadata, 46 MB doc store) |
| Index load | ~20 ms; search process RSS ~43 MB (postings + docs mmap'd) |

### Hybrid retrieval: BM25 + CodeRankEmbed

The lexical engine is unchanged. Behind `--features semantic` the index
also holds one [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed)
vector per document, and `--mode hybrid` (the default on `search`) is
**embedding-first**: the encoder's candidates are fused with BM25's.
`--mode rerank` is BM25-then-cosine, `--mode semantic` is encoder-only,
`--mode bm25` (or `--lexical`) is the plain engine. On a single-layout
index built with `index --embed` the encoder probes `ivf.bin` cluster
posting lists (same inverted-file layout and `doc_id`s as BM25; `--nprobe`
controls how many clusters to open); on the segmented indexes that
`index-repo` builds, large segments use HNSW routing and small segments use
exact scans. Candidate dot products use the existing FP16 vectors; HNSW
candidate selection is approximate. `--exact-vectors` requests exhaustive
vector retrieval, while `--ann-ef 256` increases the graph search beam.

New or compacted segments build graphs automatically at 8,192 vectors.
Upgrade an existing segmented index without re-encoding with
`hips build-ann --index <index-directory>`, then reopen readers. Missing or
invalid graphs safely fall back to exact scans. See
[the retrieval theory and bibliography](docs/THEORY.md#17-sublinear-vector-retrieval-implementation-and-literature)
for measured latency/recall, construction costs and limitations.

Inference runs in-process, no Python at query time. Two backends:

- **Candle** (`src/embedder.rs`): NomicBert on Metal, CUDA-less CPU, or
  Accelerate; portable, and the only backend off macOS. A Candle fork with
  a fused Metal SDPA kernel in NomicBert attention is patched in
  (`Cargo.toml`), 1.7x faster document encoding at sequence 512.
- **CoreML on the Apple Neural Engine** (`src/coreml.rs`): the same weights
  compiled to static-shape models (batch-8 document models at sequence
  64/128/256/512, a batch-1 query model) under `<cache>/coreml/`, produced
  by `scripts/ane-prototype/ane_convert.py`. Used automatically when the
  compiled models cover the configured token limit; incomplete families fall
  back to Candle. Document shapes also undergo a once-per-loaded-shape batch-row
  equivalence check; failed shapes are skipped, and encoding reports an error
  if no fitting shape passes. `HIPS_ENCODER=candle` forces the fallback.

#### macOS caches and restricted processes

CoreML uses two distinct caches. Compiled CodeRankEmbed shapes live at
`<CSEARCH_CACHE_DIR>/coreml` (default `~/.cache/csearch/coreml`). Loading a
shape creates another, device-specialized E5RT cache, normally under
`~/Library/Caches/hips/com.apple.e5rt.e5bundlecache/`. Changing
`CSEARCH_CACHE_DIR`, `HF_HOME`, `XDG_CACHE_HOME`, or `TMPDIR` does not relocate
that E5RT cache.

The macOS CLI supports an explicit writable cache base:

```sh
HIPS_COREML_CACHE_DIR="$PWD/.hips-coreml-cache" \
  hips search --root /path/to/repo --query "permission validation"
```

Use a directory permitted by the sandbox and ignore it in your repository.
The directory is created and checked for writes. An empty, invalid, or
unwritable override is a configuration error: exit 1, no result output.
CoreML's files appear below
`<override>/Library/Caches/hips/com.apple.e5rt.e5bundlecache/`.

Implementation limitation: [MLModelConfiguration](https://developer.apple.com/documentation/coreml/mlmodelconfiguration)
has no public E5RT cache-directory property. The CLI sets
`CFFIXED_USER_HOME` before starting threads or calling Foundation, using the
home-directory override present in [Apple's CoreFoundation implementation](https://github.com/apple-oss-distributions/CF/blob/main/CFPlatform.c).
This affects **all Foundation home-relative paths in the hips process**,
including its child processes, and takes precedence over an inherited
`CFFIXED_USER_HOME`. It does not change the shell's `HOME`, hips' index
selection, or `HF_HOME`. This is an opt-in compatibility mechanism, not a
CoreML API guarantee; verify it on your target macOS version. Library users
must arrange `CFFIXED_USER_HOME` at process startup themselves; loading an
encoder never mutates the process environment.

Cache relocation does not grant access to macOS accelerator services. In
Codex's workspace-write sandbox on macOS build `25G83`, a fresh relocated
cache removed the E5RT filesystem exception, but CoreML still failed ANE
compilation. `HIPS_COREML_COMPUTE_UNITS=cpu` selected CoreML's public CPU-only
mode, but IOSurface shared-event creation failed and predictions contained
NaNs. The default is `cpu-and-ane`; CPU-only mode is available for diagnosing
or running in environments where it works. Neither setting bypasses the
sandbox. A previously populated cache is not a guarantee that the required
runtime services are available.

hips checks cache writability, requires successful encoder warmup, and
rejects wrong-sized, non-finite, zero, or non-normalized predictions before
ranking or caching them. If CoreML initialization fails, it prints the
cause and retries CodeRankEmbed with Candle, preferring Metal when a GPU is
accessible and using CPU otherwise. The device check also prevents Candle's
empty-device-list panic when the sandbox hides every GPU. Hybrid retrieval
uses the same stored document vectors. This may increase startup time
and memory usage. `HIPS_ENCODER=candle HPS_EMBED_DEVICE=cpu` explicitly selects
that backend for one invocation if needed.

If encoder initialization is still unavailable, default/hybrid search
prints `using lexical BM25 only` to **stderr** and returns lexical results
with exit 0. Missing embeddings or a build without the semantic feature also
produce a visible fallback note, even without `-v`. Explicit `--mode semantic`
and `--mode rerank` fail with exit 1 if semantic execution is unavailable.
A prediction failure after initialization exits 1 without printing results.
JSONL stdout contains only results; fallback diagnostics go to stderr.
Buffered C stdout from synchronous CoreML calls is redirected to stderr and
flushed before returning (also protecting MCP output). This briefly redirects
the process stdout descriptor while holding Rust's stdout lock; library hosts
with unrelated native stdout writers should account for that shared descriptor.

Codex also documents `[sandbox_workspace_write].writable_roots` in its
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
Adding the existing `~/Library/Caches/hips` path there can permit cache writes
without relocation. Filesystem access does not grant accelerator-service access;
no Codex settings are changed by hips.

Fallback preserves index resolution and path rebasing: an exact `--root`
index is preferred, otherwise the nearest indexed ancestor is searched.
As before, a subdirectory root searches the entire ancestor index, with paths
relative to the requested directory (including `../` hits); it is not a
subtree filter. `--index` takes precedence. Failure never selects a different
repository or searches a global index.

The model-free regression suite covers invalid vectors, denied cache writes,
configuration errors, fallback diagnostics, exit status, JSONL output, and
ancestor-index resolution. For real CoreML coverage with cached models:

```sh
HIPS_TEST_COREML_MODELS="$HOME/.cache/csearch/coreml" \
HIPS_TEST_HF_HOME="$HOME/.cache/huggingface" \
  cargo test --features semantic --test search_runtime \
  native_coreml_cache_relocation -- --ignored
```

Run this inside the intended sandbox; it checks both CoreML compute modes,
cache relocation, finite results, and preservation of semantic execution
through the CPU retry when platform services are denied.

Cold indexing encodes each missing content key once, groups inputs by actual
token length, and overlaps CPU tokenization with inference using a bounded
two-batch queue. The model, token caps and row order are preserved. Compare
the former byte-bucketed serial scheduler with the new path using
`cargo run --release --features semantic --example cold_embed_bench -- . 128`.
This measures uncached encoding after model/shape warmup, not downloads or
the complete first index build.
  Measured on an M3: 25.4k tok/s vs Candle/Metal's 6.3k for document
  batches, an 810-chunk cold index in 13.0 s instead of 28.6 s, a hybrid
  query in 0.10 s wall including model load, and embeddings within mean
  cosine 0.99968 of the fp32 reference. That fidelity measurement supports
  reuse of these embeddings, but is not a recall guarantee for every workload.
  Model conversion and installation are documented in the
  [CoreML guide](scripts/ane-prototype/README.md); compiled models are not
  downloaded automatically.

```sh
cargo build --release --features semantic

# 1. one index directory: inverted postings + CodeRankEmbed vectors
#    (same internal doc_ids; downloads ~550 MB of weights on first run)
./target/release/hips index \
  --input data/code_eval/corpus.jsonl --out ./index-code --code --embed

# 2. search
./target/release/hips search \
  --index ./index-code --query "retry failed HTTP requests" \
  --mode hybrid --semantic-candidates 200 --fusion weighted --alpha 0.15 --top-k 10

# 3. labeled eval (BM25 vs semantic-only vs hybrid)
./target/release/hips eval-code \
  --index ./index-code \
  --queries data/code_eval/queries.tsv \
  --qrels data/code_eval/qrels.tsv

# 4. latency breakdown (candidate sweep)
./target/release/hips bench \
  --index ./index-code --queries data/code_eval/queries.txt \
  --mode hybrid --candidate-sweep 50,100,200,500,1000
```

Queries use the model's required prefix (`Represent this query for
searching relevant code: `) inside the embedder; documents are encoded
as raw code. Vectors are stored as memory-mapped FP16, 768-d,
L2-normalized. Default fusion is min-max weighted BM25+cosine
(`--fusion weighted --alpha 0.15`); reciprocal rank fusion is optional
(`--fusion rrf`).

To index a real source tree, use `index-repo` (below) rather than
`scripts/code_to_jsonl.py` — the script is kept only for reproducing the
eval corpus, and its ids carry no line numbers.

### Code search in a codebase: `index-repo`

`index-repo` walks a repository the way git does — honouring nested
`.gitignore` files, the user's global excludes file and `.git/info/exclude`,
skipping hidden directories, `target/`, `node_modules/`, symlinks, and
non-source files — and splits each file into declaration-sized chunks that
**keep their line numbers**. A document id is therefore a location:
`src/searcher.rs:120-165`.

#### Declaration-aware chunking: 41 loadable grammar variants

Chunk boundaries decide everything downstream: what one vector means,
what BM25's title boost applies to, and how many lines an agent reads
after a hit. Files are therefore parsed with tree-sitter and cut at real
definitions. Forty-one grammar variants load on demand (feature `treesitter`, on
by default): Python, JavaScript, TypeScript/TSX, Java, C, C++, C#, Go,
Rust, PHP, Ruby, Swift, Kotlin, Scala, Dart, Lua, Perl, R, Objective-C,
MATLAB, Bash, PowerShell, SQL, Haskell, Elixir, Erlang, OCaml, Julia, Zig,
Groovy, Fortran, Pascal, Ada, Solidity, HCL/Terraform, Nix, Elm, CMake,
assembly, and Markdown (chunked by heading, sections nesting by
level). Every grammar depends on the ABI-stable `tree-sitter-language`
crate, so one core version serves all of them.

What counts as a definition lives in `tree-sitters/<language>.scm`, one
small query per language in tree-sitter's own query syntax, embedded at
compile time: `@definition.<kind>` marks the node, `@name` its identifier.
The files are seeded from the grammars' own `tags.scm` where they ship one
and hand-written otherwise. Adding a language means a wrapper crate in the
grammar workspace, one registry line, and one query file. `hips chunks --file X` shows how a
file is cut (`--sexp` prints the parse tree, for writing queries).

Chunking is generic over languages once definitions are known: each
definition's span, plus the comment block above it (across one blank
line), is a cut point; the segments between cut points become chunks. A
nested definition is its own chunk with `parent` set, so a method's title
is `path::Class::method`; the class chunk keeps its header and fields. A
closing brace after the last method joins the chunk before it. Runs of
bodyless one-liners of header-like kinds — prototypes in C headers,
`#define`s, typedefs — merge into chunks of up to 24 lines so a header does
not become a thousand documents, while one-line functions in
expression-bodied languages (OCaml, Haskell, Kotlin) stay separate. A
file with no grammar, or whose parse yields no definitions, falls through
to the keyword heuristic that preceded all this, so the worst case is what
hips did before.

Measured against that heuristic on real checkouts (unnamed = chunks
carrying no declaration name; lines = per-chunk size):

```
                  files   chunks          unnamed       median lines   p90 lines
gin (Go)             99   1312 -> 1687     8% -> 7%      12 -> 10       35 -> 27
express (JS)        141    312 -> 627     59% -> 46%     41 -> 13      200 -> 104
spring-petclinic     50    124 -> 285     19% -> 16%     22 -> 12       61 -> 28
redis (C)           786   3207 -> 22741   65% -> 26%    126 -> 7       200 -> 37
flask (Python)       83   1764 -> 1960     5% -> 5%       6 -> 5        23 -> 21
this repo (Rust)     57   1072 -> 1104     5% -> 7%      10 -> 10       43 -> 42
```

C is the headline: a language with no declaration keyword went from
200-line slabs to one chunk per function. Python, Go and Rust, where the
keyword heuristic already worked, barely move. Express's remaining
unnamed chunks are anonymous test callbacks, which are not definitions.
Grammar parse tables live in separate `.dylib`/`.so` release artifacts, keeping
them out of the executable. The first file in a language loads its bundled or
cached grammar, downloading only that library if needed into
`~/.cache/csearch/grammars/<release>/<target>` (honoring `XDG_CACHE_HOME`).
Downloads are checked for SHA-256 integrity and ABI compatibility before use.
Subsequent runs reuse the cache. Missing or failed loads warn and fall back to
heuristic chunks. Maintainers can bundle any subset in `HIPS_GRAMMAR_DIR`;
those libraries take priority, with cache/download fallback for missing languages.
A broken installed library warns and uses heuristic chunks rather than being
replaced automatically. `HIPS_GRAMMAR_OFFLINE=1` disables grammar downloads
while allowing bundled and cached grammars. The local macOS arm64 default
release build measured 9.6 MiB, with about 68 MiB of grammar libraries separate;
semantic builds and other platforms have different sizes. See [grammar builds](grammars/README.md)
for the build commands and four-platform release pipeline. There is still a
one-time query compilation per language (10–60 ms). Parsing itself is cheap:
this repository's 57 Rust files index lexically in 0.06 s against 0.03 s
with the heuristic. An index built by an older chunker is re-chunked in
full on the next build when its recorded chunker version differs. Grammar
installation alone does not invalidate unchanged-file fingerprints, so files
previously indexed with the fallback are not automatically re-chunked merely
because a library becomes available.
`--no-default-features` builds the lean binary with the heuristic only. PDFs still chunk per page, as `report.pdf::page 7`.

```sh
cargo build --release --features semantic
cargo install --path . --features semantic   # installs `hips`

# Index the current repository. Hybrid by default; --lexical skips the
# model download and builds in seconds instead of a minute. A binary built
# without `--features semantic` falls back to lexical with a warning.
hips index-repo --root .
```

The index is written under `~/.cache/csearch/<repo>-<hash>/`, never into the
repository being indexed (`--index` overrides, `CSEARCH_CACHE_DIR` moves the
cache root).

Search it without knowing where the index went — `--root` resolves to the
same per-repo location `index-repo` used:

```sh
hips search --root . --query "where do we validate auth tokens"
hips search --root . --query "where do we validate auth tokens" --json   # JSONL, one hit per line
hips search --root . --query "verify_bearer" --lexical                    # BM25 only, no encoder
```

The default output is one hit per line, location then declaration name, and
nothing else, so an agent reading it pays for only what it needs:

```
src/auth/token.rs:88-140     verify_bearer
src/auth/mod.rs:12-40        AuthLayer
```

Encoder loading is silent when the model is cached. It prints a line only
when it is actually downloading (first run, ~550 MB) or when it cannot be
loaded at all (offline with an empty cache), in which case the search
degrades to BM25 with a `note:` explaining that. Pass `-v` for the old
verbose output: scores, per-stage timing, WAND stats, device and cache
lines.

PDFs under the tree are indexed too: each page's extracted text becomes a
chunk named `report.pdf::page 7`, so design docs and papers checked into a
repo are searchable alongside the code. Encrypted or malformed PDFs are
skipped with a warning, never fatal.

Every `hips search` and MCP `search_code` call appends one JSON line
(timestamp, calling agent, root, query, mode, hit ids, latency) to
`~/.cache/csearch/usage.jsonl`, for offline analysis of how agents actually
use the tool. `HIPS_NO_LOG=1` disables it.

### Installing for agents

The search skill uses a simple rule: **sandboxed shell → prefer MCP; no
shell sandbox → prefer CLI**. MCP reuses its encoder and query cache across
calls. If MCP is unavailable or serves another repository, use the CLI with
an explicit `--root`. The repo doubles
as a Claude Code plugin marketplace; the plugin gives Claude Code the `hips`
skill, a keyword-gated routing hint on code-navigation prompts (also
injected into subagents, which do not reliably see CLAUDE.md), a background
watcher that indexes the repository when a session opens and keeps the index
fresh until the last session on it closes, and the MCP server:

```sh
claude plugin marketplace add /path/to/this/repo   # or the GitHub URL
claude plugin install hips@hips -s user
```

The plugin costs about 200 always-on tokens per session (the skill line plus
one SessionStart note saying the index is handled); the hint fires only on
prompts containing words like "where", "find", "how does", "which file".

#### Always fresh: the session-leased watcher

An agent should never have to think about indexing, and several agents may
have the same repository open at once. The plugin's `SessionStart` hook runs
`hips session start`, which

- resolves the enclosing git work tree of the session's cwd (a session
  opened in `$HOME` or another non-repository directory is left alone
  unless `HIPS_WATCH_ANY_DIR=1`),
- writes a **lease** for the session under `~/.cache/csearch/watch/<index>/leases/`
  holding the session id and the pid of the Claude Code process,
- starts `hips watch --leased` for that repository if none is running
  (the watcher holds a `flock`, so a second session attaches instead of
  starting a second process), and
- answers the hook with one line of context telling the agent the index is
  handled. If the running watcher was started by a different `hips`
  binary (you ran `cargo install`), it is retired and a fresh one started,
  so an old chunker never alternates with a new one on the same index.

The watcher builds the index if it is missing, then rebuilds after each burst
of relevant filesystem events (300 ms quiet), incrementally: only changed
files are re-chunked and re-encoded. `SessionEnd` runs `hips session end`,
which drops the lease; the watcher exits 20 s after the last live lease goes
(long enough to survive `/clear` and a quick restart). A lease whose pid is
dead — a killed or crashed session — is reaped, so nothing is pinned forever,
and after five idle minutes the watcher unloads the encoder, so an idle
watcher costs a few megabytes rather than the model's memory.

Searches stay consistent with edits without any coordination on the agent's
side: `hips search` reads the watcher's status file and waits (bounded, 20 s)
for a rebuild in flight before answering, and a search issued while the very
first build is running waits up to 10 s for it. Measured on this repository
(1.1k chunks, M3): `session start` returns in 30 ms; a save is noticed within
120 ms and re-indexed in 20-170 ms with the encoder resident (3-5 s the first
time a fresh process encodes, which is Neural Engine plan loading); the idle
watcher is 11 MB without the encoder and 325 MB with it. A manual `hips index-repo`
racing the watcher waits on the index's `writer.lock` instead of failing;
whichever runs second finds the tree fingerprint current and does nothing.

```sh
hips status --root .        # index size, watcher state, sessions holding it
hips session list --root .  # leases and whether their pids are alive
hips watch --root .         # run a watcher by hand (foreground, Ctrl-C stops)
hips watch --root . --stop  # stop the running one regardless of leases
```

The watcher's log is `~/.cache/csearch/watch/<index>/watch.log`.
`HIPS_WATCH_LEXICAL=1` in the agent's environment makes session-started
watchers build lexical-only indexes (no model). The same `session start` /
`session end` pair works for any other agent or editor that can run a
command on open and close; `--id` names the session and `--pid` the process
whose exit should release it.

Because the watcher indexes the git work tree, `hips search --root .` from a
subdirectory of an indexed repository resolves to the repository's index and
prints hit paths relative to that subdirectory.

For OpenCode, Codex and any other agent that reads `~/.agents/skills`:

```sh
cp -r skills/hips ~/.agents/skills/hips
cp -r skills/hips-install ~/.agents/skills/hips-install # installation and Codex cache setup
ln -s ../../../.agents/skills/hips ~/.config/opencode/skills/hips   # OpenCode
```

`skills/hips/` and `skills/hips-install/` are the source of truth; their
copies under `plugin/hips/skills/` must be kept in sync. The install skill
includes opt-in Codex cache permissions and sandbox verification. Agents without a skill mechanism get
the same policy as a few lines in their instructions file (`AGENTS.md`,
`CLAUDE.md`): search with hips before grep for any where-is-it or
how-does-it-work question. Tested in non-interactive runs on this
repository with no mention of hips in the prompt, Claude Code and Codex
both opened with a hips search in earlier evaluations. The current skill
prefers MCP for sandboxed shell execution and the CLI otherwise.

The plugin registers the MCP server for use when shell execution is sandboxed. `hips mcp` serves the same index
over stdio, exposing three tools: `search_code` (ranked chunks with
`path:line` locations, an optional `path_glob`, and fenced snippets),
`index_status`, and `reindex`. With no `--root` it serves the git work tree
enclosing the directory the client started it in, the same tree the watcher
covers; it waits out a watcher rebuild in flight and reopens the index
whenever another process (the watcher, another agent) has rebuilt it.

```sh
# Register by hand instead of through the plugin:
claude mcp add hips -s user -- hips mcp
```

Codex (`~/.codex/config.toml`), OpenCode (`opencode.json`), Copilot CLI
(`~/.copilot/mcp-config.json`), Antigravity and Grok take the same
`hips mcp` command in their own MCP config formats:

```toml
# ~/.codex/config.toml (Grok: ~/.grok/config.toml, same shape)
[mcp_servers.hips]
command = "hips"
args = ["mcp"]
```

```jsonc
// Copilot CLI ~/.copilot/mcp-config.json, Antigravity mcp_config.json
{ "mcpServers": { "hips": { "command": "hips", "args": ["mcp"] } } }
// OpenCode ~/.config/opencode/opencode.json
{ "mcp": { "hips": { "type": "local", "command": ["hips", "mcp"], "enabled": true } } }
```

Add `"--root", "/path/to/repo"` to pin a server to one repository. The
server watches the tree itself (FSEvents / inotify) and rebuilds on the
next tool call after a change, so it stays fresh with or without the
session watcher. A client's MCP server can have a different launch context
from its sandboxed shell commands; verify runtime access through the actual
connected server. Adding filesystem cache permissions does not grant ANE,
GPU, or IOSurface access to a sandboxed CLI.

#### MCP diagnostics and scope

`verbose` and `include_snippet` both default to `false`. Set
`include_snippet: true` to include up to 4 source lines for every returned hit.
Normal results include their effective retrieval mode and report any
CoreML-to-Candle recovery. Hybrid or semantic inference failures produce
`isError: true`; the server remains available for another call. An explicit
lexical retry is labeled lexical. Failed hybrid retrieval never returns
arbitrary results as success.

```json
{"query":"permission validation","mode":"semantic","top_k":5,"verbose":true,
 "expected_root":"/absolute/path/to/repo"}
```

`verbose` adds a diagnostic text block and the same JSON data under
`structuredContent.diagnostics`. It includes server PID/version, pinned and
indexed roots, index directory, actual encoder backend, validated warmup,
CoreML loaded model paths and cache directory, fallback reasons, per-hit
scores, retrieval/total elapsed milliseconds, and query-cache reuse. Total
time includes freshness checks and rebuilds; retrieval time includes any
lazy encoder load. Repeating a query in the same server should report
`query_cache_hit: true` for successful semantic execution.

`index_status` accepts `{"verbose":true}` and reports the resident encoder
and last search without initializing a model or rebuilding. A hybrid-capable
index means vectors exist; it does not prove runtime availability.
`backend: "coreml"` after a successful semantic search proves CoreML inference.
CoreML `compute_units` are permitted devices, while
`execution_device: "not_observed"` explicitly avoids claiming that individual
operations ran on ANE. Candle reports its actual CPU or Metal device.

`expected_root` rejects a request aimed at a different repository before
retrieval or rebuilding. MCP also rejects an explicitly supplied index that
belongs to another root, so snippets cannot be read from an unrelated tree.
Use `path_glob` to filter within the pinned repository; it does not switch
repositories. Debug diagnostics can contain local paths and error details;
they are returned only when requested.

Older versions could persist invalid document embeddings after native runtime
failure: warmup errors were ignored and CoreML outputs were checked for shape,
not finite values or normalization. Content-only cache reuse and unchanged-file
shortcuts could then preserve those rows across later rebuilds. Ranked retrieval now validates stored vectors before the fast FP16
scorer can turn NaNs into large finite scores. MCP automatically rebuilds affected files before semantic/hybrid retrieval,
even when their source has not changed. Invalid cache entries become misses,
so the rebuild re-encodes them. Missing vector sidecars are repaired too;
deleted rows do not trigger repair. `index_status` with `verbose: true`
reports `invalid_live_embeddings`, and successful repair is disclosed in the
search response and `last_search.repaired_embeddings`. A failed repair is a
tool error; explicit lexical search remains available. For the CLI, run
`hips index-repo --root <repository>` to repair before retrying search.
Normal watched source edits continue to trigger incremental reindexing. A
failed source refresh remains pending and is retried; later calls cannot
silently return the old locations.

After installing an updated binary, reconnect/restart the MCP server and
refresh its tool schema. Confirm `tools/list` advertises `verbose` before
using it; an existing process continues running the old version.

### How rebuilds stay proportional to the edit

**Segmented layout (the default).** `index-repo`, the watcher and `mcp`
use the same Lucene-style segment layout the lexical engine already used
for `add`/`delete`/`merge`, extended to vectors: each segment carries its
own `embeddings.bin` plus a `keys.bin` of content-cache keys. An edit
tombstones the stale chunks (the manifest remembers which chunk ids each
file produced), appends the changed chunks as one new segment, and encodes
only those; a compaction merge runs past `codeindex::MAX_SEGMENTS` and
rebuilds the merged segment's vectors from the cache by key — no
re-encoding. An unchanged tree is detected from a fingerprint of the
walked file list (path, size, mtime), so an up-to-date rebuild costs only
the walk. This shortcut does not read and compare every file's content. Semantic
retrieval uses per-segment HNSW where available and exact scans otherwise
(no segmented IVF/PQ). Compaction rebuilds graphs for large merged segments.
Measured:

```
                                387k chunks (lexical)   this repo (embedded)
first build                            3.8 s            13 s CoreML / ~42 s Candle
one-file edit -> reindexed             0.15 s                 0.20 s
  (non-segmented rebuild)             (2.4 s+)                  --
```

The 0.15 s is dominated by walking the tree (0.11 s); the index work itself
is milliseconds. `tests/segmented_repo.rs` covers the lifecycle: edits,
line-shift renames, file deletion, and merge-with-vector-rebuild. An index
built with the old single layout migrates automatically on the next
`index-repo` (its content cache is kept, so migration re-encodes nothing
unchanged). Writers are serialized by `writer.lock`; a rebuild that finds
the lock held waits for the other writer instead of failing.
`CSEARCH_TIMING=1` prints the per-phase breakdown of any rebuild.

**Single layout (`--single`, legacy).** One index directory with a
positionally keyed `embeddings.bin` plus IVF/PQ, rebuilt as a whole and
swapped in atomically from a staging directory. Three caches keep even that
proportional to the edit:

- **Vectors.** `embcache.bin` keys FP16 embeddings by a hash of the chunk
  text, so only chunks whose content changed are re-encoded. Encoding
  now uses token-length buckets and CPU prefetch. Historical byte-length
  bucketing measured 73.4 -> 38.6 ms/chunk versus file order; that is not a
  measurement of the new scheduler. `examples/cold_embed_bench.rs` measures
  the current implementation and checks vector equivalence.
- **Quantization.** The IVF centroids and PQ codebooks are *trained once* and
  reused: k-means for IVF plus 16 x 256-centroid k-means for PQ costs ~1 s on
  a small repo and scales with the corpus, yet none of it depends on which
  chunk changed. A rebuild reads the trained parameters back out of `ivf.bin`
  and `pq.bin` and only quantizes chunks it has no cached code for — one pass
  over the centroids and codebooks each, microseconds. Cached codes are
  tagged with a hash of the parameters that produced them, so retraining
  invalidates them automatically. Retraining happens on the first build, when
  the ideal cluster count has drifted more than 2x, or on demand
  (`--retrain`, or `reindex` with `{"retrain": true}`).
- **Chunking.** Files are read and split in parallel across cores.

Measured with the Candle encoder on this repository (812 chunks, 46 files)
and on the whole crates.io source cache (358k chunks, 15k files) — Apple
M-series, 8 cores:

```
                                        this repo     crates.io cache
first build (encodes every chunk)          ~60 s              (n/a)
rebuild, nothing changed                   0.02 s            0.30 s
edit -> next query, in-server            55-95 ms              --
steady-state hybrid query                   11 ms              --
```

The 55-95 ms is everything: watcher wakeup, re-chunking the tree, encoding
the one changed chunk on the GPU, re-quantizing it, rewriting the index, and
running the search. Before the quantizer was made reusable this was ~1.1 s,
essentially all of it retraining quantizers over unchanged data. What the
single layout cannot avoid is re-tokenizing every chunk when a file did
change (~1.7 s at crates.io scale), the price of a positionally keyed
`embeddings.bin` — which is why the segmented layout became the default.

**Retrieval quality is measured, not assumed.** `eval-gen` mines a
CodeSearchNet-style benchmark from any repo's doc comments (query = the
summary line, relevant doc = the code with the comment stripped), plus an
identifier-query set. On 31k chunks of real crate source (4,567 NL queries,
800 identifier queries), R@10:

```
                    NL queries   identifier queries
bm25                   0.255          0.976
semantic               0.716          0.974
hybrid (default)       0.696          0.993
```

Three defaults were set from these measurements: fusion is min-max weighted
with `alpha 0.15` (RRF at equal weight dragged hybrid to 0.569 on NL);
`nprobe` defaults to half the IVF clusters (the old cap of 8 probes cost a
third of semantic recall — see the curve in `IvfIndex::auto_nprobe`); and
PQ/ADC scoring is no longer used automatically — measured on 31k chunks it
collapsed IVF-only semantic recall from 0.677 to 0.143, while exact
vectorized FP16 scoring of every vector costs ~16 ms. `pq.bin` is still
built: its codebooks drive the incremental-rebuild cache, and `PqMode::Force`
keeps ADC available for benchmarks.

**Scoring path.** Exact FP16 scoring is the current default, including
`PqMode::Auto`; automatic selection of PQ is disabled for recall reasons.
The following historical micro-benchmark explains the cost tradeoff, not the
current selection policy. ADC pays a fixed lookup-table cost and then little
per document; exact scoring computes a dot product per candidate. Measured
with `cargo run --release --example pq_scoring_bench`:

```
candidates   exact FP16      PQ/ADC
       100         68 us      181 us
     1 000        646 us      155 us
    10 000        6.8 ms      278 us
   200 000        132 ms      1.9 ms
```

Both paths are written to vectorize (branch-free f16 decode, four-way
accumulators); exact FP16 scoring runs at the plain-f32 ceiling. Break-even
was near 225 candidates in that benchmark. `pq::MIN_CANDIDATES` (900) and
the IVF candidate-count estimate remain benchmark helpers; they no longer
switch live queries to ADC. `PqMode::Force` is available to experimental
callers, and `--no-pq` explicitly requests exact scoring. Segmented repository
retrieval does not use PQ or IVF.

Both the MCP server (rebuild on the next call) and the watcher (rebuild
after 300 ms of quiet) coalesce a burst of edits — a branch switch, a
formatter run — into one rebuild rather than one per file, and keep the
encoder on the thread whose pipelines are already warm.

### Run the tests

From the repository root, build all native grammars and run tests without
requiring grammar release downloads:

```sh
cargo build --manifest-path grammars/Cargo.toml --release --locked
export HIPS_GRAMMAR_DIR="$PWD/grammars/target/release"
export HIPS_GRAMMAR_OFFLINE=1
cargo test --locked
cargo test --locked --features semantic
cargo check --locked --no-default-features
```

The loader tests use a localhost HTTP server and the watcher tests need
filesystem events and child-process access; restrictive sandboxes may block
those integration checks. CI builds the libraries before testing.

The test suite includes unit tests for the tokenizer, BM25 math, index
construction, block construction (including the upper-bound invariant), the
top-k heap, the watcher's lease and hook-payload handling — plus integration
tests asserting Block-Max WAND returns the same top-k as the naive BM25
oracle on handcrafted and pseudo-random corpora, that blocks are skipped
once the threshold is high, that selective queries do not scan all
documents, that segmented and external builds persist and migrate, that the
repo lifecycle (edits, renames, deletions, merges) keeps line numbers right,
that the MCP server speaks JSON-RPC over a pipe, and that a leased watcher
follows edits, is shared by two sessions, and exits with the last one.
Tests also cover all grammar declaration fixtures, checksum rejection,
invalid libraries, and concurrent download publication. The suite uses lexical
execution or synthetic vectors for encoder-related checks, so model downloads
are not required; the semantic feature run checks that build configuration.

## Debug counters

Every search returns `SearchStats`:

| Counter | Meaning |
|---|---|
| `num_docs_total` | Documents in the index (for comparison). |
| `num_query_terms` | Unique query terms after tokenization. |
| `num_postings_visited` | Posting positions the cursors actually landed on (seeks binary-search within a block, so scanned-over entries are not counted; block decode work shows up in `num_blocks_visited`). |
| `num_docs_scored` | Documents fully scored with BM25. |
| `num_blocks_visited` | Blocks whose doc_ids were decoded (tfs are fetched lazily, only for scored postings). |
| `num_blocks_skipped` | Blocks jumped over using metadata only — their postings were never read. |

Low `num_docs_scored` relative to `num_docs_total`, and high
`num_blocks_skipped`, are Block-Max WAND doing its job.

## Implementation notes

- Tokenizer: hand-rolled — lowercase, split on non-alphanumeric (Unicode-aware
  via `char::is_alphanumeric`), drop empties, remove a small hardcoded English
  stopword list (~30 words). Queries are tokenized identically.
- Duplicate query terms are deduplicated; "pizza pizza" scores like "pizza".
- Ties are broken deterministically by (score desc, doc_id asc).
- Persistence: metadata via serde + bincode (`meta.bin`), postings as
  compressed blocks (`postings.bin`, memory-mapped at load).
- `unsafe` is confined to audited areas, each with its invariant documented
  at the call site: the `mmap` call in storage (standard accepted risk,
  same as Lucene's MMapDirectory); the NEON f16 dot product in
  `src/embeddings.rs`; the CoreML bindings in `src/coreml.rs` (objc2);
  native grammar loading and symbol conversion in `src/grammar.rs` (the
  library owner outlives its language and queries); the watcher's `setsid`,
  `kill(pid, 0)` liveness probes and signal
  handlers in `src/daemon.rs` (libc); and, behind the `gpu` feature, the
  zero-copy Metal interop in `src/reorder/gpu.rs` (page-aligned shared
  allocations, no-copy buffer wrapping, and disjoint parallel writes).

### Incremental updates (segmented indexes)

Indexes created with `add` are **segmented** and updatable without full
rebuilds, Lucene-style: each segment is an ordinary immutable index;
`add` creates a new segment from a batch, `delete` tombstones a document
(an update is delete + add), and `merge` compacts everything back to one
segment, dropping tombstones — after which scores are identical to a
fresh build of the live documents (verified by test).

```sh
hips add    --index ./idx --input batch1.jsonl
hips add    --index ./idx --input batch2.jsonl --upsert  # change-detecting
hips delete --index ./idx --id doc-123
hips merge  --index ./idx
```

`--upsert` answers "which documents changed?" inside the engine: every
document stores a content hash at index time, so re-feeding a whole
corpus skips unchanged documents, replaces changed ones, and adds new
ones — re-crawling a source and piping it through `add --upsert` only
writes what actually changed.

Lexical search remains globally exact across segments: queries score under
corpus-wide statistics (live N, global average length, df summed across
segments), which the impact-based block bounds make safe. One documented
deviation, shared with Lucene: df counts tombstoned documents until a
merge removes them. `search`/`serve` open both layouts transparently;
`migrate` upgrades pre-v4 indexes in place.

## Limitations

What this engine deliberately does not do:

- single-node only
- no distributed shards
- updates are batch-granular (new segment per add; no realtime ingest buffer)
- no phrase queries
- no general fuzzy term expansion; unknown query terms have SymSpell-style
  typo correction
- no autocomplete
- no aggregations
- no advanced analyzers (no stemming, no synonyms, no language-specific analysis)
- no distributed or SSD-native graph search; per-segment HNSW topology is
  RAM-resident and approximate, with an explicit exact-vector fallback
- no highlighting
- relevance controls include title weighting, candidate pools,
  and hybrid weighted/RRF fusion; no learned reranker or tuning service

## License

MIT. See [LICENSE](LICENSE).
