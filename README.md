# High-Performance Search Engine

A single-node, Elasticsearch-like **full-text search engine written in Rust
from scratch** — no Tantivy, Lucene, or any other search-engine crate. It indexes
JSONL documents into an inverted index and answers top-k queries with **BM25
scoring executed by exact Block-Max WAND**, over **compressed
(delta + bit-packed), memory-mapped posting lists** with optional
**document reordering (recursive graph bisection)** for better compression.

```
JSONL docs ──index──▶ inverted index + block metadata ──search──▶ exact BM25 top-k
                      (compressed, mmap'd, immutable)            (Block-Max WAND)
```

The theory and original papers behind every algorithm used here are
documented in [docs/THEORY.md](docs/THEORY.md).

## Status

This is an educational and experimental search engine, not a drop-in
Elasticsearch replacement. It is useful for learning how modern lexical search
works end to end: indexing, compression, memory-mapped storage, exact dynamic
pruning, batch updates, benchmarking, and a small HTTP API.

## Requirements

- Rust stable, edition 2021.
- Python 3 if you want to use the corpus helper scripts in `scripts/`.
- macOS with Metal only if you build the optional `gpu` feature for
  `--reorder bp-gpu`; the default build is CPU-only and portable.

## Quick start

```sh
cargo test
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

- `src/cli.rs` - command routing for `index`, `search`, `repl`, `serve`,
  `bench`, `add`, `delete`, `merge`, and `migrate`.
- `src/indexer.rs`, `src/postings.rs`, `src/compress.rs` - JSONL ingestion,
  inverted-index construction, block metadata, and bit-packed postings.
- `src/searcher.rs`, `src/block_max_wand.rs`, `src/maxscore.rs`,
  `src/bm25.rs` - exact BM25 top-k query execution.
- `src/storage.rs`, `src/segments.rs`, `src/external.rs` - on-disk format,
  memory mapping, segmented updates, and sharded external builds.
- `src/api.rs` - Axum HTTP API.
- `docs/THEORY.md` - algorithm notes and paper references.
- `tests/` - correctness checks against naive BM25 oracles and persistence
  tests.

## What this engine does

- **Indexes** JSONL documents (`{"id", "title", "body"}`), tokenizing title +
  body as one searchable field, in parallel with rayon.
- **Stores** an inverted index: term dictionary (`term -> u32 term_id`),
  posting lists sorted by `doc_id` (each posting is `{doc_id: u32, tf: u32}`),
  document metadata and lengths, and per-block metadata for skipping.
- **Searches** with BM25 (`k1 = 1.2`, `b = 0.75`) using exact top-k dynamic
  pruning: Block-Max WAND for short queries, MaxScore (Turtle & Flood 1995)
  for queries of 5+ unique terms, where WAND's pivot prefix rarely clears
  the threshold. Both are provably exact; there is no naive or approximate
  mode in the CLI or HTTP API. A naive BM25 scorer exists *only inside the
  test suite* as a correctness oracle for both evaluators.
- **Serves** concurrent queries over HTTP from a read-only, `Arc`-shared index.

### "Sublinear" in the practical retrieval sense

This engine is sublinear in the practical retrieval sense because it does not
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

Posting lists are split into fixed-size blocks (128 postings by default). For
each block the index stores `min_doc_id`, `max_doc_id`, and
`block_max_score` — the maximum BM25 contribution this term can make for
*any* document in the block, computed at index time with the exact same
formula used at query time.

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

`block_max_score` is computed as the maximum actual BM25 contribution over
all postings in the block. So for any document in that block, the term's real
contribution is ≤ `block_max_score` by construction. Summing these per-term
bounds gives a number the document's real score can never exceed. If that
bound is ≤ the current k-th best score, the document cannot enter the top-k —
skipping it cannot change the result.

### Why results are still exact, not approximate

Block-Max WAND never *estimates* a score. Every skip is justified by a proven
upper bound: documents are only skipped when they provably cannot beat the
current k-th result, and every returned document was scored with the full,
exact BM25 formula. The output is therefore identical to exhaustively scoring
every matching document (verified in the test suite against a naive BM25
oracle on handcrafted and randomized corpora; ranking ties at the k-th score
boundary are the only permitted variation).

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

`--title-weight N` (default 2) applies BM25F-lite field weighting: each
title occurrence of a term counts as N occurrences in the folded tf, so
title matches outrank otherwise-equal body matches. Set 1 to disable.

`--reorder` controls doc_id assignment: `none` (input order), `path` (sort
by external id — clusters file paths/URLs), or `bp` (recursive graph
bisection, minimizes the estimated compressed size; slower to build).
Reordering is a pure renumbering and never changes search results.

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

The index directory holds three files:

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
concurrently; nothing mutates it during searches.

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

### Run the tests

```sh
cargo test
```

The test suite includes unit tests for the tokenizer, BM25 math, index
construction, block construction (including the upper-bound invariant), and
the top-k heap — plus integration tests asserting Block-Max WAND returns the
same top-k as the naive BM25 oracle on handcrafted and pseudo-random corpora,
that blocks are skipped once the threshold is high, and that selective
queries do not scan all documents.

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
- `unsafe` is confined to two audited areas: the `mmap` call in storage
  (standard accepted risk, same as Lucene's MMapDirectory) and, behind the
  `gpu` feature, the zero-copy Metal interop in
  `src/reorder/gpu.rs` (page-aligned shared allocations, no-copy buffer
  wrapping, and disjoint parallel writes — each with its invariant
  documented at the call site).

### Incremental updates (segmented indexes)

Indexes created with `add` are **segmented** and updatable without full
rebuilds, Lucene-style: each segment is an ordinary immutable index;
`add` creates a new segment from a batch, `delete` tombstones a document
(an update is delete + add), and `merge` compacts everything back to one
segment, dropping tombstones — after which scores are identical to a
fresh build of the live documents (verified by test).

```sh
high-performance-search-engine add    --index ./idx --input batch1.jsonl
high-performance-search-engine add    --index ./idx --input batch2.jsonl --upsert  # change-detecting
high-performance-search-engine delete --index ./idx --id doc-123
high-performance-search-engine merge  --index ./idx
```

`--upsert` answers "which documents changed?" inside the engine: every
document stores a content hash at index time, so re-feeding a whole
corpus skips unchanged documents, replaces changed ones, and adds new
ones — re-crawling a source and piping it through `add --upsert` only
writes what actually changed.

Search remains globally exact across segments: queries score under
corpus-wide statistics (live N, global average length, df summed across
segments), which the impact-based block bounds make safe. One documented
deviation, shared with Lucene: df counts tombstoned documents until a
merge removes them. `search`/`serve` open both layouts transparently;
`migrate` upgrades pre-v4 indexes in place.

## Limitations compared to Elasticsearch

- single-node only
- no distributed shards
- updates are batch-granular (new segment per add; no realtime ingest buffer)
- no phrase queries
- no fuzzy search
- no autocomplete
- no aggregations
- no advanced analyzers (no stemming, no synonyms, no language-specific analysis)
- no vectors
- no semantic search
- no highlighting
- no relevance tuning beyond BM25
- compression is scalar binary packing — solid but not the last word (real
  engines use SIMD-accelerated codecs like SIMD-BP128 or partitioned
  Elias-Fano for another ~1.5-2x and faster decode)

## License

MIT. See [LICENSE](LICENSE).
