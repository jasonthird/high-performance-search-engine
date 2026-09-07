# Algorithms & Theory

The algorithms and architecture implemented in this working tree, checked
against the source on 2026-09-07. File references point at the implementation.
Historical benchmark results describe the measured corpus, hardware, and
configuration; they are not fresh measurements of every current build.

The current repository-search path combines declaration chunks from loadable
tree-sitter grammars, incremental immutable segments, exact BM25 retrieval,
and optional CodeRankEmbed vectors. Single-layout indexes retain IVF/PQ for
experiments and bulk corpora. The CLI, REPL, and MCP server share ranked
retrieval; the HTTP API currently exposes lexical search.

---

## 1. Inverted index

**Where:** `src/indexer.rs`, `src/postings.rs`

The foundational data structure of lexical search (dates back to the 1950s;
the standard reference is Zobel & Moffat, *"Inverted files for text search
engines"*, ACM Computing Surveys 2006). Instead of storing documents and
scanning them at query time, store for each **term** the sorted list of
documents containing it:

```
term_id  ->  [(doc_id, tf), (doc_id, tf), ...]      sorted by doc_id
```

A query only reads the posting lists of its own terms, so documents
containing none of the query terms are never touched. Sorting by doc_id is
what enables everything else below: gap compression, binary-searchable
skipping, and the merge-style cursor advancement in WAND.

Construction streams JSONL in batches, parses and tokenizes documents in
parallel with rayon, interns terms, and scatters per-document term counts
into ordered posting lists before attaching impact metadata. Raw documents
do not accumulate for the entire corpus. `src/external.rs` adds a bounded
spill-to-disk path: sorted posting shards are merged into compressed output
when the corpus exceeds the in-memory build budget.

The single-layout hybrid path (§14) reuses this exact posting-list shape a second
time, with **cluster ids** as the vocabulary: `cluster_id → (doc_id, tf=1)`.
Both inverted files address the same `doc_id` space.

## 2. BM25 ranking

**Where:** `src/bm25.rs`

Okapi BM25 (Robertson & Walker, SIGIR 1994; survey: Robertson & Zaragoza,
*"The Probabilistic Relevance Framework: BM25 and Beyond"*, 2009). Derived
from the probabilistic relevance framework, it scores a document D for query
Q as:

```
score(D, Q) = Σ_q  idf(q) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · |D| / avgdl))
```

with `k1 = 1.2`, `b = 0.75` (the conventional defaults). Three ideas:

- **idf** — rare terms carry more information. We use the "+1" smoothed form
  `ln(1 + (N − df + 0.5)/(df + 0.5))` (the same variant Lucene adopted),
  which is always positive — important because WAND's upper-bound math
  assumes non-negative contributions.
- **tf saturation** — the `tf/(tf + k1·…)` shape grows but flattens: the
  10th occurrence of a word proves less than the 2nd. Contribution is
  bounded by `idf · (k1 + 1)`, which is what makes per-term upper bounds
  finite and tight.
- **length normalization** — an occurrence in a short document is stronger
  evidence than in a long one; `b` interpolates between no normalization
  (b=0) and full proportional normalization (b=1).

## 3. Top-k via bounded min-heap

**Where:** `TopK` in `src/block_max_wand.rs`

Selecting the k best of n scored candidates with a size-k min-heap:
O(n log k) instead of sorting everything. The heap minimum doubles as the
**threshold** for dynamic pruning — the bridge to WAND. Ties at equal score
are broken toward the smaller doc_id (the heap evicts the largest doc_id
among equal scores), making results deterministic.

## 4. WAND dynamic pruning

**Where:** the pivot-selection step in `src/block_max_wand.rs`

WAND — "Weak AND" / "Weighted AND" (Broder, Carmel, Herscovici, Soffer,
Zien, *"Efficient query evaluation using a two-level retrieval process"*,
CIKM 2003). Precompute for each term an **upper bound** U_t on its possible
BM25 contribution. At query time keep one cursor per term, sorted by current
doc_id, and find the **pivot**: the first prefix of cursors whose ΣU_t
exceeds the current threshold θ.

Soundness: a document d smaller than the pivot document can contain only
terms from a strict prefix of the sorted cursors whose ΣU_t ≤ θ, so
score(d) ≤ ΣU_t ≤ θ — it cannot enter the top-k and is skipped without
scoring. This is *document-at-a-time* (DAAT) evaluation with safe skipping;
results are provably identical to exhaustive evaluation ("rank-safe").

One correctness subtlety (caught by the oracle tests during development):
after choosing the pivot index, every cursor already positioned **on** the
pivot document must be included in the bound/evaluation prefix, otherwise
the refined bound below under-counts and a competitive document can be
skipped.

## 5. Block-Max WAND (BMW)

**Where:** `src/block_max_wand.rs`, block metadata in `src/postings.rs`

Ding & Suel, *"Faster top-k document retrieval using block-max indexes"*,
SIGIR 2011. WAND's weakness is that U_t is one number for an entire posting
list — usually a wild overestimate for any particular region of it. BMW
splits each posting list into fixed-size blocks (128 postings here) and
stores the block's last document id and an impact pair:

```
max_doc_id            skip whole blocks ending before a target
max_tf, min_doc_len   derive a safe bound under current corpus statistics
```

Block posting ranges are derived from the fixed block size. BM25 increases
with term frequency and decreases with document length, so evaluating the
pair bounds every posting in the block. The two extrema may belong to
different documents: this is a conservative bound, not necessarily a score
any document attains. Query-time evaluation keeps it safe when segment
updates change idf and average length (§11). After WAND picks a pivot, BMW
sums these bounds for the blocks containing the pivot doc. If even this refined bound is
≤ θ, the engine jumps all prefix cursors past the nearest block boundary —
skipping whole blocks *without decoding them*. The jump target is capped at
the next cursor's doc_id, because beyond it another term could contribute.

Lucene ≥ 8 (hence Elasticsearch/OpenSearch) uses this family of algorithms
for its top-k scoring. A refinement not implemented here: Variable BMW
(Mallia, Ottaviano, Porciani, Suel, Venturini, SIGIR 2017) chooses block
boundaries to minimize bound slack.

**MaxScore** (Turtle & Flood 1995, `src/maxscore.rs`) *is* implemented and
dispatched for queries of 5+ unique terms, where WAND pivoting weakens.
Terms sorted by ascending upper bound split at the threshold: the
non-essential prefix (combined bounds ≤ θ) is never iterated, only probed
for candidates that already look competitive — with early abandonment as
soon as partial score + remaining bounds ≤ θ. Both evaluators are exact and
verified against the same naive oracle.

Implementation choices that matter for speed (each verified
behavior-preserving by the oracle tests; together ~2x on the home corpus):

- **Lazy single-cursor advancement** — when cursors must move (to the pivot,
  or past a skipped block range), advance only the one with the largest
  upper bound (the heuristic from the original WAND paper) instead of all of
  them. Untouched cursors may never decode those blocks at all. Sound
  because docs the moved cursor passes were already proven non-competitive,
  and the threshold only rises — a later partial evaluation of such a doc
  can never re-enter the top-k.
- **In-block binary search** — block doc_ids are sorted, so seeks
  `partition_point` to the landing position instead of scanning linearly.
- **Lazy tf access** — fixed-width packing gives O(1) random access, so term
  frequencies are read only for postings actually scored (~1% of those
  visited), not decoded per block.

## 6. Postings compression: delta encoding + binary packing

**Where:** `src/compress.rs`

Within a block, doc_ids are strictly increasing, so store **gaps**
(`doc_id[i] − doc_id[i−1] − 1`) instead of absolute ids; the first id is kept
raw in the block header. Gaps are small where postings are dense, and the
information-theoretic cost of a gap g is ~log2(g) bits, not 32.

Each block packs its gaps at one fixed width: `doc_bits = bits_needed(max
gap)` — **Frame of Reference (FOR) / binary packing** (see Lemire & Boytsov,
*"Decoding billions of integers per second through vectorization"*, 2015 for
the modern SIMD treatment; this implementation is the scalar version).
Term frequencies, almost always 1, are packed the same way as `tf − 1`.
A block of 128 postings with all-1 gaps and all-1 tfs costs **8 bytes total**
(header only, 0-bit payloads).

On the home-directory corpus this cut postings from 8 bytes/posting raw to
~1.77 bytes/posting (4.5×). Production formats go further (PForDelta,
partitioned Elias–Fano, SIMD-BP128), trading more complexity for another
~1.5–2×.

Why fixed-width per block rather than per-integer codes (varint, gamma)?
Document gaps are decoded a block at a time, while fixed-width term
frequencies support random access for only the documents actually scored.
Fixed width keeps decoding simple and makes lazy tf reads inexpensive.

Decode extracts multiple gaps per unaligned 64-bit load — (64−7)/width
values regardless of bit alignment (2 for rare terms, dozens for dense
ones), with an all-gaps-are-1 fast path for fully dense blocks. This is
the stable-Rust equivalent of SIMD unpacking for an LSB bit-stream
(`std::simd` remains nightly-only); true lane-parallel decode would require
a planar SIMD-BP128-style layout — a format change with marginal headroom
left, since decode is no longer load-bound.

A note on GPU level-batching for BP (the one optimization considered and
*rejected*): batching all partitions of a recursion level into one kernel
dispatch needs per-partition degree arrays. Dense arrays grow as
2^depth × vocabulary — feasible only for the top ~6 levels, which the
per-partition dispatch already covers at ~8 ms each — and sparse
per-partition term remapping costs more CPU per level than the dispatch
overhead it would remove. The deep levels that dominate BP runtime are
structurally CPU-shaped at these corpus sizes.

## 7. Document reordering: recursive graph bisection (BP)

**Where:** `src/reorder.rs`

Doc_ids are arbitrary labels; compression depends on the gaps between them.
Assigning nearby ids to documents that share terms shrinks gaps. Finding the
optimal assignment is NP-hard (it generalizes minimum linear arrangement),
so heuristics:

- **Path/URL sorting** (Silvestri, ECIR 2007): sort by external id. Files in
  the same directory / pages on the same site share vocabulary. Nearly free
  and surprisingly strong. (On the home-corpus benchmark it looks like a
  no-op only because the crawler already emits files in directory order.)

- **Recursive graph bisection** — Dhulipala, Kabiljo, Karrer, Ottaviano,
  Pupyrev, Shalita, *"Compressing Graphs and Indexes with Recursive Graph
  Bisection"*, KDD 2016. The state of the art, used at Facebook and
  implemented in PISA. Model the corpus as a bipartite doc–term graph and
  minimize the **log-gap cost** — a proxy for the compressed index size:

  ```
  cost(partition of size n, term with degree d) ≈ d · log2(n / (d + 1))
  ```

  Recursively split the document set in half; within each split, iterate:
  compute for every document the **move gain** (cost delta from moving it to
  the other half, summed over its terms), sort both sides by gain, swap the
  best pairs while the combined gain is positive, repeat until convergence
  (≤ 12 iterations here); recurse on the halves (in parallel) down to
  partitions of 32 documents. The final left-to-right leaf order is the new
  doc_id assignment.

  Reordering is a pure renumbering — the tests verify it cannot change
  search results (only tie-breaks among equal scores, which are id-based).

## 8. Memory-mapped, paged index access

**Where:** `src/storage.rs`

The index is split into:

- `meta.bin` — document lengths, the term dictionary, per-term statistics,
  and impact/skip tables, loaded into RAM. Document text lives separately in
  `docs.bin`, so scoring does not need to materialize every title or snippet.
- `docs.bin` — memory-mapped document ids, titles, and snippets; only returned
  hits need their document-store records resolved.
- `postings.bin` — all compressed posting blocks. **Memory-mapped**, not
  read: `mmap(2)` maps the file into virtual address space; the OS faults
  4 KiB pages in on first access and evicts them under memory pressure
  (demand paging). Startup does no postings I/O at all, indexes larger than
  RAM work transparently, and the page cache is shared across processes.

  This is the same design choice as Lucene's `MMapDirectory` — the OS page
  cache replaces a hand-rolled buffer manager (see Kraska et al.'s caveats
  vs. the classic "mmap considered harmful" debate; for a read-only,
  immutable index, mmap is the easy win).

The synergy with BMW + compression: a skipped block is never decoded, so its
bytes are never touched, so its page is never read from disk. Logical
skipping becomes physical I/O avoidance.

A single-layout embedded index (§14) adds three more mmap sidecars in the same
directory — `embeddings.bin` (dense rows), `ivf.bin` (cluster posting
lists, same block codec as `postings.bin`), `pq.bin` (product-quantized
codes). Lexical scoring does not read vector rows. The default repository
layout instead stores vectors per segment without IVF/PQ.

## 9. GPU document reordering: Metal and the earlier experiments

**Where:** `src/reorder/gpu.rs` (feature `gpu`, `--reorder bp-gpu`)

The shipped optional backend uses zero-copy Metal through `objc2-metal`.
Burn, wgpu, and CubeCL below describe earlier implementations, not current
build dependencies. The experiment in accelerating construction began
with [Burn](https://burn.dev)'s tensor API, then moved to a single
hand-fused kernel in [CubeCL](https://github.com/tracel-ai/cubecl) (the GPU
compute DSL underneath Burn). Historical results, on the 108k doc / 19.2M edge
home corpus:

| Variant | Index time |
|---|---|
| CPU BP (rayon) | **13.9 s** |
| GPU BP, naive Burn tensor port | 197 s |
| GPU BP, Burn after optimization | 20.1 s |
| GPU BP, fused CubeCL kernel | 14.9 s |

What the measurements taught:

- **Only BP's gain computation is expressible as GPU work at all.**
  Tokenization, hashing, and posting-list merging — the bulk of plain
  indexing time — are string/hash workloads with no tensor formulation, so
  Amdahl's law caps any GPU benefit before starting.
- **Scatter-add was the Burn pathology.** Burn/wgpu's `select_assign`
  accounted for 3.7 s of each 3.9 s iteration — ~50x the cost of everything
  else combined. Moving degree counting to CPU-side native atomics and
  exploiting that the edge list is document-contiguous (per-doc gain sums
  become a linear pass) fixed it.
- **The tensor abstraction itself was the next tax.** Burn turned the gain
  formula into ~50 elementwise kernel dispatches with 76 MB intermediates
  and a 76 MB readback. The CubeCL rewrite fuses the entire pass into one
  kernel — one thread per document loops its edges, 8 logs per edge, writes
  one float — and reads back 4 bytes per *document* instead of per edge.
  Measured: **6–12 ms** per 19M-edge iteration, vs ~50 ms for optimized
  Burn and ~150 ms for the CPU. Corpus edge lists are uploaded once per run
  (the kernel resolves partitions through an `order` indirection), so per
  level only ~3 MB of degree/side arrays move.
- **Degree counting went incremental.** Recounting per-side term degrees
  every iteration (~25 ms) initially dwarfed the 6–12 ms kernel. Since only
  *swapped* documents change side, patching the counters for moved documents
  cuts that to 2–5 ms; a full 19M-edge iteration is now ~10 ms.
- **Unified-memory discipline: never re-send what didn't change, never
  round-trip what the kernel can derive.** Apple-silicon CPU and GPU share
  physical memory, but the wgpu layer doesn't expose Metal's zero-copy
  buffer import, so every upload is still a memcpy. The mitigations:
  corpus edge lists and per-partition totals upload once; `deg_left` is
  derived in-kernel from `deg_total − deg_right` (halving per-iteration
  upload); only the side assignments and right-half degrees move per
  iteration (~3 MB).
- **Keep the GPU queue fed.** Sibling partitions are independent, so the
  recursion issues them concurrently (`rayon::join`): while one partition
  is in a CPU phase (degree snapshot, sort, swap), the other has a kernel
  in flight.
- **Result: a consistent ~10% win on the reorder phase.** Three alternating
  runs: CPU 8.1/8.4/8.4 s vs GPU 7.8/7.5/7.5 s. Still bounded by Amdahl:
  ~two-thirds of BP time lives in sub-8192-doc partitions, which run on the
  CPU in both modes because per-launch overhead beats the work at that
  size. Batching entire recursion levels into single launches would push
  GPU coverage deeper, at the cost of per-partition degree-array memory
  growing with 2^depth.
- **Apple-silicon-specific economics:** unified memory makes the CPU
  unusually competitive — there is no PCIe gap for the GPU to win back, and
  the performance cores are excellent at exactly this sparse, branchy work.

Epilogue: the CubeCL kernel was subsequently replaced by a **zero-copy
Metal implementation via objc2-metal** (now the shipped `gpu` feature, macOS
only). Every GPU-visible array is page-aligned host memory wrapped with
`newBufferWithBytesNoCopy` — the CPU patches degree counters and permutes
the order array in place, the GPU reads the same physical pages, and gains
are sorted straight out of the pages the kernel wrote. Per iteration the
only explicit transfer is a 12-byte params struct. Result: the reorder
phase runs ~25% faster than the rayon CPU path (best run 36%) — the first
decisive GPU win of the experiment, and a demonstration that on unified
memory the transfer discipline matters as much as the kernel. The broader
conclusion stands: lexical index construction is CPU-shaped, and GPU
investment pays off mainly where dense math lives (embedding/vector
retrieval, where kernel libraries like CubeK — matmul, reductions,
quantization, attention on CubeCL — would slot in directly). That path is
now §14: Candle on Metal runs the encoder; retrieval itself stays an
inverted file plus (optional) product-quantized table lookups on the CPU.

## 10. Concurrency model

**Where:** `src/indexer.rs` (build), `src/api.rs` (serve)

- **Indexing**: rayon parallelizes document work and finalization. Posting
  lists are ordered by doc_id; sorted dictionary serialization and stable
  input ordering make retrieval independent of worker scheduling.
- **Serving**: the index is immutable after build (`Arc<AppState>` containing an `AnyIndex`, shared
  across handlers, no locks needed). Searches run on tokio's blocking pool so
  CPU-bound scoring doesn't starve the async accept loop. Immutability is
  what makes the whole read path trivially thread-safe — the same reason
  Lucene segments are write-once.

## 11. Segmented indexes & impact-based bounds

**Where:** `src/segments.rs`, impacts in `src/postings.rs`

The classic write path of Lucene (and every LSM system): immutability per
segment, mutability as a collection of segments. New documents form fresh
segments; deletes are tombstone bitmaps; updates are delete + re-add;
explicit or repository-threshold merges compact — decode postings, drop tombstoned docs, remap
ids densely, re-encode. A merged index scores identically to a rebuild of
the live documents.

The subtle prerequisite is **impacts** (Lucene's term): block upper
bounds must not be precomputed scores, because BM25 scores depend on
corpus statistics (idf, average length) that shift as segments come and
go — a stored bound computed under yesterday's stats can silently
under-estimate today's contribution and break rank-safety. Instead each
block stores its dominating coordinates (max tf, min doc length); the
bound is computed at query time under current global statistics. BM25's
monotonicity (increasing in tf, decreasing in length) makes the pair
bound every posting in the block, under *any* stats.

Cross-segment scoring uses global statistics (live N, global average
length, df summed across segments), so a document scores the same
regardless of which segment holds it. The one deviation, shared with
Lucene: df counts tombstoned documents until merge.

## 12. Query typo correction: Symmetric Delete (SymSpell)

**Where:** `src/spell.rs`, wired in `src/searcher.rs`

A misspelled query term ("pizzza") has document frequency 0 and retrieves
nothing. Before BM25 runs, a rewrite layer maps such terms to the nearest
real vocabulary term. "Nearest" is **Damerau-Levenshtein distance** (the
optimal string alignment variant): the fewest single-character insertions,
deletions, substitutions, and *adjacent transpositions*. Transpositions
matter because "teh" → "the" is the most common typo class, and plain
Levenshtein charges it 2 edits where Damerau charges 1.

The algorithm is Wolf Garbe's **Symmetric Delete** ("SymSpell", 2012),
reimplemented here from the description — no crate. The naive alternatives
both fail at scale: scanning the whole vocabulary is O(|vocab|) per lookup,
and Norvig-style candidate generation multiplies by the alphabet size, giving
~100k variants at edit distance 2 (and the "alphabet" is unbounded under
Unicode).

**The symmetry trick.** Deletes are alphabet-independent: a word of length n
has only n distance-1 deletes and ~n²/2 distance-2 deletes, with no ×26
blowup. Deletes alone can't express an insertion or substitution — unless
*both sides* delete. The key claim: if a dictionary term `t` and a query `q`
are within distance d, then some ≤d-delete of `t` **equals** some ≤d-delete of
`q`. Each edit class meets in the middle (d = 1 shown; d = 2 composes):

| query error | example | meeting point |
|---|---|---|
| insertion | "pizzza" (t + 1 char) | delete 1 from *q* → t |
| deletion | "piza" (t − 1 char) | delete 1 from *t* → q |
| substitution | "pizca" (1 char differs) | delete that char from *both* |
| transposition | "piazz…" (2 adjacent swapped) | delete either from *both* |

So we **precompute**, for every vocabulary term, the hashes of all its
≤d-delete variants into a map `hash(delete) -> [term ids]`. At **lookup**, we
generate the same delete variants of the query term and union the buckets they
hit — the candidate superset.

**Verification is mandatory.** A shared delete is necessary, not sufficient
("bank" and "beak" both delete to "bak" but are distance 2 apart), and
distinct deletes can hash-collide. So each candidate gets its true
Damerau-Levenshtein distance computed against the query (an O(m·n) DP over two
short words), those over budget are dropped, and survivors rank by *(distance
ascending, then document frequency descending)*. That verify step is also what
makes two memory optimizations provably safe — they can only *add* false
candidates, never drop true ones: the **prefix optimization** (generate
deletes from only the first 7 characters, bounding entries per term to
1+7+21 = 29) and **hashing the delete strings** to `u64` keys instead of
storing them.

**Design choices specific to this engine:**

- **The dictionary is the index's own vocabulary**, weighted by df — no
  external word list. Every correction therefore points at a term that
  actually retrieves something, and the on-disk index is never touched. The
  full-vocabulary walk that seeds it is `DiskIndex::for_each_term`.
- **Correction targets require df ≥ 3.** Corpus typos and OCR junk are
  overwhelmingly df-1/2 terms; excluding them both improves correction quality
  and shrinks the deletes map several-fold.
- **Length-scaled budget**, mirroring Elasticsearch `fuzziness: AUTO`: words
  under 3 chars are never corrected, 3–5 chars allow distance 1, longer allow
  distance 2. (Correcting "cat" at distance 2 would match half the
  dictionary.)
- **Only terms with df = 0 are rewritten** — a term that matches even one
  document is left alone. Clean queries pay one df lookup per term and never
  build the corrector, which is constructed lazily on the first miss
  (`OnceLock`). Rewrites are surfaced, not silent: `SearchOutcome.corrected`
  carries the rewritten query to the CLI and HTTP responses.

Cost: build is one vocabulary pass (~29 entries/term); on a full-Wikipedia
vocabulary the map is order 1–2 GB after the df filter, which is why it is
lazy and gated. The documented upgrade path when that hurts is an FST plus a
Levenshtein automaton (the Lucene/Elasticsearch approach), which intersects
the query automaton against a trie term dictionary and needs no precomputed
deletes.

## 13. Testing methodology: oracle testing

**Where:** `tests/bmw_correctness.rs`, `tests/disk_and_reorder.rs`

BMW, compression, and reordering are all *behavior-preserving
optimizations*: each must produce results identical to the simple thing it
replaces. So the tests pin them against oracles:

- BMW vs. a **naive exhaustive BM25 scan** (the oracle exists only in test
  code) over handcrafted and seeded pseudo-random corpora (LCG generator, no
  rand dependency), ~450 query/k combinations.
- DiskIndex (compressed + mmap) vs. the in-memory index it was saved from —
  results *and* skip counters must match exactly.
- Reordered indexes vs. natural order — identical scores; document sets may
  differ only among score ties cut by the k boundary (tie-breaking uses
  internal ids, which reordering legitimately renumbers).
- Property tests: every impact-derived block bound ≥ every actual contribution in its
  block (the invariant that makes skipping safe), encode/decode round-trips,
  BP outputs a valid permutation and reduces measured log-gap cost.
- Typo correction: unit tests in `src/spell.rs` pin the OSA distance
  (including transpositions), the length-scaled budget, and df tie-breaking;
  an integration test drives a real on-disk index end-to-end (typo →
  corrected query → right documents; clean and below-threshold queries left
  untouched).
- Hybrid / IVF / PQ: unit tests pin fusion (weighted + RRF), that
  encoder-first merge recovers a document BM25 never returned, that
  spherical k-means separates orthogonal vectors onto different cluster
  lists, and that `nprobe=1` does not open the orthogonal list. The
  lexical oracle suite is unchanged. IVF/PQ are approximate; segmented
  semantic search scores all live stored vectors exactly, while hybrid
  fusion combines bounded candidate pools and has a different ranking goal.
- Grammar tests: declaration fixtures cover all 41 language variants and
  nested chunks. Loader tests serve local HTTP artifacts to verify checksums,
  invalid-library rejection, concurrent atomic publication, and real parsing
  through a downloaded library. Build the grammar workspace and set
  `HIPS_GRAMMAR_DIR` before running the suite (see `../grammars/README.md`).

---

## 14. Hybrid retrieval and encoder backends

**Where:** `src/query.rs`, `src/hybrid.rs`, `src/embeddings.rs`,
`src/embedder.rs`, `src/coreml.rs`, `src/ivf.rs`, `src/pq.rs`

The `semantic` feature adds CodeRankEmbed inference. Lexical postings and
vectors share document identities; there is no external vector service.
The current retrieval paths differ by index layout:

| Layout | Construction | Vector retrieval |
|---|---|---|
| Segmented (repository default) | `index-repo`, watcher, MCP | Exact cosine over every live vector in each segment |
| Single | `index --embed`, `index-repo --single` | IVF candidate lists when present; full vector scan otherwise |

Single-layout files are `embeddings.bin` (768-dimensional FP16 rows),
`ivf.bin` (cluster postings), and `pq.bin` (codebooks and codes). Row
`i` has the same `doc_id` as lexical row `i`. In a segmented index,
alignment is local to each segment; tombstones filter both lexical and
semantic results. Each embedded segment also has `keys.bin` so merges
can recover vectors from the content cache.

### Query modes and fusion

The CLI, REPL, MCP server, and evaluation tools use `src/query.rs`.
The HTTP handler in `src/api.rs` currently calls lexical search directly.

- `hybrid`: retrieve an encoder candidate pool and BM25 candidates, union
  by document identity, then fuse. BM25 does not gate semantic retrieval.
- `semantic`: retrieve by the encoder score alone.
- `rerank`: score and fuse only the BM25 candidate pool.
- `bm25` / `--lexical`: no query encoder.

The default candidate pool is 200 (at least `top_k`). Query encoding and
BM25 overlap on separate threads; vector scoring waits for the query
embedding. A lexical-only candidate still receives a point cosine score,
so a missed IVF cluster does not exclude a BM25 hit from hybrid fusion.

Weighted fusion is the default, with `alpha = 0.15`:

```
score = alpha * normalized_bm25 + (1 - alpha) * cosine
```

BM25 is min-max normalized over the candidate union, with lexical misses
remaining zero. Cosine is already in [-1, 1]. RRF remains available via
`--fusion rrf`, using the sum of `1 / (rrf_k + rank)` with default
`rrf_k = 60` (Cormack, Clarke & Buettcher, SIGIR 2009).

Historical measurements on 31k code chunks (4,567 natural-language queries,
800 identifier queries) motivated these defaults:

| Mode | NL Recall@10 | Identifier Recall@10 |
|---|---:|---:|
| BM25 | 0.255 | 0.976 |
| Semantic | 0.716 | 0.974 |
| Hybrid, weighted alpha 0.15 | 0.696 | 0.993 |

These are retrieval-quality measurements, not proofs that one fusion
setting is optimal for every repository.

### Code tokenizer

**Where:** `src/tokenizer.rs`

The code analyzer indexes full lowercased identifiers plus camelCase,
PascalCase, snake_case, acronym, digit-boundary, and dotted-name pieces.
Thus `getUserByOrganizationId` retains an exact-name term while also
matching ordinary words. Full identifiers bypass stopword removal;
derived pieces do not. The analyzer flag is persisted in index metadata,
so queries use the analyzer that built the index.

### Encoding, caches, and CoreML

CodeRankEmbed produces 768-dimensional, CLS-pooled, L2-normalized vectors.
Documents are raw code. Queries receive the prefix
`Represent this query for searching relevant code: `.
Document tokenization defaults to 512 tokens; queries to 64.
`HPS_EMBED_MAX_SEQ` overrides the document cap.

Candle is the portable backend, using Metal F16 when available on macOS
and CPU F32 otherwise. `HPS_EMBED_DEVICE=cpu|metal` overrides Candle's
selection; unavailable Metal falls back to CPU. The current Cargo patch
uses a Candle fork with fused Metal SDPA for NomicBert.

CoreML/ANE is implemented in `src/coreml.rs`, not merely a prototype.
On macOS, compiled models under `<csearch cache>/coreml/` are detected
automatically. The intended family is batch-8 document models at sequence
64/128/256/512 and a batch-1 query model at sequence 64. Shapes load lazily,
and documents use the smallest fitting shape. Install the complete document
family and query model together; a query-only installation cannot encode
documents. `HIPS_ENCODER=candle` bypasses CoreML.

The Hugging Face model files download on first use into its cache, honoring
`HF_HOME` and `HF_ENDPOINT`. Compiled CoreML models are produced locally
with the scripts in `scripts/ane-prototype/`; the runtime does not download
or convert them. Loading CoreML still uses the Hugging Face configuration,
tokenizer, and weights paths.

Document vectors are cached persistently by chunk content in `embcache.bin`.
Encoding length-buckets chunks to reduce padding. Repeated query strings
have a separate 256-entry in-process LRU. `embeddings.bin` stores a
32-byte header and row-major FP16 vectors: 1,536 bytes per 768-dimensional
row. Exact scoring means exact dot products over these stored, rounded
vectors, not equivalence to an unrounded model output.

### IVF: approximate candidate selection in single-layout indexes

Spherical k-means partitions documents into cluster posting lists using
the same block codec as lexical postings. Default cluster count is about
`2 * sqrt(N)`, bounded by corpus size and 256. Centroids are normalized;
assignment uses cosine. Every document belongs to exactly one cluster.

A query scores centroids, opens the nearest `nprobe` lists, and scores
their documents. Automatic probing is `max(K / 2, 1)`, replacing the old
eight-probe cap. In the historical 31k-chunk sweep, half the clusters
retained about 0.977 Recall@10 relative to probing all clusters; eight
probes retained 0.677.

Unopened clusters can contain relevant documents, so IVF is approximate.
No WAND pruning is performed inside cluster lists: their uniform
`tf = 1` and disjoint membership make direct candidate enumeration the
current implementation. Segmented repository search does not use IVF.

The underlying references are Sivic & Zisserman, *Video Google*, ICCV
2003, and Jégou, Douze & Schmid, *Product Quantization for Nearest Neighbor
Search*, IEEE TPAMI 2011.

### Product quantization: retained, not automatically selected

PQ splits each vector into 16 subspaces and learns up to 256 codewords
per subspace, yielding a 16-byte code per document. ADC builds query-side
lookup tables and approximates a dot product by summing 16 entries.
This implementation quantizes raw vectors, not coarse-centroid residuals.

`PqMode::Auto` currently selects exact FP16 scoring regardless of candidate
count. The historical 900-candidate cost threshold in `src/pq.rs` remains
a benchmark helper, not the live selection policy. On a 31k-chunk benchmark,
adding PQ reduced semantic Recall@10 from 0.677 to 0.143, while full exact
scoring cost about 16 ms. `PqMode::Force` retains ADC for experiments;
`--no-pq` explicitly selects exact scoring.

Single-layout rebuilds still retain IVF centroids and PQ codebooks.
Training is separate from assignment, so unchanged vectors and their
parameter-tagged codes can be reused. A changed quantizer invalidates
cached codes. `--retrain` applies to this single-layout path; segmented
repository indexes have no IVF/PQ quantizers to retrain.

### Evaluation and exactness

`eval-code` computes MRR, Recall@5/10, and graded nDCG@10.
`eval-gen` creates doc-comment and identifier query sets from repositories.
The BM25 oracle establishes rank safety for lexical pruning, not for the
entire hybrid pipeline. IVF candidate selection and forced PQ can change
semantic top-k; segmented cosine scans have neither approximation, but
hybrid fusion still works over bounded candidate pools.

## 15. Declaration chunks and loadable grammars

**Where:** `src/repo.rs`, `src/treesit.rs`, `src/grammar.rs`,
`tree-sitters/`, and the independent `grammars/` Cargo workspace

The registry has 41 grammar variants (including separate TypeScript and
TSX). Extensions choose a language without loading it; `.m` uses content
to distinguish Objective-C and MATLAB. Only the first parse for a language
loads its library and compiles its embedded definition query.

The CLI links the tree-sitter runtime, not the grammar parse tables.
Each grammar is a separate native `cdylib` exporting `hips_language`.
A `OnceLock` retains each loaded language and library for the process
lifetime, so query/parser pointers never outlive the code and static tables
they reference. Errors are cached for that process and warn once per
language before falling back to keyword-based chunks.

Library lookup is:

1. A packaged library in `HIPS_GRAMMAR_DIR`, used in place.
2. The versioned user cache under
   `$XDG_CACHE_HOME/csearch/grammars/<release>/<target>`, defaulting to
   `~/.cache/csearch/grammars/...`.
3. An individual download from the pinned grammar release, unless
   `HIPS_GRAMMAR_OFFLINE=1`.

Maintainers can build and ship no grammars, selected packages, or all of
them independently of the CLI. Cached downloads contain only encountered,
unbundled languages. The grammar cache does not currently use
`CSEARCH_CACHE_DIR`, which configures repository/CoreML caches instead.

Downloads check a release SHA-256 sidecar, enforce a size limit and timeout,
validate the exported language and tree-sitter ABI, then publish atomically
from a unique temporary file without clobbering a concurrent winner.
Installed libraries take precedence; an invalid installed library warns
and falls back to heuristic chunking rather than silently replacing it
with a downloaded one.

The release workflow builds 41 libraries per platform for macOS and glibc
Linux on arm64 and x86_64. `src/grammar.rs` pins `grammars-v1`; its assets
must be published before that download route is usable. The locally tested
macOS arm64 default release executable is 10,101,008 bytes (9.6 MiB), with
about 68 MiB of grammar libraries separately. This is a local default-feature
measurement, not a size promise for semantic builds or other platforms.

Definition queries remain small `.scm` files embedded in the CLI.
Definitions and preceding comments define chunk boundaries; nested units
keep parent names, adjacent bodyless declarations coalesce, and files stay
covered without gaps. Markdown uses heading sections; PDFs are extracted
page by page independently of tree-sitter. Unsupported files and unusable
parses use the existing heuristic.

## 16. Incremental repository lifecycle

**Where:** `src/codeindex.rs`, `src/embcache.rs`, `src/watch.rs`,
`src/daemon.rs`, `src/mcp.rs`

Repository indexing defaults to segments. The per-file manifest records
path, size, mtime, and prior chunk ids. An edit tombstones stale chunks,
appends changed chunks, and encodes only uncached content. More than 12
segments triggers compaction; vectors are recovered by content-cache key
instead of being re-encoded. A writer lock serializes competing writers.

The unchanged-tree shortcut uses path/size/mtime fingerprints, not a fresh
content hash of every file. The manifest also records a chunker version,
which can trigger re-chunking after a chunker change. Grammar availability
and the grammar release are not currently part of that fingerprint:
installing a missing library alone does not force unchanged files to be
re-chunked.

The session watcher coalesces events after 300 ms of quiet, shares one
process across leases, exits after the final lease's grace period, and
unloads the encoder after five idle minutes. The MCP server watches for
changes itself and rebuilds on the next tool call; it also waits for a
watcher rebuild and reopens externally updated indexes. Grammar libraries
remain loaded even when the encoder is unloaded.

The legacy `--single` path rebuilds an entire index in staging and swaps
it into place, reusing content vectors and quantizers. It remains useful
for IVF/PQ experiments, but its indexing work scales with the corpus when
a file changes. Neither layout provides distributed shards, replication,
or a continuously mutable in-memory ingestion buffer.
