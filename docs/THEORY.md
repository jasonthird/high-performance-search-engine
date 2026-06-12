# Algorithms & Theory

Every algorithm implemented in this engine, why it works, and where it comes
from. File references point at the implementation.

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

Construction here is a parallel fold-and-merge (rayon): tokenize documents in
parallel, build partial `term -> postings` maps per chunk of documents, merge
the maps, sort each list by doc_id, then attach block metadata. This is a
simplified in-memory version of the classic blocked sort-based / merge-based
indexing used when corpora exceed RAM.

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
stores per block exactly the two facts the search path reads:

```
max_doc_id        (skip whole blocks that end before a target)
block_max_score   (the safe upper bound that makes skipping exact)
```

(Block posting ranges are derived from the fixed block size; nothing else
is stored.) `block_max_score` = max actual BM25 contribution over the
block's postings,
computed at index time **with the same formula used at query time** (so the
bound is exact, not estimated). After WAND picks a pivot, BMW sums the block
maxima of the blocks containing the pivot doc. If even this refined bound is
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
Random access *within* the encoded stream isn't needed — blocks are decoded
whole — but fixed width keeps decode branch-free and cheap, which matters
because BMW decodes blocks on the hot path.

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

- `meta.bin` — document metadata, term dictionary, per-term statistics and
  block skip-tables. Small and hot; deserialized into RAM (analogous to the
  term dictionary/FST that even mmap-based engines keep readily accessible).
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

## 9. GPU offload experiment (CubeCL / wgpu / Metal)

**Where:** `src/reorder/gpu.rs` (feature `gpu`, `--reorder bp-gpu`)

An experiment in accelerating index construction with the Mac's GPU, first
through [Burn](https://burn.dev)'s tensor API, then rewritten as a single
hand-fused kernel in [CubeCL](https://github.com/tracel-ai/cubecl) (the GPU
compute DSL underneath Burn). Honest results, on the 108k doc / 19.2M edge
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
quantization, attention on CubeCL — would slot in directly).

## 10. Concurrency model

**Where:** `src/indexer.rs` (build), `src/api.rs` (serve)

- **Indexing**: data-parallel map/reduce over documents (rayon work-stealing),
  then per-term parallel finalization. Deterministic output: term ids are
  assigned in sorted term order and postings are sorted by doc_id regardless
  of worker scheduling.
- **Serving**: the index is immutable after build (`Arc<DiskIndex>` shared
  across handlers, no locks needed). Searches run on tokio's blocking pool so
  CPU-bound scoring doesn't starve the async accept loop. Immutability is
  what makes the whole read path trivially thread-safe — the same reason
  Lucene segments are write-once.

## 11. Segmented indexes & impact-based bounds

**Where:** `src/segments.rs`, impacts in `src/postings.rs`

The classic write path of Lucene (and every LSM system): immutability per
segment, mutability as a collection of segments. New documents form fresh
segments; deletes are tombstone bitmaps; updates are delete + re-add;
background merges compact — decode postings, drop tombstoned docs, remap
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

## 12. Testing methodology: oracle testing

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
- Property tests: every `block_max_score` ≥ every actual contribution in its
  block (the invariant that makes skipping safe), encode/decode round-trips,
  BP outputs a valid permutation and reduces measured log-gap cost.
