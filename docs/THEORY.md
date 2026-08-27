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

The hybrid experiment (§14) reuses this exact posting-list shape a second
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

The hybrid experiment (§14) adds three more mmap sidecars in the same
directory — `embeddings.bin` (dense rows), `ivf.bin` (cluster posting
lists, same block codec as `postings.bin`), `pq.bin` (product-quantized
codes). The lexical evaluators never open them.

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
quantization, attention on CubeCL — would slot in directly). That path is
now §14: Candle on Metal runs the encoder; retrieval itself stays an
inverted file plus (optional) product-quantized table lookups on the CPU.

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
- Property tests: every `block_max_score` ≥ every actual contribution in its
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
  lexical oracle suite is unchanged — hybrid is approximate by construction
  (unopened clusters, quantized codes) and is not claimed rank-safe.

---

## 14. Hybrid code search: one inverted file, two keys

**Where:** `src/tokenizer.rs` (code analyzer), `src/embeddings.rs`,
`src/embedder.rs`, `src/ivf.rs`, `src/pq.rs`, `src/hybrid.rs`,
`src/eval.rs` (feature `semantic`)

This is still **not** a vector database. There is no HNSW graph, no FAISS
index, no Qdrant collection, and no second document-id space. The lexical
engine of §§1–13 is unchanged. What the experiment adds is a second
inverted file over the **same** `doc_id`s, plus a dense sidecar those
lists select into.

`index --code --embed` writes one directory:

```
postings.bin     term_id      →  compressed (doc_id, tf) blocks     BM25 / BMW
embeddings.bin   doc_id       →  768-d FP16 row                     cosine
ivf.bin          cluster_id   →  same compressed posting blocks     IVF probe
pq.bin           doc_id       →  M-byte PQ code                     ADC
```

Row `i` of `embeddings.bin` / `pq.bin` **is** inverted-index `doc_id == i`.
Cluster lists in `ivf.bin` are built with the same `PostingList::build`
path as BM25 (delta + bit-pack, block skip tables, impact pairs). The
vector analogue of "only read the query terms' postings" is "only read
the `nprobe` nearest clusters' postings".

That inverted-file view of vector search is the line from Sivic &
Zisserman (*"Video Google: A Text Retrieval Approach to Object Matching
in Videos"*, ICCV 2003) through Jégou, Douze & Schmid (*"Product
Quantization for Nearest Neighbor Search"*, IEEE TPAMI 2011). A service
like FAISS IVF-PQ or HNSW is the same *idea* with its own ids, quantizer,
and (for HNSW) a proximity graph. Here both engines are posting lists
in one process.

### Query path: encoder first, BM25 as helper

`--mode hybrid` is **embedding-first**. BM25 is a bonus signal on the
union, not a gate.
`--mode rerank` is the older BM25-then-cosine path, kept for comparison.
`--mode semantic` is encoder-only through IVF (full mmap scan if
`ivf.bin` is missing). `--mode bm25` is §§1–13 with no neural work.

```
query
  ├─ CodeRankEmbed(query)  →  768-d, L2-normalized     (Metal, calling thread)
  └─ BM25 / Block-Max WAND / MaxScore                  (CPU, overlapping thread)
              ↓  join
     nearest nprobe centroids                          (needs the query vector)
     decode those cluster posting lists     (disjoint: clusters partition docs)
     score survivors: FP16 dot  or  PQ-ADC
     keep top `pool`  (default 200)
              ↓
     union by doc_id
              ↓
     weighted min-max  or  RRF
              ↓
            top k
```

BM25 does not need the query vector, so it overlaps with encode
(`std::thread::scope` in `src/cli.rs`). The encoder stays on the calling
thread because Metal is where the model was warmed up; WAND runs on a
helper thread. IVF probe cannot start until encode returns. Wall time is
`max(embed, bm25) + score`. `--mode rerank` overlaps the same way;
`--mode semantic` has no BM25 side.

Documents the encoder found that WAND never saw still compete
(`bm25 = 0`). Documents WAND found that sit in an unopened cluster still
compete after a point cosine against `embeddings.bin`. Fusion cannot
hide a semantic hit behind a lexical miss, or a lexical hit behind an
IVF miss. Hybrid currently requires a single (non-segmented) index.

### Code tokenizer

**Where:** `src/tokenizer.rs`, flag bit 1 of the `meta.bin` flags word
(`flags |= 2`), so v4 indexes without the bit keep the default analyzer.

Identifiers are stored **twice**: the full lowercased form
(`getuserbyorganizationid`, `std.json.utf8parser`) and the camelCase /
PascalCase / snake_case / SCREAMING_SNAKE / dotted pieces (`get`, `user`,
`organization`, `id`, `utf`, `8`, `parser`). Digit boundaries split
(`utf8Parser` → `utf` + `8` + `parser`); acronyms split before the last
capital (`XMLHttpRequest` → `xml` + `http` + `request`). Exact-name
queries still get a rare, high-idf term; natural-language queries can
match the pieces. The original identifier is never stopword-dropped;
split pieces are.

Queries must use the same analyzer the index was built with
(`Tokenizer::code(true)`).

### Offline encoding (Candle / Metal)

**Where:** `src/embedder.rs`, sidecar layout in `src/embeddings.rs`

Behind `--features semantic`, Candle loads `nomic-ai/CodeRankEmbed`
(137M NomicBert: RoPE + SwiGLU, no official ONNX export). Documents are
encoded as raw code, in inverted-index `doc_id` order, batch 4, truncated
at 512 tokens. Queries are prefixed with
`Represent this query for searching relevant code: `; documents are not.
Pooling is CLS, then L2-normalize — matching the model card and
`1_Pooling/config.json`. L2-normalized cosine is a plain dot product.

On macOS the device is Metal F16 (`HPS_EMBED_DEVICE=cpu|metal` overrides);
elsewhere CPU F32 with Accelerate on Apple. Queries truncate RoPE at 64
tokens; document indexing keeps 512. Candle's Metal NomicBert is a few
hundred unfused kernel launches per forward — that, not FLOPs, is why a
20-token query still costs ~10 ms. CPU+Accelerate was measured *slower*
(~22 ms) on the same graph. A fused runtime (MLX, CoreML/ANE) is the
actual encoder speedup; this crate does not ship one. The Metal backend
implements `where_cond` for `(U8, F16)` but not `(U32, F16)`, so the
attention mask is stored as U8. `Embedder::load` runs a dummy
`embed_query("warmup")` so the first real query does not pay shader
compile. Query encode is the dominant term in end-to-end hybrid latency.

`embeddings.bin` is a 32-byte header (`HPSEMB01`) plus row-major FP16
vectors, memory-mapped. 768-d costs 1.5 KB/doc. Nothing in the lexical
files changes.

fastembed-rs was not used because it ships mean-pooled
`nomic-embed-text`, a different model. ORT was not used because there is
no official ONNX export of this custom `nomic_bert`.

### IVF: cluster ids as terms

**Where:** `src/ivf.rs` (`HPSIVF01`)

Spherical k-means over the stored embeddings: assignment is cosine to
the current centroids, centroids are L2-normalized after each mean
update, init is a deterministic LCG (no `rand` crate), 25 iterations.
Default `K ≈ 2√N`, clamped to `[2, min(N/2, 256)]`. Each document is
assigned to exactly one cluster, so the K posting lists **partition**
the `doc_id` space.

Each cluster is encoded as an ordinary posting list with `tf = 1` for
every member — a "term" whose df is the cluster size — including the
same impact pairs (`max_tf`, `min_len`) and block skip tables BM25 uses.
A query:

1. dots the (already L2-normalized) query against the K centroids
   (cheap: K is tens to hundreds, not N);
2. opens the `nprobe` nearest lists (`nprobe = 0` → `(K/4).clamp(1, 8)`);
3. concatenates their decoded `doc_id`s — no merge sort, the lists are
   disjoint.

This is approximate: a document whose cluster is not among the nprobe
nearest is **invisible to the encoder side**. On the 70-document labeled
code-eval set, semantic-only Recall@10 dropped 1.0 → 0.906 versus a full
scan; the hybrid union recovered the misses because BM25 still returned
them.

**WAND on cluster lists.** The on-disk layout is the same as BM25 so a
future evaluator *could* skip inside a list. Today we do not, and for a
good reason: a cluster query is a single "term" (or a tiny OR of nprobe
terms) with uniform `tf = 1`. WAND's pivot math prunes documents that
cannot beat θ given the *sum of several term upper bounds*; with one
list there is nothing to pivot against, and the interesting skip is
already "do not open the other K − nprobe lists". Scoring then walks
every member of the opened lists (FP16 dot or PQ-ADC). Treating cluster
ids as WAND query terms would reconstruct the disjoint concat
`docs_in_clusters` already does.

The sublinear claim is therefore the same one §1 makes for BM25: we do
not scan documents outside the probed lists. It is **not** rank-safe —
unlike Block-Max WAND on BM25, which never drops a document that could
enter the top-k.

On N = 70 the decode+score overhead lost to a full mmap scan of
`embeddings.bin`. On 2 500 code chunks (100 clusters) vector scoring
went 2.53 ms brute-force → 0.76 ms auto-nprobe → 0.31 ms `nprobe=1`.

### Product quantization (ADC)

**Where:** `src/pq.rs` (`HPSPQ001`)

Jégou, Douze & Schmid, TPAMI 2011. Split each 768-d vector into `M = 16`
subspaces of 48-d. Each subspace is 256-means (1 byte). A document
becomes 16 bytes instead of 1 536 (FP16) or 3 072 (FP32). Codebooks live
in the sidecar as FP16; codes are `num_docs × M` raw bytes, `doc_id`-aligned
with `embeddings.bin`.

Query scoring is **asymmetric distance computation (ADC)**: the query
stays FP32 (no query quantization error), documents are codes. Per
query, `prepare` fills `M × 256` tables of `q_sub[m] · codebook[m][c]`;
scoring a document is 16 table lookups and a sum. That sum approximates
the same L2-normalized dot `embeddings.cosine` would have computed.

PQ is a *scoring* approximation on whatever IVF (or the full scan)
already retrieved, not a second index. `--no-pq` falls back to FP16
dots. This is **not** FAISS IVF-PQ: residuals versus the coarse centroid
are not quantized, so ADC error is the full reconstruction error, not
the residual.

On the same 2 500-chunk index, ADC was about 15% *slower* than FP16
dots (table build versus saved multiplies). End-to-end hybrid stayed
~9–11 ms because the encoder (~9 ms after Metal warmup) dominates.
IVF and PQ only start to matter once N is large enough that scanning
every 768-d row costs more than encoding the query.

#### Training versus assigning

Both quantizers separate *training* (k-means over the corpus) from
*assigning* (one pass over the resulting centroids per document). This is
the same split FAISS draws between `train` and `add`, and it is what makes
`hips`'s watch-driven rebuilds cheap: centroids and codebooks are trained
once and then reused while the tree is edited, so a rebuild only assigns the
chunks whose text changed. `ivf.bin` and `pq.bin` therefore double as the
persisted quantizer — `read_centroids` / `read_codebooks` load them back,
and `assign_one` / `encode_one` reproduce exactly what training assigned.

Two details make that exactness hold:

- Both k-means loops must end with an **assignment pass against the
  centroids they return**. A loop that assigns and then updates returns an
  assignment one step stale, so the stored codes are not the nearest
  centroids of the stored codebooks and every ADC score is computed against
  a centroid the encoder would not have picked.
- `pq.bin` records how many of the 256 codes per subspace were actually
  trained. With fewer than 256 documents, training fills only part of each
  codebook, and an encoder that searched all 256 slots would select
  untrained (all-zero) centroids that training never assigned.

### Fusion

**Where:** `src/hybrid.rs`

Two combiners, both over the **union** of the encoder pool and the BM25
helper list (ranks are 0 if a document is absent from one side):

- **Weighted** — min-max normalize BM25 over the union (documents with
  `bm25 = 0` stay 0), then
  `α · n_bm25 + (1 − α) · cosine`. Cosine is already in `[-1, 1]`.
- **RRF** — Cormack, Clarke & Buettcher, *"Reciprocal Rank Fusion
  Outperforms Condorcet and Individual Rank Learning Methods"*, SIGIR
  2009: `Σ 1 / (k + rank)` over the two lists. Default `k = 60`.

RRF is the CLI default (`--fusion rrf`): it does not require comparable
raw scores, which BM25 and cosine are not.

### Evaluation

**Where:** `src/eval.rs`, CLI `eval-code`

Labeled metrics against graded qrels (`rel ∈ {3, 2, 1}`):

- **MRR** — `1 / rank` of the first relevant hit (`rel > 0`).
- **Recall@5 / Recall@10** — fraction of all relevant documents appearing
  in the top k (ungraded: any `rel > 0` counts).
- **nDCG@10** — DCG with gain `2^rel − 1` and discount `log2(rank + 1)`,
  divided by the ideal DCG of the qrel sorted by gain.

IVF and PQ are approximate, so semantic-only recall can drop versus a
brute-force scan of `embeddings.bin`; hybrid's union is the intended
backstop. The lexical BMW/MaxScore path remains exact and is still
pinned against the naive BM25 oracle in the test suite.

### What this is not

- Not HNSW / DiskANN / a graph index. Neighbors are not walked.
- Not a FAISS / Qdrant / LanceDB deployment. No separate vector service,
  no residual IVF-PQ, no GPU GEMM over the inverted lists.
- Not rank-safe on the encoder side. Unopened clusters and quantized
  codes can change the top-k versus exhaustive cosine.
- Not wired through segmented indexes or the HTTP API yet.
- Not on by default: without `--features semantic` the binary is still
  the lexical engine of §§1–13.
