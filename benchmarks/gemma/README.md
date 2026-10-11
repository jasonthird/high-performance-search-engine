# Static query embeddings experiment

This copies gemma4agentcomp's evaluation queries and reference-file labels:
129 raw issues, 129 deterministic identifier queries, and 37 first search
queries from runs F/G. `queries.source.json` records their provenance.
Reference patches are evaluation labels only; they are not included here or
passed to the search process. The word vocabulary was mined outside the four
benchmark repositories.

The hypothesis is that an average of standalone 768-dimensional CodeRankEmbed
word embeddings can search the existing full-model document vectors. This is
an approximation, not an exact query cache. Equal dimensions do not establish
equal retrieval quality.

The table is regenerated with hips' CLS pooling, L2 normalization and symmetric
int8 quantization, with no centering or PCA. Its loader normalizes each decoded
word row. Query words use hips' identifier splitting, sublinear term frequency
and final L2 normalization. The complete vocabulary has 34,937 terms: the
tokenizer vocabulary combined with the 20,000 mined identifier parts from the
existing gemma asset, after deduplication. `--query-words` optionally adds the
model's query instruction when generating individual words.

Both query encoders score the **same full-model FP16 document embeddings**.
The comparison runs BM25, full/static semantic search and full/static hybrid
search through hips' existing ranking pipeline. It uses exact vector scoring,
200 semantic candidates and the existing weighted fusion setting (alpha 0.15).
The normal CLI/full encoder retains its 64-token query limit; static queries
use the lookup vocabulary. Both receive the same input, bounded to 8,000
characters as in the original benchmark.

Each task uses its own original repository snapshot, code-only 80-line windows
with a 60-line stride, and file-level ranking with duplicate paths collapsed.
R@k means the fraction of queries whose first reference file appears within k
positions, matching gemma's evaluation; it is not recall over every patched
file. Each variant returns 20 chunks before file deduplication. Original gemma
scores are not directly comparable: its encoder, BM25 rules and reranker differ.

The benchmark worker stays resident and caches document embeddings across
snapshots. Full and approximate query vectors are each timed once per query
and reused for the semantic/hybrid comparison. Reported encoder latency
excludes full-query cache hits. Cold JSON/model load times are in `raw.jsonl`.
Unknown-only static queries are explicit errors and count as misses, without
silently invoking the full model in the measured static variant.

## Reproduce

Run from this repository. The generated word table and benchmark outputs are
ignored by Git; the copied query definitions and small result summary are kept.
The snapshot archives (21 GB) remain in gemma4agentcomp.

```sh
cargo build --release --offline --features semantic --example static_query

python3 scripts/import_gemma_bench.py \
  --source "$HOME/gitProjects/gemma4agentcomp"

HIPS_MODEL_OFFLINE=1 HIPS_ENCODER=ggml HPS_EMBED_DEVICE=cpu \
  target/release/examples/static_query build \
  --vocabulary "$HOME/gitProjects/gemma4agentcomp/submissions/EXP-031-graph-symbols/skills/repo-search/assets/static-encoder.json" \
  --out data/static-query/words-768.json

HIPS_MODEL_OFFLINE=1 HIPS_ENCODER=ggml HPS_EMBED_DEVICE=metal \
  python3 scripts/bench_static_query.py \
  --snapshots "$HOME/gitProjects/gemma4agentcomp/data/competition/snapshots" \
  --words data/static-query/words-768.json \
  --out target/static-query-bench-gpu

python3 scripts/report_static_query.py --results target/static-query-bench-gpu
```

The recorded benchmark uses Metal on an Apple M3. Use `HPS_EMBED_DEVICE=cpu`
for a CPU-only run, or the appropriate ggml GPU backend on another platform.

The benchmark supports `--limit 4` for a balanced smoke run, and resumes
completed tasks with identical binary, word-table and query hashes. Use a new
`--out` directory if these inputs change.

The reporting command checks that every requested task completed and that the
summary matches the raw records, then writes `results.md` and `results.json`.
The JSON also breaks down results by repository and records artifact hashes.

## Try an existing index

```sh
target/release/examples/static_query search \
  --index /path/to/existing/hips/index \
  --words data/static-query/words-768.json \
  --query 'release watcher lease when session ends'
```

This performs static query encoding without loading CodeRankEmbed. It reports
the approximation backend and vocabulary coverage. `--fallback-full` enables
the full model for queries with no known terms. Existing hips CLI/MCP behavior
continues to use the full encoder; adoption depends on this experiment's results.

## Validation

The completed experiment covers all 129 snapshots and 295 query records. The
report retains pure-static misses, including the numeric query `3104`, and
compares paired outcomes as well as aggregate metrics.

All 125 default library test cases passed, with the seven socket/grammar tests
rerun using localhost access and `HIPS_GRAMMAR_DIR=grammars/target/release`.
The three semantic-feature static encoder tests and five existing hybrid
ranking tests passed. Benchmark helper checks covered archive path safety,
chunk boundaries, file deduplication and metric calculations.

An existing segmented index was searched with a nonexistent `HIPS_MODEL_PATH`
to verify that static search needs no model. The `3104` query with
`--fallback-full` successfully used the full encoder. Release builds of the
example and main binary passed.

## Learn missing words on demand

This section records the earlier word-level approximation experiment. Normal
search now uses an independent [persistent full-query cache](query-cache-results.md):
match the entire query, use its saved full-model vector on a hit, or encode the
complete query and save it on a miss. Neither indexing nor reference labels
populate that cache. Knowing all words in a new issue does not make it a hit.

Use `--grow-cache` to look up every token and encode just the distinct missing
words. The encoder loads lazily, uses the base table's prefix setting, and saves
768-dimensional int8 rows to a separate JSON. A repeated query, including one
after process restart, uses those saved rows without loading the model.

```sh
target/release/examples/static_query search \
  --index /path/to/existing/hips/index \
  --words data/static-query/words-768.json \
  --grow-cache data/static-query/learned-words.json \
  --query 'newIdentifier'
```

The experimental sidecar assumes one writer. Updates validate every model row
and replace the JSON atomically; failed model calls leave the table unchanged.
This still combines word vectors into an approximate query embedding.

To compare static, growing and full-query encoders on all snapshots:

```sh
HIPS_MODEL_OFFLINE=1 HIPS_ENCODER=ggml HPS_EMBED_DEVICE=metal \
  python3 scripts/bench_static_query.py \
  --snapshots "$HOME/gitProjects/gemma4agentcomp/data/competition/snapshots" \
  --words data/static-query/words-768.json \
  --grow --out target/growing-query-bench-buffered

python3 scripts/report_static_query.py \
  --results target/growing-query-bench-buffered --out benchmarks/gemma/growing-results
```

The word cache starts empty and grows in benchmark query order, without labels.
Every query is immediately repeated; the worker asserts zero model calls and
an identical vector. First-pass timings include missing-word encoding and JSON
persistence; repeat timings measure warm lookup. Documents use the same full
CodeRankEmbed rows as the original run. A preexisting document cache may be
copied into the new output directory to skip document inference.

Five semantic-feature static encoder tests passed, including miss batching,
persistence, model-free repeat/reload, invalid/partial model batches, prefix mismatch, failed calls and failed disk persistence.
The all-unknown query `3104` was also tested on an existing segmented index:
its first call saved one word, and a new process with a nonexistent model path
returned identical hits with zero misses.

The [completed growing-lookup results](growing-results.md) cover 129 snapshots
and 295 queries. The empty sidecar learned 861 distinct words (1.66 MB); all
295 immediate repeats made zero model calls and reproduced identical vectors.
Median warm lookup was 0.0089 ms; queries with word misses took 20.6 ms including
inference and atomic persistence. These are resident-process measurements.

Hybrid top-10 file success was 83.8% for short agent queries (full: 83.8%),
65.1% for identifier queries (full: 62.0%), and 59.7% for long issues
(full: 79.1%; original static: 63.6%). Learning missing words improves some
rankings but does not restore the model's contextual encoding of a full query.
The experiment stays opt-in.
