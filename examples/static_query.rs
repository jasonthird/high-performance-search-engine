//! Build and benchmark experimental word queries against full CodeRankEmbed rows.
#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("requires --features semantic");
    std::process::exit(1);
}

#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    experiment::main()
}

#[cfg(feature = "semantic")]
mod experiment {
    use clap::{Parser, Subcommand};
    use high_performance_search_engine::{
        embcache::{key_for, EmbedCache},
        embedder::{EmbedUse, Embedder, MODEL_ID, QUERY_PREFIX},
        embeddings::{self, CODERANK_DIM},
        indexer::{self, InputDoc},
        postings::DEFAULT_BLOCK_SIZE,
        query::{self, FusionArg, PqMode, QueryEncoder, RankMode, SearchOpts},
        reorder::ReorderStrategy,
        searcher::AnyIndex,
        static_query::{GrowingQueryEncoder, StaticQueryEncoder, WordTable, SCHEMA},
        storage,
    };
    use std::{
        collections::BTreeMap,
        io::{BufRead, Write},
        path::{Path, PathBuf},
        time::Instant,
    };
    #[derive(Parser)]
    struct Args {
        #[command(subcommand)]
        command: Command,
    }
    #[derive(Subcommand)]
    enum Command {
        /// Re-encode the vocabulary of an existing JSON table, without its PCA.
        Build {
            #[arg(long)]
            vocabulary: PathBuf,
            #[arg(long)]
            out: PathBuf,
            /// Encode each word with CodeRankEmbed's query instruction prefix.
            #[arg(long)]
            query_words: bool,
            /// Limit vocabulary for a smoke test; zero means the complete vocabulary.
            #[arg(long, default_value_t = 0)]
            limit: usize,
        },
        /// JSONL requests on stdin: {corpus: JSONL path, queries: [...]}.
        /// Keeps the full encoder resident and shares a document cache across snapshots.
        Bench {
            #[arg(long)]
            words: PathBuf,
            #[arg(long)]
            cache: PathBuf,
            #[arg(long)]
            grow_cache: Option<PathBuf>,
        },
        /// Search an existing CodeRankEmbed index using the word table, without loading the model.
        Search {
            #[arg(long)]
            words: PathBuf,
            #[arg(long)]
            index: PathBuf,
            #[arg(long)]
            query: String,
            #[arg(long, value_enum, default_value = "hybrid")]
            mode: RankMode,
            #[arg(long, default_value_t = 10)]
            top_k: usize,
            /// Use full CodeRankEmbed if the query has no known words.
            #[arg(long)]
            fallback_full: bool,
            /// Persist individual word misses; load the model only when needed.
            #[arg(long, conflicts_with = "fallback_full")]
            grow_cache: Option<PathBuf>,
        },
    }
    pub fn main() -> anyhow::Result<()> {
        match Args::parse().command {
            Command::Build {
                vocabulary,
                out,
                query_words,
                limit,
            } => build(&vocabulary, &out, query_words, limit),
            Command::Bench {
                words,
                cache,
                grow_cache,
            } => bench(&words, &cache, grow_cache.as_deref()),
            Command::Search {
                words,
                index,
                query,
                mode,
                top_k,
                fallback_full,
                grow_cache,
            } => {
                anyhow::ensure!(
                    !matches!(mode, RankMode::Bm25),
                    "use hips search --mode bm25 for lexical queries"
                );
                anyhow::ensure!(top_k > 0, "top-k must be positive");
                let started = Instant::now();
                let disk = AnyIndex::open(&index)?;
                let mut learned = None;
                let approximate = if let Some(path) = grow_cache {
                    let mut growing = GrowingQueryEncoder::load(&words, &path)?;
                    let lookup = growing.lookup(&query, |words, prefix| {
                        encode_words(&Embedder::load_for(EmbedUse::Query)?, words, prefix)
                    })?;
                    learned = Some(
                        serde_json::json!({"missing_terms":lookup.missing_terms,"learned_words":growing.learned_words(),"cache":path}),
                    );
                    Ok(lookup.approximation)
                } else {
                    StaticQueryEncoder::load(&words)?.approximate(&query)
                };
                let opts = SearchOpts {
                    mode,
                    semantic_candidates: 200,
                    fusion: FusionArg::Weighted,
                    alpha: 0.15,
                    rrf_k: 60.0,
                    nprobe: 0,
                    pq: PqMode::Off,
                    ann_ef: 128,
                    exact_vectors: false,
                };
                let (run, backend, coverage) = match approximate {
                    Ok(a) => (
                        query::run_ranked_with(&disk, &Fixed(&a.vector), &query, top_k, &opts)?,
                        "static-word-approximation",
                        Some(a.known_terms as f64 / a.total_terms as f64),
                    ),
                    Err(error) if fallback_full => {
                        eprintln!("static query unavailable ({error:#}); using full CodeRankEmbed");
                        (
                            query::run_ranked_with(
                                &disk,
                                &Embedder::load()?,
                                &query,
                                top_k,
                                &opts,
                            )?,
                            "full-coderankembed",
                            None,
                        )
                    }
                    Err(error) => return Err(error),
                };
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({"backend":backend,
                    "lookup":learned,"coverage":coverage,"total_ms":started.elapsed().as_secs_f64()*1000.0,
                    "hits":run.results.iter().map(|h| serde_json::json!({"id":h.id,"score":h.score,"bm25":h.bm25,"semantic":h.semantic})).collect::<Vec<_>>()}))?
                );
                Ok(())
            }
        }
    }
    fn build(vocabulary: &Path, out: &Path, query_words: bool, limit: usize) -> anyhow::Result<()> {
        let input: serde_json::Value =
            serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(vocabulary)?))?;
        let mut words: Vec<String> = input["vectors"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("vocabulary needs a vectors object"))?
            .keys()
            .cloned()
            .collect();
        words.sort();
        if limit > 0 {
            words.truncate(limit);
        }
        anyhow::ensure!(!words.is_empty(), "empty vocabulary");
        let started = Instant::now();
        let encoder = Embedder::load_for(EmbedUse::Query)?;
        let mut vectors = BTreeMap::new();
        for batch in words.chunks(256) {
            let texts: Vec<String> = batch
                .iter()
                .map(|w| {
                    if query_words {
                        format!("{QUERY_PREFIX}{w}")
                    } else {
                        w.clone()
                    }
                })
                .collect();
            let rows = encoder.embed_docs(&texts.iter().map(String::as_str).collect::<Vec<_>>())?;
            for (word, row) in batch.iter().zip(rows) {
                vectors.insert(
                    word.clone(),
                    row.iter()
                        .map(|x| (x * 127.0).round().clamp(-127.0, 127.0) as i8)
                        .collect(),
                );
            }
            eprintln!(
                "words {}/{} ({:.1}s)",
                vectors.len(),
                words.len(),
                started.elapsed().as_secs_f64()
            );
        }
        let table = WordTable {
            schema: SCHEMA.into(),
            dimensions: CODERANK_DIM,
            source: serde_json::json!({"teacher":MODEL_ID,"pooling":"cls","transform":"none", "quantization":"symmetric-int8", "query_prefix":query_words,"vocabulary_source":vocabulary, "vocabulary_provenance":input["source"],"encoder":encoder.diagnostics()}),
            vectors,
        };
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        serde_json::to_writer(std::io::BufWriter::new(std::fs::File::create(out)?), &table)?;
        // Validate the generated artifact with the inference loader before reporting success.
        let loaded = StaticQueryEncoder::load(out)?;
        println!(
            "{}",
            serde_json::json!({"words":loaded.vocabulary_size(),"bytes":std::fs::metadata(out)?.len(),"seconds":started.elapsed().as_secs_f64(),"out":out})
        );
        Ok(())
    }
    struct Fixed<'a>(&'a [f32]);
    impl QueryEncoder for Fixed<'_> {
        fn embed_query(&self, _: &str) -> anyhow::Result<Vec<f32>> {
            Ok(self.0.to_vec())
        }
        fn cheap_query(&self, _: &str) -> bool {
            true
        }
    }
    fn encode_words(
        encoder: &Embedder,
        words: &[String],
        prefix: bool,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut rows = Vec::new();
        for batch in words.chunks(256) {
            let texts: Vec<_> = batch
                .iter()
                .map(|w| {
                    if prefix {
                        format!("{QUERY_PREFIX}{w}")
                    } else {
                        w.clone()
                    }
                })
                .collect();
            rows.extend(encoder.embed_docs(&texts.iter().map(String::as_str).collect::<Vec<_>>())?);
        }
        Ok(rows)
    }
    fn bench(words: &Path, cache_dir: &Path, grow_cache: Option<&Path>) -> anyhow::Result<()> {
        let t = Instant::now();
        let static_encoder = StaticQueryEncoder::load(words)?;
        let static_load_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut growing = grow_cache
            .map(|path| GrowingQueryEncoder::load(words, path))
            .transpose()?;
        let t = Instant::now();
        let encoder = Embedder::load_for(EmbedUse::Index)?;
        // Match the normal CLI query encoder's 64-token cap, rather than the
        // indexing encoder's 512-token cap on long issue statements.
        let query_encoder = Embedder::load_for(EmbedUse::Query)?;
        let full_load_ms = t.elapsed().as_secs_f64() * 1000.0;
        std::fs::create_dir_all(cache_dir)?;
        let mut cache = EmbedCache::load(cache_dir, CODERANK_DIM);
        let stdout = std::io::stdout();
        let mut output = stdout.lock();
        for line in std::io::stdin().lock().lines() {
            let request: serde_json::Value = serde_json::from_str(&line?)?;
            let corpus = request["corpus"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing corpus"))?;
            let docs: Vec<InputDoc> = std::io::BufReader::new(std::fs::File::open(corpus)?)
                .lines()
                .map(|l| Ok(serde_json::from_str(&l?)?))
                .collect::<anyhow::Result<_>>()?;
            anyhow::ensure!(!docs.is_empty(), "empty corpus");
            let t = Instant::now();
            let texts: Vec<String> = docs
                .iter()
                .map(|d| format!("{}\n{}", d.title, d.body))
                .collect();
            let keys: Vec<_> = texts.iter().map(|text| key_for(text)).collect();
            let misses = cache.unique_misses(&keys);
            for group in misses.chunks(256) {
                let refs: Vec<_> = group.iter().map(|&i| texts[i].as_str()).collect();
                for (&i, vector) in group.iter().zip(encoder.embed_docs(&refs)?) {
                    cache.insert(keys[i], &vector);
                }
            }
            let rows: Vec<_> = keys
                .iter()
                .map(|&k| cache.get(k).expect("encoded row"))
                .collect();
            let index = indexer::build_index_weighted_ex(
                &docs,
                true,
                2,
                DEFAULT_BLOCK_SIZE,
                ReorderStrategy::None,
                true,
            );
            let dir = tempfile::tempdir()?;
            storage::save_index(&index, dir.path())?;
            embeddings::write_f16(dir.path(), CODERANK_DIM as u32, &rows)?;
            let disk = AnyIndex::open(dir.path())?;
            cache.save(cache_dir)?;
            let index_ms = t.elapsed().as_secs_f64() * 1000.0;
            let mut results = Vec::new();
            for item in request["queries"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("missing queries"))?
            {
                let query = item["query"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("missing query text"))?;
                let cached = query_encoder.query_cached(query);
                let t = Instant::now();
                let full = query_encoder.embed_query(query)?;
                let full_ms = t.elapsed().as_secs_f64() * 1000.0;
                let t = Instant::now();
                let approx = static_encoder.approximate(query);
                let static_ms = t.elapsed().as_secs_f64() * 1000.0;
                let t = Instant::now();
                let dynamic = growing
                    .as_mut()
                    .map(|g| g.lookup(query, |w, prefix| encode_words(&query_encoder, w, prefix)))
                    .transpose()?;
                let dynamic_ms = t.elapsed().as_secs_f64() * 1000.0;
                let dynamic_stats = if let (Some(g), Some(d)) = (growing.as_mut(), dynamic.as_ref())
                {
                    let t = Instant::now();
                    let hot = g.lookup(query, |_, _| {
                        anyhow::bail!("repeat unexpectedly called model")
                    })?;
                    let hot_ms = t.elapsed().as_secs_f64() * 1000.0;
                    anyhow::ensure!(
                        hot.missing_terms == 0
                            && hot.approximation.vector == d.approximation.vector,
                        "unstable repeated lookup"
                    );
                    Some(
                        serde_json::json!({"missing_terms":d.missing_terms,"learned_words":g.learned_words(),"encode_ms":dynamic_ms,"repeat_encode_ms":hot_ms,"query_cosine":full.iter().zip(&d.approximation.vector).map(|(x,y)| x*y).sum::<f32>()}),
                    )
                } else {
                    None
                };
                let mut variants = BTreeMap::new();
                let bm25 = disk.search(query, 20);
                variants.insert("bm25", serde_json::json!({"hits":bm25.results.iter().map(|r| &r.id).collect::<Vec<_>>(),"total_ms":bm25.took_ms}));
                for (name, vector, encode_ms) in [
                    ("full", Some(full.as_slice()), full_ms),
                    (
                        "static",
                        approx.as_ref().ok().map(|a| a.vector.as_slice()),
                        static_ms,
                    ),
                    (
                        "dynamic",
                        dynamic.as_ref().map(|d| d.approximation.vector.as_slice()),
                        dynamic_ms,
                    ),
                ] {
                    let Some(vector) = vector else { continue };
                    for mode in [RankMode::Semantic, RankMode::Hybrid] {
                        let opts = SearchOpts {
                            mode,
                            semantic_candidates: 200,
                            fusion: FusionArg::Weighted,
                            alpha: 0.15,
                            rrf_k: 60.0,
                            nprobe: 0,
                            pq: PqMode::Off,
                            ann_ef: 128,
                            exact_vectors: true,
                        };
                        let run = query::run_ranked_with(&disk, &Fixed(vector), query, 20, &opts)?;
                        let key = match (name, mode) {
                            ("full", RankMode::Semantic) => "semantic_full",
                            ("full", _) => "hybrid_full",
                            ("static", RankMode::Semantic) => "semantic_static",
                            ("static", _) => "hybrid_static",
                            ("dynamic", RankMode::Semantic) => "semantic_dynamic",
                            _ => "hybrid_dynamic",
                        };
                        variants.insert(key, serde_json::json!({"hits":run.results.iter().map(|r| &r.id).collect::<Vec<_>>(),"total_ms":run.total_ms+encode_ms,"encode_ms":encode_ms}));
                    }
                }
                // The legacy BM25-then-cosine mode, and 50-chunk first-stage
                // pools that external rerankers (scripts/bench_rerank.py) reorder.
                let opts = |mode| SearchOpts {
                    mode,
                    semantic_candidates: 200,
                    fusion: FusionArg::Weighted,
                    alpha: 0.15,
                    rrf_k: 60.0,
                    nprobe: 0,
                    pq: PqMode::Off,
                    ann_ef: 128,
                    exact_vectors: true,
                };
                let run = query::run_ranked_with(&disk, &Fixed(&full), query, 20, &opts(RankMode::Rerank))?;
                variants.insert("rerank_full", serde_json::json!({"hits":run.results.iter().map(|r| &r.id).collect::<Vec<_>>(),"total_ms":run.total_ms+full_ms,"encode_ms":full_ms}));
                let run = query::run_ranked_with(&disk, &Fixed(&full), query, 50, &opts(RankMode::Hybrid))?;
                let pools = serde_json::json!({
                    "hybrid_full": run.results.iter().map(|r| &r.id).collect::<Vec<_>>(),
                    "bm25": disk.search(query, 50).results.iter().map(|r| &r.id).collect::<Vec<_>>(),
                });
                let (coverage, cosine, error) = match &approx {
                    Ok(a) => (
                        Some(a.known_terms as f64 / a.total_terms as f64),
                        Some(full.iter().zip(&a.vector).map(|(x, y)| x * y).sum::<f32>()),
                        None,
                    ),
                    Err(e) => (None, None, Some(e.to_string())),
                };
                results.push(serde_json::json!({"id":item["id"],"mode":item["mode"],"query":query,"full_cached":cached,"full_encode_ms":full_ms,"static_encode_ms":static_ms,"coverage":coverage,"query_cosine":cosine,"static_error":error,"dynamic":dynamic_stats,"variants":variants,"pools":pools}));
            }
            writeln!(
                output,
                "{}",
                serde_json::json!({"task":request["task"],"chunks":docs.len(),"encoded":misses.len(),"index_ms":index_ms,"static_load_ms":static_load_ms,"full_load_ms":full_load_ms,"encoder":encoder.diagnostics(),"results":results})
            )?;
            output.flush()?;
        }
        Ok(())
    }
}
