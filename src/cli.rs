//! Command-line interface: index, search, serve, bench.

use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};

use crate::indexer::SearchableIndex;
use crate::postings::DEFAULT_BLOCK_SIZE;
use crate::reorder::ReorderStrategy;
use crate::{api, bench, indexer, searcher, storage};

/// CLI-facing document reordering choice.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ReorderArg {
    /// Keep input order.
    None,
    /// Sort documents by their external id (clusters paths/URLs).
    Path,
    /// Recursive graph bisection (best compression, slower indexing).
    Bp,
    /// Recursive graph bisection on the GPU (Metal/Vulkan via wgpu);
    /// requires a binary built with `--features gpu`.
    BpGpu,
}

impl From<ReorderArg> for ReorderStrategy {
    fn from(arg: ReorderArg) -> Self {
        match arg {
            ReorderArg::None => ReorderStrategy::None,
            ReorderArg::Path => ReorderStrategy::Path,
            ReorderArg::Bp => ReorderStrategy::Bp,
            ReorderArg::BpGpu => ReorderStrategy::BpGpu,
        }
    }
}

/// Whether ADC scoring is used. `Auto` defers to [`crate::pq::worth_using`]:
/// below the break-even candidate count PQ costs recall and saves nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PqMode {
    Auto,
    Off,
    /// Use ADC regardless of candidate count (benchmarking).
    Force,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum FusionArg {
    Weighted,
    Rrf,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum RankMode {
    Bm25,
    /// Encoder retrieves over all vectors; BM25 is a helper on the same doc_ids.
    Hybrid,
    /// Old retrieve-then-rerank: cosine only on BM25 candidates.
    Rerank,
    Semantic,
}

#[derive(Parser)]
#[command(
    name = "high-performance-search-engine",
    about = "MVP search engine: BM25 over an inverted index with exact Block-Max WAND"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build an index from a JSONL file of documents.
    Index {
        /// Input JSONL file ({"id": ..., "title": ..., "body": ...} per line).
        #[arg(long)]
        input: PathBuf,
        /// Output index directory.
        #[arg(long)]
        out: PathBuf,
        /// Document reordering strategy (doc_id assignment for compression).
        #[arg(long, value_enum, default_value_t = ReorderArg::None)]
        reorder: ReorderArg,
        /// BM25F-lite title boost: each title occurrence of a term counts
        /// as this many occurrences (1 = no boost).
        #[arg(long, default_value_t = 2)]
        title_weight: u32,
        /// External (sharded, spill-to-disk) build for corpora larger than
        /// memory. Streams the input ("-" reads stdin), writes the index
        /// directly to disk. Incompatible with --reorder.
        #[arg(long)]
        external: bool,
        /// Code-oriented tokenizer: keep full identifiers and also split
        /// camelCase / snake_case / dotted paths into components.
        #[arg(long)]
        code: bool,
        /// Also write CodeRankEmbed vectors into the same index directory,
        /// keyed by the inverted index's internal doc_ids (needs
        /// `--features semantic`).
        #[arg(long)]
        embed: bool,
        /// IVF cluster count (0 = ~2√N). Built next to embeddings when
        /// `--embed` is set. Cluster lists use the same compressed
        /// posting-block format as BM25.
        #[arg(long, default_value_t = 0)]
        ivf_clusters: usize,
    },
    /// Index a source tree for code search: walk the repo (honouring
    /// .gitignore), split files into declaration-sized chunks that keep
    /// their line numbers, and build a hybrid index.
    IndexRepo {
        /// Repository root (default: the current directory).
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Index directory. Defaults to a per-repo directory under the
        /// user cache, so the repository itself is never written to.
        #[arg(long)]
        index: Option<PathBuf>,
        /// Skip embeddings and build a lexical-only index (no model
        /// download, seconds instead of minutes).
        #[arg(long)]
        lexical: bool,
        /// BM25F-lite title boost; the title is `path::declaration`.
        #[arg(long, default_value_t = 2)]
        title_weight: u32,
        /// IVF cluster count (0 = ~2\u{221a}N).
        #[arg(long, default_value_t = 0)]
        ivf_clusters: usize,
        /// Retrain the IVF/PQ quantizer instead of reusing the trained one.
        /// A normal rebuild reuses it, which is what makes small edits fast.
        #[arg(long)]
        retrain: bool,
    },
    /// Run the Model Context Protocol server over stdio, exposing code
    /// search to an MCP client such as Claude Code.
    Mcp {
        /// Repository root (default: the current directory).
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Index directory (default: per-repo directory under the user cache).
        #[arg(long)]
        index: Option<PathBuf>,
        /// Lexical-only: no embeddings, no model download.
        #[arg(long)]
        lexical: bool,
        /// Rebuild at startup even if an index already exists.
        #[arg(long)]
        rebuild: bool,
        /// Do not watch the tree; the index only changes on `reindex`.
        #[arg(long)]
        no_watch: bool,
        /// Default number of results per search.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Pool size before fusion.
        #[arg(long, default_value_t = 200)]
        semantic_candidates: usize,
        #[arg(long, value_enum, default_value_t = FusionArg::Rrf)]
        fusion: FusionArg,
        #[arg(long, default_value_t = 0.5)]
        alpha: f32,
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f32,
        #[arg(long, default_value_t = 0)]
        nprobe: usize,
        #[arg(long)]
        no_pq: bool,
    },
    /// Search an index from the command line (Block-Max WAND).
    Search {
        /// Index directory. Defaults to the index built for `--root`, so a
        /// repository indexed with `index-repo` needs no path here.
        #[arg(long)]
        index: Option<PathBuf>,
        /// Repository whose index to search (default: current directory).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Query string.
        #[arg(long)]
        query: String,
        /// Number of results to return.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Render each result title as a clickable terminal hyperlink using
        /// this URL template. `{id}` and `{title}` are substituted (title is
        /// percent-encoded). E.g. for Wikipedia:
        /// `--url 'https://en.wikipedia.org/?curid={id}'`.
        #[arg(long)]
        url: Option<String>,
        /// Ranking mode: lexical BM25 (default), embedding-first hybrid
        /// (encoder over all vectors + BM25 helper), BM25-then-rerank, or
        /// encoder-only.
        #[arg(long, value_enum, default_value_t = RankMode::Bm25)]
        mode: RankMode,
        /// Also accepted as an alias for `--mode hybrid`.
        #[arg(long)]
        hybrid: bool,
        /// Pool size: BM25 hits kept as helper, and encoder neighbors kept,
        /// before fusion (`hybrid` / `rerank`).
        #[arg(long, default_value_t = 200)]
        semantic_candidates: usize,
        /// Score fusion: min-max weighted mix, or reciprocal rank fusion.
        #[arg(long, value_enum, default_value_t = FusionArg::Rrf)]
        fusion: FusionArg,
        /// BM25 weight for `--fusion weighted` (semantic weight is 1-alpha).
        #[arg(long, default_value_t = 0.5)]
        alpha: f32,
        /// RRF constant k (Cormack et al.).
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f32,
        /// IVF lists to probe (0 = auto). Encoder retrieval uses the same
        /// inverted-file cluster postings as stored in `ivf.bin`.
        #[arg(long, default_value_t = 0)]
        nprobe: usize,
        /// Skip product-quantized scoring even if pq.bin exists (FP16 dots).
        #[arg(long)]
        no_pq: bool,
    },
    /// Precompute CodeRankEmbed vectors for an existing index (needs
    /// `--features semantic`). Writes `<index>/embeddings.bin`.
    Embed {
        /// Index directory (must already contain a lexical index).
        #[arg(long)]
        index: PathBuf,
        /// Same JSONL used to build the index (`id` / `title` / `body`).
        #[arg(long)]
        input: PathBuf,
    },
    /// Retrieval-quality eval on a labeled code-search query set.
    EvalCode {
        #[arg(long)]
        index: PathBuf,
        /// TSV: query_id <tab> query text.
        #[arg(long)]
        queries: PathBuf,
        /// TREC qrels: query_id 0 doc_id relevance.
        #[arg(long)]
        qrels: PathBuf,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value_t = 200)]
        semantic_candidates: usize,
        #[arg(long, value_enum, default_value_t = FusionArg::Rrf)]
        fusion: FusionArg,
        #[arg(long, default_value_t = 0.5)]
        alpha: f32,
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f32,
        #[arg(long, default_value_t = 0)]
        nprobe: usize,
        #[arg(long)]
        no_pq: bool,
    },
    /// Interactive query prompt: load the index once, then type queries and
    /// see results + per-query latency live. `\bench N` repeats the last
    /// query N times and reports latency percentiles.
    Repl {
        /// Index directory. Defaults to the index built for `--root`.
        #[arg(long)]
        index: Option<PathBuf>,
        /// Repository whose index to open (default: current directory).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Number of results to return per query.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Clickable-link URL template; see `search --url`.
        #[arg(long)]
        url: Option<String>,
    },
    /// Serve the HTTP search API.
    Serve {
        /// Index directory.
        #[arg(long)]
        index: PathBuf,
        /// Listen address.
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
    },
    /// Add documents from a JSONL file to a segmented index (creates the
    /// index if the directory is empty).
    Add {
        /// Segmented index directory.
        #[arg(long)]
        index: PathBuf,
        /// Input JSONL file.
        #[arg(long)]
        input: PathBuf,
        /// BM25F-lite title boost (used when creating a new index).
        #[arg(long, default_value_t = 2)]
        title_weight: u32,
        /// Change-detecting mode: skip documents whose content is
        /// unchanged, replace changed ones, add new ones.
        #[arg(long)]
        upsert: bool,
    },
    /// Tombstone a document by external id in a segmented index.
    Delete {
        /// Segmented index directory.
        #[arg(long)]
        index: PathBuf,
        /// External document id to delete.
        #[arg(long)]
        id: String,
    },
    /// Compact a segmented index: merge all segments, dropping tombstones.
    Merge {
        /// Segmented index directory.
        #[arg(long)]
        index: PathBuf,
    },
    /// Migrate a v3 index to the current format in place.
    Migrate {
        /// Index directory.
        #[arg(long)]
        index: PathBuf,
    },
    /// Benchmark queries against an index.
    Bench {
        /// Index directory.
        #[arg(long)]
        index: PathBuf,
        /// Text file with one query per line.
        #[arg(long)]
        queries: PathBuf,
        /// Number of results per query.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, value_enum, default_value_t = RankMode::Bm25)]
        mode: RankMode,
        #[arg(long)]
        hybrid: bool,
        #[arg(long, default_value_t = 200)]
        semantic_candidates: usize,
        #[arg(long, value_enum, default_value_t = FusionArg::Rrf)]
        fusion: FusionArg,
        #[arg(long, default_value_t = 0.5)]
        alpha: f32,
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f32,
        #[arg(long, default_value_t = 0)]
        nprobe: usize,
        #[arg(long)]
        no_pq: bool,
        /// Repeat the hybrid timing sweep over these candidate counts
        /// (comma-separated). Empty means just `--semantic-candidates`.
        #[arg(long, default_value = "")]
        candidate_sweep: String,
    },
}

pub fn run() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Index {
            input,
            out,
            reorder,
            title_weight,
            external,
            code,
            embed,
            ivf_clusters,
        } => cmd_index(
            &input,
            &out,
            reorder.into(),
            title_weight,
            external,
            code,
            embed,
            ivf_clusters,
        ),
        Command::IndexRepo {
            root,
            index,
            lexical,
            title_weight,
            ivf_clusters,
            retrain,
        } => cmd_index_repo(
            &root,
            index.as_deref(),
            !lexical,
            title_weight,
            ivf_clusters,
            retrain,
        ),
        Command::Mcp {
            root,
            index,
            lexical,
            rebuild,
            no_watch,
            top_k,
            semantic_candidates,
            fusion,
            alpha,
            rrf_k,
            nprobe,
            no_pq,
        } => cmd_mcp(
            &root,
            index.as_deref(),
            !lexical,
            rebuild,
            !no_watch,
            top_k,
            SearchOpts {
                mode: RankMode::Hybrid,
                semantic_candidates,
                fusion,
                alpha,
                rrf_k,
                nprobe,
                pq: if no_pq { PqMode::Off } else { PqMode::Auto },
            },
        ),
        Command::Search {
            index,
            root,
            query,
            top_k,
            url,
            mode,
            hybrid,
            semantic_candidates,
            fusion,
            alpha,
            rrf_k,
            nprobe,
            no_pq,
        } => cmd_search(
            &resolve_index(index.as_deref(), root.as_deref())?,
            &query,
            top_k,
            url.as_deref(),
            SearchOpts {
                mode: if hybrid { RankMode::Hybrid } else { mode },
                semantic_candidates,
                fusion,
                alpha,
                rrf_k,
                nprobe,
                pq: if no_pq { PqMode::Off } else { PqMode::Auto },
            },
        ),
        Command::Embed { index, input } => cmd_embed(&index, &input),
        Command::EvalCode {
            index,
            queries,
            qrels,
            top_k,
            semantic_candidates,
            fusion,
            alpha,
            rrf_k,
            nprobe,
            no_pq,
        } => cmd_eval_code(
            &index,
            &queries,
            &qrels,
            top_k,
            SearchOpts {
                mode: RankMode::Hybrid,
                semantic_candidates,
                fusion,
                alpha,
                rrf_k,
                nprobe,
                pq: if no_pq { PqMode::Off } else { PqMode::Auto },
            },
        ),
        Command::Repl {
            index,
            root,
            top_k,
            url,
        } => cmd_repl(
            &resolve_index(index.as_deref(), root.as_deref())?,
            top_k,
            url.as_deref(),
        ),
        Command::Serve { index, addr } => cmd_serve(&index, &addr),
        Command::Add {
            index,
            input,
            title_weight,
            upsert,
        } => cmd_add(&index, &input, title_weight, upsert),
        Command::Delete { index, id } => cmd_delete(&index, &id),
        Command::Merge { index } => cmd_merge(&index),
        Command::Migrate { index } => {
            let start = Instant::now();
            let terms = crate::migrate::migrate_v3_to_v4(&index)?;
            println!(
                "migrated {terms} terms to v4 in {:.2}s",
                start.elapsed().as_secs_f64()
            );
            Ok(())
        }
        Command::Bench {
            index,
            queries,
            top_k,
            mode,
            hybrid,
            semantic_candidates,
            fusion,
            alpha,
            rrf_k,
            nprobe,
            no_pq,
            candidate_sweep,
        } => cmd_bench(
            &index,
            &queries,
            top_k,
            SearchOpts {
                mode: if hybrid { RankMode::Hybrid } else { mode },
                semantic_candidates,
                fusion,
                alpha,
                rrf_k,
                nprobe,
                pq: if no_pq { PqMode::Off } else { PqMode::Auto },
            },
            &candidate_sweep,
        ),
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct SearchOpts {
    pub mode: RankMode,
    pub semantic_candidates: usize,
    pub fusion: FusionArg,
    pub alpha: f32,
    pub rrf_k: f32,
    pub nprobe: usize,
    pub pq: PqMode,
}

#[allow(dead_code)]
/// Pick the scoring path: ADC only when enough documents will be scored to
/// amortize its table build. See [`crate::pq::MIN_CANDIDATES`] for the
/// measurements behind the threshold.
#[allow(dead_code)]
fn choose_pq<'a>(
    index: &'a searcher::AnyIndex,
    opts: &SearchOpts,
    top_k: usize,
) -> Option<&'a crate::pq::PqIndex> {
    let pq = index.pq()?;
    match opts.pq {
        PqMode::Off => None,
        PqMode::Force => Some(pq),
        PqMode::Auto => {
            let (clusters, nprobe) = match index.ivf() {
                Some(ivf) => {
                    let n = if opts.nprobe == 0 {
                        ivf.auto_nprobe()
                    } else {
                        opts.nprobe
                    };
                    (ivf.num_clusters() as usize, n)
                }
                // No inverted file: the encoder scans everything.
                None => (0, 0),
            };
            let candidates = crate::pq::estimated_candidates(
                index.num_docs() as usize,
                clusters,
                nprobe,
            )
            .max(top_k);
            crate::pq::worth_using(candidates).then_some(pq)
        }
    }
}

pub(crate) fn fusion_from(opts: &SearchOpts) -> crate::hybrid::Fusion {
    match opts.fusion {
        FusionArg::Weighted => crate::hybrid::Fusion::weighted(opts.alpha),
        FusionArg::Rrf => crate::hybrid::Fusion::rrf(opts.rrf_k),
    }
}

fn cmd_add(index_dir: &Path, input: &Path, title_weight: u32, upsert: bool) -> anyhow::Result<()> {
    use crate::segments::SegmentedWriter;
    let start = Instant::now();
    let text =
        fs::read_to_string(input).with_context(|| format!("failed to read {}", input.display()))?;
    let docs = indexer::parse_jsonl(&text)?;
    let mut writer = SegmentedWriter::open_or_create(index_dir, true, title_weight)?;
    if upsert {
        let (added, updated, unchanged) = writer.upsert_documents(&docs)?;
        println!(
            "upserted in {:.2}s: {added} added, {updated} updated, {unchanged} unchanged",
            start.elapsed().as_secs_f64()
        );
    } else {
        let name = writer.add_documents(&docs)?;
        println!(
            "added {} docs as segment {name} in {:.2}s",
            docs.len(),
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn cmd_delete(index_dir: &Path, id: &str) -> anyhow::Result<()> {
    use crate::segments::SegmentedWriter;
    let mut writer = SegmentedWriter::open_or_create(index_dir, true, 2)?;
    if writer.delete_document(id)? {
        println!("tombstoned {id:?}");
    } else {
        println!("{id:?} not found (or already deleted)");
    }
    Ok(())
}

fn cmd_merge(index_dir: &Path) -> anyhow::Result<()> {
    use crate::segments::SegmentedWriter;
    let start = Instant::now();
    let mut writer = SegmentedWriter::open_or_create(index_dir, true, 2)?;
    writer.merge_all()?;
    println!("merged in {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn cmd_index(
    input: &Path,
    out: &Path,
    reorder: ReorderStrategy,
    title_weight: u32,
    external: bool,
    code: bool,
    embed: bool,
    ivf_clusters: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        reorder != ReorderStrategy::BpGpu || cfg!(feature = "gpu"),
        "this binary was built without GPU support; rebuild with `cargo build --release --features gpu`"
    );
    if external {
        anyhow::ensure!(
            reorder == ReorderStrategy::None,
            "--external assigns doc_ids while streaming; document reordering is not supported"
        );
        anyhow::ensure!(
            !code,
            "--code is not supported with --external in this prototype"
        );
        anyhow::ensure!(!embed, "--embed is not supported with --external");
        return cmd_index_external(input, out, title_weight);
    }
    if embed {
        require_semantic()?;
    }
    let start = Instant::now();
    // Streamed: raw text and parsed documents never accumulate beyond one
    // chunk, so peak memory tracks the index being built, not the input.
    let mut index = indexer::build_index_from_jsonl_ex(
        input,
        true,
        title_weight,
        DEFAULT_BLOCK_SIZE,
        reorder,
        code,
        embed,
    )?;
    let embed_texts = index.take_embed_texts();
    let built = start.elapsed();
    let size = storage::save_index(&index, out)?;

    let postings = index.total_postings();
    println!(
        "indexed {} docs, {} terms, {} postings in {:.2}s (build {:.2}s + save {:.2}s, reorder: {:?})",
        index.num_docs(),
        index.num_terms(),
        postings,
        start.elapsed().as_secs_f64(),
        built.as_secs_f64(),
        (start.elapsed() - built).as_secs_f64(),
        reorder
    );
    if code {
        println!("tokenizer: code (full identifiers + camelCase/snake_case/dotted splits)");
    }
    println!("avg doc len: {:.1} tokens", index.avg_doc_len());
    println!(
        "index written to {} ({} bytes, {:.2} bytes/posting incl. metadata)",
        out.display(),
        size,
        size as f64 / postings.max(1) as f64
    );
    if embed {
        let texts = embed_texts.context("internal error: embed texts were dropped")?;
        let lens: Vec<u32> = index.docs().iter().map(|d| d.doc_len).collect();
        cmd_write_embeddings(out, &texts, &lens, ivf_clusters)?;
    }
    Ok(())
}

fn cmd_index_external(input: &Path, out: &Path, title_weight: u32) -> anyhow::Result<()> {
    use crate::external;
    use crate::postings::DEFAULT_BLOCK_SIZE as BS;
    let start = Instant::now();
    let stats = if input.as_os_str() == "-" {
        let stdin = std::io::stdin();
        let mut lock = stdin.lock();
        external::build_index_external(
            &mut lock,
            out,
            true,
            title_weight,
            BS,
            external::DEFAULT_SPILL_BUDGET,
        )?
    } else {
        let file =
            fs::File::open(input).with_context(|| format!("failed to read {}", input.display()))?;
        let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
        external::build_index_external(
            &mut reader,
            out,
            true,
            title_weight,
            BS,
            external::DEFAULT_SPILL_BUDGET,
        )?
    };
    println!(
        "indexed {} docs, {} terms, {} postings in {:.2}s (external build)",
        stats.num_docs,
        stats.num_terms,
        stats.num_postings,
        start.elapsed().as_secs_f64(),
    );
    println!(
        "  stream+tokenize+spill: {:.2}s ({} shards)  merge+compress: {:.2}s",
        stats.stream_secs, stats.num_shards, stats.merge_secs
    );
    println!(
        "index written to {} ({} bytes, {:.2} bytes/posting incl. metadata)",
        out.display(),
        stats.index_bytes,
        stats.index_bytes as f64 / stats.num_postings.max(1) as f64
    );
    Ok(())
}

/// Percent-encode a string for use in a URL path/query (RFC 3986
/// unreserved set kept verbatim, everything else `%XX`).
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Render `text` as a clickable link to `url`, adapting to the terminal:
///
/// - non-TTY (piped/redirected): plain `text`, no escapes.
/// - Apple Terminal.app: ignores OSC 8 hyperlinks, but auto-linkifies a
///   *visible* URL on Cmd+click — so show the URL, dimmed, after the title.
/// - everything else (iTerm2, kitty, wezterm, VS Code, ...): an OSC 8
///   hyperlink on an underlined title — clickable, and visibly a link.
fn link(url: &str, text: &str) -> String {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        return text.to_string();
    }
    if std::env::var("TERM_PROGRAM").as_deref() == Ok("Apple_Terminal") {
        // Dim the URL so the title stays prominent; Cmd+click opens it.
        format!("{text}  \x1b[2m{url}\x1b[0m")
    } else {
        // \x1b[4m / \x1b[24m = underline on/off, inside the OSC 8 link.
        format!("\x1b]8;;{url}\x1b\\\x1b[4m{text}\x1b[24m\x1b]8;;\x1b\\")
    }
}

/// Build a result URL from a `--url` template (`{id}` / `{title}`).
fn result_url(template: &str, id: &str, title: &str) -> String {
    template
        .replace("{id}", &percent_encode(id))
        .replace("{title}", &percent_encode(title))
}

/// Render a result title, clickable if a URL template is given.
fn render_title(url_template: Option<&str>, id: &str, title: &str) -> String {
    match url_template {
        Some(t) => link(&result_url(t, id, title), title),
        None => title.to_string(),
    }
}

/// Hybrid retrieval needs the `semantic` feature. Rather than refusing to
/// run, a binary built without it degrades to lexical search and says so:
/// BM25 over the code tokenizer is still useful on its own.
fn embed_or_degrade(embed: bool) -> bool {
    if embed && !cfg!(feature = "semantic") {
        eprintln!(
            "warning: this binary was built without CodeRankEmbed, so search will be \
             lexical (BM25) only.\n         For hybrid search, reinstall with \
             `cargo install --path . --features semantic`."
        );
        return false;
    }
    embed
}

/// Resolve which index directory to open.
///
/// An explicit `--index` wins. Otherwise the per-repo index for `--root`
/// (default: the working directory) is used, which is where `index-repo`
/// puts it — so callers never have to know the cache path.
fn resolve_index(index: Option<&Path>, root: Option<&Path>) -> anyhow::Result<PathBuf> {
    if let Some(dir) = index {
        return Ok(dir.to_path_buf());
    }
    let root = root.unwrap_or_else(|| Path::new("."));
    let dir = crate::codeindex::default_index_dir(root);
    anyhow::ensure!(
        dir.join("meta.bin").exists() || crate::segments::is_segmented(&dir),
        "no index for {} yet — build one with `hips index-repo --root {}`",
        root.display(),
        root.display()
    );
    Ok(dir)
}

/// Build (or rebuild) a code-search index for a source tree.
fn cmd_index_repo(
    root: &Path,
    index: Option<&Path>,
    embed: bool,
    title_weight: u32,
    ivf_clusters: usize,
    retrain: bool,
) -> anyhow::Result<()> {
    let index_dir = match index {
        Some(dir) => dir.to_path_buf(),
        None => crate::codeindex::default_index_dir(root),
    };
    let opts = crate::codeindex::BuildOpts {
        embed: embed_or_degrade(embed),
        title_weight,
        reorder: ReorderStrategy::None,
        ivf_clusters,
        retrain,
        quiet: false,
    };
    let indexer = crate::codeindex::RepoIndexer::new(root, &index_dir, opts)?;
    let manifest = indexer.build()?;
    println!(
        "indexed {} chunks from {} files in {:.2}s",
        manifest.num_docs, manifest.num_files, manifest.build_secs
    );
    println!("index: {}", index_dir.display());
    if manifest.embedded {
        println!(
            "embeddings: {} encoded, {} reused from cache",
            manifest.encoded, manifest.cached
        );
    } else {
        // The degrade warning already explains a missing-feature build; only
        // suggest hybrid when this binary could actually do it.
        if cfg!(feature = "semantic") {
            println!("mode: lexical only (drop --lexical for hybrid search)");
        } else {
            println!("mode: lexical only");
        }
    }
    println!(
        "search it: hips search --root {} --query '...'{}",
        root.display(),
        if manifest.embedded { " --mode hybrid" } else { "" }
    );
    Ok(())
}

/// Serve code search to an MCP client over stdio.
fn cmd_mcp(
    root: &Path,
    index: Option<&Path>,
    embed: bool,
    rebuild: bool,
    watch: bool,
    top_k: usize,
    search: SearchOpts,
) -> anyhow::Result<()> {
    let root = root.to_path_buf();
    let index_dir = match index {
        Some(dir) => dir.to_path_buf(),
        None => crate::codeindex::default_index_dir(&root),
    };
    let config = crate::mcp::ServerConfig {
        root,
        index_dir,
        build: crate::codeindex::BuildOpts {
            embed: embed_or_degrade(embed),
            title_weight: 2,
            reorder: ReorderStrategy::None,
            ivf_clusters: 0,
            retrain: false,
            // Build progress is useful during the first (slow) index; it
            // goes to stderr, since stdout carries the JSON-RPC frames.
            quiet: false,
        },
        force_rebuild: rebuild,
        watch,
        default_top_k: top_k,
        search,
    };
    let mut server = crate::mcp::Server::start(config)?;
    server.serve()
}

fn cmd_search(
    index_dir: &Path,
    query: &str,
    top_k: usize,
    url_template: Option<&str>,
    opts: SearchOpts,
) -> anyhow::Result<()> {
    let index = searcher::AnyIndex::open(index_dir)?;
    if opts.mode == RankMode::Bm25 {
        let outcome = index.search(query, top_k);
        print_outcome(query, &outcome, url_template);
        return Ok(());
    }
    let timed = run_ranked(&index, query, top_k, &opts)?;
    println!(
        "query: {query:?}  ({:.3} ms  bm25 {:.3} + embed {:.3} + score {:.3})  mode={:?}",
        timed.total_ms, timed.bm25_ms, timed.embed_ms, timed.score_ms, opts.mode
    );
    if timed.results.is_empty() {
        println!("no results");
    }
    for (rank, r) in timed.results.iter().enumerate() {
        println!(
            "{:>3}. {:<12} {:>8.4}  bm25={:>8.4}  sem={:>7.4}  {}",
            rank + 1,
            r.id,
            r.score,
            r.bm25,
            r.semantic,
            render_title(url_template, &r.id, &r.title),
        );
    }
    let s = &timed.stats;
    println!(
        "stats: docs_total={} query_terms={} postings_visited={} docs_scored={} blocks_visited={} blocks_skipped={}",
        s.num_docs_total,
        s.num_query_terms,
        s.num_postings_visited,
        s.num_docs_scored,
        s.num_blocks_visited,
        s.num_blocks_skipped
    );
    Ok(())
}

fn print_outcome(query: &str, outcome: &searcher::SearchOutcome, url_template: Option<&str>) {
    println!("query: {query:?}  ({:.3} ms)", outcome.took_ms);
    if let Some(corrected) = &outcome.corrected {
        println!("corrected to: {corrected:?}");
    }
    if outcome.results.is_empty() {
        println!("no results");
    }
    for (rank, result) in outcome.results.iter().enumerate() {
        println!(
            "{:>3}. {:<12} {:>8.4}  {}",
            rank + 1,
            result.id,
            result.score,
            render_title(url_template, &result.id, &result.title),
        );
    }
    let s = &outcome.stats;
    println!(
        "stats: docs_total={} query_terms={} postings_visited={} docs_scored={} blocks_visited={} blocks_skipped={}",
        s.num_docs_total,
        s.num_query_terms,
        s.num_postings_visited,
        s.num_docs_scored,
        s.num_blocks_visited,
        s.num_blocks_skipped
    );
}

pub struct RankedHit {
    pub id: String,
    pub title: String,
    pub score: f32,
    pub bm25: f32,
    pub semantic: f32,
}

pub struct RankedRun {
    pub results: Vec<RankedHit>,
    pub stats: crate::block_max_wand::SearchStats,
    pub bm25_ms: f64,
    pub embed_ms: f64,
    pub score_ms: f64,
    pub total_ms: f64,
}

pub(crate) fn require_semantic() -> anyhow::Result<()> {
    if cfg!(feature = "semantic") {
        Ok(())
    } else {
        anyhow::bail!(
            "this binary was built without CodeRankEmbed; rebuild with `cargo build --release --features semantic`"
        )
    }
}

fn run_ranked(
    index: &searcher::AnyIndex,
    query: &str,
    top_k: usize,
    opts: &SearchOpts,
) -> anyhow::Result<RankedRun> {
    require_semantic()?;
    #[cfg(feature = "semantic")]
    {
        let embedder = crate::embedder::Embedder::load()?;
        run_ranked_with(index, &embedder, query, top_k, opts)
    }
    #[cfg(not(feature = "semantic"))]
    {
        let _ = (index, query, top_k, opts);
        unreachable!("require_semantic already returned")
    }
}

/// Query embed and BM25 do not depend on each other. Encode on this
/// thread (Metal stays where the model was warmed up) and run WAND on a
/// helper thread; IVF/fusion still wait for the vector.
#[cfg(feature = "semantic")]
fn embed_and_bm25(
    disk: &storage::DiskIndex,
    embedder: &crate::embedder::Embedder,
    query: &str,
    k: usize,
) -> anyhow::Result<(
    Vec<f32>,
    Vec<crate::block_max_wand::SearchHit>,
    crate::block_max_wand::SearchStats,
    f64,
    f64,
)> {
    std::thread::scope(|s| {
        let bm25 = s.spawn(|| searcher::search_hits(disk, query, k));
        let t_embed = Instant::now();
        let qvec = embedder.embed_query(query)?;
        let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
        let (hits, stats, bm25_ms) = bm25
            .join()
            .map_err(|_| anyhow::anyhow!("BM25 helper thread panicked"))?;
        Ok((qvec, hits, stats, bm25_ms, embed_ms))
    })
}

#[cfg(feature = "semantic")]
pub fn run_ranked_with(
    index: &searcher::AnyIndex,
    embedder: &crate::embedder::Embedder,
    query: &str,
    top_k: usize,
    opts: &SearchOpts,
) -> anyhow::Result<RankedRun> {
    use crate::hybrid;
    use crate::indexer::SearchableIndex as _;
    let embeddings = index
        .embeddings()
        .context("no embeddings.bin — run `embed --index ... --input ...` first")?;
    let disk = index
        .as_single()
        .context("hybrid/semantic modes currently support a single (non-segmented) index")?;

    let total = Instant::now();

    let (hits, stats, bm25_ms, embed_ms, score_ms) = match opts.mode {
        RankMode::Bm25 => unreachable!(),
        RankMode::Hybrid => {
            let pool = opts.semantic_candidates.max(top_k);
            let (qvec, bm25_hits, stats, bm25_ms, embed_ms) =
                embed_and_bm25(disk, embedder, query, pool)?;
            let t_score = Instant::now();
            let fused = hybrid::merge_retrieve_ivf(
                &bm25_hits,
                embeddings,
                &qvec,
                fusion_from(opts),
                top_k,
                pool,
                index.ivf(),
                opts.nprobe,
                choose_pq(index, opts, top_k),
            );
            let score_ms = t_score.elapsed().as_secs_f64() * 1000.0;
            (fused, stats, bm25_ms, embed_ms, score_ms)
        }
        RankMode::Rerank => {
            let candidates = opts.semantic_candidates.max(top_k);
            let (qvec, bm25_hits, stats, bm25_ms, embed_ms) =
                embed_and_bm25(disk, embedder, query, candidates)?;
            let t_score = Instant::now();
            let fused = hybrid::rerank(&bm25_hits, embeddings, &qvec, fusion_from(opts), top_k);
            let score_ms = t_score.elapsed().as_secs_f64() * 1000.0;
            (fused, stats, bm25_ms, embed_ms, score_ms)
        }
        RankMode::Semantic => {
            let t_embed = Instant::now();
            let qvec = embedder.embed_query(query)?;
            let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
            let t_score = Instant::now();
            let fused = hybrid::semantic_pool(
                embeddings,
                &qvec,
                top_k,
                index.ivf(),
                opts.nprobe,
                choose_pq(index, opts, top_k),
            );
            let score_ms = t_score.elapsed().as_secs_f64() * 1000.0;
            (
                fused,
                crate::block_max_wand::SearchStats {
                    num_docs_total: disk.num_docs(),
                    ..Default::default()
                },
                0.0,
                embed_ms,
                score_ms,
            )
        }
    };
    let results = hits
        .into_iter()
        .map(|h| {
            let s = disk.doc_summary(h.doc_id);
            RankedHit {
                id: s.id,
                title: s.title,
                score: h.score,
                bm25: h.bm25,
                semantic: h.semantic,
            }
        })
        .collect();
    Ok(RankedRun {
        results,
        stats,
        bm25_ms,
        embed_ms,
        score_ms,
        total_ms: total.elapsed().as_secs_f64() * 1000.0,
    })
}

fn cmd_write_embeddings(
    index_dir: &Path,
    texts: &[String],
    doc_lens: &[u32],
    ivf_clusters: usize,
) -> anyhow::Result<()> {
    require_semantic()?;
    #[cfg(feature = "semantic")]
    {
        let embedder = crate::embedder::Embedder::load_for(crate::embedder::EmbedUse::Index)?;
        let n = texts.len();
        println!(
            "embedding {n} documents with CodeRankEmbed (same doc_ids as postings.bin)..."
        );
        let start = Instant::now();
        let bytes = embedder.embed_index_docs(index_dir, texts)?;
        println!(
            "wrote {} ({} docs, {} bytes, {:.1} KB, {:.2}s)",
            index_dir.join(crate::embeddings::EMBEDDINGS_FILE).display(),
            n,
            bytes,
            bytes as f64 / 1024.0,
            start.elapsed().as_secs_f64()
        );
        let store = crate::embeddings::EmbeddingStore::open(index_dir)?;
        let k = if ivf_clusters == 0 {
            crate::ivf::default_num_clusters(n)
        } else {
            ivf_clusters
        };
        let ivf_bytes = crate::ivf::build_and_write(index_dir, &store, doc_lens, k)?;
        println!(
            "wrote {} ({} clusters, {} bytes, same inverted-file blocks as BM25)",
            index_dir.join(crate::ivf::IVF_FILE).display(),
            k,
            ivf_bytes
        );
        let pq_bytes = crate::pq::build_and_write(index_dir, &store, 16)?;
        println!(
            "wrote {} (PQ M=16, 256 codes/subspace, {} bytes)",
            index_dir.join(crate::pq::PQ_FILE).display(),
            pq_bytes
        );
        let _ = start;
        Ok(())
    }
    #[cfg(not(feature = "semantic"))]
    {
        let _ = (index_dir, texts, doc_lens, ivf_clusters);
        unreachable!()
    }
}

fn cmd_embed(index_dir: &Path, input: &Path) -> anyhow::Result<()> {
    require_semantic()?;
    #[cfg(feature = "semantic")]
    {
        let index = crate::storage::load_index(index_dir)?;
        let text = fs::read_to_string(input)
            .with_context(|| format!("failed to read {}", input.display()))?;
        let docs = indexer::parse_jsonl(&text)?;
        let mut by_id: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for d in docs {
            by_id.insert(d.id, format!("{}\n{}", d.title, d.body));
        }
        let n = index.num_docs();
        let mut texts = Vec::with_capacity(n);
        for doc_id in 0..n as u32 {
            let summary = index.doc_summary(doc_id);
            texts.push(
                by_id
                    .get(&summary.id)
                    .cloned()
                    .unwrap_or_else(|| format!("{}\n{}", summary.title, summary.id)),
            );
        }
        let lens: Vec<u32> = (0..n as u32).map(|id| index.doc_len(id)).collect();
        cmd_write_embeddings(index_dir, &texts, &lens, 0)
    }
    #[cfg(not(feature = "semantic"))]
    {
        let _ = (index_dir, input);
        unreachable!()
    }
}

fn cmd_eval_code(
    index_dir: &Path,
    queries_path: &Path,
    qrels_path: &Path,
    top_k: usize,
    opts: SearchOpts,
) -> anyhow::Result<()> {
    require_semantic()?;
    let queries = load_query_tsv(queries_path)?;
    let qrels = load_qrels(qrels_path)?;
    let index = searcher::AnyIndex::open(index_dir)?;

    let mut qrel_by_id: std::collections::HashMap<String, Vec<(String, u32)>> =
        std::collections::HashMap::new();
    for (qid, doc, rel) in qrels {
        qrel_by_id.entry(qid).or_default().push((doc, rel));
    }

    let modes = [
        RankMode::Bm25,
        RankMode::Semantic,
        RankMode::Rerank,
        RankMode::Hybrid,
    ];
    println!(
        "{:<10} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "mode", "n", "MRR", "R@5", "R@10", "nDCG@10"
    );
    #[cfg(feature = "semantic")]
    let embedder = crate::embedder::Embedder::load()?;
    for mode in modes {
        let mut ranked: Vec<Vec<String>> = Vec::new();
        let mut rels: Vec<Vec<(String, u32)>> = Vec::new();
        #[cfg_attr(not(feature = "semantic"), allow(unused_variables))]
        let run_opts = SearchOpts { mode, ..opts };
        for (qid, query) in &queries {
            let ids = match mode {
                RankMode::Bm25 => index
                    .search(query, top_k)
                    .results
                    .into_iter()
                    .map(|r| r.id)
                    .collect(),
                _ => {
                    #[cfg(feature = "semantic")]
                    {
                        run_ranked_with(&index, &embedder, query, top_k, &run_opts)?
                            .results
                            .into_iter()
                            .map(|r| r.id)
                            .collect()
                    }
                    #[cfg(not(feature = "semantic"))]
                    Vec::new()
                }
            };
            ranked.push(ids);
            rels.push(qrel_by_id.get(qid).cloned().unwrap_or_default());
        }
        let m = crate::eval::evaluate(&ranked, &rels);
        println!(
            "{:<10} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            format!("{mode:?}").to_lowercase(),
            m.n,
            m.mrr,
            m.recall_at_5,
            m.recall_at_10,
            m.ndcg_at_10
        );
    }
    Ok(())
}

fn load_query_tsv(path: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let (id, q) = l.split_once('\t')?;
            Some((id.to_owned(), q.to_owned()))
        })
        .collect())
}

fn load_qrels(path: &Path) -> anyhow::Result<Vec<(String, String, u32)>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        anyhow::ensure!(
            parts.len() >= 4,
            "qrels line {}: want `qid 0 docid rel`",
            i + 1
        );
        out.push((
            parts[0].to_owned(),
            parts[2].to_owned(),
            parts[3].parse().unwrap_or(0),
        ));
    }
    Ok(out)
}

fn cmd_repl(index_dir: &Path, top_k: usize, url_template: Option<&str>) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};
    use std::time::Instant;

    let load_start = Instant::now();
    let index = searcher::AnyIndex::open(index_dir)?;
    println!(
        "loaded {} docs, {} bytes in {:.0} ms",
        index.num_docs(),
        index.size_bytes(),
        load_start.elapsed().as_secs_f64() * 1000.0,
    );
    println!("type a query and press enter; `\\bench N` repeats the last query N times; Ctrl-D to quit\n");

    let stdin = std::io::stdin();
    let mut last_query = String::new();
    print!("search> ");
    std::io::stdout().flush().ok();

    for line in stdin.lock().lines() {
        let line = line?;
        let query = line.trim();

        if let Some(rest) = query.strip_prefix("\\bench") {
            let n: usize = rest.trim().parse().unwrap_or(1000);
            if last_query.is_empty() {
                println!("(run a query first, then \\bench N)");
            } else {
                bench_one(&index, &last_query, top_k, n);
            }
        } else if query.is_empty() {
            // ignore
        } else {
            last_query = query.to_string();
            let outcome = index.search(query, top_k);
            if let Some(corrected) = &outcome.corrected {
                println!("corrected to: {corrected:?}");
            }
            for (rank, r) in outcome.results.iter().enumerate() {
                println!(
                    "{:>3}. {:>8.3}  {}",
                    rank + 1,
                    r.score,
                    render_title(url_template, &r.id, &r.title),
                );
            }
            if outcome.results.is_empty() {
                println!("(no results)");
            }
            let s = &outcome.stats;
            println!(
                "  {:.3} ms · {} scored / {} docs · {} blocks skipped",
                outcome.took_ms, s.num_docs_scored, s.num_docs_total, s.num_blocks_skipped,
            );
        }
        print!("\nsearch> ");
        std::io::stdout().flush().ok();
    }
    println!();
    Ok(())
}

/// Repeat one query `n` times and report latency percentiles.
fn bench_one(index: &searcher::AnyIndex, query: &str, top_k: usize, n: usize) {
    use std::time::Instant;
    // Warm-up so page faults don't pollute the measured run.
    for _ in 0..(n / 10).max(1) {
        index.search(query, top_k);
    }
    let mut samples: Vec<f64> = Vec::with_capacity(n);
    let wall = Instant::now();
    for _ in 0..n {
        let t = Instant::now();
        index.search(query, top_k);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let total = wall.elapsed().as_secs_f64();
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| samples[((p * n as f64) as usize).min(n - 1)];
    let mean = samples.iter().sum::<f64>() / n as f64;
    println!(
        "  {n} runs of {query:?}: mean {:.3} ms · p50 {:.3} · p95 {:.3} · p99 {:.3} · min {:.3} · {:.0} queries/s",
        mean,
        pct(0.50),
        pct(0.95),
        pct(0.99),
        samples[0],
        n as f64 / total,
    );
}

fn cmd_serve(index_dir: &Path, addr: &str) -> anyhow::Result<()> {
    let addr: SocketAddr = addr
        .parse()
        .with_context(|| format!("invalid listen address {addr:?}"))?;
    let index = searcher::AnyIndex::open(index_dir)?;
    println!(
        "loaded index: {} docs, {} bytes (memory-mapped)",
        index.num_docs(),
        index.size_bytes()
    );
    let runtime = tokio::runtime::Runtime::new().context("failed to start tokio runtime")?;
    runtime.block_on(api::serve(index, addr))
}

fn cmd_bench(
    index_dir: &Path,
    queries_path: &Path,
    top_k: usize,
    opts: SearchOpts,
    candidate_sweep: &str,
) -> anyhow::Result<()> {
    if opts.mode == RankMode::Bm25 {
        let index = storage::load_index(index_dir)?;
        let text = fs::read_to_string(queries_path)
            .with_context(|| format!("failed to read {}", queries_path.display()))?;
        let queries: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        anyhow::ensure!(!queries.is_empty(), "no queries found in file");
        for query in &queries {
            searcher::search(&index, query, top_k);
        }
        let report = bench::run(&index, &queries, top_k);
        bench::print_report(&report, &index);
        return Ok(());
    }

    require_semantic()?;
    #[cfg(feature = "semantic")]
    {
        let index = searcher::AnyIndex::open(index_dir)?;
        let text = fs::read_to_string(queries_path)
            .with_context(|| format!("failed to read {}", queries_path.display()))?;
        let queries: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        anyhow::ensure!(!queries.is_empty(), "no queries found in file");
        let embedder = crate::embedder::Embedder::load()?;
        let mut counts: Vec<usize> = candidate_sweep
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if counts.is_empty() {
            counts.push(opts.semantic_candidates);
        }
        let emb_bytes = index.embeddings().map(|e| e.size_bytes()).unwrap_or(0);
        println!(
            "hybrid bench: {} queries, lexical index {} bytes, embeddings.bin {} bytes ({:.1} KB), RSS ~{}",
            queries.len(),
            index.size_bytes(),
            emb_bytes,
            emb_bytes as f64 / 1024.0,
            rss_human(),
        );
        for cand in counts {
            let run_opts = SearchOpts {
                semantic_candidates: cand,
                ..opts
            };
            for q in &queries {
                let _ = run_ranked_with(&index, &embedder, q, top_k, &run_opts)?;
            }
            let mut bm25 = Vec::new();
            let mut embed = Vec::new();
            let mut score = Vec::new();
            let mut total = Vec::new();
            for q in &queries {
                let r = run_ranked_with(&index, &embedder, q, top_k, &run_opts)?;
                bm25.push(r.bm25_ms);
                embed.push(r.embed_ms);
                score.push(r.score_ms);
                total.push(r.total_ms);
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            println!(
                "candidates={:<5}  bm25 {:.3} ms  embed {:.3} ms  score {:.3} ms  total {:.3} ms  (score/embed = {:.4})",
                cand,
                mean(&bm25),
                mean(&embed),
                mean(&score),
                mean(&total),
                mean(&score) / mean(&embed).max(1e-9),
            );
        }
        println!("RSS after bench: {}", rss_human());
        Ok(())
    }
    #[cfg(not(feature = "semantic"))]
    {
        let _ = (index_dir, queries_path, top_k, opts, candidate_sweep);
        unreachable!()
    }
}

#[allow(dead_code)]
fn rss_human() -> String {
    let kb = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    if kb >= 1024 {
        format!("{:.1} MB", kb as f64 / 1024.0)
    } else {
        format!("{kb} KB")
    }
}
