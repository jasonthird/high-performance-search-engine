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
    },
    /// Search an index from the command line (Block-Max WAND).
    Search {
        /// Index directory.
        #[arg(long)]
        index: PathBuf,
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
    },
    /// Interactive query prompt: load the index once, then type queries and
    /// see results + per-query latency live. `\bench N` repeats the last
    /// query N times and reports latency percentiles.
    Repl {
        /// Index directory.
        #[arg(long)]
        index: PathBuf,
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
        } => cmd_index(&input, &out, reorder.into(), title_weight, external),
        Command::Search {
            index,
            query,
            top_k,
            url,
        } => cmd_search(&index, &query, top_k, url.as_deref()),
        Command::Repl {
            index,
            top_k,
            url,
        } => cmd_repl(&index, top_k, url.as_deref()),
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
        } => cmd_bench(&index, &queries, top_k),
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
        return cmd_index_external(input, out, title_weight);
    }
    let start = Instant::now();
    // Streamed: raw text and parsed documents never accumulate beyond one
    // chunk, so peak memory tracks the index being built, not the input.
    let index =
        indexer::build_index_from_jsonl(input, true, title_weight, DEFAULT_BLOCK_SIZE, reorder)?;
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
    println!("avg doc len: {:.1} tokens", index.avg_doc_len());
    println!(
        "index written to {} ({} bytes, {:.2} bytes/posting incl. metadata)",
        out.display(),
        size,
        size as f64 / postings.max(1) as f64
    );
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

fn cmd_search(
    index_dir: &Path,
    query: &str,
    top_k: usize,
    url_template: Option<&str>,
) -> anyhow::Result<()> {
    let index = searcher::AnyIndex::open(index_dir)?;
    let outcome = index.search(query, top_k);

    println!("query: {query:?}  ({:.3} ms)", outcome.took_ms);
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
    Ok(())
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

fn cmd_bench(index_dir: &Path, queries_path: &Path, top_k: usize) -> anyhow::Result<()> {
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

    // Warm-up pass so first-touch effects do not distort latencies.
    for query in &queries {
        searcher::search(&index, query, top_k);
    }
    let report = bench::run(&index, &queries, top_k);
    bench::print_report(&report, &index);
    Ok(())
}
