//! Model Context Protocol server over stdio.
//!
//! Exposes this engine's code search to an MCP client (Claude Code, or any
//! other) as three tools: `search_code`, `index_status`, and `reindex`.
//!
//! The transport is newline-delimited JSON-RPC 2.0 on stdin/stdout, which is
//! small enough to implement directly on `serde_json` — consistent with a
//! project that writes its own posting lists. **stdout is the protocol
//! channel**: every diagnostic goes to stderr, or it would corrupt the
//! stream.
//!
//! Freshness is handled by [`crate::watch`]: the watcher marks the tree
//! dirty, and the next tool call rebuilds before answering. See that module
//! for why the rebuild is not done on the watcher thread.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::{json, Value};

use crate::codeindex::{BuildOpts, Manifest, RepoIndexer};
use crate::repo;
use crate::usagelog;
use crate::searcher::AnyIndex;
use crate::watch::TreeWatcher;

/// Protocol version implemented here. If the client asks for a different
/// one, we echo theirs back when it is a version we can speak.
const PROTOCOL_VERSION: &str = "2025-06-18";
const SUPPORTED_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Snippet lines returned per snippeted hit. Enough to judge relevance,
/// small enough that hits do not flood the caller's context: results are
/// pointers the caller can open, not a substitute for reading the file.
const SNIPPET_LINES: usize = 4;

pub struct ServerConfig {
    pub root: PathBuf,
    pub index_dir: PathBuf,
    pub build: BuildOpts,
    /// Rebuild at startup even when an index already exists.
    pub force_rebuild: bool,
    /// Watch the tree and rebuild on change.
    pub watch: bool,
    pub default_top_k: usize,
    pub search: crate::query::SearchOpts,
}

pub struct Server {
    indexer: RepoIndexer,
    /// `None` until the first build. A repository with no index yet is
    /// built on the first tool call, not at startup: a cold hybrid build
    /// takes tens of seconds, and building before `initialize` would trip
    /// the client's connect timeout on every first launch in a new repo.
    index: Option<AnyIndex>,
    watcher: Option<TreeWatcher>,
    manifest: Manifest,
    config: ServerConfig,
    rebuilds: u64,
    last_search: Option<Value>,
    refresh_pending: bool,
    /// Modification time of the on-disk manifest when `index` was opened.
    /// The background watcher (or another agent's `index-repo`) rebuilds
    /// the index behind this process; a changed mtime means reopen.
    manifest_mtime: Option<std::time::SystemTime>,
}

impl Server {
    /// Prepare the index (building it if absent) and start watching.
    pub fn start(mut config: ServerConfig) -> anyhow::Result<Self> {
        // Automatic freshness/repair preserves an existing index's layout.
        // In particular, a failed encoder load must not trigger the legacy
        // destructive single-to-segmented migration during a tool call.
        if config.index_dir.join("meta.bin").exists()
            && !crate::segments::is_segmented(&config.index_dir)
        {
            config.build.segmented = false;
        }
        let indexer = RepoIndexer::new(&config.root, &config.index_dir, config.build.clone())?;
        config.root = indexer.root().to_path_buf();
        // The encoder loads lazily on the first search: eager loading here
        // delays the MCP initialize handshake by model-load time (~1-2s) in
        // every session, including sessions that never search — measured as
        // a real overhead for one-shot `codex exec` style clients. Set
        // HIPS_PRELOAD=1 to restore eager loading for long-lived sessions
        // that want the first search warm.
        if std::env::var_os("HIPS_PRELOAD").is_some_and(|v| v == "1") {
            indexer.preload_embedder()?;
        }
        // An existing, loadable index is opened now. Anything else (absent,
        // pre-manifest, or a forced rebuild) is deferred to the first call.
        let existing = if config.force_rebuild || !index_exists(&config.index_dir) {
            None
        } else {
            Manifest::load(&config.index_dir).ok()
        };
        let (manifest, index) = match existing {
            Some(m) => {
                validate_index_root(&m, &config.root)?;
                let idx = AnyIndex::open(&config.index_dir)?;
                (m, Some(idx))
            }
            None => (Manifest::pending(&config.root, config.build.embed), None),
        };
        let watcher = if config.watch {
            match TreeWatcher::start(&config.root) {
                Ok(w) => Some(w),
                Err(e) => {
                    // A missing watch is a degradation, not a failure: the
                    // caller can still use `reindex`.
                    eprintln!("warning: filesystem watch unavailable ({e:#}); use `reindex`");
                    None
                }
            }
        } else {
            None
        };
        let manifest_mtime = manifest_mtime(&config.index_dir);
        Ok(Self {
            indexer,
            index,
            watcher,
            manifest,
            config,
            rebuilds: 0,
            last_search: None,
            refresh_pending: false,
            manifest_mtime,
        })
    }

    /// Serve until stdin closes.
    pub fn serve(&mut self) -> anyhow::Result<()> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        if self.index.is_some() {
            eprintln!(
                "hips MCP server ready: {} chunks from {} ({} search)",
                self.manifest.num_docs,
                self.config.root.display(),
                if self.hybrid_enabled() {
                    "hybrid"
                } else {
                    "lexical"
                }
            );
        } else {
            eprintln!(
                "hips MCP server ready: {} has no index yet; it is built on the first search",
                self.config.root.display()
            );
        }
        for line in stdin.lock().lines() {
            let line = line.context("stdin read failed")?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some(response) = self.handle_line(line) else {
                continue; // notification: no reply
            };
            writeln!(stdout, "{response}")?;
            stdout.flush()?;
        }
        Ok(())
    }

    /// Parse one line and produce its response, if any. A line may hold a
    /// single message or a JSON-RPC batch (array); a batch answers with an
    /// array of the responses to its non-notification members.
    fn handle_line(&mut self, line: &str) -> Option<String> {
        let message: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                return Some(
                    error_response(Value::Null, -32700, &format!("parse error: {e}")).to_string(),
                )
            }
        };
        match message {
            Value::Array(items) => {
                if items.is_empty() {
                    return Some(
                        error_response(Value::Null, -32600, "invalid request: empty batch")
                            .to_string(),
                    );
                }
                let responses: Vec<Value> = items
                    .into_iter()
                    .filter_map(|m| self.handle_message(m))
                    .collect();
                (!responses.is_empty()).then(|| Value::Array(responses).to_string())
            }
            m => self.handle_message(m).map(|v| v.to_string()),
        }
    }

    /// Handle one (non-batch) message; None for notifications.
    fn handle_message(&mut self, message: Value) -> Option<Value> {
        if !message.is_object() {
            // A client waiting on a malformed request would block forever
            // on silence; answer with invalid-request instead.
            return Some(error_response(Value::Null, -32600, "invalid request"));
        }
        let id = message.get("id").cloned();
        let method = message.get("method").and_then(Value::as_str).unwrap_or("");
        let params = message.get("params").cloned().unwrap_or(json!({}));

        // A message without an id is a notification: never answer it.
        let id = id?;

        let result = match method {
            "initialize" => Ok(self.on_initialize(&params)),
            "tools/list" => Ok(self.on_tools_list()),
            "tools/call" => self.on_tools_call(&params),
            "ping" => Ok(json!({})),
            other => {
                return Some(error_response(id, -32601, &format!("method not found: {other}")))
            }
        };
        Some(match result {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err(e) => error_response(id, -32603, &format!("{e:#}")),
        })
    }

    fn on_initialize(&self, params: &Value) -> Value {
        let requested = params
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or(PROTOCOL_VERSION);
        let version = if SUPPORTED_VERSIONS.contains(&requested) {
            requested
        } else {
            PROTOCOL_VERSION
        };
        json!({
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": false}},
            "serverInfo": {"name": "hips", "version": env!("CARGO_PKG_VERSION")},
            "instructions": format!(
                "Code search over {} ({}, {} retrieval). For ANY question about where \
                 something is implemented, how a mechanism works, or which file is \
                 responsible, call search_code FIRST, before grep or file listing. Describe \
                 what the code does; exact identifiers work too. Results are \
                 `path:startLine-endLine` locations you can open directly. The index follows \
                 the working tree automatically. Use grep only for an exact literal you \
                 already know.",
                self.config.root.display(),
                if self.index.is_some() {
                    format!("{} chunks", self.manifest.num_docs)
                } else {
                    "indexed on first call".to_string()
                },
                if self.hybrid_enabled() { "hybrid BM25 + CodeRankEmbed" } else { "BM25" },
            ),
        })
    }

    fn hybrid_enabled(&self) -> bool {
        self.manifest.embedded && self.config.build.embed
    }

    fn on_tools_list(&self) -> Value {
        let hybrid = self.hybrid_enabled();
        let modes: Vec<&str> = if hybrid {
            vec!["hybrid", "lexical", "semantic"]
        } else {
            vec!["lexical"]
        };
        json!({"tools": [
            {
                "name": "search_code",
                "description": format!(
                    "Search the indexed codebase and return ranked code chunks with their \
                     file path and line range. {} Prefer this over grep when you do not know \
                     the exact string to look for.",
                    if hybrid {
                        "Ranking fuses BM25 with CodeRankEmbed vectors, so both natural-language \
                         intent (\"where do we validate auth tokens\") and exact identifiers work."
                    } else {
                        "Ranking is exact BM25 over a code-aware tokenizer that also splits \
                         camelCase and snake_case identifiers."
                    }
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look for: a description of behaviour, or an identifier."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": format!("Number of results (default {}, max 50).", self.config.default_top_k.clamp(1, 50)),
                            "minimum": 1,
                            "maximum": 50
                        },
                        "mode": {
                            "type": "string",
                            "enum": modes,
                            "description": "Retrieval mode. Defaults to the best available."
                        },
                        "path_glob": {
                            "type": "string",
                            "description": "Optional glob over the repo-relative path, e.g. `src/**/*.rs`."
                        },
                        "include_snippet": {
                            "type": "boolean",
                            "default": false,
                            "description": "Include up to 4 source lines for every returned hit (default false)."
                        },
                        "verbose": {
                            "type": "boolean",
                            "default": false,
                            "description": "Include runtime backend, fallback, timing, cache reuse and score diagnostics (default false)."
                        },
                        "expected_root": {
                            "type": "string",
                            "description": "Optional absolute repository root; fail if this server serves a different root."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "index_status",
                "description": "Report what is indexed: root, chunk count, retrieval mode, \
                                freshness, and rebuild statistics.",
                "inputSchema": {"type": "object", "properties": {
                    "verbose": {"type": "boolean", "default": false, "description": "Inspect resident encoder and last search diagnostics without loading the model (default false)."}
                }}
            },
            {
                "name": "reindex",
                "description": "Rebuild the index now. Rarely needed — the index rebuilds \
                                itself when watched files change. Unchanged chunks reuse \
                                cached embeddings, so this is far cheaper than a first build.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "retrain": {
                            "type": "boolean",
                            "description": "Also retrain the vector quantizer. Slow (seconds); \
                                            only worth it after the codebase has changed a lot."
                        }
                    }
                }
            }
        ]})
    }

    fn on_tools_call(&mut self, params: &Value) -> anyhow::Result<Value> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .context("tools/call missing `name`")?
            .to_string();
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        let verbose = match args.get("verbose") {
            None => false,
            Some(Value::Bool(v)) => *v,
            Some(_) => return Ok(tool_error("`verbose` must be a boolean")),
        };
        let started = std::time::Instant::now();
        if name == "search_code" {
            self.last_search = Some(json!({
                "requested_mode": args.get("mode").cloned().unwrap_or(json!("auto")),
                "effective_mode": null,
                "success": false,
            }));
        }
        // Failed semantic execution is a tool error, never a successful
        // lexical response. CoreML -> Candle is still semantic execution.
        let mut result = match self.dispatch(&name, &args) {
            Ok(text) => json!({"content": [{"type": "text", "text": text}], "isError": false}),
            Err(e) => {
                if name == "search_code" {
                    if let Some(search) = self.last_search.as_mut() {
                        search["error"] = json!(format!("{e:#}"));
                    }
                }
                tool_failure(&e)
            }
        };
        if name == "search_code" {
            if let Some(search) = self.last_search.as_mut() {
                search["total_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        if verbose {
            append_diagnostics(&mut result, self.diagnostics())?;
        }
        Ok(result)
    }

    fn diagnostics(&self) -> Value {
        json!({
            "server": {"pid": std::process::id(), "version": env!("CARGO_PKG_VERSION")},
            "root": self.config.root,
            "index_dir": self.config.index_dir,
            "indexed_root": self.manifest.root,
            "index_loaded": self.index.is_some(),
            "index_has_embeddings": self.manifest.embedded,
            "semantic_enabled": self.hybrid_enabled(),
            "refresh_pending": self.refresh_pending,
            "invalid_live_embeddings": if self.manifest.embedded {
                self.index.as_ref().map(|i| i.invalid_embedding_ids().len())
            } else { None },
            "encoder": self.indexer.encoder_diagnostics(),
            "last_search": self.last_search,
        })
    }

    fn dispatch(&mut self, name: &str, args: &Value) -> anyhow::Result<String> {
        match name {
            "search_code" => self.tool_search(args),
            "index_status" => self.tool_status(),
            "reindex" => {
                let before = self.manifest.num_docs;
                let retrain = args.get("retrain").and_then(Value::as_bool).unwrap_or(false);
                // This rebuild covers all watcher events seen so far; clear
                // the dirty flag first so the next search doesn't rebuild
                // again (an edit during the rebuild re-sets it).
                if let Some(w) = self.watcher.as_ref() {
                    w.take_dirty();
                }
                self.rebuild_with(retrain)?;
                Ok(format!(
                    "Rebuilt in {:.2}s: {} chunks ({:+}), {} encoded, {} reused from cache.",
                    self.manifest.build_secs,
                    self.manifest.num_docs,
                    self.manifest.num_docs as i64 - before as i64,
                    self.manifest.encoded,
                    self.manifest.cached
                ))
            }
            other => anyhow::bail!("unknown tool: {other}"),
        }
    }

    /// Rebuild if the watcher saw a change since the last call.
    fn ensure_fresh(&mut self, repair_vectors: bool) -> anyhow::Result<usize> {
        // A session-leased watcher may be rebuilding right now; let it
        // finish rather than race it for the writer lock.
        crate::daemon::wait_for_idle(&self.config.index_dir, std::time::Duration::from_secs(20));
        let invalid_before = if repair_vectors && self.hybrid_enabled() {
            self.index.as_ref().map(|i| i.invalid_embedding_ids().len()).unwrap_or(0)
        } else { 0 };
        let dirty = self.refresh_pending || self.indexer.recovery_pending()
            || self.watcher.as_ref().is_some_and(|w| w.take_dirty());
        if dirty || self.index.is_none() {
            // Keep failed source refreshes pending: taking the watcher flag
            // must not let the following query silently use stale locations.
            self.refresh_pending = true;
            self.rebuild()?;
        } else if manifest_mtime(&self.config.index_dir) != self.manifest_mtime {
            // Rebuilt by someone else: pick up their index.
            let manifest = Manifest::load(&self.config.index_dir)?;
            validate_index_root(&manifest, &self.config.root)?;
            self.index = Some(AnyIndex::open(&self.config.index_dir)?);
            self.manifest = manifest;
            self.manifest_mtime = manifest_mtime(&self.config.index_dir);
        }
        if repair_vectors && self.hybrid_enabled() {
            let invalid = self.index()?.invalid_embedding_ids().len();
            if invalid > 0 {
                self.rebuild().with_context(|| format!("repair of {invalid} invalid document embeddings failed"))?;
            }
            anyhow::ensure!(self.index()?.invalid_embedding_ids().is_empty(), "document embedding repair incomplete; results withheld");
        }
        Ok(invalid_before)
    }

    fn index(&self) -> anyhow::Result<&AnyIndex> {
        self.index.as_ref().context("index not built yet")
    }

    fn rebuild(&mut self) -> anyhow::Result<()> {
        self.rebuild_with(false)
    }

    fn rebuild_with(&mut self, retrain: bool) -> anyhow::Result<()> {
        self.manifest = self.indexer.build_with(retrain)?;
        // Reopen after the swap: the old handle still maps the retired files.
        self.index = Some(AnyIndex::open(&self.config.index_dir)?);
        self.manifest_mtime = manifest_mtime(&self.config.index_dir);
        self.rebuilds += 1;
        self.refresh_pending = false;
        Ok(())
    }

    fn tool_search(&mut self, args: &Value) -> anyhow::Result<String> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .context("`query` is required")?
            .trim()
            .to_string();
        anyhow::ensure!(!query.is_empty(), "`query` is empty");
        if let Some(expected) = args.get("expected_root") {
            let expected = expected.as_str().context("`expected_root` must be an absolute path string")?;
            anyhow::ensure!(Path::new(expected).is_absolute(), "`expected_root` must be absolute");
            let expected = Path::new(expected).canonicalize().context("resolve expected_root")?;
            anyhow::ensure!(expected == self.config.root,
                "MCP root mismatch: this server serves {}; requested {}. Connect a server for the requested repository or use hips search --root.",
                self.config.root.display(), expected.display());
        }
        let top_k = match args.get("top_k") {
            None => self.config.default_top_k.clamp(1, 50),
            Some(v) => v.as_u64().filter(|k| (1..=50).contains(k))
                .context("`top_k` must be an integer between 1 and 50")? as usize,
        };
        let include_snippet = match args.get("include_snippet") {
            None => false,
            Some(v) => v.as_bool().context("`include_snippet` must be a boolean")?,
        };
        let path_glob = args.get("path_glob").map(|v|
            v.as_str().map(str::to_string).context("`path_glob` must be a string")
        ).transpose()?;
        let mode = args.get("mode").map(|v|
            v.as_str().context("`mode` must be hybrid, semantic, or lexical")
        ).transpose()?;
        anyhow::ensure!(mode.is_none_or(|m| matches!(m, "hybrid" | "semantic" | "lexical")),
            "`mode` must be hybrid, semantic, or lexical");

        let repaired_embeddings = self.ensure_fresh(mode != Some("lexical"))?;

        // A path filter is applied after retrieval, so over-fetch to keep
        // `top_k` results reachable through the filter.
        let fetch = if path_glob.is_some() {
            (top_k * 8).min(500)
        } else {
            top_k
        };
        let started = std::time::Instant::now();
        let effective_mode = mode.unwrap_or(if self.hybrid_enabled() { "hybrid" } else { "lexical" });
        let query_cache_hit = effective_mode != "lexical" && self.indexer.query_cached(&query);
        let hits = self.retrieve(&query, fetch, mode)?;
        anyhow::ensure!(hits.iter().all(|h| h.score.is_finite() &&
            h.components.is_none_or(|(b, s)| b.is_finite() && s.is_finite())),
            "search returned non-finite scores; results withheld");
        let hits: Vec<Hit> = hits
            .into_iter()
            .filter(|h| match &path_glob {
                Some(glob) => repo::glob_match(glob, &h.path),
                None => true,
            })
            .take(top_k)
            .collect();
        usagelog::record(&usagelog::UsageEvent {
            ts: usagelog::now_secs(),
            source: "mcp",
            agent: usagelog::detect_agent(),
            cwd: usagelog::cwd_string(),
            root: usagelog::path_string(Some(&self.config.root)),
            index_dir: self.config.index_dir.to_string_lossy().to_string(),
            query: &query,
            mode: mode
                .map(str::to_string)
                .unwrap_or_else(|| if self.hybrid_enabled() { "hybrid" } else { "lexical" }.to_string()),
            top_k,
            path_glob: path_glob.as_deref(),
            n_hits: hits.len(),
            took_ms: started.elapsed().as_secs_f64() * 1e3,
            hits: hits.iter().map(|h| h.id.clone()).collect(),
        });

        self.last_search = Some(json!({
            "requested_mode": mode.unwrap_or("auto"),
            "effective_mode": effective_mode,
            "success": true,
            "query_cache_hit": query_cache_hit,
            "repaired_embeddings": repaired_embeddings,
            "retrieval_ms": started.elapsed().as_secs_f64() * 1000.0,
            "path_glob": path_glob,
            "hits": hits.iter().map(|h| json!({
                "path": h.path, "start_line": h.start_line, "end_line": h.end_line,
                "score": h.score,
                "bm25": h.components.map(|(b, _)| b),
                "semantic": h.components.map(|(_, s)| s),
            })).collect::<Vec<_>>(),
        }));
        let mut text = format!("Retrieval: {effective_mode}\n");
        if repaired_embeddings > 0 {
            text.push_str(&format!("Reindexed {repaired_embeddings} invalid document embeddings before searching.\n"));
        }
        if effective_mode != "lexical" {
            if let Some(fallbacks) = self.indexer.encoder_diagnostics()["fallbacks"].as_array() {
                if !fallbacks.is_empty() {
                    text.push_str("Note: CoreML unavailable; using Candle (semantic search retained). Use verbose=true for details.\n");
                }
            }
        }
        if hits.is_empty() {
            text.push_str(&format!("No matches for {query:?} in {}{}.\n", self.config.root.display(),
                path_glob.as_ref().map(|g| format!(" under `{g}`")).unwrap_or_default()));
        } else {
            text.push_str(&self.render(&query, &hits, include_snippet));
        }
        Ok(text)
    }

    /// Run the configured retrieval mode and normalize to [`Hit`].
    fn retrieve(&self, query: &str, k: usize, mode: Option<&str>) -> anyhow::Result<Vec<Hit>> {
        let lexical = match mode {
            Some("lexical") | Some("bm25") => true,
            Some(_) => false,
            None => !self.hybrid_enabled(),
        };
        if lexical {
            let outcome = self.index()?.search(query, k);
            return Ok(outcome
                .results
                .into_iter()
                .filter_map(|r| Hit::new(r.id, r.title, r.score, None))
                .collect());
        }
        anyhow::ensure!(
            self.hybrid_enabled(),
            "semantic retrieval is disabled or this index has no embeddings; pass mode=\"lexical\" or rebuild with embeddings"
        );
        self.retrieve_ranked(query, k, mode)
    }

    #[cfg(feature = "semantic")]
    fn retrieve_ranked(
        &self,
        query: &str,
        k: usize,
        mode: Option<&str>,
    ) -> anyhow::Result<Vec<Hit>> {
        let mut opts = self.config.search;
        opts.mode = match mode {
            Some("semantic") => crate::query::RankMode::Semantic,
            Some("rerank") => crate::query::RankMode::Rerank,
            _ => crate::query::RankMode::Hybrid,
        };
        let embedder = self.indexer.embedder()?;
        let run = crate::query::run_ranked_with(self.index()?, embedder, query, k, &opts)?;
        Ok(run
            .results
            .into_iter()
            .filter_map(|r| Hit::new(r.id, r.title, r.score, Some((r.bm25, r.semantic))))
            .collect())
    }

    #[cfg(not(feature = "semantic"))]
    fn retrieve_ranked(
        &self,
        _query: &str,
        _k: usize,
        _mode: Option<&str>,
    ) -> anyhow::Result<Vec<Hit>> {
        anyhow::bail!("this binary was built without CodeRankEmbed; use mode=\"lexical\"")
    }

    fn render(&self, query: &str, hits: &[Hit], include_snippet: bool) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{} result{} for {query:?} in {}:\n\n",
            hits.len(),
            if hits.len() == 1 { "" } else { "s" },
            self.config.root.display()
        ));
        for (rank, hit) in hits.iter().enumerate() {
            out.push_str(&format!(
                "{}. {}:{}-{}",
                rank + 1,
                hit.path,
                hit.start_line,
                hit.end_line
            ));
            if let Some(name) = &hit.name {
                out.push_str(&format!("  {name}"));
            }
            out.push('\n');
            // Scores are logged (usagelog) rather than shown: they are for
            // tuning, not for deciding which hit to open.
            let _ = (hit.score, hit.components);
            if include_snippet {
                if let Some(snippet) =
                    repo::snippet_for(&self.config.root, &hit.id, SNIPPET_LINES)
                {
                    let lang = language_for(&hit.path);
                    out.push_str(&format!("\n```{lang}\n{snippet}\n```\n"));
                }
            }
            out.push('\n');
        }
        out
    }

    fn tool_status(&mut self) -> anyhow::Result<String> {
        let watching = match &self.watcher {
            Some(w) => format!(
                "watching (events seen: {}, pending rebuild: {})",
                w.events_seen(),
                w.is_dirty()
            ),
            None => "not watching (call `reindex` after edits)".to_string(),
        };
        Ok(format!(
            "root: {}\nindex: {}\nchunks: {} from {} files\nretrieval: {}\nencoder: {}\nfreshness: {}\n\
             rebuilds this session: {}\nlast build: {:.2}s ({} encoded, {} from cache)",
            self.config.root.display(),
            self.config.index_dir.display(),
            self.manifest.num_docs,
            self.manifest.num_files,
            if self.hybrid_enabled() {
                "hybrid-capable index (BM25 + CodeRankEmbed vectors; encoder loads on demand)"
            } else {
                "lexical (BM25, code tokenizer)"
            },
            {
                let encoder = self.indexer.encoder_diagnostics();
                format!("{}{}", encoder["state"].as_str().unwrap_or("unknown"),
                    encoder["backend"].as_str().map(|b| format!(" ({b})")).unwrap_or_default())
            },
            watching,
            self.rebuilds,
            self.manifest.build_secs,
            self.manifest.encoded,
            self.manifest.cached,
        ))
    }
}

/// One normalized search hit, resolved back to a source location.
struct Hit {
    id: String,
    path: String,
    start_line: usize,
    end_line: usize,
    name: Option<String>,
    score: f32,
    /// (bm25, semantic) when the mode produced both.
    components: Option<(f32, f32)>,
}

impl Hit {
    fn new(id: String, title: String, score: f32, components: Option<(f32, f32)>) -> Option<Self> {
        let (path, start_line, end_line) = repo::parse_id(&id)?;
        let name = title.rsplit_once("::").map(|(_, n)| n.to_string());
        Some(Self {
            path: path.to_string(),
            start_line,
            end_line,
            name,
            id,
            score,
            components,
        })
    }
}

fn manifest_mtime(index_dir: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(index_dir.join(crate::codeindex::MANIFEST_FILE))
        .and_then(|m| m.modified())
        .ok()
}

fn index_exists(dir: &Path) -> bool {
    dir.join("meta.bin").exists() || crate::segments::is_segmented(dir)
}

fn language_for(path: &str) -> &'static str {
    match path.rsplit_once('.').map(|(_, e)| e) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("js") | Some("mjs") | Some("cjs") => "javascript",
        Some("jsx") => "jsx",
        Some("ts") => "typescript",
        Some("tsx") => "tsx",
        Some("go") => "go",
        Some("java") => "java",
        Some("kt") | Some("kts") => "kotlin",
        Some("rb") => "ruby",
        Some("c") | Some("h") => "c",
        Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("hh") => "cpp",
        Some("cs") => "csharp",
        Some("swift") => "swift",
        Some("scala") => "scala",
        Some("php") => "php",
        Some("sh") | Some("bash") | Some("zsh") => "bash",
        Some("sql") => "sql",
        Some("md") => "markdown",
        Some("toml") => "toml",
        Some("yaml") | Some("yml") => "yaml",
        _ => "",
    }
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn tool_error(message: &str) -> Value {
    json!({"content": [{"type": "text", "text": message}], "isError": true})
}

fn tool_failure(error: &anyhow::Error) -> Value {
    let cause = format!("{error:#}");
    let storage_full = error.chain().any(|cause| cause.downcast_ref::<std::io::Error>()
        .is_some_and(|io| io.kind() == std::io::ErrorKind::StorageFull));
    if !storage_full { return tool_error(&cause); }
    let message = "Not enough disk space.";
    let action = "Free space on the affected filesystem, then retry.";
    let mut result = tool_error(&format!("{message} {action}\nCause: {cause}"));
    // Operational errors are actionable without enabling verbose diagnostics.
    result["structuredContent"] = json!({"error": {
        "code": "insufficient_disk_space", "message": message,
        "action": action, "cause": cause,
    }});
    result
}

fn append_diagnostics(result: &mut Value, diagnostics: Value) -> anyhow::Result<()> {
    let text = format!("Debug diagnostics:\n{}", serde_json::to_string_pretty(&diagnostics)?);
    result["structuredContent"]["diagnostics"] = diagnostics;
    result["content"].as_array_mut().unwrap().push(json!({"type": "text", "text": text}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_full_is_actionable_without_verbose_and_survives_diagnostics() {
        #[cfg(unix)]
        assert_eq!(tool_failure(&std::io::Error::from_raw_os_error(libc::ENOSPC).into())
            ["structuredContent"]["error"]["code"], "insufficient_disk_space");
        let error = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::StorageFull))
            .context("failed to create /cache/seg-000046/embeddings.bin")
            .context("document embedding repair failed");
        let mut result = tool_failure(&error);
        assert_eq!(result["isError"], true);
        assert_eq!(result["structuredContent"]["error"]["code"], "insufficient_disk_space");
        assert!(result["structuredContent"].get("diagnostics").is_none());
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("Not enough disk space."));
        assert!(text.contains("Free space"));
        assert!(text.contains("/cache/seg-000046/embeddings.bin"));
        let operational_error = result["structuredContent"]["error"].clone();
        append_diagnostics(&mut result, json!({"encoder": {"state": "ready"}})).unwrap();
        assert_eq!(result["structuredContent"]["error"], operational_error);
        assert_eq!(result["structuredContent"]["diagnostics"]["encoder"]["state"], "ready");
        assert_eq!(result["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn other_failures_are_not_mislabeled_as_disk_full() {
        for error in [
            anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
            anyhow::anyhow!("missing keys.bin after a previous No space left on device error"),
        ] {
            let result = tool_failure(&error);
            assert_eq!(result["isError"], true);
            assert!(result.get("structuredContent").is_none());
            assert_eq!(result["content"][0]["text"], format!("{error:#}"));
        }
    }

    #[test]
    fn hits_resolve_to_locations() {
        let hit = Hit::new(
            "src/searcher.rs:10-42".into(),
            "src/searcher.rs::query_terms".into(),
            1.5,
            Some((3.0, 0.8)),
        )
        .unwrap();
        assert_eq!(hit.path, "src/searcher.rs");
        assert_eq!((hit.start_line, hit.end_line), (10, 42));
        assert_eq!(hit.name.as_deref(), Some("query_terms"));
        // A non-repo id (e.g. an index built from plain JSONL) is skipped
        // rather than rendered as a bogus location.
        assert!(Hit::new("doc-7".into(), "Doc 7".into(), 1.0, None).is_none());
    }

    #[test]
    fn languages_are_fenced_correctly() {
        assert_eq!(language_for("a/b.rs"), "rust");
        assert_eq!(language_for("a/b.tsx"), "tsx");
        assert_eq!(language_for("Makefile"), "");
    }
}

/// An explicit index must belong to the pinned repository; otherwise even
/// valid hit IDs would resolve snippets against unrelated source files.
fn validate_index_root(manifest: &Manifest, root: &Path) -> anyhow::Result<()> {
    let indexed = Path::new(&manifest.root).canonicalize().context("resolve indexed root")?;
    anyhow::ensure!(indexed == root, "MCP index root mismatch: index belongs to {}, server root is {}",
        indexed.display(), root.display());
    Ok(())
}

#[cfg(test)]
mod freshness_regression_tests {
    use super::*;

    #[test]
    fn restarted_server_recovers_pending_publication_without_a_watch_event() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("repo");
        std::fs::create_dir(&root).unwrap();
        let file = root.join("settings.toml");
        std::fs::write(&file, "permission = true\n").unwrap();
        let config = ServerConfig {
            root,
            index_dir: temp.path().join("index"),
            build: BuildOpts { embed: false, segmented: true, quiet: true, ..Default::default() },
            force_rebuild: false, watch: false, default_top_k: 5, search: Default::default(),
        };
        RepoIndexer::new(&config.root, &config.index_dir, config.build.clone()).unwrap().build().unwrap();
        // A previous writer tombstoned a row and stopped before repo.json.
        {
            let mut writer = crate::segments::SegmentedWriter::open_or_create_ex(&config.index_dir, false, 2, true).unwrap();
            writer.delete_documents(&["settings.toml:1-1".into()]).unwrap();
        }
        std::fs::write(config.index_dir.join("build.pending"), b"interrupted").unwrap();
        let mut server = Server::start(config).unwrap();
        assert!(!server.refresh_pending);
        let text = server.tool_search(&json!({"query":"permission"})).unwrap();
        assert!(text.contains("settings.toml:1-1"), "{text}");
        assert!(!server.indexer.recovery_pending());
        assert_eq!(server.rebuilds, 1);
    }

    #[test]
    fn failed_source_refresh_is_retried_before_results_are_served() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("repo");
        std::fs::create_dir(&root).unwrap();
        let file = root.join("settings.toml");
        std::fs::write(&file, "permission = true\n").unwrap();
        let mut server = Server::start(ServerConfig {
            root,
            index_dir: temp.path().join("index"),
            build: BuildOpts { embed: false, segmented: false, quiet: true, ..Default::default() },
            force_rebuild: false,
            watch: false,
            default_top_k: 5,
            search: Default::default(),
        }).unwrap();
        server.tool_search(&json!({"query":"permission"})).unwrap();
        std::fs::remove_file(&file).unwrap();
        server.refresh_pending = true; // the state set when consuming an edit event
        for _ in 0..2 {
            assert!(server.tool_search(&json!({"query":"permission"})).is_err());
            assert!(server.refresh_pending);
        }
        std::fs::write(&file, "financial_validation = true\n").unwrap();
        let text = server.tool_search(&json!({"query":"financial_validation", "include_snippet":true})).unwrap();
        assert!(text.contains("financial_validation = true"));
        assert!(!server.refresh_pending);
    }
}
