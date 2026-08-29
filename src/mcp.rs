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
use crate::searcher::AnyIndex;
use crate::watch::TreeWatcher;

/// Protocol version implemented here. If the client asks for a different
/// one, we echo theirs back when it is a version we can speak.
const PROTOCOL_VERSION: &str = "2025-06-18";
const SUPPORTED_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Snippet lines returned per hit. Enough to judge relevance, small enough
/// that ten hits do not flood the caller's context.
const SNIPPET_LINES: usize = 40;

pub struct ServerConfig {
    pub root: PathBuf,
    pub index_dir: PathBuf,
    pub build: BuildOpts,
    /// Rebuild at startup even when an index already exists.
    pub force_rebuild: bool,
    /// Watch the tree and rebuild on change.
    pub watch: bool,
    pub default_top_k: usize,
    pub search: crate::cli::SearchOpts,
}

pub struct Server {
    indexer: RepoIndexer,
    index: AnyIndex,
    watcher: Option<TreeWatcher>,
    manifest: Manifest,
    config: ServerConfig,
    rebuilds: u64,
}

impl Server {
    /// Prepare the index (building it if absent) and start watching.
    pub fn start(config: ServerConfig) -> anyhow::Result<Self> {
        let indexer = RepoIndexer::new(&config.root, &config.index_dir, config.build.clone())?;
        // Pay for model load at startup, not on the first search after an edit.
        indexer.preload_embedder()?;
        let needs_build = config.force_rebuild || !index_exists(&config.index_dir);
        let manifest = if needs_build {
            indexer.build()?
        } else {
            match Manifest::load(&config.index_dir) {
                Ok(m) => m,
                // Index directory exists but predates the manifest, or was
                // built for a different tree: rebuild rather than guess.
                Err(_) => indexer.build()?,
            }
        };
        let index = AnyIndex::open(&config.index_dir)?;
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
        Ok(Self {
            indexer,
            index,
            watcher,
            manifest,
            config,
            rebuilds: 0,
        })
    }

    /// Serve until stdin closes.
    pub fn serve(&mut self) -> anyhow::Result<()> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        eprintln!(
            "hips MCP server ready: {} chunks from {} ({} search)",
            self.manifest.num_docs,
            self.config.root.display(),
            if self.manifest.embedded {
                "hybrid"
            } else {
                "lexical"
            }
        );
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
                "Code search over {} ({} chunks, {} retrieval). Use search_code with a \
                 natural-language description of what you are looking for, or with exact \
                 identifiers — both work. Results are `path:startLine-endLine` locations you \
                 can open directly. The index follows the working tree automatically.",
                self.config.root.display(),
                self.manifest.num_docs,
                if self.manifest.embedded { "hybrid BM25 + CodeRankEmbed" } else { "BM25" },
            ),
        })
    }

    fn on_tools_list(&self) -> Value {
        let hybrid = self.manifest.embedded;
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
                            "description": "Number of results (default 10, max 50).",
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
                            "description": "Include the source of each hit (default true)."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "index_status",
                "description": "Report what is indexed: root, chunk count, retrieval mode, \
                                freshness, and rebuild statistics.",
                "inputSchema": {"type": "object", "properties": {}}
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
        // Tool failures are reported in-band so the model can react to them,
        // rather than as transport errors.
        let text = match self.dispatch(&name, &args) {
            Ok(text) => text,
            Err(e) => return Ok(tool_error(&format!("{e:#}"))),
        };
        Ok(json!({"content": [{"type": "text", "text": text}], "isError": false}))
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
    fn ensure_fresh(&mut self) -> anyhow::Result<()> {
        let dirty = self.watcher.as_ref().is_some_and(|w| w.take_dirty());
        if dirty {
            self.rebuild()?;
        }
        Ok(())
    }

    fn rebuild(&mut self) -> anyhow::Result<()> {
        self.rebuild_with(false)
    }

    fn rebuild_with(&mut self, retrain: bool) -> anyhow::Result<()> {
        self.manifest = self.indexer.build_with(retrain)?;
        // Reopen after the swap: the old handle still maps the retired files.
        self.index = AnyIndex::open(&self.config.index_dir)?;
        self.rebuilds += 1;
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
        let top_k = args
            .get("top_k")
            .and_then(Value::as_u64)
            .map(|k| k.clamp(1, 50) as usize)
            .unwrap_or(self.config.default_top_k);
        let include_snippet = args
            .get("include_snippet")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let path_glob = args
            .get("path_glob")
            .and_then(Value::as_str)
            .map(str::to_string);
        let mode = args.get("mode").and_then(Value::as_str);

        self.ensure_fresh()?;

        // A path filter is applied after retrieval, so over-fetch to keep
        // `top_k` results reachable through the filter.
        let fetch = if path_glob.is_some() {
            (top_k * 8).min(500)
        } else {
            top_k
        };
        let hits = self.retrieve(&query, fetch, mode)?;
        let hits: Vec<Hit> = hits
            .into_iter()
            .filter(|h| match &path_glob {
                Some(glob) => repo::glob_match(glob, &h.path),
                None => true,
            })
            .take(top_k)
            .collect();

        if hits.is_empty() {
            return Ok(format!(
                "No matches for {query:?}{}.",
                match &path_glob {
                    Some(g) => format!(" under `{g}`"),
                    None => String::new(),
                }
            ));
        }
        Ok(self.render(&query, &hits, include_snippet))
    }

    /// Run the configured retrieval mode and normalize to [`Hit`].
    fn retrieve(&self, query: &str, k: usize, mode: Option<&str>) -> anyhow::Result<Vec<Hit>> {
        let lexical = match mode {
            Some("lexical") | Some("bm25") => true,
            Some(_) => false,
            None => !self.manifest.embedded,
        };
        if lexical {
            let outcome = self.index.search(query, k);
            return Ok(outcome
                .results
                .into_iter()
                .filter_map(|r| Hit::new(r.id, r.title, r.score, None))
                .collect());
        }
        anyhow::ensure!(
            self.manifest.embedded,
            "this index has no embeddings; pass mode=\"lexical\" or rebuild with embeddings"
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
            Some("semantic") => crate::cli::RankMode::Semantic,
            Some("rerank") => crate::cli::RankMode::Rerank,
            _ => crate::cli::RankMode::Hybrid,
        };
        let embedder = self.indexer.embedder()?;
        let run = crate::cli::run_ranked_with(&self.index, embedder, query, k, &opts)?;
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
                out.push_str(&format!("  ({name})"));
            }
            match hit.components {
                Some((bm25, semantic)) => out.push_str(&format!(
                    "\n   score {:.4}  bm25 {:.3}  semantic {:.3}\n",
                    hit.score, bm25, semantic
                )),
                None => out.push_str(&format!("\n   score {:.4}\n", hit.score)),
            }
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
            "root: {}\nindex: {}\nchunks: {} from {} files\nretrieval: {}\nfreshness: {}\n\
             rebuilds this session: {}\nlast build: {:.2}s ({} encoded, {} from cache)",
            self.config.root.display(),
            self.config.index_dir.display(),
            self.manifest.num_docs,
            self.manifest.num_files,
            if self.manifest.embedded {
                "hybrid (BM25 + CodeRankEmbed, IVF/PQ)"
            } else {
                "lexical (BM25, code tokenizer)"
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

#[cfg(test)]
mod tests {
    use super::*;

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
