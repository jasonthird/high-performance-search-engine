//! Append-only usage log for search calls.
//!
//! Every `hips search` and every MCP `search_code` call appends one JSON
//! line to `<cache_root>/usage.jsonl` (`~/.cache/csearch/usage.jsonl` by
//! default; `CSEARCH_CACHE_DIR` moves it). The point is offline analysis of
//! how agents actually use the tool: which queries, which modes, how many
//! hits, how long — so the skill text and defaults can be tuned from data
//! rather than guesswork. Set `HIPS_NO_LOG=1` to disable.
//!
//! Logging is best-effort: a failure to write never fails the search.

use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

pub const LOG_FILE: &str = "usage.jsonl";

#[derive(Serialize)]
pub struct UsageEvent<'a> {
    /// Unix time, seconds.
    pub ts: u64,
    /// `cli` or `mcp`.
    pub source: &'static str,
    /// Best-effort guess at the calling agent from environment variables.
    pub agent: &'static str,
    pub cwd: String,
    pub root: Option<String>,
    pub index_dir: String,
    pub query: &'a str,
    pub mode: String,
    pub top_k: usize,
    pub path_glob: Option<&'a str>,
    pub n_hits: usize,
    pub took_ms: f64,
    /// Ids (`path:start-end`) of the returned hits, in rank order.
    pub hits: Vec<String>,
}

/// Which agent is driving us, when it announces itself in the environment.
pub fn detect_agent() -> &'static str {
    let set = |k: &str| std::env::var_os(k).is_some_and(|v| !v.is_empty());
    // OpenCode is checked first: launched from inside a Claude Code shell it
    // inherits Claude's variables too, and the innermost agent is the caller.
    if set("OPENCODE") {
        "opencode"
    } else if set("CLAUDECODE") || set("CLAUDE_CODE_ENTRYPOINT") {
        "claude-code"
    } else if set("CODEX_SANDBOX") || set("CODEX_THREAD_ID") {
        "codex"
    } else if set("CURSOR_TRACE_ID") {
        "cursor"
    } else if set("GEMINI_CLI") {
        "gemini-cli"
    } else {
        "unknown"
    }
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn enabled() -> bool {
    !std::env::var_os("HIPS_NO_LOG").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Append one event. Silent on every failure.
pub fn record(event: &UsageEvent<'_>) {
    if !enabled() {
        return;
    }
    let dir = crate::codeindex::cache_root();
    let _ = std::fs::create_dir_all(&dir);
    let Ok(mut line) = serde_json::to_string(event) else {
        return;
    };
    line.push('\n');
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(LOG_FILE))
    {
        let _ = f.write_all(line.as_bytes());
    }
}

pub fn path_string(p: Option<&Path>) -> Option<String> {
    p.map(|p| {
        p.canonicalize()
            .unwrap_or_else(|_| p.to_path_buf())
            .to_string_lossy()
            .to_string()
    })
}

pub fn cwd_string() -> String {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default()
}
