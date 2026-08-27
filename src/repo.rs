//! Source-tree ingestion: walk a repository, honour `.gitignore`, and split
//! each source file into retrievable chunks.
//!
//! This replaces `scripts/code_to_jsonl.py` with a native walker that keeps
//! **line numbers**, so a hit can be reported as `path:line` and its snippet
//! re-read from the file on demand. Chunk ids have the form
//! `relative/path.rs:START-END` (1-based, inclusive), which is both the
//! external document id and a location the caller can open directly.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use rayon::prelude::*;

use crate::indexer::InputDoc;

/// Files larger than this are skipped: generated bundles and vendored blobs
/// dominate retrieval otherwise.
pub const MAX_FILE_BYTES: u64 = 1 << 20;
/// A chunk longer than this many lines is split, so one huge function does
/// not swallow a whole file's worth of ranking mass.
pub const MAX_CHUNK_LINES: usize = 200;
/// Bodies are truncated here before indexing; the encoder truncates at its
/// own sequence limit anyway and BM25 gains nothing from the tail.
pub const MAX_CHUNK_BYTES: usize = 8 * 1024;

/// Source extensions worth indexing.
pub const SOURCE_EXTS: &[&str] = &[
    "rs", "py", "js", "jsx", "ts", "tsx", "mjs", "cjs", "go", "java", "kt", "kts", "rb", "c", "h",
    "cpp", "cc", "cxx", "hpp", "hh", "cs", "swift", "scala", "php", "sh", "bash", "zsh", "sql",
    "lua", "ml", "hs", "ex", "exs", "erl", "clj", "vue", "svelte", "proto", "tf", "md", "toml",
    "yaml", "yml",
];

/// Directories never worth walking, even when not gitignored.
const ALWAYS_SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    ".csearch",
    "node_modules",
    "target",
    "dist",
    "build",
    "vendor",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
    ".nuxt",
];

/// One indexable unit of source: a declaration, or a slice of a file with no
/// recognizable declarations.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Repo-relative path, forward-slashed.
    pub path: String,
    /// 1-based inclusive line range.
    pub start_line: usize,
    pub end_line: usize,
    /// Declaration name, when one was found.
    pub name: Option<String>,
    pub body: String,
}

impl Chunk {
    /// `path:start-end` — the external document id, and a location string.
    pub fn id(&self) -> String {
        format!("{}:{}-{}", self.path, self.start_line, self.end_line)
    }

    /// Human-facing label: the declaration name qualified by its file.
    pub fn title(&self) -> String {
        match &self.name {
            Some(name) => format!("{}::{}", self.path, name),
            None => self.path.clone(),
        }
    }

    pub fn into_doc(self) -> InputDoc {
        let id = self.id();
        let title = self.title();
        InputDoc {
            id,
            title,
            body: self.body,
        }
    }
}

/// Parse a chunk id back into its location.
pub fn parse_id(id: &str) -> Option<(&str, usize, usize)> {
    let (path, range) = id.rsplit_once(':')?;
    let (start, end) = range.split_once('-')?;
    Some((path, start.parse().ok()?, end.parse().ok()?))
}

/// Re-read the lines a chunk id points at. Returns `None` if the file is
/// gone or the range no longer exists (a stale hit after an edit).
pub fn snippet_for(root: &Path, id: &str, max_lines: usize) -> Option<String> {
    let (path, start, end) = parse_id(id)?;
    let text = fs::read_to_string(root.join(path)).ok()?;
    let take = (end + 1).saturating_sub(start).min(max_lines);
    let body: Vec<&str> = text
        .lines()
        .skip(start.saturating_sub(1))
        .take(take)
        .collect();
    if body.is_empty() {
        return None;
    }
    Some(body.join("\n"))
}

// ---------------------------------------------------------------------------
// gitignore
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct IgnoreRule {
    /// Pattern with any leading `!` and trailing `/` stripped.
    pattern: String,
    negated: bool,
    dir_only: bool,
    /// Pattern contained a non-trailing `/`, so it matches against the path
    /// relative to the `.gitignore` rather than against the basename.
    anchored: bool,
}

/// The rules from one `.gitignore`, together with the directory it governs.
#[derive(Debug, Clone, Default)]
pub struct IgnoreSet {
    /// Directory prefix (repo-relative, `""` for the root) -> rules.
    layers: Vec<(String, Vec<IgnoreRule>)>,
}

impl IgnoreSet {
    fn parse_layer(prefix: &str, text: &str) -> Vec<IgnoreRule> {
        let mut rules = Vec::new();
        let _ = prefix;
        for raw in text.lines() {
            let line = raw.trim_end();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (negated, rest) = match line.strip_prefix('!') {
                Some(r) => (true, r),
                None => (false, line),
            };
            let dir_only = rest.ends_with('/');
            let rest = rest.trim_end_matches('/');
            let anchored = rest.trim_start_matches('/').contains('/');
            let pattern = rest.trim_start_matches('/').to_string();
            if pattern.is_empty() {
                continue;
            }
            rules.push(IgnoreRule {
                pattern,
                negated,
                dir_only,
                anchored,
            });
        }
        rules
    }

    /// Add the `.gitignore` found in `dir` (repo-relative `prefix`), if any.
    fn push_dir(&mut self, dir: &Path, prefix: &str) {
        let path = dir.join(".gitignore");
        let Ok(text) = fs::read_to_string(&path) else {
            return;
        };
        let rules = Self::parse_layer(prefix, &text);
        if !rules.is_empty() {
            self.layers.push((prefix.to_string(), rules));
        }
    }

    /// Is `rel` (repo-relative, forward-slashed) ignored? Later layers and
    /// later rules win, matching git's last-match-wins semantics.
    pub fn is_ignored(&self, rel: &str, is_dir: bool) -> bool {
        let mut ignored = false;
        for (prefix, rules) in &self.layers {
            let Some(sub) = strip_prefix_path(rel, prefix) else {
                continue;
            };
            for rule in rules {
                if rule.dir_only && !is_dir {
                    continue;
                }
                let hit = if rule.anchored {
                    glob_match(&rule.pattern, sub)
                } else {
                    let base = sub.rsplit('/').next().unwrap_or(sub);
                    glob_match(&rule.pattern, base)
                        || sub.split('/').any(|seg| glob_match(&rule.pattern, seg))
                };
                if hit {
                    ignored = !rule.negated;
                }
            }
        }
        ignored
    }
}

/// Strip a directory prefix from a repo-relative path.
fn strip_prefix_path<'a>(rel: &'a str, prefix: &str) -> Option<&'a str> {
    if prefix.is_empty() {
        return Some(rel);
    }
    let rest = rel.strip_prefix(prefix)?;
    rest.strip_prefix('/')
}

/// Glob matcher for the gitignore subset: `*` (not crossing `/`), `**`,
/// `?`, and character classes are not supported beyond `*`/`?`.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    glob_inner(pattern.as_bytes(), text.as_bytes())
}

fn glob_inner(p: &[u8], t: &[u8]) -> bool {
    // `**` crosses path separators; a single `*` and `?` do not. Nested
    // wildcards need real backtracking (a single remembered star position is
    // not enough for `**/*.log`), so each wildcard recurses over the suffixes
    // it could consume. Gitignore patterns are short, so this stays cheap.
    if p.first() == Some(&b'*') {
        let double = p.get(1) == Some(&b'*');
        let mut rest = p;
        while rest.first() == Some(&b'*') {
            rest = &rest[1..];
        }
        if double {
            // `**/` may also match zero directories.
            if rest.first() == Some(&b'/') && glob_inner(&rest[1..], t) {
                return true;
            }
            let mut i = 0;
            loop {
                if glob_inner(rest, &t[i..]) {
                    return true;
                }
                if i == t.len() {
                    return false;
                }
                i += 1;
            }
        }
        let mut i = 0;
        loop {
            if glob_inner(rest, &t[i..]) {
                return true;
            }
            // A single `*` stops at a path separator.
            if i == t.len() || t[i] == b'/' {
                return false;
            }
            i += 1;
        }
    }
    let Some(&pc) = p.first() else {
        return t.is_empty();
    };
    let Some(&tc) = t.first() else {
        return false;
    };
    if pc == tc || (pc == b'?' && tc != b'/') {
        return glob_inner(&p[1..], &t[1..]);
    }
    false
}

// ---------------------------------------------------------------------------
// walking
// ---------------------------------------------------------------------------

/// A source file found in the tree, with the mtime used for staleness checks.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub rel: String,
    pub abs: PathBuf,
    pub len: u64,
    /// Modification time in nanoseconds since the epoch (0 if unavailable).
    /// Together with `len` this fingerprints the file cheaply: a rebuild
    /// whose walk fingerprint matches the manifest can skip everything.
    pub mtime_ns: u128,
}

/// Walk `root`, returning indexable source files in a deterministic order.
pub fn walk(root: &Path) -> anyhow::Result<Vec<SourceFile>> {
    let root = root
        .canonicalize()
        .with_context(|| format!("cannot open {}", root.display()))?;
    let mut ignores = IgnoreSet::default();
    ignores.push_dir(&root, "");
    let mut out = Vec::new();
    walk_dir(&root, "", &mut ignores, &mut out)?;
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

fn walk_dir(
    dir: &Path,
    prefix: &str,
    ignores: &mut IgnoreSet,
    out: &mut Vec<SourceFile>,
) -> anyhow::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        // An unreadable directory is not fatal: skip it and keep indexing.
        Err(_) => return Ok(()),
    };
    let mut dirs = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let rel = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        let Ok(ft) = entry.file_type() else { continue };
        // Symlinks are not followed: they can escape the repo or form cycles.
        if ft.is_symlink() {
            continue;
        }
        if ft.is_dir() {
            if ALWAYS_SKIP_DIRS.contains(&name.as_str()) || ignores.is_ignored(&rel, true) {
                continue;
            }
            dirs.push((entry.path(), rel));
        } else if ft.is_file() {
            if !is_source(&name) || ignores.is_ignored(&rel, false) {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if meta.len() == 0 || meta.len() > MAX_FILE_BYTES {
                continue;
            }
            let mtime_ns = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            out.push(SourceFile {
                rel,
                abs: entry.path(),
                len: meta.len(),
                mtime_ns,
            });
        }
    }
    for (path, rel) in dirs {
        // Nested .gitignore layers apply only below their own directory, and
        // are popped again on the way out so siblings do not inherit them.
        let before = ignores.layers.len();
        ignores.push_dir(&path, &rel);
        walk_dir(&path, &rel, ignores, out)?;
        ignores.layers.truncate(before);
    }
    Ok(())
}

fn is_source(name: &str) -> bool {
    match name.rsplit_once('.') {
        Some((_, ext)) => SOURCE_EXTS.contains(&ext),
        None => false,
    }
}

// ---------------------------------------------------------------------------
// doc-comment extraction (self-supervised eval data)
// ---------------------------------------------------------------------------

/// A chunk's leading documentation, separated from its code.
///
/// This exists for CodeSearchNet-style evaluation: the doc comment becomes a
/// natural-language *query* and the code it documents becomes the *relevant
/// document*. The comment must be stripped from the indexed body, or lexical
/// retrieval trivially matches the query against its own text and every
/// score inflates.
#[derive(Debug, Clone)]
pub struct SplitDoc {
    /// The doc-comment text, comment markers removed.
    pub doc: String,
    /// The chunk body with the leading doc comment removed.
    pub code: String,
}

/// Split a chunk body into its leading doc comment and the remaining code.
/// Returns `None` when there is no leading documentation.
///
/// Recognized: `///` and `//!` (Rust), `/** ... */` (JS/TS/Java/C++),
/// `#` comment blocks immediately preceding code (Python/Ruby/shell), and
/// Python docstrings are deliberately *not* handled here — they follow the
/// `def`, not precede it, and stripping them is a per-language job for the
/// tree-sitter chunker (roadmap phase 3).
pub fn split_doc_comment(body: &str) -> Option<SplitDoc> {
    let lines: Vec<&str> = body.lines().collect();
    let mut doc_lines: Vec<String> = Vec::new();
    let mut idx = 0;

    // Block comment form: /** ... */ or /* ... */ at the top.
    if lines.first().map_or(false, |l| l.trim_start().starts_with("/*")) {
        let mut closed = false;
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim();
            let cleaned = t
                .trim_start_matches("/**")
                .trim_start_matches("/*")
                .trim_end_matches("*/")
                .trim_start_matches('*')
                .trim();
            if !cleaned.is_empty() {
                doc_lines.push(cleaned.to_string());
            }
            if t.ends_with("*/") {
                idx = i + 1;
                closed = true;
                break;
            }
        }
        if !closed {
            return None;
        }
    } else {
        // Line-comment form: a run of ///, //!, or # lines at the top.
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim_start();
            let stripped = if let Some(r) = t.strip_prefix("///") {
                Some(r)
            } else if let Some(r) = t.strip_prefix("//!") {
                Some(r)
            } else if t.starts_with("#!") || t.starts_with("#[") {
                // Shebang or Rust attribute, not documentation.
                None
            } else {
                t.strip_prefix('#')
            };
            match stripped {
                Some(text) => doc_lines.push(text.trim().to_string()),
                None => {
                    idx = i;
                    break;
                }
            }
            idx = i + 1;
        }
    }
    if doc_lines.is_empty() || idx >= lines.len() {
        return None;
    }
    // First paragraph only: that is the summary sentence CodeSearchNet uses;
    // later paragraphs are examples, arguments, and errata.
    let first_para: Vec<&String> = doc_lines
        .iter()
        .take_while(|l| !l.is_empty())
        .collect();
    let doc = first_para
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let code = lines[idx..].join("\n");
    if code.trim().is_empty() {
        return None;
    }
    Some(SplitDoc { doc, code })
}

// ---------------------------------------------------------------------------
// chunking
// ---------------------------------------------------------------------------

/// Keywords that introduce a named declaration worth its own chunk.
const DECL_KEYWORDS: &[&str] = &[
    "fn",
    "def",
    "function",
    "func",
    "class",
    "struct",
    "impl",
    "interface",
    "enum",
    "mod",
    "trait",
    "type",
    "package",
];

/// Leading modifiers skipped before looking for a declaration keyword.
const MODIFIERS: &[&str] = &[
    "pub",
    "export",
    "default",
    "async",
    "public",
    "private",
    "protected",
    "internal",
    "static",
    "final",
    "abstract",
    "unsafe",
    "extern",
    "const",
    "open",
    "override",
    "suspend",
    "inline",
];

/// If a line opens a declaration, return its name.
fn declaration_name(line: &str) -> Option<String> {
    let mut rest = line.trim_start();
    if rest.is_empty() || rest.starts_with("//") || rest.starts_with('#') && !rest.starts_with("#[")
    {
        return None;
    }
    // Skip modifiers, including `pub(crate)` forms.
    loop {
        let word_end = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let word = &rest[..word_end];
        if word.is_empty() {
            return None;
        }
        if DECL_KEYWORDS.contains(&word) {
            let after = rest[word_end..].trim_start();
            // `type` and `const` only count when they look like declarations,
            // not like `const x = ...` inside a body of another language.
            let name = read_ident(after)?;
            return if name.is_empty() { None } else { Some(name) };
        }
        if !MODIFIERS.contains(&word) {
            return None;
        }
        rest = rest[word_end..].trim_start();
        // Consume a `(crate)` / `(super)` visibility qualifier.
        if let Some(stripped) = rest.strip_prefix('(') {
            let close = stripped.find(')')?;
            rest = stripped[close + 1..].trim_start();
        }
    }
}

fn read_ident(s: &str) -> Option<String> {
    let end = s
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == ':'))
        .unwrap_or(s.len());
    let ident = s[..end].trim_end_matches(':').to_string();
    if ident.is_empty() {
        None
    } else {
        Some(ident)
    }
}

/// Split one file's text into chunks with line ranges.
pub fn chunk_text(rel: &str, text: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    // Declaration start lines (0-based).
    let mut starts: Vec<(usize, String)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some(name) = declaration_name(line) {
            starts.push((i, name));
        }
    }
    let mut chunks = Vec::new();
    if starts.is_empty() {
        push_split(&mut chunks, rel, &lines, 0, lines.len(), None);
        return chunks;
    }
    // A declaration's doc comment sits *above* it, so each chunk start is
    // pulled upward over contiguous comment/attribute lines. Without this
    // the documentation lands at the tail of the previous chunk — the one
    // place it helps neither retrieval nor the reader.
    let mut adj: Vec<(usize, String)> = Vec::with_capacity(starts.len());
    let mut floor = 0usize;
    for (i, (start, name)) in starts.iter().enumerate() {
        let mut at = *start;
        while at > floor && is_doc_or_attr(lines[at - 1]) {
            at -= 1;
        }
        adj.push((at, name.clone()));
        floor = starts.get(i).map(|(s, _)| *s).unwrap_or(at).max(at);
    }
    // Everything before the first declaration (imports, module docs) is its
    // own chunk: it is where `use`/`import` lines live and is searchable.
    if adj[0].0 > 0 {
        push_split(&mut chunks, rel, &lines, 0, adj[0].0, None);
    }
    for (idx, (start, name)) in adj.iter().enumerate() {
        let end = adj
            .get(idx + 1)
            .map(|(s, _)| *s)
            .unwrap_or_else(|| lines.len());
        push_split(&mut chunks, rel, &lines, *start, end, Some(name.clone()));
    }
    chunks
}

/// Lines that document or annotate the declaration below them.
fn is_doc_or_attr(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("///")
        || t.starts_with("//!")
        || t.starts_with("//")
        || t.starts_with('*')
        || t.starts_with("/*")
        || t.starts_with("#[")
        || t.starts_with('@')
        || (t.starts_with('#') && !t.starts_with("#!"))
}

/// Append `lines[start..end]` as one or more chunks, splitting overlong runs.
fn push_split(
    out: &mut Vec<Chunk>,
    rel: &str,
    lines: &[&str],
    start: usize,
    end: usize,
    name: Option<String>,
) {
    let mut at = start;
    while at < end {
        let stop = (at + MAX_CHUNK_LINES).min(end);
        let body = lines[at..stop].join("\n");
        if !body.trim().is_empty() {
            let mut body = body;
            if body.len() > MAX_CHUNK_BYTES {
                let cut = body
                    .char_indices()
                    .take_while(|(i, _)| *i <= MAX_CHUNK_BYTES)
                    .last()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                body.truncate(cut);
            }
            out.push(Chunk {
                path: rel.to_string(),
                start_line: at + 1,
                end_line: stop,
                // Only the first slice of a split declaration keeps the name.
                name: if at == start { name.clone() } else { None },
                body,
            });
        }
        at = stop;
    }
}

/// Order-sensitive fingerprint of a walked file list: any added, removed,
/// renamed, resized, or touched file changes it. `walk` returns files
/// sorted, so the same tree always fingerprints the same.
pub fn fingerprint(files: &[SourceFile]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for f in files {
        feed(f.rel.as_bytes());
        feed(&f.len.to_le_bytes());
        feed(&f.mtime_ns.to_le_bytes());
        feed(&[0xFE]);
    }
    h
}

/// Read and chunk every source file under `root`.
pub fn collect_chunks(root: &Path) -> anyhow::Result<Vec<Chunk>> {
    Ok(chunk_files(&walk(root)?))
}

/// Chunk an already-walked file list, so a caller that needs both the file
/// count and the chunks does not walk the tree twice.
///
/// Files are read and split in parallel — they are independent, and on a
/// large tree this is otherwise a serial I/O-plus-string-scanning phase.
/// `files` is sorted, and the per-file results are concatenated in that same
/// order, so doc_id assignment stays deterministic.
pub fn chunk_files(files: &[SourceFile]) -> Vec<Chunk> {
    let per_file: Vec<Vec<Chunk>> = files
        .par_iter()
        .map(|file| match fs::read_to_string(&file.abs) {
            Ok(text) => chunk_text(&file.rel, &text),
            // Binary or non-UTF-8: skip it, as the walker's extension filter
            // cannot rule this out on its own.
            Err(_) => Vec::new(),
        })
        .collect();
    // `collect` on an indexed parallel iterator preserves input order, so
    // flattening here yields the same sequence a serial walk would.
    per_file.into_iter().flatten().collect()
}

/// Chunk a tree into `InputDoc`s, de-duplicating ids defensively.
pub fn collect_docs(root: &Path) -> anyhow::Result<Vec<InputDoc>> {
    docs_from_chunks(collect_chunks(root)?)
}

/// Turn chunks into indexable documents, keeping ids unique.
pub fn docs_from_chunks(chunks: Vec<Chunk>) -> anyhow::Result<Vec<InputDoc>> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut docs = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let mut doc = chunk.into_doc();
        let count = seen.entry(doc.id.clone()).or_insert(0);
        if *count > 0 {
            doc.id = format!("{}#{}", doc.id, count);
        }
        *count += 1;
        docs.push(doc);
    }
    Ok(docs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_carry_line_ranges() {
        let text = "use std::fs;\n\nfn alpha() {\n    1\n}\n\nfn beta() {\n    2\n}\n";
        let chunks = chunk_text("src/x.rs", text);
        assert_eq!(chunks.len(), 3, "preamble + two fns: {chunks:?}");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[1].name.as_deref(), Some("alpha"));
        assert_eq!(chunks[1].start_line, 3);
        assert_eq!(chunks[2].name.as_deref(), Some("beta"));
        assert_eq!(chunks[2].id(), "src/x.rs:7-9");
    }

    #[test]
    fn doc_comment_travels_with_its_declaration() {
        let text = "use std::fs;\n\n/// Adds one.\n#[inline]\nfn alpha(x: u32) -> u32 {\n    x + 1\n}\n";
        let chunks = chunk_text("src/x.rs", text);
        assert_eq!(chunks.len(), 2, "{chunks:?}");
        assert!(!chunks[0].body.contains("Adds one"), "doc stuck in preamble");
        let alpha = &chunks[1];
        assert_eq!(alpha.name.as_deref(), Some("alpha"));
        assert!(alpha.body.starts_with("/// Adds one."), "{}", alpha.body);
        assert_eq!(alpha.start_line, 3);
        // And split_doc_comment can now lift it back out.
        let split = split_doc_comment(&alpha.body).unwrap();
        assert_eq!(split.doc, "Adds one.");
        assert!(split.code.contains("fn alpha"));
        assert!(!split.code.contains("#[inline]") || split.code.starts_with("#["));
    }

    #[test]
    fn parallel_chunking_preserves_file_order() {
        // doc_ids are assigned in chunk order, and the embedding sidecar is
        // keyed positionally by doc_id, so this ordering is load-bearing.
        let files = walk(Path::new("src")).unwrap();
        assert!(files.len() > 5);
        let chunks = chunk_files(&files);
        let mut seen: Vec<&str> = Vec::new();
        for chunk in &chunks {
            if seen.last() != Some(&chunk.path.as_str()) {
                assert!(
                    !seen.contains(&chunk.path.as_str()),
                    "{} is interleaved, not contiguous",
                    chunk.path
                );
                seen.push(&chunk.path);
            }
        }
        let mut sorted = seen.clone();
        sorted.sort();
        assert_eq!(seen, sorted, "files must stay in walk order");
        // And it must be stable across runs.
        let again = chunk_files(&files);
        let ids: Vec<String> = chunks.iter().map(|c| c.id()).collect();
        let ids_again: Vec<String> = again.iter().map(|c| c.id()).collect();
        assert_eq!(ids, ids_again);
    }

    #[test]
    fn id_roundtrips() {
        let (path, start, end) = parse_id("a/b/c.rs:10-20").unwrap();
        assert_eq!((path, start, end), ("a/b/c.rs", 10, 20));
    }

    #[test]
    fn declaration_forms() {
        assert_eq!(declaration_name("pub async fn run()").as_deref(), Some("run"));
        assert_eq!(
            declaration_name("pub(crate) struct Thing {").as_deref(),
            Some("Thing")
        );
        assert_eq!(declaration_name("    def handle(self):").as_deref(), Some("handle"));
        assert_eq!(
            declaration_name("export default class App {").as_deref(),
            Some("App")
        );
        assert_eq!(declaration_name("// fn commented()"), None);
        assert_eq!(declaration_name("let x = 1;"), None);
    }

    #[test]
    fn splits_doc_comments_from_code() {
        let rust = "/// Compute the retry backoff.\n///\n/// Doubles each attempt.\npub fn backoff(n: u32) -> u64 {\n    1 << n\n}\n";
        let split = split_doc_comment(rust).unwrap();
        assert_eq!(split.doc, "Compute the retry backoff.");
        assert!(split.code.starts_with("pub fn backoff"));
        assert!(!split.code.contains("Doubles"), "comment must be stripped");

        let block = "/** Parse a header value. */\nfunction parse(h) {}\n";
        let split = split_doc_comment(block).unwrap();
        assert_eq!(split.doc, "Parse a header value.");
        assert!(split.code.starts_with("function parse"));

        assert!(split_doc_comment("fn plain() {}\n").is_none());
        assert!(split_doc_comment("#!/bin/sh\necho hi\n").is_none());
        // Comment with no code after it documents nothing.
        assert!(split_doc_comment("/// orphan comment\n").is_none());
    }

    #[test]
    fn globs_and_ignores() {
        assert!(glob_match("*.log", "debug.log"));
        assert!(!glob_match("*.log", "a/debug.log"));
        assert!(glob_match("**/*.log", "a/b/debug.log"));
        assert!(glob_match("build", "build"));

        let mut set = IgnoreSet::default();
        set.layers.push((
            String::new(),
            IgnoreSet::parse_layer("", "target/\n*.tmp\n!keep.tmp\n"),
        ));
        assert!(set.is_ignored("target", true));
        assert!(!set.is_ignored("target", false), "dir-only rule");
        assert!(set.is_ignored("src/a.tmp", false));
        assert!(!set.is_ignored("src/keep.tmp", false), "negation wins");
    }
}
