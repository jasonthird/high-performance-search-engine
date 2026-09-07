---
name: hips
description: Primary code-search tool. Use it FIRST, before grep/ripgrep/find, for any question about where something is implemented, how a mechanism works, or which file/function is responsible — in any repository. Hybrid BM25 + CodeRankEmbed over declaration-sized chunks (and PDF pages) that keep their line numbers; one query returns ranked `path:line` locations and replaces several grep rounds. Fall back to ripgrep only for an exact literal you already know.
compatibility: Requires the `hips` binary (`cargo install --git https://github.com/jasonthird/high-performance-search-engine --features semantic`). First use in a repo builds an index under ~/.cache/csearch.
allowed-tools: Bash(hips *)
metadata:
  version: "0.5.2"
  argument-hint: <describe what the code does>
---

# Code navigation policy

For ANY question about where something is implemented, how a mechanism works,
or which file/function is responsible, your FIRST action is:

    hips search --root . --query "<describe what the code does>" --top-k 10

- It is a hybrid semantic+lexical search: natural-language descriptions and
  exact identifiers both work. Output is one ranked hit per line,
  `path:startLine-endLine` then the declaration name, nothing else:

      src/repo.rs:700-722    chunk_pdf
      src/repo.rs:663-675    pdf_text

  A `note:` line means the encoder was unavailable and results are BM25 only.
  `--lexical` forces BM25 (exact identifiers, no encoder load); `--json`
  gives one `{"id","path","start","end","name","score"}` object per line;
  `-v` adds scores, timing and stats.
- Open only the top 1-3 hits to confirm; do not fall back to grep/find
  unless hips returned nothing relevant.
- One good hips query usually replaces several grep rounds. Keep queries
  descriptive ("heal live counts after crash"), not keyword soup.

## Before running

If `hips` is not on PATH, install it; do not fall back to grep instead:

    cargo install --git https://github.com/jasonthird/high-performance-search-engine --features semantic

In Claude Code with the hips plugin, the index is built when the session
opens and kept fresh by a background watcher for as long as any session has
the repository open — do not run `index-repo`; a search issued right after
an edit waits for the watcher's rebuild (about a second). Outside that setup,
if a search says "no index for . yet":

    hips index-repo --root .

Builds once into `~/.cache/csearch/` (never inside the repo); later runs are
incremental and take under a second for unchanged code. Without a watcher,
re-run it after you edit files, or hits will point at old line numbers. Add
`--lexical` to skip the embedding model when you only need exact-word
matching. `hips status --root .` shows what is indexed, whether a watcher is
running, and which sessions hold it; `hips watch --root .` runs one by hand.

## Notes

- `--root .` from a subdirectory of an indexed repository finds the
  repository's index; hits are then printed relative to that subdirectory.
- Honours `.gitignore`; skips `target/`, `node_modules/`, `.venv/`, symlinks.
- Chunks are real declarations: 41 tree-sitter grammars (C, C++, Java, Go,
  Python, JS/TS, Rust, C#, Ruby, PHP, Swift, Kotlin, ... and Markdown by
  heading) cut files at functions, classes and methods, so a hit is one
  unit and its name is qualified (`path::Class::method`). Other files use
  a keyword heuristic; `.toml`/`.yaml` are indexed whole; PDFs per page
  (`report.pdf::page 7`). `hips chunks --file X` shows how a file is cut.
- Every search appends one JSON line to `~/.cache/csearch/usage.jsonl` for
  later analysis. `HIPS_NO_LOG=1` disables it.
