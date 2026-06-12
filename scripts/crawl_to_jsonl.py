#!/usr/bin/env python3
"""Crawl a directory tree and emit a JSONL corpus for high-performance-search-engine.

Each text-like file becomes one document:
    {"id": "<path>", "title": "<path relative to root>", "body": "<content>"}

Only files with an allowlisted text extension are read. Noisy/binary/system
directories are skipped. File content is truncated to keep documents bounded.

Usage:
    python3 scripts/crawl_to_jsonl.py <root_dir> <out.jsonl> [max_docs]
"""

import json
import os
import sys

TEXT_EXTENSIONS = {
    ".c", ".cc", ".cfg", ".conf", ".cpp", ".css", ".csv", ".go", ".h", ".hpp",
    ".html", ".ini", ".java", ".js", ".json", ".jsx", ".kt", ".log", ".lua",
    ".m", ".md", ".mjs", ".php", ".pl", ".py", ".r", ".rb", ".rs", ".rst",
    ".scala", ".sh", ".sql", ".svelte", ".swift", ".tex", ".toml", ".ts",
    ".tsx", ".txt", ".vue", ".xml", ".yaml", ".yml", ".zsh",
}

SKIP_DIRS = {
    ".Trash", ".cache", ".cargo", ".conda", ".cpan", ".docker", ".gem",
    ".git", ".gradle", ".hg", ".ivy2", ".m2", ".npm", ".nvm", ".pyenv",
    ".rustup", ".svn", ".venv", ".vscode", "Applications", "Library",
    "Movies", "Music", "Pictures", "Public", "__pycache__", "bower_components",
    "dist", "node_modules", "site-packages", "target", "venv",
}

MAX_FILE_BYTES = 200_000   # skip huge files entirely
MAX_BODY_CHARS = 20_000    # truncate long documents
DEFAULT_MAX_DOCS = 150_000


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    root = os.path.abspath(os.path.expanduser(sys.argv[1]))
    out_path = sys.argv[2]
    max_docs = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MAX_DOCS

    num_docs = 0
    total_bytes = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                if num_docs >= max_docs:
                    print(f"reached max_docs={max_docs}")
                    print(f"wrote {num_docs} docs, {total_bytes} body bytes -> {out_path}")
                    return
                ext = os.path.splitext(name)[1].lower()
                if ext not in TEXT_EXTENSIONS:
                    continue
                path = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(path) > MAX_FILE_BYTES:
                        continue
                    with open(path, "r", encoding="utf-8", errors="strict") as f:
                        body = f.read(MAX_BODY_CHARS)
                except (OSError, UnicodeDecodeError, ValueError):
                    continue  # unreadable or not really text
                if not body.strip():
                    continue
                rel = os.path.relpath(path, root)
                out.write(
                    json.dumps({"id": path, "title": rel, "body": body}) + "\n"
                )
                num_docs += 1
                total_bytes += len(body)

    print(f"wrote {num_docs} docs, {total_bytes} body bytes -> {out_path}")


if __name__ == "__main__":
    main()
