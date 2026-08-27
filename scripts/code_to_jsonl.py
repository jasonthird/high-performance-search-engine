#!/usr/bin/env python3
"""Turn a source tree into JSONL chunks for the search engine.

This is a deliberately simple regex splitter, not a real parser: enough to
evaluate retrieval. Each function / class / struct / impl / module becomes
one document: {"id", "title", "body"}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DECL = re.compile(
    r"^(?P<indent>[ \t]*)(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?"
    r"(?:fn|def|function|class|struct|impl|interface|enum|mod|trait)\s+"
    r"(?P<name>[\w.]+)",
    re.MULTILINE,
)

SOURCE_EXT = {
    ".rs",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".java",
    ".rb",
    ".c",
    ".h",
    ".cpp",
    ".cc",
}
SKIP_DIRS = {
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".git",
    ".venv",
    "venv",
    "target",
    ".pnpm-store",
    "pdf_venv",
    ".playwright-mcp",
}


def chunks_from_file(path: Path) -> list[tuple[str, str, str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"skip {path}: {e}", file=sys.stderr)
        return []
    matches = list(DECL.finditer(text))
    if not matches:
        rel = str(path)
        return [(rel, rel, text)]
    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].rstrip() + "\n"
        name = m.group("name")
        ident = f"{path}::{name}"
        out.append((ident, ident, body))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="directory of source files")
    p.add_argument("out", type=Path, help="output JSONL path")
    args = p.parse_args()
    n = 0
    with args.out.open("w", encoding="utf-8") as w:
        for path in sorted(args.root.rglob("*")):
            if any(p in SKIP_DIRS for p in path.parts):
                continue
            if path.suffix.lower() not in SOURCE_EXT or not path.is_file():
                continue
            for ident, title, body in chunks_from_file(path):
                w.write(json.dumps({"id": ident, "title": title, "body": body}) + "\n")
                n += 1
    print(f"wrote {n} chunks to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
