#!/usr/bin/env python3
"""Sample realistic queries from a JSONL corpus' titles.

Produces a mix: full titles, 2-word prefixes, and single salient words —
roughly matching real query-length distributions.

Usage: python3 scripts/make_queries.py corpus.jsonl out.txt [count]
"""
import json
import sys

STOP = set("a an and are as at be but by for if in into is it no not of on or the to with".split())

def main():
    src, out_path = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    queries = []
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 1499 != 0:  # spread samples across the corpus
                continue
            try:
                title = json.loads(line).get("title", "")
            except json.JSONDecodeError:
                continue
            words = [w.lower() for w in title.split() if w.isalnum() and w.lower() not in STOP]
            if not words:
                continue
            n = len(queries)
            if n % 3 == 0 and len(words) >= 2:
                queries.append(" ".join(words[:4]))      # full-ish title
            elif n % 3 == 1 and len(words) >= 2:
                queries.append(" ".join(words[:2]))      # 2-word
            else:
                queries.append(max(words, key=len))       # salient single word
            if len(queries) >= count:
                break
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("\n".join(queries) + "\n")
    print(f"wrote {len(queries)} queries -> {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
