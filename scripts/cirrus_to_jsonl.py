#!/usr/bin/env python3
"""Convert a Wikimedia CirrusSearch content dump to high-performance-search-engine JSONL.

CirrusSearch content dumps are newline-delimited JSON in Elasticsearch bulk
format: alternating action lines ({"index": {"_id": ...}}) and document
lines ({"title": ..., "text": ..., ...}). The `text` field is already plain
text (no wikitext), which maps directly onto our {"id", "title", "body"}.

Usage:
    python3 scripts/cirrus_to_jsonl.py dump.json.gz out.jsonl [max_docs]
    gunzip -c dump.json.gz | python3 scripts/cirrus_to_jsonl.py - out.jsonl [max_docs]
"""

import gzip
import json
import sys


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out_path = sys.argv[1], sys.argv[2]
    max_docs = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if src == "-":
        stream = sys.stdin.buffer
    elif src.endswith(".gz"):
        stream = gzip.open(src, "rb")
    else:
        stream = open(src, "rb")

    num_docs = 0
    pending_id = None
    out_stream = sys.stdout if out_path == "-" else open(out_path, "w", encoding="utf-8")
    with out_stream as out:
        for raw in stream:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "index" in record and "title" not in record:
                pending_id = record["index"].get("_id")
                continue
            title = record.get("title")
            text = record.get("text")
            if not title or not text:
                pending_id = None
                continue
            doc_id = str(pending_id) if pending_id is not None else f"doc-{num_docs}"
            pending_id = None
            out.write(
                json.dumps(
                    {"id": doc_id, "title": title, "body": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
            num_docs += 1
            if num_docs % 100000 == 0:
                print(f"{num_docs} docs...", file=sys.stderr)
            if max_docs is not None and num_docs >= max_docs:
                break

    print(f"wrote {num_docs} documents -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
