//! Fast CirrusSearch-dump → JSONL converter (Rust + simd-json), replacing
//! scripts/cirrus_to_jsonl.py when throughput matters: the Python version
//! tops out around 6 MB/s of output; this one keeps up with a fast mirror.
//!
//! ```sh
//! curl -s <dump.json.gz> | gunzip -c | \
//!   cargo run --release --example cirrus_convert -- out.jsonl [max_docs]
//! ```

use std::io::{BufRead, BufWriter, Write};

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Action {
    index: Option<ActionIndex>,
}

#[derive(Deserialize)]
struct ActionIndex {
    #[serde(rename = "_id")]
    id: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct CirrusDoc {
    title: Option<String>,
    text: Option<String>,
}

#[derive(Serialize)]
struct OutDoc<'a> {
    id: &'a str,
    title: &'a str,
    body: &'a str,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_path = args
        .next()
        .expect("usage: cirrus_convert OUT.jsonl [max_docs]");
    let max_docs: usize = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);

    let stdin = std::io::stdin();
    let mut input = std::io::BufReader::with_capacity(1 << 22, stdin.lock());
    let mut out = BufWriter::with_capacity(
        1 << 22,
        std::fs::File::create(&out_path).expect("failed to create output"),
    );

    let mut line: Vec<u8> = Vec::with_capacity(1 << 16);
    let mut pending_id: Option<String> = None;
    let mut num_docs = 0usize;
    while num_docs < max_docs {
        line.clear();
        let n = input.read_until(b'\n', &mut line).expect("read failed");
        if n == 0 {
            break;
        }
        if line.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        // Action lines are tiny and start with {"index":
        if line.starts_with(b"{\"index\"") {
            if let Ok(action) = simd_json::serde::from_slice::<Action>(&mut line.clone()) {
                pending_id = action.index.and_then(|i| i.id).map(|v| match v {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                });
            }
            continue;
        }
        let Ok(doc) = simd_json::serde::from_slice::<CirrusDoc>(&mut line) else {
            pending_id = None;
            continue;
        };
        let (Some(title), Some(text)) = (doc.title.as_deref(), doc.text.as_deref()) else {
            pending_id = None;
            continue;
        };
        if title.is_empty() || text.is_empty() {
            pending_id = None;
            continue;
        }
        let fallback = format!("doc-{num_docs}");
        let id = pending_id.take().unwrap_or(fallback);
        serde_json::to_writer(
            &mut out,
            &OutDoc {
                id: &id,
                title,
                body: text,
            },
        )
        .expect("write failed");
        out.write_all(b"\n").expect("write failed");
        num_docs += 1;
        if num_docs % 500_000 == 0 {
            eprintln!("{num_docs} docs...");
        }
    }
    out.flush().expect("flush failed");
    eprintln!("wrote {num_docs} documents -> {out_path}");
}
