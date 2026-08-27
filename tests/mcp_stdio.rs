//! End-to-end test of the MCP stdio transport: spawn the real binary, speak
//! JSON-RPC to it over a pipe, and check the handshake, tool listing, and a
//! search that must resolve to an openable `path:line` location.
//!
//! Runs lexically (`--lexical`) so it needs no model download.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Client {
    fn start(cache: &std::path::Path) -> Self {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        let mut child = Command::new(env!("CARGO_BIN_EXE_hips"))
            .args(["mcp", "--root", root, "--lexical"])
            .env("CSEARCH_CACHE_DIR", cache)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn hips mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn send(&mut self, message: Value) {
        writeln!(self.stdin, "{message}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn call(&mut self, id: u32, method: &str, params: Value) -> Value {
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("read response");
        let response: Value = serde_json::from_str(&line).expect("valid JSON-RPC");
        assert_eq!(response["id"], id, "response id must match request");
        response
    }

    fn tool_text(&mut self, id: u32, name: &str, args: Value) -> String {
        let response = self.call(id, "tools/call", json!({"name": name, "arguments": args}));
        let result = &response["result"];
        assert_eq!(result["isError"], json!(false), "tool failed: {result}");
        result["content"][0]["text"].as_str().unwrap().to_string()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

fn temp_cache(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("csearch-mcp-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

#[test]
fn serves_the_mcp_handshake_and_searches_code() {
    let cache = temp_cache("main");
    let mut client = Client::start(&cache);

    let init = client.call(
        1,
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1"}}),
    );
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(init["result"]["serverInfo"]["name"], "hips");
    assert!(init["result"]["capabilities"]["tools"].is_object());

    // A notification carries no id and must not be answered; if the server
    // replied, the next read would return this stale response instead.
    client.send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));

    let tools = client.call(2, "tools/list", json!({}));
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["search_code", "index_status", "reindex"]);

    let text = client.tool_text(
        3,
        "search_code",
        json!({"query": "block max wand pivot threshold", "top_k": 3}),
    );
    // Every hit must be an openable location, and the snippet must be fenced.
    let location = text
        .lines()
        .find(|l| l.starts_with("1. "))
        .unwrap_or_else(|| panic!("no first hit in:\n{text}"));
    assert!(location.contains(".rs:"), "not a location: {location}");
    let range = location.rsplit(':').next().unwrap();
    let (start, end) = range.split_whitespace().next().unwrap().split_once('-').unwrap();
    assert!(
        start.parse::<usize>().unwrap() <= end.parse::<usize>().unwrap(),
        "bad line range: {location}"
    );
    assert!(text.contains("```rust"), "snippet not fenced:\n{text}");

    let status = client.tool_text(4, "index_status", json!({}));
    assert!(status.contains("chunks:"), "{status}");
    assert!(status.contains("lexical"), "{status}");

    std::fs::remove_dir_all(&cache).ok();
}

#[test]
fn reports_errors_without_dropping_the_connection() {
    let cache = temp_cache("errors");
    let mut client = Client::start(&cache);
    client.call(
        1,
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1"}}),
    );

    // Unknown method: a JSON-RPC error, not a crash.
    let response = client.call(2, "no/such/method", json!({}));
    assert_eq!(response["error"]["code"], -32601);

    // A failing tool reports in-band so the model can react to it.
    let response = client.call(
        3,
        "tools/call",
        json!({"name": "search_code", "arguments": {"query": "   "}}),
    );
    assert_eq!(response["result"]["isError"], json!(true));

    let response = client.call(4, "tools/call", json!({"name": "nope", "arguments": {}}));
    assert_eq!(response["result"]["isError"], json!(true));

    // Still alive and answering after all of that.
    let status = client.tool_text(5, "index_status", json!({}));
    assert!(status.contains("root:"), "{status}");

    std::fs::remove_dir_all(&cache).ok();
}

#[test]
fn path_glob_restricts_results() {
    let cache = temp_cache("glob");
    let mut client = Client::start(&cache);
    client.call(
        1,
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1"}}),
    );
    let text = client.tool_text(
        2,
        "search_code",
        json!({"query": "tokenize", "top_k": 5,
               "path_glob": "tokenizer.rs", "include_snippet": false}),
    );
    let hits: Vec<&str> = text
        .lines()
        .filter(|l| l.split_once(". ").is_some_and(|(n, _)| n.parse::<u32>().is_ok()))
        .collect();
    assert!(!hits.is_empty(), "no hits to check:\n{text}");
    for line in hits {
        assert!(line.contains("tokenizer.rs"), "leaked past the glob: {line}");
    }
    std::fs::remove_dir_all(&cache).ok();
}
