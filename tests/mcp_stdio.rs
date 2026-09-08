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
    // Default output contains openable locations without snippets.
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
    assert!(!text.contains("```"), "snippets must be opt-in:\n{text}");
    let snippets = client.tool_text(6, "search_code", json!({
        "query":"search", "top_k":10, "include_snippet":true
    }));
    let bodies: Vec<&str> = snippets.split("```rust\n").skip(1)
        .map(|s| s.split("\n```").next().unwrap()).collect();
    assert_eq!(bodies.len(), 10, "all k hits get snippets: {snippets}");
    assert!(bodies.iter().all(|s| s.lines().count() <= 4), "{snippets}");

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

#[test]
fn answers_json_rpc_batches() {
    let cache = temp_cache("batch");
    let mut client = Client::start(&cache);
    client.call(
        1,
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1"}}),
    );

    // A batch of two requests answers with an array of two responses
    // (previously the whole batch was silently dropped and a batching
    // client would block forever).
    client.send(json!([
        {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
    ]));
    let mut line = String::new();
    client.stdout.read_line(&mut line).expect("read batch response");
    let response: Value = serde_json::from_str(&line).expect("valid JSON");
    let batch = response.as_array().expect("batch answers with an array");
    assert_eq!(batch.len(), 2);
    let ids: Vec<&Value> = batch.iter().map(|r| &r["id"]).collect();
    assert!(ids.contains(&&json!(2)) && ids.contains(&&json!(3)));

    // An empty batch is an invalid request, not silence.
    client.send(json!([]));
    let mut line = String::new();
    client.stdout.read_line(&mut line).expect("read error response");
    let response: Value = serde_json::from_str(&line).expect("valid JSON");
    assert_eq!(response["error"]["code"], -32600);

    // Still alive.
    let status = client.tool_text(4, "index_status", json!({}));
    assert!(status.contains("root:"), "{status}");

    std::fs::remove_dir_all(&cache).ok();
}

#[test]
fn verbose_diagnostics_are_opt_in_and_reject_wrong_roots_and_modes() {
    let cache = temp_cache("diagnostics");
    let mut client = Client::start(&cache);
    let tools = client.call(1, "tools/list", json!({}));
    for tool in tools["result"]["tools"].as_array().unwrap().iter().take(2) {
        assert_eq!(tool["inputSchema"]["properties"]["verbose"]["type"], "boolean");
        assert_eq!(tool["inputSchema"]["properties"]["verbose"]["default"], false);
    }
    let status = client.call(2, "tools/call", json!({"name":"index_status", "arguments":{"verbose":true}}));
    let debug = &status["result"]["structuredContent"]["diagnostics"];
    assert_eq!(debug["index_loaded"], false, "status must not build the index");
    assert_ne!(debug["encoder"]["state"], "ready");
    assert_eq!(debug["last_search"], Value::Null);
    let root = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")).canonicalize().unwrap();
    assert_eq!(debug["root"], root.to_str().unwrap());
    for args in [
        json!({"query":"tokenizer", "mode":"typo", "verbose":true}),
        json!({"query":"tokenizer", "expected_root":env!("CARGO_MANIFEST_DIR"), "verbose":true}),
        json!({"query":"tokenizer", "top_k":0}),
        json!({"query":"tokenizer", "verbose":"yes"}),
        json!({"query":"tokenizer", "mode":42}),
    ] {
        let response = client.call(3, "tools/call", json!({"name":"search_code", "arguments":args}));
        assert_eq!(response["result"]["isError"], true, "{response}");
    }
    let response = client.call(4, "tools/call", json!({"name":"search_code", "arguments":{
        "query":"tokenizer", "verbose":true, "expected_root":root, "top_k":2, "include_snippet":false
    }}));
    assert_eq!(response["result"]["isError"], false, "{response}");
    let debug = &response["result"]["structuredContent"]["diagnostics"];
    let search = &debug["last_search"];
    assert_eq!(search["effective_mode"], "lexical");
    assert_eq!(search["success"], true);
    assert_eq!(search["query_cache_hit"], false);
    assert!(search["total_ms"].as_f64().unwrap() >= search["retrieval_ms"].as_f64().unwrap());
    assert!(!search["hits"].as_array().unwrap().is_empty());
    for hit in search["hits"].as_array().unwrap() {
        assert!(root.join(hit["path"].as_str().unwrap()).is_file());
        assert!(hit["score"].as_f64().unwrap().is_finite());
    }
    let compact = client.call(5, "tools/call", json!({"name":"search_code", "arguments":{"query":"tokenizer"}}));
    assert!(compact["result"].get("structuredContent").is_none());
    assert_eq!(compact["result"]["content"].as_array().unwrap().len(), 1);
    assert!(!compact["result"]["content"][0]["text"].as_str().unwrap().contains("```"));
    let explicit = client.call(6, "tools/call", json!({"name":"search_code", "arguments":{
        "query":"tokenizer", "verbose":false, "include_snippet":false
    }}));
    assert_eq!(compact["result"], explicit["result"]);
    std::fs::remove_dir_all(&cache).ok();
}
