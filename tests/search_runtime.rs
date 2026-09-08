//! CLI fallback contracts, without network access or model downloads.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    cache: PathBuf,
    hf: PathBuf,
    index: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("repo");
        let cache = temp.path().join("cache");
        let hf = temp.path().join("hf");
        let index = temp.path().join("index");
        std::fs::create_dir_all(root.join("server")).unwrap();
        std::fs::write(
            root.join("server/inside.toml"),
            "permission_validation = true\n",
        )
        .unwrap();
        std::fs::write(root.join("outside.toml"), "permission_workflow = true\n").unwrap();
        let manifest = high_performance_search_engine::codeindex::RepoIndexer::new(
            &root,
            &index,
            high_performance_search_engine::codeindex::BuildOpts {
                embed: false,
                segmented: false,
                quiet: true,
                ..Default::default()
            },
        )
        .unwrap()
        .build()
        .unwrap();
        // Synthetic document vectors make CLI initialize its query encoder.
        // No inference test relies on their meaning.
        let mut vector = vec![0.0; 768];
        vector[0] = 1.0;
        high_performance_search_engine::embeddings::write_f16(
            &index,
            768,
            &vec![vector; manifest.num_docs],
        )
        .unwrap();
        // Fully cached but invalid model: deterministic initialization failure,
        // with no calls to the network and no dependency on the user's cache.
        let repo = hf.join("hub/models--nomic-ai--CodeRankEmbed");
        let snapshot = repo.join("snapshots/fixture");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), "fixture").unwrap();
        for file in ["config.json", "tokenizer.json", "model.safetensors"] {
            std::fs::write(snapshot.join(file), "invalid fixture").unwrap();
        }
        Self {
            _temp: temp,
            root,
            cache,
            hf,
            index,
        }
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_hips"));
        cmd.env("CSEARCH_CACHE_DIR", &self.cache)
            .env("HF_HOME", &self.hf)
            .env("HIPS_NO_LOG", "1");
        cmd
    }

    fn search(&self, mode: &str, root: &Path, index: Option<&Path>, json: bool) -> Output {
        let mut cmd = self.command();
        cmd.args([
            "search",
            "--query",
            "permission",
            "--top-k",
            "10",
            "--mode",
            mode,
        ])
        .arg("--root")
        .arg(root);
        if let Some(index) = index {
            cmd.arg("--index").arg(index);
        }
        if json {
            cmd.arg("--json");
        }
        cmd.output().unwrap()
    }
}

#[test]
fn initialization_fallback_is_visible_and_keeps_the_same_index_and_paths() {
    let f = Fixture::new();
    for root in [&f.root, &f.root.join("server")] {
        for json in [false, true] {
            let lexical = f.search("bm25", root, Some(&f.index), json);
            let hybrid = f.search("hybrid", root, Some(&f.index), json);
            assert!(lexical.status.success());
            assert!(
                hybrid.status.success(),
                "{}",
                String::from_utf8_lossy(&hybrid.stderr)
            );
            assert_eq!(hybrid.stdout, lexical.stdout);
            let stderr = String::from_utf8_lossy(&hybrid.stderr);
            assert!(stderr.contains("lexical BM25"), "{stderr}");
            #[cfg(feature = "semantic")]
            assert!(
                stderr.contains("parse CodeRankEmbed config.json"),
                "{stderr}"
            );
            assert!(!String::from_utf8_lossy(&hybrid.stdout).contains("NaN"));
            assert!(!hybrid.stdout.is_empty());
        }
    }
}

#[test]
fn explicit_semantic_and_rerank_fail_without_results_when_encoder_is_unavailable() {
    let f = Fixture::new();
    for mode in ["semantic", "rerank"] {
        let output = f.search(mode, &f.root, Some(&f.index), true);
        assert!(!output.status.success());
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("cannot proceed"), "{error}");
    }
}

#[test]
fn missing_vectors_fallback_is_visible_without_verbose() {
    let f = Fixture::new();
    std::fs::remove_file(f.index.join("embeddings.bin")).unwrap();
    let output = f.search("hybrid", &f.root, Some(&f.index), true);
    assert!(output.status.success());
    assert!(!output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("lexical BM25"));
}

#[test]
fn ancestor_index_resolution_is_identical_for_fallback_and_lexical() {
    let f = Fixture::new();
    // Build at the hashed default location, then install synthetic vectors.
    let out = f
        .command()
        .args(["index-repo", "--lexical", "--root"])
        .arg(&f.root)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let index = std::fs::read_dir(&f.cache)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.join("repo.json").exists())
        .unwrap();
    // This index may be segmented; leaving it lexical verifies the no-vector
    // fallback path through automatic ancestor resolution.
    let root = f.root.join("server");
    let hybrid = f.search("hybrid", &root, None, false);
    let lexical = f.search("bm25", &root, None, false);
    assert!(hybrid.status.success());
    assert_eq!(hybrid.stdout, lexical.stdout);
    let text = String::from_utf8_lossy(&hybrid.stdout);
    assert!(text.contains("inside.toml:"), "{text}");
    assert!(text.contains("../outside.toml:"), "{text}");
    assert!(index.starts_with(&f.cache));

    // An unindexed sibling repository must never inherit this index.
    let sibling = f.root.with_file_name("other");
    std::fs::create_dir(&sibling).unwrap();
    let output = f.search("hybrid", &sibling, None, false);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no index"));
}

#[cfg(all(target_os = "macos", feature = "semantic"))]
#[test]
fn cache_override_rejects_empty_file_and_unwritable_paths_with_exit_one() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new();
    let file = f._temp.path().join("file");
    std::fs::write(&file, "occupied").unwrap();
    let denied = f._temp.path().join("denied");
    std::fs::create_dir(&denied).unwrap();
    std::fs::set_permissions(&denied, std::fs::Permissions::from_mode(0o555)).unwrap();
    for dir in [Path::new(""), &file, &denied] {
        if dir == denied && unsafe { libc::geteuid() } == 0 {
            continue;
        }
        let output = f
            .command()
            .env("HIPS_COREML_CACHE_DIR", dir)
            .arg("--version")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("HIPS_COREML_CACHE_DIR"), "{error}");
        if !dir.as_os_str().is_empty() {
            assert!(error.contains(dir.to_str().unwrap()), "{error}");
        }
    }
    std::fs::set_permissions(&denied, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(all(target_os = "macos", feature = "semantic"))]
#[test]
fn cache_override_preserves_repository_and_model_cache_selection() {
    let f = Fixture::new();
    let dir = f._temp.path().join("relocated cache");
    let output = f
        .command()
        .env("HIPS_COREML_CACHE_DIR", &dir)
        .args(["search", "--query", "permission", "--json", "--index"])
        .arg(&f.index)
        .arg("--root")
        .arg(&f.root)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(dir.is_dir());
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "probe must be removed"
    );
    assert_eq!(
        output.stdout,
        f.search("bm25", &f.root, Some(&f.index), true).stdout
    );
    // Reading this exact fixture proves HF_HOME was not accidentally relocated.
    assert!(String::from_utf8_lossy(&output.stderr).contains("parse CodeRankEmbed config.json"));
}

/// Real CoreML predictions under the invoking process's sandbox. Explicitly
/// opt in with local model paths; never download 550 MB in the normal suite.
#[cfg(all(target_os = "macos", feature = "semantic"))]
#[test]
#[ignore = "requires HIPS_TEST_COREML_MODELS and HIPS_TEST_HF_HOME with cached CodeRankEmbed"]
fn native_coreml_cache_relocation_never_returns_invalid_hybrid_results() {
    let models = std::env::var_os("HIPS_TEST_COREML_MODELS").expect("set HIPS_TEST_COREML_MODELS");
    let hf = std::env::var_os("HIPS_TEST_HF_HOME").expect("set HIPS_TEST_HF_HOME");
    let f = Fixture::new();
    std::fs::create_dir_all(&f.cache).unwrap();
    std::os::unix::fs::symlink(models, f.cache.join("coreml")).unwrap();
    let relocated = f._temp.path().join("coreml-cache");
    for units in ["cpu-and-ane", "cpu"] {
        let output = f
            .command()
            .env("HF_HOME", &hf)
            .env("HIPS_COREML_CACHE_DIR", &relocated)
            .env("HIPS_COREML_COMPUTE_UNITS", units)
            .args([
                "search",
                "--query",
                "permission validation",
                "--json",
                "-v",
                "--index",
            ])
            .arg(&f.index)
            .arg("--root")
            .arg(f.root.join("server"))
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stderr}");
        assert!(
            stderr.contains("CoreML compute units:"),
            "CoreML must be attempted: {stderr}"
        );
        assert!(stderr.contains(relocated.to_str().unwrap()), "{stderr}");
        assert!(
            !stderr.contains("using lexical BM25"),
            "must retain semantic execution: {stderr}"
        );
        assert!(
            !stderr.contains("in create_directories: Operation not permitted"),
            "{stderr}"
        );
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let hit: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(hit["score"].as_f64().is_some_and(f64::is_finite), "{hit}");
        }
        assert!(!output.stdout.is_empty());
        assert!(relocated
            .join("Library/Caches/hips/com.apple.e5rt.e5bundlecache")
            .is_dir());
    }
}

#[cfg(all(target_os = "macos", feature = "semantic"))]
#[test]
#[ignore = "requires HIPS_TEST_HF_HOME with cached CodeRankEmbed"]
fn native_candle_metal_request_handles_an_inaccessible_gpu_without_panicking() {
    let hf = std::env::var_os("HIPS_TEST_HF_HOME").expect("set HIPS_TEST_HF_HOME");
    let f = Fixture::new();
    let output = f
        .command()
        .env("HF_HOME", hf)
        .env("HIPS_ENCODER", "candle")
        .env("HPS_EMBED_DEVICE", "metal")
        .args([
            "search",
            "--query",
            "permission validation",
            "--json",
            "-v",
            "--index",
        ])
        .arg(&f.index)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(!stderr.contains("using lexical BM25"), "{stderr}");
    assert!(
        stderr.contains("CodeRankEmbed device: Metal") || stderr.contains("falling back to CPU"),
        "{stderr}"
    );
    assert!(!output.stdout.is_empty());
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let hit: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(hit["score"].as_f64().is_some_and(f64::is_finite), "{hit}");
    }
}

/// Feed several calls to the actual stdio server; parsing every output line
/// also catches native runtime diagnostics leaking into protocol stdout.
fn mcp_calls(mut cmd: Command, calls: &[serde_json::Value]) -> (Output, Vec<serde_json::Value>) {
    use std::io::Write;
    use std::process::Stdio;
    cmd.env_remove("HIPS_PRELOAD");
    let mut child = cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for (id, args) in calls.iter().enumerate() {
        writeln!(stdin, "{}", serde_json::json!({"jsonrpc":"2.0", "id":id, "method":"tools/call", "params":args})).unwrap();
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    let responses = String::from_utf8_lossy(&output.stdout).lines().map(|line|
        serde_json::from_str(line).unwrap_or_else(|e| panic!("non-protocol stdout: {e}: {line}"))
    ).collect();
    (output, responses)
}

impl Fixture {
    fn mcp_command(&self) -> Command {
        let path = self.index.join("repo.json");
        let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest["embedded"] = serde_json::json!(true);
        std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let mut cmd = self.command();
        cmd.args(["mcp", "--no-watch", "--root"]).arg(&self.root).arg("--index").arg(&self.index);
        cmd
    }
}

#[cfg(feature = "semantic")]
#[test]
fn mcp_encoder_failure_is_an_error_and_explicit_lexical_recovery_preserves_scope() {
    use serde_json::json;
    let f = Fixture::new();
    let (output, responses) = mcp_calls(f.mcp_command(), &[
        json!({"name":"search_code", "arguments":{"query":"permission", "mode":"hybrid", "verbose":true}}),
        json!({"name":"index_status", "arguments":{"verbose":true}}),
        json!({"name":"search_code", "arguments":{"query":"permission", "mode":"lexical", "verbose":true}}),
    ]);
    assert!(output.status.success(), "server should remain alive after tool failure");
    assert_eq!(responses.len(), 3);
    let failed = &responses[0]["result"];
    assert_eq!(failed["isError"], true);
    assert!(failed["content"][0]["text"].as_str().unwrap().contains("parse CodeRankEmbed config.json"));
    let debug = &failed["structuredContent"]["diagnostics"];
    assert_eq!(debug["last_search"]["success"], false);
    assert_eq!(debug["last_search"]["effective_mode"], serde_json::Value::Null);
    assert_eq!(debug["encoder"]["state"], "not_loaded");
    assert_eq!(debug, &responses[1]["result"]["structuredContent"]["diagnostics"]);
    let recovered = &responses[2]["result"];
    assert_eq!(recovered["isError"], false);
    let debug = &recovered["structuredContent"]["diagnostics"];
    assert_eq!(debug["root"], f.root.canonicalize().unwrap().to_str().unwrap());
    assert_eq!(debug["root"], debug["indexed_root"]);
    assert_eq!(debug["last_search"]["effective_mode"], "lexical");
    let hits = debug["last_search"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    for hit in hits { assert!(f.root.join(hit["path"].as_str().unwrap()).is_file()); }
}

#[test]
fn mcp_rejects_an_index_from_a_different_repository() {
    let f = Fixture::new();
    let mut cmd = f.mcp_command();
    // Change manifest root to an existing sibling. Loading this index would
    // otherwise render valid hit IDs against the wrong repository.
    let path = f.index.join("repo.json");
    let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["root"] = serde_json::json!(f.root.join("server"));
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    cmd.arg("--lexical");
    let output = cmd.output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("MCP index root mismatch"));
}

#[cfg(all(target_os = "macos", feature = "semantic"))]
#[test]
#[ignore = "requires HIPS_TEST_COREML_MODELS and HIPS_TEST_HF_HOME with cached CodeRankEmbed"]
fn native_mcp_reports_cache_failure_candle_recovery_and_query_reuse() {
    use serde_json::json;
    let f = Fixture::new();
    std::fs::create_dir_all(&f.cache).unwrap();
    std::os::unix::fs::symlink(std::env::var_os("HIPS_TEST_COREML_MODELS").expect("set HIPS_TEST_COREML_MODELS"), f.cache.join("coreml")).unwrap();
    let relocated = f._temp.path().join("coreml-home");
    // A file at the cache path reliably denies create_dir_all, including root
    // test runners. The override itself remains writable and passes startup.
    let blocked = relocated.join("Library/Caches/hips/com.apple.e5rt.e5bundlecache");
    std::fs::create_dir_all(blocked.parent().unwrap()).unwrap();
    std::fs::write(&blocked, "not a directory").unwrap();
    let mut cmd = f.mcp_command();
    cmd.env("HF_HOME", std::env::var_os("HIPS_TEST_HF_HOME").expect("set HIPS_TEST_HF_HOME"))
        .env("HIPS_COREML_CACHE_DIR", &relocated).env("HPS_EMBED_DEVICE", "cpu").env_remove("HIPS_ENCODER");
    let query = json!({"name":"search_code", "arguments":{"query":"permission validation", "mode":"semantic", "verbose":true}});
    let (output, responses) = mcp_calls(cmd, &[
        json!({"name":"index_status", "arguments":{"verbose":true}}), query.clone(), query,
    ]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(responses.len(), 3);
    assert_eq!(responses[0]["result"]["structuredContent"]["diagnostics"]["encoder"]["state"], "not_loaded");
    for (i, response) in responses.iter().enumerate().skip(1) {
        let result = &response["result"];
        assert_eq!(result["isError"], false, "{result}");
        let debug = &result["structuredContent"]["diagnostics"];
        assert_eq!(debug["encoder"]["backend"], "candle");
        assert_eq!(debug["encoder"]["candle_device"], "cpu");
        assert_eq!(debug["encoder"]["warmup_validated"], true);
        assert!(debug["encoder"]["fallbacks"][0].as_str().unwrap().contains(blocked.to_str().unwrap()));
        assert!(result["content"][0]["text"].as_str().unwrap().contains("CoreML unavailable"));
        assert_eq!(debug["last_search"]["effective_mode"], "semantic");
        assert_eq!(debug["last_search"]["query_cache_hit"], i == 2);
        for hit in debug["last_search"]["hits"].as_array().unwrap() {
            assert!(hit["score"].as_f64().unwrap().is_finite());
            assert!(f.root.join(hit["path"].as_str().unwrap()).is_file());
        }
    }
    // A legacy invalid document vector must never turn into a plausible
    // finite score through the optimized FP16 decoder.
    high_performance_search_engine::embeddings::write_f16(&f.index, 768, &vec![vec![f32::NAN; 768]; 2]).unwrap();
    let mut cmd = f.mcp_command();
    cmd.env("HF_HOME", std::env::var_os("HIPS_TEST_HF_HOME").unwrap())
        .env("HIPS_ENCODER", "candle").env("HPS_EMBED_DEVICE", "cpu");
    let (output, responses) = mcp_calls(cmd, &[
        json!({"name":"search_code", "arguments":{"query":"permission", "mode":"semantic", "verbose":true}}),
        json!({"name":"search_code", "arguments":{"query":"permission", "mode":"lexical", "verbose":true}}),
    ]);
    assert!(output.status.success());
    assert_eq!(responses[0]["result"]["isError"], false, "{}", responses[0]);
    let debug = &responses[0]["result"]["structuredContent"]["diagnostics"];
    assert_eq!(debug["last_search"]["repaired_embeddings"], 2);
    assert_eq!(debug["invalid_live_embeddings"], 0);
    assert!(responses[0]["result"]["content"][0]["text"].as_str().unwrap().contains("Reindexed 2"));
    assert_eq!(responses[1]["result"]["isError"], false);
}

#[test]
fn mcp_lexical_flag_keeps_an_existing_vector_index_lexical() {
    use serde_json::json;
    let f = Fixture::new();
    let mut cmd = f.mcp_command();
    cmd.arg("--lexical");
    let (output, responses) = mcp_calls(cmd, &[
        json!({"name":"search_code", "arguments":{"query":"permission", "verbose":true}}),
        json!({"name":"search_code", "arguments":{"query":"permission", "mode":"semantic", "verbose":true}}),
    ]);
    assert!(output.status.success());
    assert_eq!(responses[0]["result"]["isError"], false);
    let debug = &responses[0]["result"]["structuredContent"]["diagnostics"];
    assert_eq!(debug["index_has_embeddings"], true);
    assert_eq!(debug["semantic_enabled"], false);
    assert_eq!(debug["last_search"]["effective_mode"], "lexical");
    assert_ne!(debug["encoder"]["state"], "ready");
    assert_eq!(responses[1]["result"]["isError"], true);
}

#[cfg(feature = "semantic")]
#[test]
fn mcp_failed_document_repair_is_visible_and_lexical_still_works() {
    use serde_json::json;
    let f = Fixture::new();
    high_performance_search_engine::embeddings::write_f16(&f.index, 768, &vec![vec![f32::NAN; 768]; 2]).unwrap();
    let (output, responses) = mcp_calls(f.mcp_command(), &[
        json!({"name":"search_code", "arguments":{"query":"permission", "mode":"hybrid", "verbose":true}}),
        json!({"name":"search_code", "arguments":{"query":"permission", "mode":"lexical", "verbose":true}}),
    ]);
    assert!(output.status.success());
    assert_eq!(responses[0]["result"]["isError"], true);
    assert!(responses[0]["result"]["content"][0]["text"].as_str().unwrap().contains("repair of 2 invalid document embeddings failed"));
    assert_eq!(responses[1]["result"]["isError"], false, "{responses:#?} stderr={}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(responses[1]["result"]["structuredContent"]["diagnostics"]["last_search"]["effective_mode"], "lexical");
}

#[cfg(all(target_os = "macos", feature = "semantic"))]
#[test]
#[ignore = "requires HIPS_TEST_HF_HOME with cached CodeRankEmbed"]
fn native_mcp_repairs_missing_segment_vectors_and_reindexes_source_edits() {
    use high_performance_search_engine::codeindex::{BuildOpts, RepoIndexer};
    use serde_json::json;
    let f = Fixture::new();
    std::fs::remove_dir_all(&f.index).unwrap();
    RepoIndexer::new(&f.root, &f.index, BuildOpts {
        embed: false, segmented: true, quiet: true, ..Default::default()
    }).unwrap().build().unwrap();
    let command = || {
        let mut cmd = f.mcp_command();
        cmd.env("HF_HOME", std::env::var_os("HIPS_TEST_HF_HOME").expect("set HIPS_TEST_HF_HOME"))
            .env("HIPS_ENCODER", "candle").env("HPS_EMBED_DEVICE", "cpu");
        cmd
    };
    let (output, responses) = mcp_calls(command(), &[
        json!({"name":"index_status", "arguments":{"verbose":true}}),
        json!({"name":"search_code", "arguments":{"query":"permission", "mode":"semantic", "verbose":true}}),
    ]);
    assert!(output.status.success());
    assert_eq!(responses[0]["result"]["structuredContent"]["diagnostics"]["invalid_live_embeddings"], 2);
    assert_eq!(responses[1]["result"]["isError"], false, "{} stderr={}", responses[1], String::from_utf8_lossy(&output.stderr));
    let debug = &responses[1]["result"]["structuredContent"]["diagnostics"];
    assert_eq!(debug["invalid_live_embeddings"], 0);
    assert_eq!(debug["last_search"]["repaired_embeddings"], 2);
    // Repair retained the source root and document IDs. A later source edit
    // still replaces the changed chunk and its vector through normal reindex.
    std::fs::write(f.root.join("server/inside.toml"), "financial_validation_new_code = true\n").unwrap();
    let (output, responses) = mcp_calls(command(), &[
        json!({"name":"reindex", "arguments":{}}),
        json!({"name":"search_code", "arguments":{"query":"financial_validation_new_code", "mode":"hybrid", "verbose":true, "include_snippet":true}}),
    ]);
    assert!(output.status.success());
    for response in &responses { assert_eq!(response["result"]["isError"], false, "{response}"); }
    let text = responses[1]["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("financial_validation_new_code = true"), "{text}");
    assert_eq!(responses[1]["result"]["structuredContent"]["diagnostics"]["invalid_live_embeddings"], 0);
}
