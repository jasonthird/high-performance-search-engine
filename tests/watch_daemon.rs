//! End-to-end test of the background watcher: spawn the real binary in
//! leased mode against a scratch repository, check that an edit is picked
//! up without anyone asking, that a search issued straight after the edit
//! sees it, that a second session shares the same watcher, and that the
//! watcher exits on its own once the last lease is gone.
//!
//! Runs lexically so it needs no model download.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hips-watch-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn hips(cache: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_hips"));
    cmd.env("CSEARCH_CACHE_DIR", cache)
        .env("HIPS_NO_LOG", "1")
        .stdin(Stdio::null());
    cmd
}

fn run(cache: &Path, args: &[&str]) -> (bool, String, String) {
    let out = hips(cache).args(args).output().expect("run hips");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn status_json(cache: &Path, root: &Path) -> serde_json::Value {
    let (ok, out, err) = run(
        cache,
        &["status", "--root", root.to_str().unwrap(), "--json"],
    );
    assert!(ok, "status failed: {err}");
    serde_json::from_str(&out).expect("status --json is one JSON object")
}

fn wait_until(what: &str, max: Duration, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + max;
    while Instant::now() < deadline {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("timed out waiting for {what}");
}

#[test]
fn leased_watcher_follows_edits_and_exits_with_the_last_session() {
    let cache = scratch("cache");
    let repo = scratch("repo");
    // `session start` only watches git work trees.
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(repo.join("src/deep")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "/// Parse the config file.\npub fn parse_config(text: &str) -> u32 {\n    text.len() as u32\n}\n",
    )
    .unwrap();
    let repo_s = repo.to_str().unwrap();

    // Session A opens the repo from a subdirectory: the watcher must cover
    // the git toplevel, and the lease must carry our (live) pid.
    let sub = repo.join("src/deep");
    let (ok, out, err) = run(
        &cache,
        &[
            "session",
            "start",
            "--root",
            sub.to_str().unwrap(),
            "--id",
            "session-a",
            "--lexical",
            "--grace-secs",
            "2",
            "--pid",
            &std::process::id().to_string(),
        ],
    );
    assert!(ok, "session start failed: {err}");
    assert!(
        out.contains("started"),
        "first start launches the watcher: {out}"
    );

    wait_until("initial build", Duration::from_secs(30), || {
        let s = status_json(&cache, &repo);
        s["watcher"]["state"] == "idle" && s["index"]["chunks"].as_u64().unwrap_or(0) > 0
    });
    let s = status_json(&cache, &repo);
    assert_eq!(s["watcher"]["leased"], true);
    assert_eq!(s["sessions"].as_array().unwrap().len(), 1);
    let watcher_pid = s["watcher"]["pid"].as_u64().unwrap();

    // Session B (a second agent on the same repo) attaches to the same watcher.
    let (ok, out, err) = run(
        &cache,
        &[
            "session",
            "start",
            "--root",
            repo_s,
            "--id",
            "session-b",
            "--lexical",
            "--pid",
            &std::process::id().to_string(),
        ],
    );
    assert!(ok, "second start failed: {err}");
    assert!(
        out.contains("already running"),
        "second start must attach: {out}"
    );
    assert_eq!(
        status_json(&cache, &repo)["watcher"]["pid"]
            .as_u64()
            .unwrap(),
        watcher_pid
    );

    // Search from the subdirectory resolves to the ancestor index and shows
    // paths relative to where the caller stands.
    let (ok, out, err) = run(
        &cache,
        &[
            "search",
            "--root",
            sub.to_str().unwrap(),
            "--query",
            "parse config",
            "--mode",
            "bm25",
        ],
    );
    assert!(ok, "search failed: {err}");
    assert!(
        out.contains("../lib.rs:"),
        "rebased path expected, got: {out}"
    );
    let (_, out, _) = run(
        &cache,
        &[
            "search",
            "--root",
            repo_s,
            "--query",
            "parse config",
            "--mode",
            "bm25",
        ],
    );
    assert!(
        out.contains("src/lib.rs:"),
        "repo-relative path expected, got: {out}"
    );

    // An edit lands; nobody runs index-repo; the next search sees it.
    std::fs::write(
        repo.join("src/net.rs"),
        "/// Retry the upload with exponential backoff.\npub fn retry_upload_backoff(n: u32) -> u64 {\n    1 << n\n}\n",
    )
    .unwrap();
    wait_until(
        "watcher rebuild after edit",
        Duration::from_secs(20),
        || {
            status_json(&cache, &repo)["watcher"]["rebuilds"]
                .as_u64()
                .unwrap_or(0)
                >= 2
        },
    );
    let (ok, out, err) = run(
        &cache,
        &[
            "search",
            "--root",
            repo_s,
            "--query",
            "retry upload backoff",
            "--mode",
            "bm25",
        ],
    );
    assert!(ok, "search failed: {err}");
    assert!(
        out.contains("src/net.rs:"),
        "new file must be searchable: {out}"
    );

    // Sessions end one by one; the watcher outlives the first, not the last.
    let (ok, _, err) = run(
        &cache,
        &["session", "end", "--root", repo_s, "--id", "session-a"],
    );
    assert!(ok, "session end failed: {err}");
    std::thread::sleep(Duration::from_secs(3));
    assert!(
        status_json(&cache, &repo)["watcher"].is_object(),
        "one lease left keeps the watcher"
    );
    let (ok, _, _) = run(
        &cache,
        &["session", "end", "--root", repo_s, "--id", "session-b"],
    );
    assert!(ok);
    wait_until(
        "watcher exit after last lease",
        Duration::from_secs(15),
        || status_json(&cache, &repo)["watcher"].is_null(),
    );
    assert_eq!(
        status_json(&cache, &repo)["sessions"]
            .as_array()
            .unwrap()
            .len(),
        0
    );

    // A restart finds the index current and reports it ready.
    let (ok, out, _) = run(
        &cache,
        &[
            "session",
            "start",
            "--root",
            repo_s,
            "--id",
            "session-c",
            "--lexical",
            "--grace-secs",
            "1",
            "--pid",
            &std::process::id().to_string(),
        ],
    );
    assert!(ok);
    assert!(out.contains("index ready"), "{out}");
    wait_until("restart idle", Duration::from_secs(20), || {
        status_json(&cache, &repo)["watcher"]["state"] == "idle"
    });
    let (ok, out, _) = run(&cache, &["watch", "--root", repo_s, "--stop"]);
    assert!(ok);
    assert!(out.contains("stopping"), "{out}");
    wait_until("stop", Duration::from_secs(10), || {
        status_json(&cache, &repo)["watcher"].is_null()
    });

    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&cache);
}

#[test]
fn dead_session_pid_releases_its_lease() {
    let cache = scratch("cache2");
    let repo = scratch("repo2");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(repo.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();
    let repo_s = repo.to_str().unwrap();

    // A lease held by a pid that is not running (a crashed agent) does not
    // keep the watcher alive.
    let (ok, _, err) = run(
        &cache,
        &[
            "session",
            "start",
            "--root",
            repo_s,
            "--id",
            "crashed",
            "--lexical",
            "--grace-secs",
            "1",
            "--pid",
            &i32::MAX.to_string(),
        ],
    );
    assert!(ok, "{err}");
    wait_until(
        "watcher exits with only a dead lease",
        Duration::from_secs(30),
        || status_json(&cache, &repo)["watcher"].is_null(),
    );
    assert!(
        status_json(&cache, &repo)["index"].is_object(),
        "the index was still built first"
    );
    assert_eq!(
        status_json(&cache, &repo)["sessions"]
            .as_array()
            .unwrap()
            .len(),
        0,
        "dead lease reaped"
    );

    // Outside a git repository nothing is watched.
    let plain = scratch("plain");
    let (ok, out, _) = run(
        &cache,
        &[
            "session",
            "start",
            "--root",
            plain.to_str().unwrap(),
            "--id",
            "x",
            "--lexical",
        ],
    );
    assert!(ok);
    assert!(out.contains("not inside a git repository"), "{out}");
    assert!(status_json(&cache, &plain)["watcher"].is_null());

    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&plain);
    let _ = std::fs::remove_dir_all(&cache);
}

#[test]
fn hook_mode_reads_claude_code_payload_and_answers_with_context() {
    let cache = scratch("cache3");
    let repo = scratch("repo3");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(repo.join("a.rs"), "fn main() {}\n").unwrap();
    let payload = serde_json::json!({
        "session_id": "hook-session", "cwd": repo, "hook_event_name": "SessionStart", "source": "startup"
    })
    .to_string();
    let mut child = hips(&cache)
        .args([
            "session",
            "start",
            "--hook",
            "--lexical",
            "--grace-secs",
            "1",
            "--pid",
            &std::process::id().to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), payload.as_bytes()).unwrap();
    drop(child.stdin.take());
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let reply: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("hook JSON on stdout");
    let context = reply["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(context.contains("hips search --root ."), "{context}");
    assert_eq!(reply["hookSpecificOutput"]["hookEventName"], "SessionStart");
    let s = status_json(&cache, &repo);
    assert_eq!(s["sessions"][0]["session"], "hook-session");

    // SessionEnd with the same payload shape releases it, quietly.
    let end_payload = serde_json::json!({"session_id": "hook-session", "cwd": repo, "hook_event_name": "SessionEnd", "reason": "other"}).to_string();
    let mut child = hips(&cache)
        .args(["session", "end", "--hook"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), end_payload.as_bytes()).unwrap();
    drop(child.stdin.take());
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    assert!(out.stdout.is_empty(), "session end --hook prints nothing");
    wait_until("exit", Duration::from_secs(15), || {
        status_json(&cache, &repo)["watcher"].is_null()
    });

    // Garbage on stdin never fails the hook.
    let mut child = hips(&cache)
        .args(["session", "start", "--hook"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), b"not json").unwrap();
    drop(child.stdin.take());
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "hook mode exits 0 on bad input");

    let _ = std::fs::remove_dir_all(&repo);
    let _ = std::fs::remove_dir_all(&cache);
}
