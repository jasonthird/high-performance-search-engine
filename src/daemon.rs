//! Background index keeper: one watcher process per repository, shared by
//! every agent session that has the repository open.
//!
//! Agents do not want to think about indexing. A Claude Code session starts,
//! files change while it works, and its searches should simply be current.
//! This module gives that with three pieces:
//!
//! - **Leases.** Each session that wants the index kept fresh writes a lease
//!   file (`<state>/leases/<session>.json`) holding its id and the pid of
//!   the agent process. Several sessions on one repository each hold their
//!   own lease; the watcher lives as long as any live lease exists. A lease
//!   whose pid is gone is reaped, so a crashed or killed agent never pins
//!   the watcher.
//! - **The watcher.** `hips watch --root .` runs the loop: build (or
//!   refresh) the index once, then rebuild after each burst of relevant
//!   filesystem events. It holds `watch.lock` (flock) for its lifetime, so
//!   a second start for the same repository is a no-op rather than a
//!   second process. A leased watcher exits on its own once no live lease
//!   remains for a grace period; a foreground one runs until Ctrl-C. The
//!   encoder is unloaded after a few idle minutes so an idle watcher costs
//!   megabytes, not the model's hundreds.
//! - **Status.** The watcher publishes `watch.json` (state, rebuild count,
//!   last error, lease ids). `hips search` reads it to wait out a rebuild
//!   in flight instead of answering from a half-updated index, and `hips
//!   status` renders it for people.
//!
//! State lives beside the index under the user cache
//! (`<cache>/watch/<index-name>/`), never inside the index directory: a
//! single-layout rebuild swaps that whole directory, and a segmented
//! migration wipes it.
//!
//! Writers are serialized by the segmented index's own `writer.lock`, so a
//! manual `hips index-repo` racing the watcher waits for it rather than
//! failing (see [`crate::codeindex`]).

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::codeindex::{self, BuildOpts, RepoIndexer};
use crate::watch::TreeWatcher;

/// Seconds a leased watcher lingers after its last lease disappears. Long
/// enough to bridge `/clear` (SessionEnd immediately followed by a new
/// SessionStart) and a quick restart of the agent, short enough that an
/// abandoned watcher is gone before anyone notices it.
pub const DEFAULT_GRACE_SECS: u64 = 20;

/// Idle seconds before the watcher drops the encoder. Reloading costs
/// about a second on the next rebuild; holding it costs the model's RAM.
pub const DEFAULT_UNLOAD_SECS: u64 = 300;

/// Filesystem events are bursty (a save is several events, a branch switch
/// hundreds). A rebuild starts only after the tree has been quiet this long.
const QUIET: Duration = Duration::from_millis(300);
/// How often the loop looks at the dirty flag, leases, and stop signal.
const TICK: Duration = Duration::from_millis(100);
/// How often leases are re-read and reaped.
const LEASE_CHECK: Duration = Duration::from_secs(2);

/// One session's claim on the watcher.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Lease {
    pub session: String,
    /// Pid of the agent process; 0 when unknown (then only an explicit
    /// `session end` releases it).
    pub pid: u32,
    pub agent: String,
    pub since: u64,
}

/// What the watcher publishes for `hips status` and for `hips search`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Status {
    pub pid: u32,
    pub root: String,
    pub index_dir: String,
    pub since: u64,
    /// `starting` (initial build in progress), `idle`, or `indexing`.
    pub state: String,
    pub leased: bool,
    pub embed: bool,
    pub rebuilds: u64,
    pub last_rebuild: u64,
    pub last_rebuild_secs: f64,
    pub last_error: Option<String>,
    pub leases: Vec<String>,
    pub events: u64,
    pub encoder_loaded: bool,
}

/// Where a repository's watcher keeps its lock, leases, log, and status.
pub fn state_dir(index_dir: &Path) -> PathBuf {
    let name = index_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "repo".to_string());
    let base = index_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(codeindex::cache_root);
    base.join("watch").join(name)
}

fn leases_dir(state: &Path) -> PathBuf {
    state.join("leases")
}

pub fn log_path(state: &Path) -> PathBuf {
    state.join("watch.log")
}

fn status_path(state: &Path) -> PathBuf {
    state.join("watch.json")
}

fn lock_path(state: &Path) -> PathBuf {
    state.join("watch.lock")
}

/// Session ids come from the outside (hook JSON, command line); keep them
/// to a filename-safe alphabet.
fn lease_file_name(session: &str) -> String {
    let safe: String = session
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{}.json", if safe.is_empty() { "session" } else { &safe })
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Is a process with this pid alive? `kill(pid, 0)` probes without
/// signalling; EPERM means it exists but belongs to someone else.
#[cfg(unix)]
pub fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
pub fn pid_alive(_pid: u32) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Leases
// ---------------------------------------------------------------------------

/// Record (or refresh) a session's lease.
pub fn add_lease(state: &Path, lease: &Lease) -> anyhow::Result<()> {
    let dir = leases_dir(state);
    fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let path = dir.join(lease_file_name(&lease.session));
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec(lease)?)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

/// Drop a session's lease. Missing is fine: the hook may fire twice.
pub fn remove_lease(state: &Path, session: &str) -> anyhow::Result<bool> {
    let path = leases_dir(state).join(lease_file_name(session));
    match fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e).with_context(|| format!("cannot remove {}", path.display())),
    }
}

/// All leases on disk, live or not.
pub fn read_leases(state: &Path) -> Vec<Lease> {
    let Ok(entries) = fs::read_dir(leases_dir(state)) else {
        return Vec::new();
    };
    let mut leases: Vec<Lease> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| {
            let text = fs::read_to_string(e.path()).ok()?;
            serde_json::from_str::<Lease>(&text).ok()
        })
        .collect();
    leases.sort_by(|a, b| a.since.cmp(&b.since).then(a.session.cmp(&b.session)));
    leases
}

/// Leases whose agent process is still running; the others are deleted.
/// A lease with pid 0 is trusted until explicitly ended.
pub fn live_leases(state: &Path) -> Vec<Lease> {
    let mut live = Vec::new();
    for lease in read_leases(state) {
        if lease.pid == 0 || pid_alive(lease.pid) {
            live.push(lease);
        } else {
            let _ = remove_lease(state, &lease.session);
        }
    }
    live
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

pub fn read_status(state: &Path) -> Option<Status> {
    let text = fs::read_to_string(status_path(state)).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_status(state: &Path, status: &Status) {
    let path = status_path(state);
    let tmp = path.with_extension("tmp");
    let Ok(bytes) = serde_json::to_vec_pretty(status) else {
        return;
    };
    if fs::write(&tmp, bytes).is_ok() {
        let _ = fs::rename(&tmp, &path);
    }
}

/// The watcher currently serving this index, if one is alive. The lock is
/// the truth: a status file can outlive a killed process, a held lock
/// cannot.
pub fn running(state: &Path) -> Option<Status> {
    if lock_is_held(state) {
        read_status(state).filter(|s| pid_alive(s.pid))
    } else {
        None
    }
}

fn lock_is_held(state: &Path) -> bool {
    let Ok(file) = File::options()
        .read(true)
        .write(true)
        .open(lock_path(state))
    else {
        return false;
    };
    match file.try_lock() {
        Ok(()) => {
            let _ = file.unlock();
            false
        }
        Err(std::fs::TryLockError::WouldBlock) => true,
        Err(_) => false,
    }
}

/// Block (briefly) while the watcher is mid-rebuild, so a search issued
/// right after an edit sees the edit. Returns the status if a watcher is
/// running. Never waits longer than `max`; a stuck watcher must not turn
/// into a stuck search.
pub fn wait_for_idle(index_dir: &Path, max: Duration) -> Option<Status> {
    let state = state_dir(index_dir);
    let deadline = Instant::now() + max;
    loop {
        let status = running(&state)?;
        if status.state == "idle" || Instant::now() >= deadline {
            return Some(status);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

// ---------------------------------------------------------------------------
// Session start / end (what the hooks call)
// ---------------------------------------------------------------------------

/// How the watcher should be started for a session.
#[derive(Debug, Clone)]
pub struct StartOpts {
    /// Lexical-only index (no encoder, no model download).
    pub lexical: bool,
    pub grace_secs: u64,
    pub unload_secs: u64,
    /// Watch even when `root` is not inside a git repository. Off by
    /// default: a session opened in `$HOME` must not index the home dir.
    pub any_dir: bool,
}

impl Default for StartOpts {
    fn default() -> Self {
        Self {
            lexical: false,
            grace_secs: DEFAULT_GRACE_SECS,
            unload_secs: DEFAULT_UNLOAD_SECS,
            any_dir: false,
        }
    }
}

/// What `session start` did.
#[derive(Debug)]
pub struct StartReport {
    pub root: PathBuf,
    pub index_dir: PathBuf,
    pub state_dir: PathBuf,
    /// True when this call launched the watcher (false: already running).
    pub spawned: bool,
    pub watcher_pid: u32,
    pub live_leases: usize,
    /// Whether an index already exists (else the watcher is building it).
    pub index_exists: bool,
}

/// The directory the watcher should cover for a session opened at `cwd`:
/// the enclosing git work tree when there is one (an agent launched in a
/// subdirectory still wants the whole repository), else `cwd` itself.
pub fn project_root(cwd: &Path) -> Option<PathBuf> {
    let cwd = cwd.canonicalize().ok()?;
    let mut dir = cwd.as_path();
    loop {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

/// Take a lease for `session` on the repository at `cwd` and make sure a
/// watcher is running for it. Returns `None` when there is nothing to
/// watch (not a git repository and `any_dir` is off).
pub fn session_start(
    cwd: &Path,
    session: &str,
    pid: u32,
    opts: &StartOpts,
) -> anyhow::Result<Option<StartReport>> {
    let root = match project_root(cwd) {
        Some(r) => r,
        None if opts.any_dir => cwd
            .canonicalize()
            .with_context(|| format!("cannot open {}", cwd.display()))?,
        None => return Ok(None),
    };
    let index_dir = codeindex::default_index_dir(&root);
    let state = state_dir(&index_dir);
    fs::create_dir_all(&state).with_context(|| format!("cannot create {}", state.display()))?;
    add_lease(
        &state,
        &Lease {
            session: session.to_string(),
            pid,
            agent: crate::usagelog::detect_agent().to_string(),
            since: now_secs(),
        },
    )?;
    let (spawned, watcher_pid) = match running(&state) {
        Some(s) => (false, s.pid),
        None => (true, spawn_watcher(&root, &state, opts)?),
    };
    Ok(Some(StartReport {
        index_exists: index_exists(&index_dir),
        live_leases: live_leases_count(&state),
        root,
        index_dir,
        state_dir: state,
        spawned,
        watcher_pid,
    }))
}

fn live_leases_count(state: &Path) -> usize {
    // Do not reap here: `running()` above may have raced a watcher that is
    // still starting; counting is enough for the report.
    read_leases(state).len()
}

pub fn index_exists(index_dir: &Path) -> bool {
    index_dir.join("meta.bin").exists() || crate::segments::is_segmented(index_dir)
}

/// Release `session`'s lease. The watcher notices within a couple of
/// seconds and exits after the grace period if it was the last one.
pub fn session_end(cwd: &Path, session: &str) -> anyhow::Result<bool> {
    let Some(root) = project_root(cwd).or_else(|| cwd.canonicalize().ok()) else {
        return Ok(false);
    };
    let state = state_dir(&codeindex::default_index_dir(&root));
    remove_lease(&state, session)
}

/// Launch `hips watch --leased` for `root` as a detached process: its own
/// session (so the agent's exit, terminal hangup, or process-group kill
/// never reaches it), stdio on the log file. Returns the child's pid.
#[cfg(unix)]
fn spawn_watcher(root: &Path, state: &Path, opts: &StartOpts) -> anyhow::Result<u32> {
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    let exe = std::env::current_exe().context("cannot locate the hips binary")?;
    let log = File::options()
        .create(true)
        .append(true)
        .open(log_path(state))
        .with_context(|| format!("cannot open {}", log_path(state).display()))?;
    let err = log.try_clone()?;
    let mut cmd = Command::new(exe);
    cmd.arg("watch")
        .arg("--root")
        .arg(root)
        .arg("--leased")
        .arg("--grace-secs")
        .arg(opts.grace_secs.to_string())
        .arg("--unload-secs")
        .arg(opts.unload_secs.to_string());
    if opts.lexical {
        cmd.arg("--lexical");
    }
    cmd.stdin(Stdio::null()).stdout(log).stderr(err);
    // A new session detaches from the controlling terminal and from the
    // agent's process group before exec.
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let child = cmd.spawn().context("cannot start the watcher")?;
    // Wait (briefly) for the lock to be held so a second `session start`
    // fired in the same instant sees the watcher instead of spawning one
    // more. A late start is still fine: the lock rejects the duplicate.
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && !lock_is_held(state) {
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(child.id())
}

#[cfg(not(unix))]
fn spawn_watcher(_root: &Path, _state: &Path, _opts: &StartOpts) -> anyhow::Result<u32> {
    anyhow::bail!("the background watcher is only supported on unix")
}

// ---------------------------------------------------------------------------
// The watcher loop
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WatchOpts {
    pub embed: bool,
    /// Exit when no live lease remains (plus grace). Off: run until a
    /// stop signal.
    pub leased: bool,
    pub grace_secs: u64,
    pub unload_secs: u64,
    pub title_weight: u32,
}

static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_stop_signal(_sig: libc::c_int) {
    STOP.store(true, Ordering::Release);
}

#[cfg(unix)]
fn install_signals(leased: bool) {
    unsafe {
        libc::signal(
            libc::SIGTERM,
            on_stop_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGINT,
            on_stop_signal as *const () as libc::sighandler_t,
        );
        // A leased watcher belongs to no terminal; ignore hangups so a
        // closing window cannot take it down while sessions still lease it.
        if leased {
            libc::signal(libc::SIGHUP, libc::SIG_IGN);
        } else {
            libc::signal(
                libc::SIGHUP,
                on_stop_signal as *const () as libc::sighandler_t,
            );
        }
    }
}

#[cfg(not(unix))]
fn install_signals(_leased: bool) {}

fn stamp() -> String {
    let secs = now_secs();
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}Z")
}

fn log(msg: &str) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "[{}] {msg}", stamp());
    let _ = err.flush();
}

/// Run the watcher for `root` until stopped. See the module docs.
pub fn run_watch(root: &Path, opts: WatchOpts) -> anyhow::Result<()> {
    let root = root
        .canonicalize()
        .with_context(|| format!("cannot open {}", root.display()))?;
    let index_dir = codeindex::default_index_dir(&root);
    let state = state_dir(&index_dir);
    fs::create_dir_all(&state).with_context(|| format!("cannot create {}", state.display()))?;

    let lock = File::create(lock_path(&state)).context("cannot create watch.lock")?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            let who = read_status(&state)
                .map(|s| format!(" (pid {})", s.pid))
                .unwrap_or_default();
            anyhow::bail!("a watcher is already running for {}{who}", root.display());
        }
        Err(std::fs::TryLockError::Error(e)) => return Err(e).context("cannot lock watch.lock"),
    }
    install_signals(opts.leased);
    STOP.store(false, Ordering::Release);

    let mut status = Status {
        pid: std::process::id(),
        root: root.to_string_lossy().to_string(),
        index_dir: index_dir.to_string_lossy().to_string(),
        since: now_secs(),
        state: "starting".to_string(),
        leased: opts.leased,
        embed: opts.embed,
        ..Status::default()
    };
    status.leases = read_leases(&state).into_iter().map(|l| l.session).collect();
    write_status(&state, &status);
    log(&format!(
        "watching {} ({} index, {}) pid {}",
        root.display(),
        if opts.embed { "hybrid" } else { "lexical" },
        if opts.leased { "leased" } else { "foreground" },
        status.pid
    ));

    // Watch before the first build so nothing that lands during it is lost.
    let watcher = TreeWatcher::start(&root)?;
    let build = BuildOpts {
        embed: opts.embed,
        title_weight: opts.title_weight,
        segmented: true,
        quiet: true,
        ..BuildOpts::default()
    };
    let mut indexer = RepoIndexer::new(&root, &index_dir, build)?;

    let rebuild = |status: &mut Status, indexer: &RepoIndexer, why: &str| {
        status.state = "indexing".to_string();
        write_status(&state, status);
        let started = Instant::now();
        match indexer.build() {
            Ok(m) => {
                status.rebuilds += 1;
                status.last_rebuild = now_secs();
                status.last_rebuild_secs = started.elapsed().as_secs_f64();
                status.last_error = None;
                log(&format!(
                    "{why}: {} chunks from {} files in {:.2}s ({} encoded, {} cached)",
                    m.num_docs, m.num_files, status.last_rebuild_secs, m.encoded, m.cached
                ));
            }
            Err(e) => {
                status.last_error = Some(format!("{e:#}"));
                log(&format!("{why} failed: {e:#}"));
            }
        }
        status.state = "idle".to_string();
        status.events = watcher.events_seen();
        status.encoder_loaded = indexer.encoder_loaded();
        write_status(&state, status);
    };

    rebuild(&mut status, &indexer, "initial build");
    let mut last_activity = Instant::now();
    let mut last_lease_check = Instant::now();
    let mut leaseless_since: Option<Instant> = None;
    let grace = Duration::from_secs(opts.grace_secs);
    let unload_after = Duration::from_secs(opts.unload_secs);

    let reason = loop {
        if STOP.load(Ordering::Acquire) {
            break "stop signal";
        }
        if watcher.take_dirty() {
            // Announce the rebuild before the quiet wait, so a search that
            // follows an edit by more than one tick already waits for it.
            status.state = "indexing".to_string();
            write_status(&state, &status);
            // Let the burst finish: wait until no new event for QUIET.
            loop {
                let seen = watcher.events_seen();
                std::thread::sleep(QUIET);
                if watcher.events_seen() == seen {
                    break;
                }
            }
            watcher.take_dirty();
            rebuild(&mut status, &indexer, "reindex");
            last_activity = Instant::now();
            continue;
        }
        if last_lease_check.elapsed() >= LEASE_CHECK {
            last_lease_check = Instant::now();
            let live = live_leases(&state);
            let ids: Vec<String> = live.iter().map(|l| l.session.clone()).collect();
            if ids != status.leases {
                log(&format!(
                    "leases: {}",
                    if ids.is_empty() {
                        "none".to_string()
                    } else {
                        ids.join(", ")
                    }
                ));
                status.leases = ids;
                write_status(&state, &status);
            }
            if opts.leased {
                if live.is_empty() {
                    let since = *leaseless_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= grace {
                        break "no sessions left";
                    }
                } else {
                    leaseless_since = None;
                }
            }
            if indexer.encoder_loaded() && last_activity.elapsed() >= unload_after {
                indexer.unload_embedder();
                status.encoder_loaded = false;
                write_status(&state, &status);
                log("encoder unloaded after idle period");
            }
        }
        std::thread::sleep(TICK);
    };
    log(&format!("exiting: {reason}"));
    let _ = fs::remove_file(status_path(&state));
    drop(lock);
    Ok(())
}

/// Ask a running watcher to stop; true if one was signalled.
#[cfg(unix)]
pub fn stop_watcher(index_dir: &Path) -> bool {
    let state = state_dir(index_dir);
    match running(&state) {
        Some(s) => unsafe { libc::kill(s.pid as libc::pid_t, libc::SIGTERM) == 0 },
        None => false,
    }
}

#[cfg(not(unix))]
pub fn stop_watcher(_index_dir: &Path) -> bool {
    false
}

/// Fields of a Claude Code hook payload that the session commands use.
#[derive(Debug, Deserialize, Default)]
pub struct HookInput {
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub cwd: String,
    #[serde(default)]
    pub hook_event_name: String,
}

impl HookInput {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(text).context("hook input is not the expected JSON object")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_state(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hips-daemon-{name}-{}-{}",
            std::process::id(),
            now_secs()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn lease(session: &str, pid: u32) -> Lease {
        Lease {
            session: session.to_string(),
            pid,
            agent: "test".to_string(),
            since: now_secs(),
        }
    }

    #[test]
    fn leases_round_trip_and_reap_dead_pids() {
        let state = temp_state("leases");
        add_lease(&state, &lease("a", std::process::id())).unwrap();
        add_lease(&state, &lease("b", 0)).unwrap();
        // A pid nobody can be running: the maximum a 32-bit pid_t allows.
        add_lease(&state, &lease("dead", i32::MAX as u32)).unwrap();
        assert_eq!(read_leases(&state).len(), 3);

        let live = live_leases(&state);
        let ids: Vec<&str> = live.iter().map(|l| l.session.as_str()).collect();
        assert_eq!(
            ids,
            ["a", "b"],
            "own pid and pid 0 are live; dead pid reaped"
        );
        assert_eq!(read_leases(&state).len(), 2, "reaping deletes the file");

        assert!(remove_lease(&state, "a").unwrap());
        assert!(!remove_lease(&state, "a").unwrap(), "second end is a no-op");
        assert_eq!(read_leases(&state).len(), 1);
        fs::remove_dir_all(&state).unwrap();
    }

    #[test]
    fn session_ids_become_safe_file_names() {
        assert_eq!(lease_file_name("abc-123_x.y"), "abc-123_x.y.json");
        assert_eq!(lease_file_name("../etc/passwd"), ".._etc_passwd.json");
        assert_eq!(lease_file_name(""), "session.json");
    }

    #[test]
    fn hook_input_parses_claude_code_payload() {
        let text = r#"{"session_id":"s1","transcript_path":"/x","cwd":"/repo","hook_event_name":"SessionStart","source":"startup"}"#;
        let input = HookInput::parse(text).unwrap();
        assert_eq!(input.session_id, "s1");
        assert_eq!(input.cwd, "/repo");
        assert_eq!(input.hook_event_name, "SessionStart");
        assert_eq!(HookInput::parse("  ").unwrap().session_id, "");
        assert!(HookInput::parse("nope").is_err());
    }

    #[test]
    fn project_root_climbs_to_the_git_toplevel() {
        let state = temp_state("root");
        fs::create_dir_all(state.join(".git")).unwrap();
        let nested = state.join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        let found = project_root(&nested).unwrap();
        assert_eq!(found, state.canonicalize().unwrap());
        fs::remove_dir_all(&state).unwrap();
    }

    #[test]
    fn state_dir_is_beside_the_index_not_inside_it() {
        let state = state_dir(Path::new("/cache/csearch/repo-abc"));
        assert_eq!(state, PathBuf::from("/cache/csearch/watch/repo-abc"));
    }

    #[test]
    fn no_watcher_means_no_wait() {
        let index_dir = temp_state("wait").join("idx");
        let started = Instant::now();
        assert!(wait_for_idle(&index_dir, Duration::from_secs(5)).is_none());
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
