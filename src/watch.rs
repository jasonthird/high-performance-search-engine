//! Filesystem watching: keep a repository index fresh as files change.
//!
//! The watcher does not rebuild by itself. It owns a native OS event stream
//! (FSEvents / inotify) and collapses every relevant event into a single
//! "dirty" flag; the server rebuilds on the next query that observes the
//! flag set. That ordering is deliberate:
//!
//! - a burst of edits (a branch switch, a formatter run, a build) costs one
//!   rebuild rather than one per file;
//! - the encoder stays on the thread that loaded it, which matters because
//!   its Metal pipelines are compiled and warmed there;
//! - a query never reads a half-written index, because rebuilds are
//!   sequenced with the query that triggered them.
//!
//! The cost is that the first query after an edit pays for the rebuild —
//! which the content-keyed embedding cache keeps proportional to the edit.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Context;
use notify::{Event, EventKind, RecursiveMode, Watcher};

use crate::repo;

/// A live watch over a source tree.
pub struct TreeWatcher {
    dirty: Arc<AtomicBool>,
    events: Arc<AtomicU64>,
    /// Dropping this ends the watch.
    _watcher: notify::RecommendedWatcher,
}

impl TreeWatcher {
    /// Start watching `root` recursively.
    pub fn start(root: &Path) -> anyhow::Result<Self> {
        let root = root
            .canonicalize()
            .with_context(|| format!("cannot watch {}", root.display()))?;
        let dirty = Arc::new(AtomicBool::new(false));
        let events = Arc::new(AtomicU64::new(0));
        let (flag, counter, watch_root) = (dirty.clone(), events.clone(), root.clone());

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            let Ok(event) = res else { return };
            if !is_content_change(&event.kind) {
                return;
            }
            if event.paths.iter().any(|p| is_relevant(&watch_root, p)) {
                counter.fetch_add(1, Ordering::Relaxed);
                flag.store(true, Ordering::Release);
            }
        })
        .context("failed to create filesystem watcher")?;
        watcher
            .watch(&root, RecursiveMode::Recursive)
            .with_context(|| format!("failed to watch {}", root.display()))?;

        Ok(Self {
            dirty,
            events,
            _watcher: watcher,
        })
    }

    /// True if the tree changed since the last [`Self::take_dirty`].
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    /// Consume the dirty flag, returning whether a rebuild is owed.
    ///
    /// The flag is cleared *before* the rebuild reads the tree, so an edit
    /// racing with a rebuild sets it again and is picked up next time rather
    /// than being lost.
    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::AcqRel)
    }

    /// Total relevant events seen — diagnostics for `index_status`.
    pub fn events_seen(&self) -> u64 {
        self.events.load(Ordering::Relaxed)
    }
}

/// Only changes to file contents matter; access times and metadata do not.
fn is_content_change(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) | EventKind::Any
    )
}

/// Would this path have been indexed? Editor swap files, build output, and
/// anything gitignored must not trigger rebuilds.
fn is_relevant(root: &Path, path: &Path) -> bool {
    let Ok(rel) = path.strip_prefix(root) else {
        return false;
    };
    let rel = rel.to_string_lossy().replace('\\', "/");
    if rel.is_empty() {
        return false;
    }
    if rel.split('/').any(is_skipped_component) {
        return false;
    }
    let name = rel.rsplit('/').next().unwrap_or(&rel);
    // Editors write `.foo.swp`, `foo.rs~`, `#foo#` next to the real file.
    if name.starts_with('.') || name.ends_with('~') || name.starts_with('#') {
        return false;
    }
    match name.rsplit_once('.') {
        Some((_, ext)) => repo::SOURCE_EXTS.contains(&ext),
        None => false,
    }
}

fn is_skipped_component(seg: &str) -> bool {
    matches!(
        seg,
        ".git"
            | ".hg"
            | ".svn"
            | ".csearch"
            | "node_modules"
            | "target"
            | "dist"
            | "build"
            | "vendor"
            | "__pycache__"
            | ".venv"
            | "venv"
            | ".mypy_cache"
            | ".pytest_cache"
            | ".next"
            | ".nuxt"
    )
}

/// Paths the watcher would react to, for tests and `--explain`-style output.
pub fn would_trigger(root: &Path, path: &Path) -> bool {
    is_relevant(root, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_noise_from_real_edits() {
        let root = Path::new("/repo");
        assert!(would_trigger(root, Path::new("/repo/src/main.rs")));
        assert!(would_trigger(root, Path::new("/repo/pkg/app.tsx")));

        assert!(!would_trigger(root, Path::new("/repo/target/debug/x.rs")));
        assert!(!would_trigger(root, Path::new("/repo/.git/index")));
        assert!(!would_trigger(root, Path::new("/repo/node_modules/a/b.js")));
        assert!(!would_trigger(root, Path::new("/repo/src/.main.rs.swp")));
        assert!(!would_trigger(root, Path::new("/repo/src/main.rs~")));
        assert!(!would_trigger(root, Path::new("/repo/README.png")));
        assert!(!would_trigger(root, Path::new("/elsewhere/src/main.rs")));
    }

    #[test]
    fn dirty_flag_is_edge_triggered() {
        let dirty = Arc::new(AtomicBool::new(false));
        let watcher = || dirty.swap(false, Ordering::AcqRel);
        assert!(!watcher());
        dirty.store(true, Ordering::Release);
        assert!(watcher(), "first take sees the change");
        assert!(!watcher(), "second take does not");
    }
}
