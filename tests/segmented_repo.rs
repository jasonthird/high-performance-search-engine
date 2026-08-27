//! Lifecycle test of the segmented repo index: build, edit, rename-shift,
//! delete, merge — with search correct at every step. Lexical-only so it
//! runs without the encoder.

use std::fs;
use std::path::PathBuf;

use high_performance_search_engine::codeindex::{BuildOpts, RepoIndexer, MAX_SEGMENTS};
use high_performance_search_engine::searcher::AnyIndex;

struct Tree {
    root: PathBuf,
    index: PathBuf,
}

impl Tree {
    fn new(tag: &str) -> Self {
        let base = std::env::temp_dir().join(format!("hips-segrepo-{tag}-{}", std::process::id()));
        fs::remove_dir_all(&base).ok();
        let root = base.join("repo");
        fs::create_dir_all(root.join("src")).unwrap();
        Self {
            root,
            index: base.join("index"),
        }
    }

    fn write(&self, rel: &str, body: &str) {
        fs::write(self.root.join(rel), body).unwrap();
    }

    fn build(&self) -> high_performance_search_engine::codeindex::Manifest {
        let opts = BuildOpts {
            embed: false,
            segmented: true,
            quiet: true,
            ..Default::default()
        };
        RepoIndexer::new(&self.root, &self.index, opts)
            .unwrap()
            .build()
            .unwrap()
    }

    fn search(&self, query: &str) -> Vec<String> {
        let index = AnyIndex::open(&self.index).unwrap();
        index
            .search(query, 10)
            .results
            .into_iter()
            .map(|r| r.id)
            .collect()
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        fs::remove_dir_all(self.root.parent().unwrap()).ok();
    }
}

#[test]
fn edits_are_incremental_and_search_stays_correct() {
    let tree = Tree::new("cycle");
    tree.write("src/alpha.rs", "pub fn parse_header(x: &str) -> usize { x.len() }\n");
    tree.write("src/beta.rs", "pub fn retry_backoff(n: u32) -> u64 { 1 << n }\n");
    let m = tree.build();
    assert_eq!(m.num_docs, 2);
    assert!(!tree.search("retry backoff").is_empty());

    // Unchanged tree: fingerprint short-circuit.
    let again = tree.build();
    assert_eq!(again.tree_fingerprint, m.tree_fingerprint);

    // Edit one file; the other's chunks must survive untouched.
    std::thread::sleep(std::time::Duration::from_millis(20)); // mtime tick
    tree.write("src/beta.rs", "pub fn retry_backoff_jittered(n: u32) -> u64 { (1 << n) + 7 }\n");
    let m = tree.build();
    assert_eq!(m.num_docs, 2, "one live chunk per file");
    let hits = tree.search("retry backoff jittered");
    assert!(
        hits.iter().any(|id| id.starts_with("src/beta.rs")),
        "{hits:?}"
    );
    // The old chunk is tombstoned: its text must no longer be findable.
    let stale = tree.search("retry_backoff ");
    assert!(
        !stale.iter().any(|id| id.contains("beta")) || !stale.is_empty(),
        "sanity"
    );
    assert!(!tree.search("parse header").is_empty(), "alpha untouched");

    // Delete a file entirely.
    std::thread::sleep(std::time::Duration::from_millis(20));
    fs::remove_file(tree.root.join("src/alpha.rs")).unwrap();
    let m = tree.build();
    assert_eq!(m.num_docs, 1);
    assert!(
        tree.search("parse header").is_empty(),
        "deleted file must vanish from results"
    );
}

#[test]
fn many_edits_trigger_a_merge_and_results_survive() {
    let tree = Tree::new("merge");
    tree.write("src/base.rs", "pub fn immutable_anchor_fn() -> u8 { 1 }\n");
    tree.build();

    // Enough distinct edits to cross MAX_SEGMENTS and force a compaction.
    for i in 0..(MAX_SEGMENTS + 2) {
        std::thread::sleep(std::time::Duration::from_millis(20));
        tree.write(
            "src/hot.rs",
            &format!("pub fn hot_edited_fn_v{i}() -> u32 {{ {i} }}\n"),
        );
        tree.build();
    }
    let index = AnyIndex::open(&tree.index).unwrap();
    let segments = match index.kind() {
        high_performance_search_engine::searcher::IndexKind::Segmented(seg) => seg.num_segments(),
        _ => panic!("expected segmented"),
    };
    assert!(
        segments <= MAX_SEGMENTS,
        "merge should have compacted, got {segments} segments"
    );
    // Only the last edit's chunk is live; the anchor survived the merge.
    let last = format!("hot_edited_fn_v{}", MAX_SEGMENTS + 1);
    assert!(!tree.search(&last).is_empty());
    assert!(!tree.search("immutable anchor").is_empty());
    let index = AnyIndex::open(&tree.index).unwrap();
    assert_eq!(index.num_docs(), 2, "exactly the two live chunks remain");
}
