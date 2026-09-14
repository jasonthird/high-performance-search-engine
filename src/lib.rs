//! MVP single-node search engine.
//!
//! Lexical BM25 retrieval over an inverted index, with exact top-k query
//! execution using Block-Max WAND. No existing search-engine crates are used.

pub mod api;
pub mod bench;
pub mod block_max_wand;
pub mod bm25;
pub mod cli;
pub mod codeindex;
pub mod daemon;
pub mod compress;
#[cfg(all(target_os = "macos", feature = "semantic"))]
pub mod coreml;
#[cfg(feature = "semantic")]
pub mod embedder;
pub mod embcache;
#[cfg(any(feature = "treesitter", feature = "semantic"))]
mod fetch;
#[cfg(feature = "semantic")]
pub mod ggml;
pub mod embeddings;
pub mod eval;
pub mod external;
pub mod hash;
pub mod hybrid;
pub mod hnsw;
pub mod indexer;
pub mod ivf;
pub mod pq;
pub mod maxscore;
pub mod mcp;
pub mod migrate;
pub mod postings;
pub mod query;
pub mod repo;
pub mod reorder;
pub mod searcher;
pub mod segments;
pub mod spell;
pub mod storage;
pub mod tokenizer;
#[cfg(feature = "treesitter")]
pub mod treesit;
#[cfg(feature = "treesitter")]
mod grammar;
pub mod usagelog;
#[cfg(feature = "semantic")]
pub mod wordpiece;

/// Process-wide verbosity, set once from the CLI's `-v/--verbose`. Off by
/// default: an agent reading the output pays for every line in context,
/// so the CLI prints only what changes the caller's next action.
/// Shared by the model-dependent tests: the encoder is part of what the
/// suite verifies, so a missing model is a failure, not a skip.
#[cfg(all(test, feature = "semantic"))]
pub(crate) mod test_util {
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Environment-driven behavior is process-global: tests that set
    /// variables hold this while they do.
    pub static ENV: Mutex<()> = Mutex::new(());

    /// The CodeRankEmbed GGUF, resolved exactly as the program resolves it
    /// (`HIPS_MODEL_PATH`, the csearch cache, else a download from the
    /// pinned release). `None` only with `HIPS_TEST_SKIP_MODEL=1`.
    pub fn model() -> Option<PathBuf> {
        if std::env::var_os("HIPS_TEST_SKIP_MODEL").is_some_and(|v| v == "1") {
            eprintln!("skipped: HIPS_TEST_SKIP_MODEL=1");
            return None;
        }
        let _lock = ENV.lock().unwrap_or_else(|e| e.into_inner());
        match crate::embedder::model_path() {
            Ok(path) => Some(path),
            Err(err) => panic!(
                "this test needs the CodeRankEmbed GGUF and could not get it: {err:#}\n\
                 set HIPS_MODEL_PATH to a local copy, let it download from the release, \
                 or set HIPS_TEST_SKIP_MODEL=1 to skip the model tests deliberately"
            ),
        }
    }
}

/// The encoder's model file name, for tests and tools that place or
/// inspect it without the semantic feature compiled in.
pub fn embedder_model_file() -> &'static str {
    "CodeRankEmbed-f16.gguf"
}

pub mod verbosity {
    use std::sync::atomic::{AtomicBool, Ordering};
    static VERBOSE: AtomicBool = AtomicBool::new(false);
    pub fn set(on: bool) {
        VERBOSE.store(on, Ordering::Relaxed);
    }
    pub fn verbose() -> bool {
        VERBOSE.load(Ordering::Relaxed)
    }
}
pub mod watch;
