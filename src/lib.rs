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

/// Process-wide verbosity, set once from the CLI's `-v/--verbose`. Off by
/// default: an agent reading the output pays for every line in context,
/// so the CLI prints only what changes the caller's next action.
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
