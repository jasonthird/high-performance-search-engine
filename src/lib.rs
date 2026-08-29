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
pub mod compress;
#[cfg(feature = "semantic")]
pub mod embedder;
pub mod embcache;
pub mod embeddings;
pub mod eval;
pub mod external;
pub mod hash;
pub mod hybrid;
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
pub mod watch;
