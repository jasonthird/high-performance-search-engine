//! MVP single-node search engine.
//!
//! Lexical BM25 retrieval over an inverted index, with exact top-k query
//! execution using Block-Max WAND. No existing search-engine crates are used.

pub mod api;
pub mod bench;
pub mod block_max_wand;
pub mod bm25;
pub mod cli;
pub mod compress;
pub mod external;
pub mod indexer;
pub mod maxscore;
pub mod migrate;
pub mod postings;
pub mod reorder;
pub mod searcher;
pub mod segments;
pub mod storage;
pub mod tokenizer;
