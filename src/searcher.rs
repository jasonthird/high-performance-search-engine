//! Query execution: tokenize the query the same way documents were indexed,
//! deduplicate the terms, and run an exact dynamic-pruning evaluator —
//! Block-Max WAND for short queries, MaxScore for long ones. Both return
//! provably exact top-k; there is no approximate mode.

use std::time::Instant;

use crate::block_max_wand::{self, SearchStats};
use crate::indexer::SearchableIndex;
use crate::maxscore;
use crate::tokenizer::Tokenizer;

/// Queries with at least this many unique terms run MaxScore instead of
/// Block-Max WAND: with many cursors the WAND pivot prefix rarely clears
/// the threshold, while MaxScore's essential/non-essential split keeps
/// skipping effective. Both are exact.
const MAXSCORE_MIN_TERMS: usize = 5;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub title: String,
}

#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub stats: SearchStats,
    pub took_ms: f64,
}

/// Tokenize a query and deduplicate terms, preserving first-seen order.
/// Duplicate query terms would otherwise double-count BM25 contributions.
pub fn query_terms<I: SearchableIndex + ?Sized>(index: &I, query: &str) -> Vec<String> {
    let tokenizer = Tokenizer::new(index.remove_stopwords());
    let mut terms = Vec::new();
    for token in tokenizer.tokenize(query) {
        if !terms.contains(&token) {
            terms.push(token);
        }
    }
    terms
}

/// A search handle over either layout: a single immutable index or a
/// segmented (incrementally updatable) one.
pub enum AnyIndex {
    Single(Box<crate::storage::DiskIndex>),
    Segmented(crate::segments::SegmentedIndex),
}

impl AnyIndex {
    pub fn open(dir: &std::path::Path) -> anyhow::Result<Self> {
        if crate::segments::is_segmented(dir) {
            Ok(Self::Segmented(crate::segments::SegmentedIndex::open(dir)?))
        } else {
            Ok(Self::Single(Box::new(crate::storage::load_index(dir)?)))
        }
    }

    pub fn search(&self, query: &str, k: usize) -> SearchOutcome {
        match self {
            Self::Single(index) => search(index.as_ref(), query, k),
            Self::Segmented(index) => index.search(query, k),
        }
    }

    pub fn num_docs(&self) -> u64 {
        match self {
            Self::Single(index) => {
                use crate::indexer::SearchableIndex as _;
                index.num_docs() as u64
            }
            Self::Segmented(index) => index.num_docs_live(),
        }
    }

    pub fn size_bytes(&self) -> u64 {
        match self {
            Self::Single(index) => index.size_bytes(),
            Self::Segmented(index) => index.size_bytes(),
        }
    }
}

/// Run a query through Block-Max WAND and resolve internal doc_ids to
/// external document metadata.
pub fn search<I: SearchableIndex + ?Sized>(index: &I, query: &str, k: usize) -> SearchOutcome {
    let terms = query_terms(index, query);
    let start = Instant::now();
    let (hits, stats) = if terms.len() >= MAXSCORE_MIN_TERMS {
        maxscore::search(index, &terms, k)
    } else {
        block_max_wand::search(index, &terms, k)
    };
    let took_ms = start.elapsed().as_secs_f64() * 1000.0;

    let results = hits
        .into_iter()
        .map(|hit| {
            let summary = index.doc_summary(hit.doc_id);
            SearchResult {
                id: summary.id,
                score: hit.score,
                title: summary.title,
            }
        })
        .collect();

    SearchOutcome {
        results,
        stats,
        took_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::{build_index, Index, InputDoc};
    use crate::postings::DEFAULT_BLOCK_SIZE;
    use crate::reorder::ReorderStrategy;

    fn index() -> Index {
        let docs = vec![
            InputDoc {
                id: "doc-0".into(),
                title: "Cheap pizza in Montreal".into(),
                body: "The best cheap pizza montreal has to offer".into(),
            },
            InputDoc {
                id: "doc-1".into(),
                title: "Sushi guide".into(),
                body: "Fresh sushi downtown".into(),
            },
        ];
        build_index(&docs, true, DEFAULT_BLOCK_SIZE, ReorderStrategy::None)
    }

    #[test]
    fn deduplicates_query_terms() {
        let index = index();
        assert_eq!(
            query_terms(&index, "pizza Pizza PIZZA cheap"),
            vec!["pizza", "cheap"]
        );
    }

    #[test]
    fn resolves_external_ids() {
        let index = index();
        let outcome = search(&index, "cheap pizza montreal", 10);
        assert_eq!(outcome.results.len(), 1);
        assert_eq!(outcome.results[0].id, "doc-0");
        assert_eq!(outcome.results[0].title, "Cheap pizza in Montreal");
        assert!(outcome.results[0].score > 0.0);
        assert_eq!(outcome.stats.num_query_terms, 3);
    }

    #[test]
    fn empty_or_unknown_query_returns_no_results() {
        let index = index();
        assert!(search(&index, "", 10).results.is_empty());
        assert!(search(&index, "zebra unicorn", 10).results.is_empty());
    }
}
