//! Query execution: tokenize the query the same way documents were indexed,
//! deduplicate the terms, and run an exact dynamic-pruning evaluator —
//! Block-Max WAND for short queries, MaxScore for long ones. Both return
//! provably exact top-k; there is no approximate mode.

use std::sync::OnceLock;
use std::time::Instant;

use crate::block_max_wand::{self, SearchStats};
use crate::indexer::SearchableIndex;
use crate::maxscore;
use crate::spell::SpellCorrector;
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
    /// The rewritten query when typo correction changed it, else None.
    pub corrected: Option<String>,
}

/// Tokenize a query and deduplicate terms, preserving first-seen order.
/// Duplicate query terms would otherwise double-count BM25 contributions.
pub fn query_terms<I: SearchableIndex + ?Sized>(index: &I, query: &str) -> Vec<String> {
    let tokenizer = Tokenizer::with_flags(index.remove_stopwords(), index.code_mode());
    let mut seen = std::collections::HashSet::new();
    tokenizer
        .tokenize(query)
        .into_iter()
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// The underlying index layout: a single immutable index or a segmented
/// (incrementally updatable) one.
pub enum IndexKind {
    Single(Box<crate::storage::DiskIndex>),
    Segmented(crate::segments::SegmentedIndex),
}

/// A search handle over either layout, with a lazily-built spelling
/// corrector for query-time typo correction.
pub struct AnyIndex {
    kind: IndexKind,
    /// Per-segment embedding stores for a segmented index (None entries
    /// for segments without an embeddings sidecar).
    seg_stores: Option<crate::hybrid::SegmentStores>,
    /// Built on the first query that contains an unmatched term; clean
    /// workloads never pay for it.
    spell: OnceLock<SpellCorrector>,
    /// Optional CodeRankEmbed sidecar; hybrid search no-ops without it.
    embeddings: Option<crate::embeddings::EmbeddingStore>,
    /// Optional IVF cluster inverted file (same doc_ids as postings).
    ivf: Option<crate::ivf::IvfIndex>,
    pq: Option<crate::pq::PqIndex>,
}

impl AnyIndex {
    pub fn open(dir: &std::path::Path) -> anyhow::Result<Self> {
        let kind = if crate::segments::is_segmented(dir) {
            IndexKind::Segmented(crate::segments::SegmentedIndex::open(dir)?)
        } else {
            IndexKind::Single(Box::new(crate::storage::load_index(dir)?))
        };
        let embeddings = crate::embeddings::EmbeddingStore::open(dir).ok();
        // A sidecar left behind by an older build maps rows to renumbered
        // doc_ids: silently wrong scores, or panics past the end. Drop it
        // (with a warning) rather than serve stale vectors.
        let embeddings = match (&kind, embeddings) {
            (IndexKind::Single(index), Some(store))
                if store.num_docs() as usize != index.num_docs() =>
            {
                eprintln!(
                    "warning: embeddings.bin has {} rows but the index has {} docs; \
                     ignoring stale vector sidecar (rebuild with --embed)",
                    store.num_docs(),
                    index.num_docs()
                );
                None
            }
            (_, e) => e,
        };
        let ivf = crate::ivf::IvfIndex::open(dir).ok();
        let pq = crate::pq::PqIndex::open(dir).ok();
        let seg_stores = match &kind {
            IndexKind::Segmented(seg) => {
                let stores = seg
                    .segment_names()
                    .iter()
                    .enumerate()
                    .map(|(si, name)| {
                        let store = crate::embeddings::EmbeddingStore::open(&dir.join(name)).ok()?;
                        if store.num_docs() != seg.num_docs_in(si) {
                            eprintln!(
                                "warning: segment {name} embeddings have {} rows for {} docs; \
                                 ignoring stale vector sidecar",
                                store.num_docs(),
                                seg.num_docs_in(si)
                            );
                            return None;
                        }
                        Some(store)
                    })
                    .collect::<Vec<_>>();
                stores
                    .iter()
                    .any(Option::is_some)
                    .then_some(crate::hybrid::SegmentStores { stores })
            }
            IndexKind::Single(_) => None,
        };
        Ok(Self {
            kind,
            seg_stores,
            spell: OnceLock::new(),
            embeddings,
            ivf,
            pq,
        })
    }

    pub fn embeddings(&self) -> Option<&crate::embeddings::EmbeddingStore> {
        self.embeddings.as_ref()
    }

    /// Per-segment embedding stores, present only for segmented indexes
    /// that were built with embeddings.
    pub fn segment_stores(&self) -> Option<&crate::hybrid::SegmentStores> {
        self.seg_stores.as_ref()
    }

    /// True when some form of vector search is available (single-index
    /// sidecar or per-segment stores).
    pub fn has_vectors(&self) -> bool {
        self.embeddings.is_some() || self.seg_stores.is_some()
    }

    pub fn ivf(&self) -> Option<&crate::ivf::IvfIndex> {
        self.ivf.as_ref()
    }

    pub fn pq(&self) -> Option<&crate::pq::PqIndex> {
        self.pq.as_ref()
    }

    pub fn kind(&self) -> &IndexKind {
        &self.kind
    }

    fn remove_stopwords(&self) -> bool {
        match &self.kind {
            IndexKind::Single(index) => index.remove_stopwords(),
            IndexKind::Segmented(index) => index.remove_stopwords(),
        }
    }

    fn code_mode(&self) -> bool {
        match &self.kind {
            IndexKind::Single(index) => {
                use crate::indexer::SearchableIndex as _;
                index.code_mode()
            }
            IndexKind::Segmented(index) => index.code_mode(),
        }
    }

    /// Global document frequency of a term (0 if it matches nothing).
    fn term_df(&self, term: &str) -> u32 {
        match &self.kind {
            IndexKind::Single(index) => index.term_df(term),
            IndexKind::Segmented(index) => index.term_df(term),
        }
    }

    /// The spelling corrector, built from the full vocabulary on first use.
    fn corrector(&self) -> &SpellCorrector {
        self.spell.get_or_init(|| {
            let mut dfs: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
            match &self.kind {
                IndexKind::Single(index) => index.for_each_term(|t, df| {
                    dfs.insert(t.to_owned(), df);
                }),
                // A term recurs once per segment; sum to the global df.
                IndexKind::Segmented(index) => index.for_each_term(|t, df| {
                    *dfs.entry(t.to_owned()).or_insert(0) += df;
                }),
            }
            SpellCorrector::build(dfs)
        })
    }

    pub fn search(&self, query: &str, k: usize) -> SearchOutcome {
        // Rewrite only terms that match nothing, correcting toward the
        // closest known vocabulary term. Clean queries pay one df lookup
        // per term and never build the corrector.
        let tokenizer = Tokenizer::with_flags(self.remove_stopwords(), self.code_mode());
        let mut terms: Vec<String> = Vec::new();
        for token in tokenizer.tokenize(query) {
            if !terms.contains(&token) {
                terms.push(token);
            }
        }
        let mut rewritten = Vec::with_capacity(terms.len());
        let mut changed = false;
        for term in &terms {
            if self.term_df(term) == 0 {
                if let Some(fix) = self.corrector().correct(term) {
                    rewritten.push(fix.to_owned());
                    changed = true;
                    continue;
                }
            }
            rewritten.push(term.clone());
        }

        if changed {
            let new_query = rewritten.join(" ");
            let mut outcome = self.search_raw(&new_query, k);
            outcome.corrected = Some(new_query);
            outcome
        } else {
            self.search_raw(query, k)
        }
    }

    fn search_raw(&self, query: &str, k: usize) -> SearchOutcome {
        match &self.kind {
            IndexKind::Single(index) => search(index.as_ref(), query, k),
            IndexKind::Segmented(index) => index.search(query, k),
        }
    }

    pub fn num_docs(&self) -> u64 {
        match &self.kind {
            IndexKind::Single(index) => {
                use crate::indexer::SearchableIndex as _;
                index.num_docs() as u64
            }
            IndexKind::Segmented(index) => index.num_docs_live(),
        }
    }

    pub fn size_bytes(&self) -> u64 {
        match &self.kind {
            IndexKind::Single(index) => index.size_bytes(),
            IndexKind::Segmented(index) => index.size_bytes(),
        }
    }

    pub fn as_single(&self) -> Option<&crate::storage::DiskIndex> {
        match &self.kind {
            IndexKind::Single(index) => Some(index.as_ref()),
            IndexKind::Segmented(_) => None,
        }
    }
}

/// Exact BM25 top-k as internal doc_ids (used by hybrid reranking).
pub fn search_hits<I: SearchableIndex + ?Sized>(
    index: &I,
    query: &str,
    k: usize,
) -> (Vec<crate::block_max_wand::SearchHit>, SearchStats, f64) {
    let terms = query_terms(index, query);
    let start = Instant::now();
    let (hits, stats) = if terms.len() >= MAXSCORE_MIN_TERMS {
        maxscore::search(index, &terms, k)
    } else {
        block_max_wand::search(index, &terms, k)
    };
    let took_ms = start.elapsed().as_secs_f64() * 1000.0;
    (hits, stats, took_ms)
}

/// Run a query through Block-Max WAND and resolve internal doc_ids to
/// external document metadata.
pub fn search<I: SearchableIndex + ?Sized>(index: &I, query: &str, k: usize) -> SearchOutcome {
    let (hits, stats, took_ms) = search_hits(index, query, k);

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
        corrected: None,
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
