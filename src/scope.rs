//! Query-time document scope: the set of documents one search may return.
//!
//! A scope is a per-segment bitmap built from a predicate over file paths
//! (the MCP `path_glob`). It is applied *inside* the evaluators, the way
//! tombstones are: Block-Max WAND never scores an out-of-scope document, the
//! vector pool never returns one, and hybrid fusion normalizes over in-scope
//! candidates only. A scoped top-k is therefore exact — not a post-filter
//! over a global cutoff — and a scope that matches nothing is known before
//! any retrieval runs.
//!
//! Resolving a glob must not cost a pass over every chunk per query, or a
//! filter would make queries slower instead of cheaper. [`PathIndex`] maps
//! each file to the runs of doc ids its chunks occupy; it is built once
//! when an index is opened, and a scope is then one glob match per file
//! plus word-wise bit fills.
//!
//! Hiding documents can only remove candidates, so every upper bound the
//! dynamic-pruning evaluators rely on stays valid.

use crate::indexer::{DocSummary, SearchableIndex};
use crate::postings::TermPostings;

/// Which documents are in scope, addressed by (segment, segment-local doc).
/// A single-layout index is segment 0.
pub struct DocScope {
    allowed: Vec<Vec<u64>>,
    matched: usize,
}

impl DocScope {
    /// An empty scope over segments of the given sizes.
    pub fn new(segment_sizes: &[u32]) -> Self {
        Self {
            allowed: segment_sizes
                .iter()
                .map(|&n| vec![0u64; (n as usize).div_ceil(64)])
                .collect(),
            matched: 0,
        }
    }

    pub fn allow(&mut self, segment: usize, doc: u32) {
        let words = &mut self.allowed[segment];
        if !bit_is_set(words, doc) {
            set_bit(words, doc);
            self.matched += 1;
        }
    }

    /// Allow `len` consecutive documents from `start`, word-wise.
    pub fn allow_range(&mut self, segment: usize, start: u32, len: u32) {
        let words = &mut self.allowed[segment];
        let (mut doc, end) = (start, start + len);
        while doc < end {
            let word = (doc / 64) as usize;
            let lo = doc % 64;
            let take = (64 - lo).min(end - doc);
            let mask = if take == 64 { u64::MAX } else { ((1u64 << take) - 1) << lo };
            self.matched += (mask & !words[word]).count_ones() as usize;
            words[word] |= mask;
            doc += take;
        }
    }

    pub fn contains(&self, segment: usize, doc: u32) -> bool {
        self.allowed
            .get(segment)
            .is_some_and(|words| bit_is_set(words, doc))
    }

    /// In-scope documents (tombstones excluded by the builder).
    pub fn matched(&self) -> usize {
        self.matched
    }

    /// Drop tombstoned documents: `deleted` is the segment's tombstone
    /// bitmap (empty when nothing is deleted). Word-wise, so the cost is
    /// the bitmap length, not the document count.
    pub fn exclude(&mut self, segment: usize, deleted: &[u64]) {
        let words = &mut self.allowed[segment];
        for (w, d) in words.iter_mut().zip(deleted) {
            let dropped = (*w & *d).count_ones() as usize;
            *w &= !*d;
            self.matched -= dropped;
        }
    }

    pub fn num_segments(&self) -> usize {
        self.allowed.len()
    }

    /// The bitmap for one segment, for adapters that hide documents.
    pub fn segment_bits(&self, segment: usize) -> &[u64] {
        self.allowed.get(segment).map(Vec::as_slice).unwrap_or(&[])
    }
}

pub fn bit_is_set(words: &[u64], doc: u32) -> bool {
    words
        .get((doc / 64) as usize)
        .is_some_and(|w| (w >> (doc % 64)) & 1 == 1)
}

pub fn set_bit(words: &mut [u64], doc: u32) {
    words[(doc / 64) as usize] |= 1 << (doc % 64);
}

/// The first set bit at or after `doc`, or `u32::MAX` when none remains.
/// Word-wise: a sparse scope lets the evaluators leap over whole runs of
/// hidden documents in one call.
pub fn next_set_bit(words: &[u64], doc: u32) -> u32 {
    let mut word = (doc / 64) as usize;
    if word >= words.len() {
        return u32::MAX;
    }
    let mut w = words[word] & (u64::MAX << (doc % 64));
    loop {
        if w != 0 {
            return (word as u32) * 64 + w.trailing_zeros();
        }
        word += 1;
        if word >= words.len() {
            return u32::MAX;
        }
        w = words[word];
    }
}

/// A [`SearchableIndex`] adapter that hides every document outside `allowed`.
/// The evaluators treat hidden documents exactly like tombstones.
pub struct Scoped<'a, I: ?Sized> {
    pub inner: &'a I,
    pub allowed: &'a [u64],
}

impl<I: SearchableIndex + ?Sized> SearchableIndex for Scoped<'_, I> {
    fn num_docs(&self) -> usize {
        self.inner.num_docs()
    }

    fn num_terms(&self) -> usize {
        self.inner.num_terms()
    }

    fn avg_doc_len(&self) -> f32 {
        self.inner.avg_doc_len()
    }

    fn remove_stopwords(&self) -> bool {
        self.inner.remove_stopwords()
    }

    fn code_mode(&self) -> bool {
        self.inner.code_mode()
    }

    fn doc_len(&self, doc_id: u32) -> u32 {
        self.inner.doc_len(doc_id)
    }

    fn doc_summary(&self, doc_id: u32) -> DocSummary {
        self.inner.doc_summary(doc_id)
    }

    fn is_deleted(&self, doc_id: u32) -> bool {
        self.inner.is_deleted(doc_id) || !bit_is_set(self.allowed, doc_id)
    }

    fn next_live(&self, doc: u32) -> u32 {
        next_set_bit(self.allowed, self.inner.next_live(doc))
    }

    fn total_postings(&self) -> u64 {
        self.inner.total_postings()
    }

    fn term_postings(&self, term: &str) -> Option<TermPostings<'_>> {
        self.inner.term_postings(term)
    }
}

/// Visits `(doc, external id)` for every document of one segment within a
/// range; what [`PathIndex::build`] runs over each range.
pub type IdVisitor<'a> = dyn Fn(usize, std::ops::Range<u32>, &mut dyn FnMut(u32, &str)) + Sync + 'a;

/// A run of consecutive chunks of one file: `(segment, first doc, count)`.
pub type DocRun = (usize, u32, u32);

/// Where each file's chunks live: runs of consecutive doc ids per segment.
///
/// Built once per opened index with one pass over the doc store, then kept
/// with the index handle. Resolving a glob afterwards costs one match per
/// *file* (a repository has far fewer files than chunks) plus word-wise bit
/// fills, so a scoped query does less work than an unscoped one, as a
/// filter should. Chunks of one file are appended together, so a file is a
/// handful of runs even after incremental edits spread it over segments.
pub struct PathIndex {
    segment_sizes: Vec<u32>,
    /// `(path, runs)`, sorted by path. For documents whose id is not a
    /// `path:start-end` location the whole id is the path.
    files: Vec<(String, Vec<DocRun>)>,
}

impl PathIndex {
    pub fn build(segment_sizes: &[u32], for_each_id: &IdVisitor<'_>) -> Self {
        use rayon::prelude::*;
        // Ranges, not segments: after a compaction merge the index is one
        // big segment and per-segment parallelism would leave the scan on
        // one core exactly when it is largest.
        const RANGE: u32 = 8192;
        let mut ranges: Vec<(usize, u32, u32)> = Vec::new();
        for (si, &n) in segment_sizes.iter().enumerate() {
            let mut at = 0u32;
            while at < n {
                let end = (at + RANGE).min(n);
                ranges.push((si, at, end));
                at = end;
            }
        }
        // Each range yields its runs in doc order; consecutive docs of one
        // path coalesce into one run.
        let partial: Vec<Vec<(String, usize, u32, u32)>> = ranges
            .par_iter()
            .map(|&(si, start, end)| {
                let mut runs: Vec<(String, usize, u32, u32)> = Vec::new();
                for_each_id(si, start..end, &mut |doc, id| {
                    let path = crate::repo::parse_id(id).map(|(p, _, _)| p).unwrap_or(id);
                    match runs.last_mut() {
                        Some((p, _, first, len)) if p == path && *first + *len == doc => *len += 1,
                        _ => runs.push((path.to_string(), si, doc, 1)),
                    }
                });
                runs
            })
            .collect();
        let mut by_path: std::collections::HashMap<String, Vec<DocRun>> =
            std::collections::HashMap::new();
        for runs in partial {
            for (path, si, first, len) in runs {
                let entry = by_path.entry(path).or_default();
                match entry.last_mut() {
                    // A run split only by a range boundary rejoins.
                    Some((s, f, l)) if *s == si && *f + *l == first => *l += len,
                    _ => entry.push((si, first, len)),
                }
            }
        }
        let mut files: Vec<(String, Vec<DocRun>)> = by_path.into_iter().collect();
        files.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Self {
            segment_sizes: segment_sizes.to_vec(),
            files,
        }
    }

    /// Distinct paths in the index.
    pub fn num_files(&self) -> usize {
        self.files.len()
    }

    /// The scope of every live document whose path `accept`s. `tombstones`
    /// returns a segment's deletion bitmap (empty when nothing is deleted).
    pub fn scope<'t>(
        &self,
        accept: &(dyn Fn(&str) -> bool + Sync),
        tombstones: &dyn Fn(usize) -> &'t [u64],
    ) -> DocScope {
        use rayon::prelude::*;
        let matched: Vec<&Vec<DocRun>> = self
            .files
            .par_iter()
            .filter(|(path, _)| accept(path))
            .map(|(_, runs)| runs)
            .collect();
        let mut scope = DocScope::new(&self.segment_sizes);
        for runs in matched {
            for &(si, first, len) in runs {
                scope.allow_range(si, first, len);
            }
        }
        for si in 0..self.segment_sizes.len() {
            scope.exclude(si, tombstones(si));
        }
        scope
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_and_contains_track_matches() {
        let mut scope = DocScope::new(&[70, 3]);
        assert_eq!(scope.matched(), 0);
        scope.allow(0, 69);
        scope.allow(1, 0);
        scope.allow(1, 0);
        assert_eq!(scope.matched(), 2);
        assert!(scope.contains(0, 69));
        assert!(!scope.contains(0, 68));
        assert!(scope.contains(1, 0));
        assert!(!scope.contains(2, 0), "unknown segment is out of scope");
        assert!(!scope.contains(0, 1000), "out-of-range doc is out of scope");
    }

    #[test]
    fn next_set_bit_leaps_over_hidden_runs() {
        let mut words = vec![0u64; 4];
        for doc in [3, 64, 130, 255] {
            set_bit(&mut words, doc);
        }
        assert_eq!(next_set_bit(&words, 0), 3);
        assert_eq!(next_set_bit(&words, 3), 3, "an admitted doc is its own successor");
        assert_eq!(next_set_bit(&words, 4), 64);
        assert_eq!(next_set_bit(&words, 64), 64);
        assert_eq!(next_set_bit(&words, 65), 130);
        assert_eq!(next_set_bit(&words, 131), 255);
        assert_eq!(next_set_bit(&words, 256), u32::MAX);
        assert_eq!(next_set_bit(&words, 10_000), u32::MAX);
        assert_eq!(next_set_bit(&[], 0), u32::MAX);
        assert_eq!(next_set_bit(&[u64::MAX], 63), 63);
    }

    #[test]
    fn ranges_and_exclusions_count_word_wise() {
        let mut scope = DocScope::new(&[200]);
        scope.allow_range(0, 60, 70); // crosses two word boundaries
        assert_eq!(scope.matched(), 70);
        assert!(!scope.contains(0, 59) && scope.contains(0, 60) && scope.contains(0, 129) && !scope.contains(0, 130));
        scope.allow_range(0, 100, 10); // overlap is not double counted
        assert_eq!(scope.matched(), 70);
        let mut deleted = vec![0u64; 4];
        set_bit(&mut deleted, 64);
        set_bit(&mut deleted, 65);
        set_bit(&mut deleted, 10); // not in scope: no effect
        scope.exclude(0, &deleted);
        assert_eq!(scope.matched(), 68);
        assert!(!scope.contains(0, 64) && scope.contains(0, 66));
        scope.exclude(0, &[]); // a segment with no tombstones
        assert_eq!(scope.matched(), 68);
    }

    #[test]
    fn path_index_resolves_globs_per_file() {
        // Three segments; files are runs of chunks, one file straddles a
        // parallel range boundary, one id is not a location at all.
        let sizes = [20_000u32, 100, 0];
        let ids = |si: usize, doc: u32| {
            if si == 1 && doc == 99 {
                "plain-doc-id".to_string()
            } else {
                format!("seg{si}/file{}.rs:{doc}-{doc}", doc / 2500)
            }
        };
        let index = PathIndex::build(&sizes, &|si, range, f| {
            for doc in range {
                f(doc, &ids(si, doc));
            }
        });
        assert_eq!(index.num_files(), 8 + 1 + 1);
        let runs = &index.files.iter().find(|(p, _)| p == "seg0/file3.rs").unwrap().1;
        assert_eq!(runs, &[(0, 7500, 2500)], "one run despite the 8192 range boundary");

        let mut deleted0 = vec![0u64; (20_000usize).div_ceil(64)];
        set_bit(&mut deleted0, 7500);
        let tombstones = |si: usize| -> &[u64] { if si == 0 { &deleted0 } else { &[] } };
        let scope = index.scope(&|path| crate::repo::glob_match("seg0/file3.rs", path), &tombstones);
        assert_eq!(scope.matched(), 2499);
        assert!(!scope.contains(0, 7500) && scope.contains(0, 7501) && scope.contains(0, 9999));
        assert!(!scope.contains(0, 10_000) && !scope.contains(1, 0));

        let all_rs = index.scope(&|path| path.ends_with(".rs"), &tombstones);
        assert_eq!(all_rs.matched(), 20_000 + 99 - 1);
        assert!(!all_rs.contains(1, 99), "the plain id is not a .rs path");
        assert_eq!(index.scope(&|_| false, &tombstones).matched(), 0);
    }
}
