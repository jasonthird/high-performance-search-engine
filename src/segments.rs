//! Segmented (incrementally updatable) indexes.
//!
//! The single-index format is immutable by design — that immutability is
//! where much of the engine's speed comes from. Mutability is layered *on
//! top*, Lucene-style, never inside:
//!
//! - an index directory holds a `manifest.bin` plus segment subdirectories,
//!   each an ordinary immutable index;
//! - **adding** documents builds a fresh segment from the batch;
//! - **deleting** sets a bit in the owning segment's tombstone file (an
//!   update is delete + add);
//! - **merging** compacts segments into one, dropping tombstoned documents
//!   and renumbering — after which the result is byte-equivalent to a fresh
//!   build of the live documents.
//!
//! Scoring is **globally exact**: queries compute corpus-wide statistics
//! (N over live documents, global average length, df summed across
//! segments) and every segment scores under those, which the impact-based
//! bounds (format v4) make safe. One documented deviation, shared with
//! Lucene: a term's df still counts tombstoned documents until a merge
//! physically removes them.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::block_max_wand::{self, SearchStats};
use crate::indexer::{DocSummary, InputDoc, SearchableIndex};
use crate::postings::{self, Posting, TermPostings, DEFAULT_BLOCK_SIZE};
use crate::reorder::ReorderStrategy;
use crate::searcher::{SearchOutcome, SearchResult};
use crate::storage::{self, DiskIndex};
use crate::tokenizer::Tokenizer;
use crate::{bm25, maxscore};

const MANIFEST_FILE: &str = "manifest.bin";

/// Queries with at least this many unique terms run MaxScore (mirrors the
/// single-index searcher).
const MAXSCORE_MIN_TERMS: usize = 5;

#[derive(Serialize, Deserialize, Clone)]
struct SegmentEntry {
    name: String,
    num_docs: u32,
    /// Total token length of all documents at build time.
    total_len: u64,
    /// Documents not tombstoned.
    live_docs: u32,
    /// Token length of live documents.
    live_len: u64,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    next_segment: u64,
    remove_stopwords: bool,
    title_weight: u32,
    /// Code-oriented tokenizer (identifier splitting); must match between
    /// build and query, and across every segment.
    #[serde(default)]
    code_mode: bool,
    segments: Vec<SegmentEntry>,
}

struct Segment {
    entry: SegmentEntry,
    index: DiskIndex,
    /// Tombstone bitmap (one bit per doc_id), empty when nothing deleted.
    deleted: Vec<u64>,
}

impl Segment {
    fn is_deleted(&self, doc_id: u32) -> bool {
        bit_is_set(&self.deleted, doc_id)
    }
}

/// A read view over all live segments, searched with global statistics.
pub struct SegmentedIndex {
    manifest: Manifest,
    segments: Vec<Segment>,
}

/// Per-segment adapter that scores under global statistics and hides
/// tombstoned documents. The generic evaluators see it as just another
/// index.
struct SegmentView<'a> {
    segment: &'a Segment,
    num_docs_global: usize,
    avg_doc_len_global: f32,
    /// Global idf per query term (df summed across segments).
    idfs: &'a HashMap<String, f32>,
    remove_stopwords: bool,
}

impl SearchableIndex for SegmentView<'_> {
    fn num_docs(&self) -> usize {
        self.num_docs_global
    }

    fn num_terms(&self) -> usize {
        self.segment.index.num_terms()
    }

    fn avg_doc_len(&self) -> f32 {
        self.avg_doc_len_global
    }

    fn remove_stopwords(&self) -> bool {
        self.remove_stopwords
    }

    fn doc_len(&self, doc_id: u32) -> u32 {
        self.segment.index.doc_len(doc_id)
    }

    fn doc_summary(&self, doc_id: u32) -> DocSummary {
        self.segment.index.doc_summary(doc_id)
    }

    fn is_deleted(&self, doc_id: u32) -> bool {
        self.segment.is_deleted(doc_id)
    }

    fn total_postings(&self) -> u64 {
        self.segment.index.total_postings()
    }

    fn term_postings(&self, term: &str) -> Option<TermPostings<'_>> {
        let &idf = self.idfs.get(term)?;
        self.segment
            .index
            .term_postings_with(term, idf, self.avg_doc_len_global)
    }
}

fn read_manifest(dir: &Path) -> anyhow::Result<Manifest> {
    let file = File::open(dir.join(MANIFEST_FILE)).context("failed to open manifest")?;
    bincode::deserialize_from(BufReader::new(file)).context("failed to parse manifest")
}

fn write_manifest(dir: &Path, manifest: &Manifest) -> anyhow::Result<()> {
    let tmp = dir.join("manifest.tmp");
    let mut w = BufWriter::new(File::create(&tmp).context("failed to create manifest")?);
    bincode::serialize_into(&mut w, manifest).context("failed to write manifest")?;
    w.flush()?;
    drop(w);
    fs::rename(&tmp, dir.join(MANIFEST_FILE)).context("failed to commit manifest")?;
    Ok(())
}


/// Test one document's bit in a tombstone bitmap.
fn bit_is_set(words: &[u64], doc: u32) -> bool {
    words
        .get((doc / 64) as usize)
        .is_some_and(|w| (w >> (doc % 64)) & 1 == 1)
}

/// Set one document's bit (the bitmap must already be sized to cover it).
fn set_bit(words: &mut [u64], doc: u32) {
    words[(doc / 64) as usize] |= 1 << (doc % 64);
}

fn tombstone_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.del"))
}

fn read_tombstones(dir: &Path, name: &str, num_docs: u32) -> anyhow::Result<Vec<u64>> {
    let path = tombstone_path(dir, name);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = fs::read(&path)?;
    let mut words = vec![0u64; (num_docs as usize).div_ceil(64)];
    for (i, chunk) in bytes.chunks(8).enumerate() {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        if i < words.len() {
            words[i] = u64::from_le_bytes(buf);
        }
    }
    Ok(words)
}

fn write_tombstones(dir: &Path, name: &str, words: &[u64]) -> anyhow::Result<()> {
    let tmp = dir.join(format!("{name}.del.tmp"));
    let mut w = BufWriter::new(File::create(&tmp)?);
    for word in words {
        w.write_all(&word.to_le_bytes())?;
    }
    w.flush()?;
    drop(w);
    fs::rename(&tmp, tombstone_path(dir, name))?;
    Ok(())
}

/// Does this directory hold a segmented index?
pub fn is_segmented(dir: &Path) -> bool {
    dir.join(MANIFEST_FILE).exists()
}

/// Recompute a segment's live doc count and live length from its tombstone
/// bitmap, given the segment's open index for per-doc lengths.
///
/// Tombstone files and the manifest are written in separate steps; a crash
/// between them leaves the manifest's cached `live_docs`/`live_len` stale.
/// The bitmaps are the source of truth, so both open paths recount from
/// them and the counts self-heal.
fn recount_live(entry: &SegmentEntry, index: &DiskIndex, deleted: &[u64]) -> (u32, u64) {
    let mut dead = 0u32;
    let mut dead_len = 0u64;
    for (wi, &word) in deleted.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            let bit = word.trailing_zeros();
            let doc = wi as u32 * 64 + bit;
            if doc < entry.num_docs {
                dead += 1;
                dead_len += index.doc_len(doc) as u64;
            }
            word &= word - 1;
        }
    }
    (entry.num_docs - dead, entry.total_len - dead_len)
}

impl SegmentedIndex {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let manifest = read_manifest(dir)?;
        let segments = manifest
            .segments
            .iter()
            .map(|entry| {
                let index = storage::load_index(&dir.join(&entry.name))?;
                let deleted = read_tombstones(dir, &entry.name, entry.num_docs)?;
                // Trust the bitmaps, not the manifest's cached counts: a
                // crash between tombstone and manifest writes leaves the
                // cache stale, and global BM25 stats come from these.
                let mut entry = entry.clone();
                (entry.live_docs, entry.live_len) = recount_live(&entry, &index, &deleted);
                Ok(Segment {
                    entry,
                    index,
                    deleted,
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self { manifest, segments })
    }

    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    pub fn num_docs_live(&self) -> u64 {
        self.segments.iter().map(|s| s.entry.live_docs as u64).sum()
    }

    pub fn size_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.index.size_bytes()).sum()
    }

    pub fn remove_stopwords(&self) -> bool {
        self.manifest.remove_stopwords
    }

    pub fn code_mode(&self) -> bool {
        self.manifest.code_mode
    }

    /// Global document frequency of a term (summed across all segments,
    /// tombstones included, matching the idf statistics used at search time).
    pub fn term_df(&self, term: &str) -> u32 {
        self.segments.iter().map(|s| s.index.term_df(term)).sum()
    }

    /// Stream (term, per-segment df) pairs across every segment. The same
    /// term recurs once per segment it appears in; the caller sums the dfs.
    pub fn for_each_term(&self, mut f: impl FnMut(&str, u32)) {
        for seg in &self.segments {
            seg.index.for_each_term(|t, df| f(t, df));
        }
    }

    fn global_stats(&self) -> (usize, f32) {
        let live_docs: u64 = self.num_docs_live();
        let live_len: u64 = self.segments.iter().map(|s| s.entry.live_len).sum();
        let avg = if live_docs == 0 {
            1.0
        } else {
            live_len as f32 / live_docs as f32
        };
        (live_docs as usize, avg)
    }

    /// The shared query pipeline: tokenize + dedup terms, compute global
    /// statistics and per-term global idf (df counts tombstoned docs until
    /// merge, as in Lucene), search every segment in parallel under those
    /// statistics, and merge-sort-truncate the per-segment top-k heaps.
    /// Returns the merged (segment, doc_id, score) triples plus aggregated
    /// engine statistics.
    fn search_core(&self, query: &str, k: usize) -> (Vec<(usize, u32, f32)>, SearchStats) {
        let remove_stopwords = self.manifest.remove_stopwords;
        let tokenizer = Tokenizer::with_flags(remove_stopwords, self.manifest.code_mode);
        let mut terms: Vec<String> = Vec::new();
        tokenizer.for_each_token(query, |t| {
            if !terms.iter().any(|x| x == t) {
                terms.push(t.to_owned());
            }
        });

        let (num_docs_global, avg_doc_len_global) = self.global_stats();
        let mut idfs: HashMap<String, f32> = HashMap::new();
        for term in &terms {
            let df: u64 = self
                .segments
                .iter()
                .map(|s| s.index.term_df(term) as u64)
                .sum();
            if df > 0 {
                idfs.insert(term.clone(), bm25::idf(num_docs_global, df as usize));
            }
        }

        let per_segment: Vec<(usize, Vec<crate::block_max_wand::SearchHit>, SearchStats)> = self
            .segments
            .par_iter()
            .enumerate()
            .map(|(si, segment)| {
                let view = SegmentView {
                    segment,
                    num_docs_global,
                    avg_doc_len_global,
                    idfs: &idfs,
                    remove_stopwords,
                };
                let (hits, stats) = if terms.len() >= MAXSCORE_MIN_TERMS {
                    maxscore::search(&view, &terms, k)
                } else {
                    block_max_wand::search(&view, &terms, k)
                };
                (si, hits, stats)
            })
            .collect();

        let mut stats = SearchStats {
            num_docs_total: num_docs_global,
            num_query_terms: terms.len(),
            ..SearchStats::default()
        };
        let mut merged: Vec<(usize, u32, f32)> = Vec::new();
        for (si, hits, seg_stats) in per_segment {
            stats.num_postings_visited += seg_stats.num_postings_visited;
            stats.num_docs_scored += seg_stats.num_docs_scored;
            stats.num_blocks_visited += seg_stats.num_blocks_visited;
            stats.num_blocks_skipped += seg_stats.num_blocks_skipped;
            for hit in hits {
                merged.push((si, hit.doc_id, hit.score));
            }
        }
        merged.sort_unstable_by(|a, b| {
            b.2.total_cmp(&a.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        merged.truncate(k);
        (merged, stats)
    }

    /// Exact top-k search across all segments under global statistics.
    /// Raw BM25 top-k as (segment, doc_id, score) triples, for callers
    /// (hybrid fusion) that need scores joined with other per-document
    /// signals before summaries are materialized.
    pub fn search_hits_raw(&self, query: &str, k: usize) -> Vec<(usize, u32, f32)> {
        self.search_core(query, k).0
    }

    /// Per-segment directory names, in manifest (and doc-compaction) order.
    pub fn segment_names(&self) -> Vec<String> {
        self.manifest
            .segments
            .iter()
            .map(|e| e.name.clone())
            .collect()
    }

    /// Live (not tombstoned) check for one segment-local doc.
    pub fn is_live(&self, segment: usize, doc_id: u32) -> bool {
        self.segments
            .get(segment)
            .is_some_and(|s| !s.is_deleted(doc_id))
    }

    pub fn doc_summary_in(&self, segment: usize, doc_id: u32) -> DocSummary {
        self.segments[segment].index.doc_summary(doc_id)
    }

    pub fn num_docs_in(&self, segment: usize) -> u32 {
        self.segments[segment].entry.num_docs
    }

    pub fn search(&self, query: &str, k: usize) -> SearchOutcome {
        let start = Instant::now();
        let (merged, stats) = self.search_core(query, k);
        let results = merged
            .into_iter()
            .map(|(si, doc_id, score)| {
                let summary = self.segments[si].index.doc_summary(doc_id);
                SearchResult {
                    id: summary.id,
                    score,
                    title: summary.title,
                }
            })
            .collect();

        SearchOutcome {
            results,
            stats,
            took_ms: start.elapsed().as_secs_f64() * 1000.0,
            corrected: None,
        }
    }
}

/// Mutating operations on a segmented index directory.
/// Outcome of an upsert batch: counts, plus the segment the changed and
/// new documents landed in (with those documents in the segment's doc_id
/// order), so callers can build per-segment sidecars — embeddings, keys —
/// for exactly the documents that moved.
pub struct UpsertOutcome {
    pub added: usize,
    pub updated: usize,
    pub unchanged: usize,
    /// (segment name, documents in segment doc order), when anything was
    /// written.
    pub new_segment: Option<(String, Vec<InputDoc>)>,
}

pub struct SegmentedWriter {
    dir: PathBuf,
    manifest: Manifest,
    /// Exclusive advisory lock on `writer.lock`, held for the writer's
    /// lifetime. Two concurrent writers would otherwise both read the same
    /// `next_segment`, claim the same `seg-NNNNNN` directory, and clobber
    /// each other's manifest. Released on drop (close).
    _lock: File,
}

impl SegmentedWriter {
    /// Open an existing segmented index, or initialize a new one.
    pub fn open_or_create(
        dir: &Path,
        remove_stopwords: bool,
        title_weight: u32,
    ) -> anyhow::Result<Self> {
        Self::open_or_create_ex(dir, remove_stopwords, title_weight, false)
    }

    /// As [`Self::open_or_create`], choosing the tokenizer. `code_mode` is
    /// recorded at creation and must not change for the index's lifetime.
    pub fn open_or_create_ex(
        dir: &Path,
        remove_stopwords: bool,
        title_weight: u32,
        code_mode: bool,
    ) -> anyhow::Result<Self> {
        fs::create_dir_all(dir)?;
        let lock = File::create(dir.join("writer.lock")).context("failed to create writer.lock")?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!(
                "{} is locked by another writer (writer.lock held)",
                dir.display()
            ),
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(e).context("failed to lock writer.lock")
            }
        }
        let manifest = if is_segmented(dir) {
            let mut manifest = read_manifest(dir)?;
            Self::recover(dir, &mut manifest)?;
            manifest
        } else {
            anyhow::ensure!(
                !dir.join("meta.bin").exists(),
                "{} holds a single (non-segmented) index; segmented and single \
                 layouts cannot mix",
                dir.display()
            );
            let manifest = Manifest {
                version: 1,
                next_segment: 0,
                remove_stopwords,
                title_weight,
                code_mode,
                segments: Vec::new(),
            };
            write_manifest(dir, &manifest)?;
            manifest
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
            _lock: lock,
        })
    }

    /// Crash recovery, run once at writer open with the lock held:
    /// self-heal the manifest's cached live counts from the tombstone
    /// bitmaps, and garbage-collect segment directories and tombstone
    /// files that no manifest entry references (a crash mid-`merge_all`
    /// or mid-add leaves them behind).
    fn recover(dir: &Path, manifest: &mut Manifest) -> anyhow::Result<()> {
        let mut healed = false;
        for entry in manifest.segments.iter_mut() {
            let deleted = read_tombstones(dir, &entry.name, entry.num_docs)?;
            if deleted.iter().all(|&w| w == 0) {
                continue;
            }
            let index = storage::load_index(&dir.join(&entry.name))?;
            let (live_docs, live_len) = recount_live(entry, &index, &deleted);
            if (live_docs, live_len) != (entry.live_docs, entry.live_len) {
                (entry.live_docs, entry.live_len) = (live_docs, live_len);
                healed = true;
            }
        }
        if healed {
            write_manifest(dir, manifest)?;
        }

        let referenced: std::collections::HashSet<&str> =
            manifest.segments.iter().map(|e| e.name.as_str()).collect();
        for e in fs::read_dir(dir)? {
            let e = e?;
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            if let Some(seg) = name.strip_suffix(".del") {
                if !referenced.contains(seg) {
                    fs::remove_file(e.path()).ok();
                }
            } else if name.starts_with("seg-")
                && e.file_type()?.is_dir()
                && !referenced.contains(name)
            {
                fs::remove_dir_all(e.path()).ok();
            }
        }
        Ok(())
    }

    /// Add a batch of documents as one new segment.
    pub fn add_documents(&mut self, docs: &[InputDoc]) -> anyhow::Result<String> {
        anyhow::ensure!(!docs.is_empty(), "no documents to add");
        let name = format!("seg-{:06}", self.manifest.next_segment);
        let index = crate::indexer::build_index_weighted_ex(
            docs,
            self.manifest.remove_stopwords,
            self.manifest.title_weight,
            DEFAULT_BLOCK_SIZE,
            ReorderStrategy::None,
            self.manifest.code_mode,
        );
        storage::save_index(&index, &self.dir.join(&name))?;
        let total_len: u64 = index.docs().iter().map(|d| d.doc_len as u64).sum();
        self.manifest.segments.push(SegmentEntry {
            name: name.clone(),
            num_docs: index.docs().len() as u32,
            total_len,
            live_docs: index.docs().len() as u32,
            live_len: total_len,
        });
        self.manifest.next_segment += 1;
        write_manifest(&self.dir, &self.manifest)?;
        Ok(name)
    }

    /// Tombstone a batch of documents by external id in one pass: each
    /// segment is opened once and its tombstone file written once, however
    /// many ids it holds. A 200-file refactor tombstones hundreds of chunks;
    /// per-id deletion would reload every segment for each of them.
    /// Returns how many ids were found and tombstoned.
    pub fn delete_documents(&mut self, external_ids: &[String]) -> anyhow::Result<usize> {
        if external_ids.is_empty() {
            return Ok(0);
        }
        let mut remaining: std::collections::HashSet<&str> =
            external_ids.iter().map(String::as_str).collect();
        let mut deleted = 0usize;
        for entry in self.manifest.segments.iter_mut() {
            if remaining.is_empty() {
                break;
            }
            let index = storage::load_index(&self.dir.join(&entry.name))?;
            let mut tombstones: Option<Vec<u64>> = None;
            let mut found: Vec<&str> = Vec::new();
            for &id in remaining.iter() {
                let Some(doc_id) = index.find_by_external_id(id) else {
                    continue;
                };
                let words = match &mut tombstones {
                    Some(w) => w,
                    None => {
                        let mut w = read_tombstones(&self.dir, &entry.name, entry.num_docs)?;
                        w.resize((entry.num_docs as usize).div_ceil(64), 0);
                        tombstones.insert(w)
                    }
                };
                if bit_is_set(words, doc_id) {
                    continue; // already tombstoned
                }
                set_bit(words, doc_id);
                entry.live_docs -= 1;
                entry.live_len -= index.doc_len(doc_id) as u64;
                deleted += 1;
                found.push(id);
            }
            if let Some(words) = tombstones {
                write_tombstones(&self.dir, &entry.name, &words)?;
            }
            for id in found {
                remaining.remove(id);
            }
        }
        if deleted > 0 {
            write_manifest(&self.dir, &self.manifest)?;
        }
        Ok(deleted)
    }

    /// Tombstone a document by external id. Returns true if found.
    pub fn delete_document(&mut self, external_id: &str) -> anyhow::Result<bool> {
        let id = external_id.to_owned();
        Ok(self.delete_documents(std::slice::from_ref(&id))? > 0)
    }

    /// Update = delete (if present) + add as a new segment.
    pub fn update_document(&mut self, doc: InputDoc) -> anyhow::Result<()> {
        self.delete_document(&doc.id)?;
        self.add_documents(&[doc])?;
        Ok(())
    }

    /// Upsert a batch with change detection: documents whose stored content
    /// hash matches are skipped entirely; changed documents are tombstoned
    /// and re-added; new documents are added. Re-feeding an unchanged
    /// corpus is a no-op. Returns (added, updated, unchanged).
    pub fn upsert_documents(&mut self, docs: &[InputDoc]) -> anyhow::Result<(usize, usize, usize)> {
        let outcome = self.upsert_documents_full(docs)?;
        Ok((outcome.added, outcome.updated, outcome.unchanged))
    }

    /// As [`Self::upsert_documents`], reporting the created segment too.
    pub fn upsert_documents_full(&mut self, docs: &[InputDoc]) -> anyhow::Result<UpsertOutcome> {
        // Resolve against current live segments once.
        let segments: Vec<(String, DiskIndex, Vec<u64>, u32)> = self
            .manifest
            .segments
            .iter()
            .map(|entry| {
                let index = storage::load_index(&self.dir.join(&entry.name))?;
                let deleted = read_tombstones(&self.dir, &entry.name, entry.num_docs)?;
                Ok((entry.name.clone(), index, deleted, entry.num_docs))
            })
            .collect::<anyhow::Result<_>>()?;
        let live_lookup = |id: &str| -> Option<(usize, u32, u64)> {
            for (si, (_, index, deleted, _)) in segments.iter().enumerate() {
                if let Some(doc_id) = index.find_by_external_id(id) {
                    if !bit_is_set(deleted, doc_id) {
                        return Some((si, doc_id, index.content_hash(doc_id)));
                    }
                }
            }
            None
        };

        // Within one batch the last occurrence of an id wins; without this
        // dedup every copy would be classified `added` and land live in the
        // new segment, and later deletes would only find the first.
        let docs: Vec<&InputDoc> = {
            let mut last: HashMap<&str, usize> = HashMap::new();
            for (i, d) in docs.iter().enumerate() {
                last.insert(d.id.as_str(), i);
            }
            docs.iter()
                .enumerate()
                .filter(|(i, d)| last[d.id.as_str()] == *i)
                .map(|(_, d)| d)
                .collect()
        };

        let mut to_add: Vec<InputDoc> = Vec::new();
        let mut to_delete: Vec<String> = Vec::new();
        let mut added = 0usize;
        let mut updated = 0usize;
        let mut unchanged = 0usize;
        for &doc in &docs {
            let new_hash = crate::indexer::content_hash(&doc.title, &doc.body);
            match live_lookup(&doc.id) {
                Some((_, _, stored)) if stored == new_hash => unchanged += 1,
                Some(_) => {
                    to_delete.push(doc.id.clone());
                    to_add.push(doc.clone());
                    updated += 1;
                }
                None => {
                    to_add.push(doc.clone());
                    added += 1;
                }
            }
        }
        self.delete_documents(&to_delete)?;
        let new_segment = if to_add.is_empty() {
            None
        } else {
            let name = self.add_documents(&to_add)?;
            Some((name, to_add))
        };
        Ok(UpsertOutcome {
            added,
            updated,
            unchanged,
            new_segment,
        })
    }

    /// Merge every segment into one, dropping tombstoned documents.
    ///
    /// Works at the postings level: term dictionaries k-way merge by name,
    /// posting lists decode, doc_ids remap past tombstones (compacting in
    /// segment order, exactly like a fresh sequential build), and blocks
    /// re-encode with fresh impacts. Original tf and doc_len data carry
    /// through untouched, and df no longer counts deleted documents — the
    /// merged segment scores identically to a from-scratch rebuild of the
    /// live documents.
    pub fn merge_all(&mut self) -> anyhow::Result<()> {
        let needs_merge = self.manifest.segments.len() > 1
            || self
                .manifest
                .segments
                .first()
                .is_some_and(|e| e.live_docs != e.num_docs);
        if !needs_merge {
            return Ok(());
        }

        let old = self.manifest.segments.clone();
        let segments: Vec<(DiskIndex, Vec<u64>)> = old
            .iter()
            .map(|entry| {
                let index = storage::load_index(&self.dir.join(&entry.name))?;
                let deleted = read_tombstones(&self.dir, &entry.name, entry.num_docs)?;
                Ok((index, deleted))
            })
            .collect::<anyhow::Result<_>>()?;


        // Doc id remap: live docs renumber densely in (segment, doc) order.
        let mut remap: Vec<Vec<Option<u32>>> = Vec::with_capacity(segments.len());
        let mut next_id = 0u32;
        for ((index, deleted), entry) in segments.iter().zip(&old) {
            let _ = index;
            let mut seg_map = Vec::with_capacity(entry.num_docs as usize);
            for doc in 0..entry.num_docs {
                if bit_is_set(deleted, doc) {
                    seg_map.push(None);
                } else {
                    seg_map.push(Some(next_id));
                    next_id += 1;
                }
            }
            remap.push(seg_map);
        }

        let name = format!("seg-{:06}", self.manifest.next_segment);
        let out_dir = self.dir.join(&name);
        fs::create_dir_all(&out_dir)?;

        // --- docs.bin + doc_lens + ids.bin: raw-copy live records --------
        let mut doc_lens: Vec<u32> = Vec::new();
        let mut doc_offsets: Vec<u64> = vec![0];
        let mut id_hashes: Vec<u64> = Vec::new();
        let mut content_hashes: Vec<u64> = Vec::new();
        {
            let docs_path = out_dir.join("docs.bin");
            let mut w = BufWriter::new(File::create(&docs_path)?);
            for ((index, _), seg_map) in segments.iter().zip(&remap) {
                for (doc, mapped) in seg_map.iter().enumerate() {
                    if mapped.is_none() {
                        continue;
                    }
                    let record = index.doc_record_bytes(doc as u32);
                    w.write_all(record)?;
                    doc_offsets.push(doc_offsets.last().unwrap() + record.len() as u64);
                    doc_lens.push(index.doc_len(doc as u32));
                    id_hashes.push(storage::id_hash(&index.doc_summary(doc as u32).id));
                    content_hashes.push(index.content_hash(doc as u32));
                }
            }
            w.flush()?;
        }
        storage::write_ids_file(&out_dir, id_hashes.into_iter())?;
        storage::write_hashes_file(&out_dir, content_hashes.into_iter())?;

        let num_docs = doc_lens.len();
        let total_len: u64 = doc_lens.iter().map(|&l| l as u64).sum();

        // --- k-way dictionary merge by term name --------------------------
        // Cursors over each segment's name-sorted dictionary.
        let mut cursors: Vec<(usize, usize, String)> = segments
            .iter()
            .enumerate()
            .filter(|(_, (index, _))| index.num_terms() > 0)
            .map(|(si, (index, _))| (si, 0usize, index.term_at(0)))
            .collect();

        let block_size = DEFAULT_BLOCK_SIZE;
        let postings_path = out_dir.join("postings.bin");
        let mut postings_writer = BufWriter::with_capacity(1 << 20, File::create(&postings_path)?);
        let mut term_names: Vec<String> = Vec::new();
        let mut term_dfs: Vec<u32> = Vec::new();
        let mut region_offsets: Vec<u64> = vec![0];
        let mut block_rows: Vec<u32> = vec![0];
        let mut term_max_tfs: Vec<u32> = Vec::new();
        let mut term_min_lens: Vec<u32> = Vec::new();
        let mut block_max_doc_ids: Vec<u32> = Vec::new();
        let mut block_max_tfs: Vec<u32> = Vec::new();
        let mut block_min_lens: Vec<u32> = Vec::new();
        let mut block_byte_offsets: Vec<u32> = Vec::new();

        let mut decoded: Vec<Posting> = Vec::new();
        let mut merged_list: Vec<Posting> = Vec::new();
        let mut encoded: Vec<u8> = Vec::new();
        while !cursors.is_empty() {
            // Smallest current name across cursors.
            // Min by &str, cloning only the winner (not every cursor's term).
            let term = cursors
                .iter()
                .map(|(_, _, name)| name.as_str())
                .min()
                .expect("non-empty")
                .to_owned();

            merged_list.clear();
            for (si, _, name) in cursors.iter() {
                if *name != term {
                    continue;
                }
                let (index, _) = &segments[*si];
                let tp = index
                    .term_postings(&term)
                    .expect("term listed in dictionary");
                for b in 0..tp.num_blocks() {
                    tp.decode_block(b, &mut decoded);
                    for p in &decoded {
                        if let Some(new_id) = remap[*si][p.doc_id as usize] {
                            merged_list.push(Posting {
                                doc_id: new_id,
                                tf: p.tf,
                            });
                        }
                    }
                }
            }
            // Advance the cursors that were on this term.
            for c in cursors.iter_mut() {
                if c.2 == term {
                    c.1 += 1;
                    if c.1 < segments[c.0].0.num_terms() {
                        c.2 = segments[c.0].0.term_at(c.1);
                    }
                }
            }
            cursors.retain(|c| c.1 < segments[c.0].0.num_terms());

            if merged_list.is_empty() {
                continue; // every posting belonged to tombstoned docs
            }
            // Segment order + dense remap keeps doc_ids ascending.
            let (max_ids, max_tfs, min_lens) =
                postings::build_blocks(&merged_list, &doc_lens, block_size);
            encoded.clear();
            for chunk in merged_list.chunks(block_size) {
                block_byte_offsets.push(encoded.len() as u32);
                crate::compress::encode_block(chunk, &mut encoded);
            }
            postings_writer.write_all(&encoded)?;
            term_names.push(term);
            term_dfs.push(merged_list.len() as u32);
            region_offsets.push(region_offsets.last().unwrap() + encoded.len() as u64);
            block_rows.push(block_rows.last().unwrap() + max_ids.len() as u32);
            term_max_tfs.push(max_tfs.iter().copied().max().unwrap_or(1));
            term_min_lens.push(min_lens.iter().copied().min().unwrap_or(1));
            block_max_doc_ids.extend_from_slice(&max_ids);
            block_max_tfs.extend_from_slice(&max_tfs);
            block_min_lens.extend_from_slice(&min_lens);
        }
        postings_writer.flush()?;
        drop(postings_writer);

        let num_terms = term_names.len();
        let (dict_groups, dict_bytes) =
            storage::front_code_dict(term_names.iter().map(|t| t.as_str()), num_terms);
        let meta = storage::MetaSections {
            avg_doc_len: if num_docs == 0 {
                1.0
            } else {
                total_len as f32 / num_docs as f32
            },
            remove_stopwords: self.manifest.remove_stopwords,
            code_mode: self.manifest.code_mode,
            block_size: block_size as u32,
            doc_lens,
            doc_offsets,
            term_dfs,
            region_offsets,
            block_rows,
            term_max_tfs,
            term_min_lens,
            name_to_slot: (0..num_terms as u32).collect(),
            block_max_doc_ids,
            block_max_tfs,
            block_min_lens,
            block_byte_offsets,
        };
        storage::write_meta(&meta, (&dict_groups, &dict_bytes), &out_dir)?;

        // Commit: new manifest first, then remove the old segments.
        self.manifest.segments = vec![SegmentEntry {
            name: name.clone(),
            num_docs: num_docs as u32,
            total_len,
            live_docs: num_docs as u32,
            live_len: total_len,
        }];
        self.manifest.next_segment += 1;
        write_manifest(&self.dir, &self.manifest)?;
        for entry in &old {
            fs::remove_dir_all(self.dir.join(&entry.name)).ok();
            fs::remove_file(tombstone_path(&self.dir, &entry.name)).ok();
        }
        Ok(())
    }
}
