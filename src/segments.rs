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
use crate::indexer::{build_index_weighted, DocSummary, InputDoc, SearchableIndex};
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
        let word = (doc_id / 64) as usize;
        self.deleted
            .get(word)
            .is_some_and(|w| (w >> (doc_id % 64)) & 1 == 1)
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

impl SegmentedIndex {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let manifest = read_manifest(dir)?;
        let segments = manifest
            .segments
            .iter()
            .map(|entry| {
                let index = storage::load_index(&dir.join(&entry.name))?;
                let deleted = read_tombstones(dir, &entry.name, entry.num_docs)?;
                Ok(Segment {
                    entry: entry.clone(),
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

    /// Exact top-k search across all segments under global statistics.
    pub fn search(&self, query: &str, k: usize) -> SearchOutcome {
        let remove_stopwords = self.manifest.remove_stopwords;
        let tokenizer = Tokenizer::new(remove_stopwords);
        let mut terms: Vec<String> = Vec::new();
        tokenizer.for_each_token(query, |t| {
            if !terms.iter().any(|x| x == t) {
                terms.push(t.to_owned());
            }
        });

        let start = Instant::now();
        let (num_docs_global, avg_doc_len_global) = self.global_stats();

        // Global df per term -> global idf (df counts tombstoned docs until
        // merge, as in Lucene).
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

        // Each segment is searched independently (in parallel) and the
        // per-segment top-k heaps merge into the final ranking.
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
        let mut merged: Vec<(f32, usize, u32)> = Vec::new();
        for (si, hits, seg_stats) in per_segment {
            stats.num_postings_visited += seg_stats.num_postings_visited;
            stats.num_docs_scored += seg_stats.num_docs_scored;
            stats.num_blocks_visited += seg_stats.num_blocks_visited;
            stats.num_blocks_skipped += seg_stats.num_blocks_skipped;
            for hit in hits {
                merged.push((hit.score, si, hit.doc_id));
            }
        }
        merged.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        merged.truncate(k);

        let results = merged
            .into_iter()
            .map(|(score, si, doc_id)| {
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
        }
    }
}

/// Mutating operations on a segmented index directory.
pub struct SegmentedWriter {
    dir: PathBuf,
    manifest: Manifest,
}

impl SegmentedWriter {
    /// Open an existing segmented index, or initialize a new one.
    pub fn open_or_create(
        dir: &Path,
        remove_stopwords: bool,
        title_weight: u32,
    ) -> anyhow::Result<Self> {
        fs::create_dir_all(dir)?;
        let manifest = if is_segmented(dir) {
            read_manifest(dir)?
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
                segments: Vec::new(),
            };
            write_manifest(dir, &manifest)?;
            manifest
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
        })
    }

    /// Add a batch of documents as one new segment.
    pub fn add_documents(&mut self, docs: &[InputDoc]) -> anyhow::Result<String> {
        anyhow::ensure!(!docs.is_empty(), "no documents to add");
        let name = format!("seg-{:06}", self.manifest.next_segment);
        let index = build_index_weighted(
            docs,
            self.manifest.remove_stopwords,
            self.manifest.title_weight,
            DEFAULT_BLOCK_SIZE,
            ReorderStrategy::None,
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

    /// Tombstone a document by external id. Returns true if found.
    pub fn delete_document(&mut self, external_id: &str) -> anyhow::Result<bool> {
        for entry in self.manifest.segments.iter_mut() {
            let index = storage::load_index(&self.dir.join(&entry.name))?;
            let Some(doc_id) = index.find_by_external_id(external_id) else {
                continue;
            };
            let mut deleted = read_tombstones(&self.dir, &entry.name, entry.num_docs)?;
            deleted.resize((entry.num_docs as usize).div_ceil(64), 0);
            let word = (doc_id / 64) as usize;
            if (deleted[word] >> (doc_id % 64)) & 1 == 1 {
                continue; // already tombstoned; treat as not-found here
            }
            deleted[word] |= 1 << (doc_id % 64);
            write_tombstones(&self.dir, &entry.name, &deleted)?;
            entry.live_docs -= 1;
            entry.live_len -= index.doc_len(doc_id) as u64;
            write_manifest(&self.dir, &self.manifest)?;
            return Ok(true);
        }
        Ok(false)
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
                    let dead = deleted
                        .get((doc_id / 64) as usize)
                        .is_some_and(|w| (w >> (doc_id % 64)) & 1 == 1);
                    if !dead {
                        return Some((si, doc_id, index.content_hash(doc_id)));
                    }
                }
            }
            None
        };

        let mut to_add: Vec<InputDoc> = Vec::new();
        let mut added = 0usize;
        let mut updated = 0usize;
        let mut unchanged = 0usize;
        for doc in docs {
            let new_hash = crate::indexer::content_hash(&doc.title, &doc.body);
            match live_lookup(&doc.id) {
                Some((_, _, stored)) if stored == new_hash => unchanged += 1,
                Some(_) => {
                    self.delete_document(&doc.id)?;
                    to_add.push(doc.clone());
                    updated += 1;
                }
                None => {
                    to_add.push(doc.clone());
                    added += 1;
                }
            }
        }
        if !to_add.is_empty() {
            self.add_documents(&to_add)?;
        }
        Ok((added, updated, unchanged))
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

        let is_dead = |deleted: &Vec<u64>, doc: u32| -> bool {
            deleted
                .get((doc / 64) as usize)
                .is_some_and(|w| (w >> (doc % 64)) & 1 == 1)
        };

        // Doc id remap: live docs renumber densely in (segment, doc) order.
        let mut remap: Vec<Vec<Option<u32>>> = Vec::with_capacity(segments.len());
        let mut next_id = 0u32;
        for ((index, deleted), entry) in segments.iter().zip(&old) {
            let _ = index;
            let mut seg_map = Vec::with_capacity(entry.num_docs as usize);
            for doc in 0..entry.num_docs {
                if is_dead(deleted, doc) {
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
            let term = cursors
                .iter()
                .map(|(_, _, name)| name.clone())
                .min()
                .expect("non-empty");

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
