//! Multithreaded construction of the inverted index.
//!
//! Pipeline:
//! 1. Parse JSONL lines in parallel (rayon).
//! 2. Tokenize documents in parallel; record per-document term counts.
//! 3. Build partial inverted indexes per chunk of documents in parallel.
//! 4. Merge the partial indexes, sort every posting list by doc_id, and
//!    build per-block metadata.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::RwLock;

use anyhow::Context;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::postings::{BlockSource, Posting, PostingList, TermPostings};

/// The fields of a document needed to present a search hit.
#[derive(Debug, Clone)]
pub struct DocSummary {
    pub id: String,
    pub title: String,
}
use crate::reorder::{self, ReorderStrategy};
use crate::tokenizer::Tokenizer;

/// Read-only interface every searchable index implements: the in-memory
/// [`Index`] produced by the builder and the mmap-backed
/// [`crate::storage::DiskIndex`] used in production.
pub trait SearchableIndex: Sync {
    fn num_docs(&self) -> usize;
    fn num_terms(&self) -> usize;
    fn avg_doc_len(&self) -> f32;
    fn remove_stopwords(&self) -> bool;
    /// Length in tokens of one document (the only per-doc fact scoring needs).
    fn doc_len(&self, doc_id: u32) -> u32;
    /// External id and title of one document (only fetched for the top-k
    /// hits actually returned).
    fn doc_summary(&self, doc_id: u32) -> DocSummary;
    /// Tombstone check (segmented indexes); immutable indexes never delete.
    fn is_deleted(&self, _doc_id: u32) -> bool {
        false
    }
    fn total_postings(&self) -> u64;
    /// Posting data for one term, or None if the term is not in the index.
    fn term_postings(&self, term: &str) -> Option<TermPostings<'_>>;
}

/// One input document, as found in the JSONL file.
#[derive(Debug, Clone, Deserialize)]
pub struct InputDoc {
    pub id: String,
    pub title: String,
    pub body: String,
}

/// Per-document metadata stored in the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMeta {
    /// External document id (e.g. "doc-123").
    pub external_id: String,
    pub title: String,
    /// First few hundred characters of the body.
    pub snippet: String,
    /// Document length in tokens (after tokenization).
    pub doc_len: u32,
    /// Hash of (title, body) at index time — change detection for upserts.
    pub content_hash: u64,
}

/// Content hash over title and body (order-sensitive, separator-safe).
pub fn content_hash(title: &str, body: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    feed(title.as_bytes());
    feed(&[0xFF]);
    feed(body.as_bytes());
    h
}

/// The immutable, searchable index.
#[derive(Serialize, Deserialize)]
pub struct Index {
    docs: Vec<DocMeta>,
    avg_doc_len: f32,
    /// term string -> term_id (term_id indexes into `posting_lists`).
    term_dict: HashMap<String, u32>,
    posting_lists: Vec<PostingList>,
    /// Tokenizer setting used at index time; queries must match it.
    remove_stopwords: bool,
    /// Postings per block (fixed for the whole index).
    block_size: usize,
}

impl Index {
    pub fn docs(&self) -> &[DocMeta] {
        &self.docs
    }

    pub fn term_id(&self, term: &str) -> Option<u32> {
        self.term_dict.get(term).copied()
    }

    pub fn term_dict(&self) -> &HashMap<String, u32> {
        &self.term_dict
    }

    pub fn posting_list(&self, term_id: u32) -> &PostingList {
        &self.posting_lists[term_id as usize]
    }

    pub fn posting_list_for(&self, term: &str) -> Option<&PostingList> {
        self.term_id(term).map(|id| self.posting_list(id))
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

impl SearchableIndex for Index {
    fn num_docs(&self) -> usize {
        self.docs.len()
    }

    fn num_terms(&self) -> usize {
        self.posting_lists.len()
    }

    fn avg_doc_len(&self) -> f32 {
        self.avg_doc_len
    }

    fn remove_stopwords(&self) -> bool {
        self.remove_stopwords
    }

    fn doc_len(&self, doc_id: u32) -> u32 {
        self.docs[doc_id as usize].doc_len
    }

    fn doc_summary(&self, doc_id: u32) -> DocSummary {
        let meta = &self.docs[doc_id as usize];
        DocSummary {
            id: meta.external_id.clone(),
            title: meta.title.clone(),
        }
    }

    fn total_postings(&self) -> u64 {
        self.posting_lists
            .iter()
            .map(|l| l.postings.len() as u64)
            .sum()
    }

    fn term_postings(&self, term: &str) -> Option<TermPostings<'_>> {
        let list = self.posting_list_for(term)?;
        if list.postings.is_empty() {
            return None;
        }
        Some(TermPostings {
            idf: list.idf,
            avg_doc_len: self.avg_doc_len,
            df: list.df(),
            block_size: self.block_size,
            block_max_doc_ids: &list.block_max_doc_ids,
            block_max_tfs: &list.block_max_tfs,
            block_min_lens: &list.block_min_lens,
            term_max_tf: list.term_max_tf,
            term_min_len: list.term_min_len,
            source: BlockSource::Memory(&list.postings),
        })
    }
}

/// Parse JSONL text into documents, one JSON object per non-empty line.
pub fn parse_jsonl(text: &str) -> anyhow::Result<Vec<InputDoc>> {
    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .collect();
    lines
        .par_iter()
        .map(|(line_no, line)| {
            let mut bytes = line.as_bytes().to_vec();
            simd_json::serde::from_slice::<InputDoc>(&mut bytes)
                .with_context(|| format!("invalid JSON on line {}", line_no + 1))
        })
        .collect()
}

/// Documents processed per streamed chunk: large enough to keep every core
/// busy, small enough that raw text never accumulates.
const INDEX_CHUNK_SIZE: usize = 8192;

/// Concurrent string interner: term -> dense u32 id. Sharded so parallel
/// tokenizers rarely contend; hits (the overwhelming majority once the
/// vocabulary is warm) take only a shard read-lock and never allocate.
pub(crate) struct Interner {
    shards: Vec<RwLock<HashMap<Box<str>, u32>>>,
    next_id: AtomicU32,
}

const INTERNER_SHARDS: usize = 64;

impl Interner {
    pub(crate) fn new() -> Self {
        Self {
            shards: (0..INTERNER_SHARDS)
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
            next_id: AtomicU32::new(0),
        }
    }

    fn shard_of(term: &str) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        term.hash(&mut hasher);
        (hasher.finish() as usize) % INTERNER_SHARDS
    }

    pub(crate) fn get_or_insert(&self, term: &str) -> u32 {
        let shard = &self.shards[Self::shard_of(term)];
        if let Some(&id) = shard.read().expect("interner lock").get(term) {
            return id;
        }
        let mut write = shard.write().expect("interner lock");
        if let Some(&id) = write.get(term) {
            return id;
        }
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        write.insert(term.into(), id);
        id
    }

    /// All interned terms, indexed by id.
    pub(crate) fn into_names(self) -> Vec<String> {
        let n = self.next_id.into_inner() as usize;
        let mut names = vec![String::new(); n];
        for shard in self.shards {
            for (term, id) in shard.into_inner().expect("interner lock") {
                names[id as usize] = term.into();
            }
        }
        names
    }
}

/// One tokenized document: metadata plus its distinct (term_id, tf) pairs —
/// 8 bytes per posting instead of a String-keyed map.
type TokenizedDoc = (DocMeta, Vec<(u32, u32)>);

/// Incremental index builder: feed it chunks of parsed documents, then
/// `finish`. Raw document text never outlives its chunk.
pub struct IndexBuilder {
    tokenizer: Tokenizer,
    interner: Interner,
    docs: Vec<TokenizedDoc>,
    remove_stopwords: bool,
    title_weight: u32,
    block_size: usize,
    reorder: ReorderStrategy,
}

impl IndexBuilder {
    pub fn new(
        remove_stopwords: bool,
        title_weight: u32,
        block_size: usize,
        reorder: ReorderStrategy,
    ) -> Self {
        Self {
            tokenizer: Tokenizer::new(remove_stopwords),
            interner: Interner::new(),
            docs: Vec::new(),
            remove_stopwords,
            title_weight: title_weight.max(1),
            block_size,
            reorder,
        }
    }

    /// Tokenize one chunk of documents in parallel and append the compact
    /// results.
    pub fn add_documents(&mut self, chunk: &[InputDoc]) {
        let tokenizer = &self.tokenizer;
        let interner = &self.interner;
        let title_weight = self.title_weight;
        let mut tokenized: Vec<TokenizedDoc> = chunk
            .par_iter()
            .map(|doc| {
                // Title and body are indexed together as one searchable
                // field, BM25F-lite style: a title occurrence counts as
                // `title_weight` occurrences in the folded tf.
                let mut ids: Vec<(u32, u32)> = Vec::new();
                let mut doc_len = 0u32;
                tokenizer.for_each_token(&doc.title, |t| {
                    ids.push((interner.get_or_insert(t), title_weight));
                    doc_len += 1;
                });
                tokenizer.for_each_token(&doc.body, |t| {
                    ids.push((interner.get_or_insert(t), 1));
                    doc_len += 1;
                });
                // Distinct (term, tf) pairs via sort + run-length sum.
                ids.sort_unstable_by_key(|&(id, _)| id);
                let mut pairs: Vec<(u32, u32)> = Vec::new();
                for &(id, w) in &ids {
                    match pairs.last_mut() {
                        Some((last, tf)) if *last == id => *tf += w,
                        _ => pairs.push((id, w)),
                    }
                }
                let meta = DocMeta {
                    external_id: doc.id.clone(),
                    title: doc.title.clone(),
                    snippet: doc.body.chars().take(200).collect(),
                    doc_len,
                    content_hash: content_hash(&doc.title, &doc.body),
                };
                (meta, pairs)
            })
            .collect();
        self.docs.append(&mut tokenized);
    }

    pub fn finish(self) -> Index {
        let Self {
            interner,
            docs,
            remove_stopwords,
            block_size,
            reorder,
            ..
        } = self;

        // Document reordering (doc_id assignment). A pure renumbering: it
        // can change index size, never search results.
        let reorder_start = std::time::Instant::now();
        let docs = apply_reordering(docs, reorder);
        if reorder != ReorderStrategy::None {
            println!(
                "reordering ({reorder:?}) took {:.2}s",
                reorder_start.elapsed().as_secs_f64()
            );
        }

        let num_docs = docs.len();
        let total_len: u64 = docs.iter().map(|(m, _)| m.doc_len as u64).sum();
        let avg_doc_len = if total_len == 0 {
            1.0 // avoid division by zero; irrelevant when there are no postings
        } else {
            total_len as f32 / num_docs as f32
        };

        // Final term ids are sorted by term text (deterministic, and the
        // on-disk dictionary requires sorted order); remap interner ids.
        let names = interner.into_names();
        let num_terms = names.len();
        let mut sorted_ids: Vec<u32> = (0..num_terms as u32).collect();
        sorted_ids.sort_unstable_by(|&a, &b| names[a as usize].cmp(&names[b as usize]));
        let mut remap = vec![0u32; num_terms];
        for (new_id, &old_id) in sorted_ids.iter().enumerate() {
            remap[old_id as usize] = new_id as u32;
        }

        // Document frequencies, then one sequential scatter pass in
        // ascending doc order: every posting lands in its (preallocated)
        // list already sorted by doc_id — no hash maps, no merge, no sort.
        let mut dfs = vec![0u32; num_terms];
        for (_, pairs) in &docs {
            for &(old_id, _) in pairs {
                dfs[remap[old_id as usize] as usize] += 1;
            }
        }
        let mut postings_per_term: Vec<Vec<Posting>> = dfs
            .iter()
            .map(|&df| Vec::with_capacity(df as usize))
            .collect();
        for (doc_id, (_, pairs)) in docs.iter().enumerate() {
            for &(old_id, tf) in pairs {
                postings_per_term[remap[old_id as usize] as usize].push(Posting {
                    doc_id: doc_id as u32,
                    tf,
                });
            }
        }

        let doc_lens: Vec<u32> = docs.iter().map(|(m, _)| m.doc_len).collect();
        let posting_lists: Vec<PostingList> = postings_per_term
            .into_par_iter()
            .map(|postings| PostingList::build(postings, num_docs, &doc_lens, block_size))
            .collect();

        let term_dict: HashMap<String, u32> = sorted_ids
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (names[old_id as usize].clone(), new_id as u32))
            .collect();

        Index {
            docs: docs.into_iter().map(|(meta, _)| meta).collect(),
            avg_doc_len,
            term_dict,
            posting_lists,
            remove_stopwords,
            block_size,
        }
    }
}

/// Permute tokenized documents according to the chosen reordering strategy.
fn apply_reordering(docs: Vec<TokenizedDoc>, strategy: ReorderStrategy) -> Vec<TokenizedDoc> {
    let order: Vec<u32> = match strategy {
        ReorderStrategy::None => return docs,
        ReorderStrategy::Path => {
            // Lexicographic order of external ids clusters files from the
            // same directory / pages from the same site.
            let mut order: Vec<u32> = (0..docs.len() as u32).collect();
            order.sort_by(|&a, &b| {
                docs[a as usize]
                    .0
                    .external_id
                    .cmp(&docs[b as usize].0.external_id)
            });
            order
        }
        ReorderStrategy::Bp | ReorderStrategy::BpGpu => {
            // The compact pairs already carry dense term ids.
            let doc_terms: Vec<Vec<u32>> = docs
                .iter()
                .map(|(_, pairs)| pairs.iter().map(|&(t, _)| t).collect())
                .collect();
            if strategy == ReorderStrategy::BpGpu {
                #[cfg(feature = "gpu")]
                {
                    reorder::gpu::bp_order_gpu(&doc_terms)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    // The CLI refuses bp-gpu without the feature; this guards
                    // direct library callers.
                    panic!("BpGpu requires building with `--features gpu`");
                }
            } else {
                reorder::bp_order(&doc_terms)
            }
        }
    };

    let mut slots: Vec<Option<TokenizedDoc>> = docs.into_iter().map(Some).collect();
    order
        .iter()
        .map(|&old| {
            slots[old as usize]
                .take()
                .expect("order must be a permutation")
        })
        .collect()
}

/// Build the inverted index from parsed documents.
///
/// Internal doc_ids are assigned by position after reordering: documents
/// placed next to each other share small doc_id gaps, which is what the
/// reordering strategies optimize for compression.
pub fn build_index(
    docs: &[InputDoc],
    remove_stopwords: bool,
    block_size: usize,
    reorder: ReorderStrategy,
) -> Index {
    build_index_weighted(docs, remove_stopwords, 1, block_size, reorder)
}

/// [`build_index`] with a BM25F-lite title weight: each title occurrence of
/// a term counts as `title_weight` occurrences in the folded tf (document
/// length stays the plain token count).
pub fn build_index_weighted(
    docs: &[InputDoc],
    remove_stopwords: bool,
    title_weight: u32,
    block_size: usize,
    reorder: ReorderStrategy,
) -> Index {
    let mut builder = IndexBuilder::new(remove_stopwords, title_weight, block_size, reorder);
    for chunk in docs.chunks(INDEX_CHUNK_SIZE) {
        builder.add_documents(chunk);
    }
    builder.finish()
}

/// Build an index by streaming a JSONL file chunk by chunk: raw text and
/// parsed documents never accumulate beyond one chunk.
pub fn build_index_from_jsonl(
    path: &Path,
    remove_stopwords: bool,
    title_weight: u32,
    block_size: usize,
    reorder: ReorderStrategy,
) -> anyhow::Result<Index> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to read {}", path.display()))?;
    let reader = std::io::BufReader::with_capacity(1 << 20, file);
    let mut builder = IndexBuilder::new(remove_stopwords, title_weight, block_size, reorder);

    let mut lines: Vec<String> = Vec::with_capacity(INDEX_CHUNK_SIZE);
    let mut line_no = 0usize;
    let flush = |builder: &mut IndexBuilder,
                 lines: &mut Vec<String>,
                 first_line: usize|
     -> anyhow::Result<()> {
        let parsed: Vec<InputDoc> = lines
            .par_iter()
            .enumerate()
            .map(|(i, line)| {
                serde_json::from_str::<InputDoc>(line)
                    .with_context(|| format!("invalid JSON on line {}", first_line + i + 1))
            })
            .collect::<anyhow::Result<_>>()?;
        builder.add_documents(&parsed);
        lines.clear();
        Ok(())
    };

    let mut chunk_start = 0usize;
    for line in std::io::BufRead::lines(reader) {
        let line = line.context("failed to read input")?;
        line_no += 1;
        if line.trim().is_empty() {
            continue;
        }
        if lines.is_empty() {
            chunk_start = line_no - 1;
        }
        lines.push(line);
        if lines.len() >= INDEX_CHUNK_SIZE {
            flush(&mut builder, &mut lines, chunk_start)?;
        }
    }
    if !lines.is_empty() {
        flush(&mut builder, &mut lines, chunk_start)?;
    }
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::DEFAULT_BLOCK_SIZE;

    fn doc(id: &str, title: &str, body: &str) -> InputDoc {
        InputDoc {
            id: id.to_owned(),
            title: title.to_owned(),
            body: body.to_owned(),
        }
    }

    fn small_index() -> Index {
        let docs = vec![
            doc("d0", "pizza place", "cheap pizza pizza montreal"),
            doc("d1", "sushi bar", "fresh sushi downtown"),
            doc("d2", "pizza oven", "wood fired pizza"),
        ];
        build_index(&docs, false, DEFAULT_BLOCK_SIZE, ReorderStrategy::None)
    }

    #[test]
    fn builds_document_metadata_and_averages() {
        let index = small_index();
        assert_eq!(index.num_docs(), 3);
        assert_eq!(index.doc_summary(0).id, "d0");
        assert_eq!(index.doc_len(0), 6); // "pizza place cheap pizza pizza montreal"
        assert_eq!(index.doc_len(1), 5);
        assert_eq!(index.doc_len(2), 5);
        let expected_avg = (6.0 + 5.0 + 5.0) / 3.0;
        assert!((index.avg_doc_len() - expected_avg).abs() < 1e-6);
    }

    #[test]
    fn computes_term_frequencies_and_document_frequencies() {
        let index = small_index();
        let pizza = index.posting_list_for("pizza").expect("pizza indexed");
        assert_eq!(pizza.df(), 2); // d0 and d2
        let d0 = pizza.postings.iter().find(|p| p.doc_id == 0).unwrap();
        assert_eq!(d0.tf, 3); // once in title, twice in body
        let d2 = pizza.postings.iter().find(|p| p.doc_id == 2).unwrap();
        assert_eq!(d2.tf, 2);
        assert!(index.posting_list_for("nonexistent").is_none());
    }

    #[test]
    fn posting_lists_are_sorted_by_doc_id() {
        let docs: Vec<InputDoc> = (0..3000)
            .map(|i| doc(&format!("doc-{i}"), "", &format!("common word{}", i % 7)))
            .collect();
        let index = build_index(&docs, false, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);
        let common = index.posting_list_for("common").unwrap();
        assert_eq!(common.df(), 3000);
        for pair in common.postings.windows(2) {
            assert!(pair[0].doc_id < pair[1].doc_id);
        }
        // Blocks must tile the full posting list.
        // Blocks must tile the full posting list: ceil(df / block_size)
        // entries, the last one covering the final doc_id.
        assert_eq!(
            common.block_max_doc_ids.len(),
            common.postings.len().div_ceil(DEFAULT_BLOCK_SIZE)
        );
        assert_eq!(
            *common.block_max_doc_ids.last().unwrap(),
            common.postings.last().unwrap().doc_id
        );
    }

    #[test]
    fn title_weight_boosts_title_matches() {
        // Two docs of identical length; "pizza" appears once in d0's title
        // and once in d1's body. With weight 1 they tie; with weight 3 the
        // title match must rank first.
        let docs = vec![
            doc("d0", "pizza guide", "food and other words here"),
            doc("d1", "eating guide", "pizza and other words here"),
        ];
        let unweighted =
            build_index_weighted(&docs, false, 1, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);
        let boosted =
            build_index_weighted(&docs, false, 3, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);

        let tf = |index: &Index, doc: u32| {
            let list = index.posting_list_for("pizza").unwrap();
            list.postings.iter().find(|p| p.doc_id == doc).unwrap().tf
        };
        assert_eq!(tf(&unweighted, 0), 1);
        assert_eq!(tf(&boosted, 0), 3); // title occurrence counts triple
        assert_eq!(tf(&boosted, 1), 1); // body occurrence unchanged

        let results = crate::searcher::search(&boosted, "pizza", 2).results;
        assert_eq!(results[0].id, "d0", "title match must outrank body match");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn parses_jsonl() {
        let text = r#"{"id":"a","title":"T","body":"B"}

{"id":"b","title":"U","body":"C"}"#;
        let docs = parse_jsonl(text).unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[1].id, "b");
        assert!(parse_jsonl("not json").is_err());
    }
}
