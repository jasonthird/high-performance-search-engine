//! External (sharded, spill-to-disk) index construction for corpora whose
//! postings do not fit in memory — e.g. all of English Wikipedia on a
//! laptop.
//!
//! The in-memory builder holds every `(term, doc, tf)` triple until the end;
//! at ~2.5B postings that is tens of gigabytes. This builder instead:
//!
//! 1. **Streams** documents (file or stdin): each chunk is parsed and
//!    tokenized in parallel, document records go straight to `docs.bin`,
//!    and the chunk's postings triples accumulate in a bounded buffer.
//! 2. **Spills** the buffer to a sorted shard file whenever it reaches the
//!    budget: triples are sorted by (term, doc) — doc order is insertion
//!    order, so postings stay doc-ascending — and written as
//!    `(term, count, [doc, tf]...)` runs.
//! 3. **Merges** the shards with a k-way heap walk over ascending term ids.
//!    Shards were filled in document order, so concatenating a term's runs
//!    in shard order yields its full posting list already sorted by doc_id;
//!    blocks are compressed and written directly to the final
//!    `postings.bin`, never materializing more than one posting list.
//!
//! Peak memory is the spill budget plus the term dictionary and per-doc
//! lengths — independent of corpus size. The final index is byte-compatible
//! with the in-memory builder's output except that posting regions appear
//! in first-seen term order (the dictionary's `name_to_slot` indirection
//! covers that).
//!
//! Document reordering is not supported in this mode: doc_ids are assigned
//! while streaming, before the corpus is fully known.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::Context;
use rayon::prelude::*;

use crate::compress;
use crate::indexer::{InputDoc, Interner};
use crate::postings;
use crate::postings::Posting;
use crate::storage::{self, MetaSections};
use crate::tokenizer::Tokenizer;

/// Documents per streamed chunk.
const CHUNK_DOCS: usize = 4096;

/// Default spill budget in postings triples (12 bytes each): 192M triples
/// ≈ 2.3 GB of buffer.
pub const DEFAULT_SPILL_BUDGET: usize = 192_000_000;

pub struct ExternalStats {
    pub num_docs: usize,
    pub num_terms: usize,
    pub num_postings: u64,
    pub num_shards: usize,
    pub index_bytes: u64,
    /// Phase 1: stream + parse + tokenize + spill.
    pub stream_secs: f64,
    /// Phase 2: k-way shard merge + block compression + write.
    pub merge_secs: f64,
}

/// One in-flight posting before the merge.
#[derive(Clone, Copy)]
struct Triple {
    term: u32,
    doc: u32,
    tf: u32,
}

/// Build an index directly on disk from a JSONL stream.
pub fn build_index_external(
    input: &mut dyn BufRead,
    out_dir: &Path,
    remove_stopwords: bool,
    title_weight: u32,
    block_size: usize,
    spill_budget: usize,
) -> anyhow::Result<ExternalStats> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create index directory {}", out_dir.display()))?;
    let shard_dir = out_dir.join("shards.tmp");
    fs::create_dir_all(&shard_dir).context("failed to create shard directory")?;

    let tokenizer = Tokenizer::new(remove_stopwords);
    let interner = Interner::new();
    let title_weight = title_weight.max(1);

    // docs.bin is written as documents stream past; only offsets and
    // lengths stay in memory (8 bytes per document).
    let docs_path = out_dir.join(storage::DOCS_FILE);
    let mut docs_writer = BufWriter::new(
        File::create(&docs_path)
            .with_context(|| format!("failed to create {}", docs_path.display()))?,
    );
    let mut doc_offsets: Vec<u64> = vec![0];
    let mut doc_lens: Vec<u32> = Vec::new();
    let mut id_hashes: Vec<u64> = Vec::new();
    let mut content_hashes: Vec<u64> = Vec::new();

    let mut pending: Vec<Triple> = Vec::new();
    let mut num_shards = 0usize;
    let mut record = Vec::new();

    let spill = |pending: &mut Vec<Triple>, num_shards: &mut usize| -> anyhow::Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        // Triples were appended in ascending doc order, so sorting by
        // (term, doc) keeps every term's run doc-ascending.
        pending.par_sort_unstable_by_key(|t| (t.term, t.doc));
        let path = shard_dir.join(format!("shard-{num_shards:05}.bin"));
        let mut writer = BufWriter::with_capacity(1 << 20, File::create(&path)?);
        let mut i = 0;
        while i < pending.len() {
            let term = pending[i].term;
            let mut j = i;
            while j < pending.len() && pending[j].term == term {
                j += 1;
            }
            writer.write_all(&term.to_le_bytes())?;
            writer.write_all(&((j - i) as u32).to_le_bytes())?;
            for t in &pending[i..j] {
                writer.write_all(&t.doc.to_le_bytes())?;
                writer.write_all(&t.tf.to_le_bytes())?;
            }
            i = j;
        }
        writer.flush()?;
        *num_shards += 1;
        pending.clear();
        Ok(())
    };

    // ---- Phase 1: stream, tokenize, spill ------------------------------
    let phase1_start = std::time::Instant::now();
    let mut lines: Vec<String> = Vec::with_capacity(CHUNK_DOCS);
    let mut line_no = 0usize;
    let mut eof = false;
    while !eof {
        lines.clear();
        let mut buf = String::new();
        while lines.len() < CHUNK_DOCS {
            buf.clear();
            if input.read_line(&mut buf).context("failed to read input")? == 0 {
                eof = true;
                break;
            }
            line_no += 1;
            if !buf.trim().is_empty() {
                lines.push(buf.clone());
            }
        }
        if lines.is_empty() {
            continue;
        }
        let first_line = line_no - lines.len();
        let parsed: Vec<InputDoc> = std::mem::take(&mut lines)
            .into_par_iter()
            .enumerate()
            .map(|(i, line)| {
                // simd-json parses in place (NEON on Apple silicon).
                let mut bytes = line.into_bytes();
                simd_json::serde::from_slice::<InputDoc>(&mut bytes)
                    .with_context(|| format!("invalid JSON on line {}", first_line + i + 1))
            })
            .collect::<anyhow::Result<_>>()?;

        // Tokenize the chunk in parallel into per-doc (term, tf) pairs.
        let chunk_pairs: Vec<(u32, Vec<(u32, u32)>)> = parsed
            .par_iter()
            .map(|doc| {
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
                ids.sort_unstable_by_key(|&(id, _)| id);
                let mut pairs: Vec<(u32, u32)> = Vec::new();
                for &(id, w) in &ids {
                    match pairs.last_mut() {
                        Some((last, tf)) if *last == id => *tf += w,
                        _ => pairs.push((id, w)),
                    }
                }
                (doc_len, pairs)
            })
            .collect();

        for (doc, (doc_len, pairs)) in parsed.iter().zip(&chunk_pairs) {
            let doc_id = doc_lens.len() as u32;
            record.clear();
            storage::write_str(&mut record, &doc.id);
            storage::write_str(&mut record, &doc.title);
            let snippet: String = doc.body.chars().take(200).collect();
            storage::write_str(&mut record, &snippet);
            docs_writer
                .write_all(&record)
                .context("failed to write docs")?;
            doc_offsets.push(doc_offsets.last().unwrap() + record.len() as u64);
            doc_lens.push(*doc_len);
            id_hashes.push(storage::id_hash(&doc.id));
            content_hashes.push(crate::indexer::content_hash(&doc.title, &doc.body));
            for &(term, tf) in pairs {
                pending.push(Triple {
                    term,
                    doc: doc_id,
                    tf,
                });
            }
        }
        if pending.len() >= spill_budget {
            spill(&mut pending, &mut num_shards)?;
        }
    }
    spill(&mut pending, &mut num_shards)?;
    docs_writer.flush().context("failed to flush docs")?;
    drop(docs_writer);
    storage::write_ids_file(out_dir, id_hashes.into_iter())?;
    storage::write_hashes_file(out_dir, content_hashes.into_iter())?;

    let num_docs = doc_lens.len();
    let total_len: u64 = doc_lens.iter().map(|&l| l as u64).sum();
    let avg_doc_len = if total_len == 0 {
        1.0
    } else {
        total_len as f32 / num_docs as f32
    };

    let stream_secs = phase1_start.elapsed().as_secs_f64();

    // ---- Phase 2: k-way merge of shards into the final index -----------
    let merge_start = std::time::Instant::now();
    let names = interner.into_names();
    let num_terms = names.len();

    let postings_path = out_dir.join(storage::POSTINGS_FILE);
    let mut postings_writer = BufWriter::with_capacity(
        1 << 20,
        File::create(&postings_path)
            .with_context(|| format!("failed to create {}", postings_path.display()))?,
    );

    let mut readers: Vec<ShardReader> = (0..num_shards)
        .map(|i| ShardReader::open(&shard_dir.join(format!("shard-{i:05}.bin"))))
        .collect::<anyhow::Result<_>>()?;

    let mut term_dfs = Vec::with_capacity(num_terms);
    let mut region_offsets: Vec<u64> = Vec::with_capacity(num_terms + 1);
    let mut block_rows: Vec<u32> = Vec::with_capacity(num_terms + 1);
    let mut term_max_tfs: Vec<u32> = Vec::with_capacity(num_terms);
    let mut term_min_lens: Vec<u32> = Vec::with_capacity(num_terms);
    let mut block_max_doc_ids = Vec::new();
    let mut block_max_tfs = Vec::new();
    let mut block_min_lens = Vec::new();
    let mut block_byte_offsets = Vec::new();
    let mut num_postings = 0u64;
    region_offsets.push(0);
    block_rows.push(0);

    // The gather stays sequential (each shard is one forward scan), but
    // block building and bit-packing — the CPU-heavy part — run in parallel
    // over batches of terms, and each batch is written in order.
    const MERGE_BATCH: usize = 4096;
    let mut batch: Vec<Vec<Posting>> = Vec::with_capacity(MERGE_BATCH);
    let mut term = 0u32;
    while (term as usize) < num_terms {
        batch.clear();
        while (term as usize) < num_terms && batch.len() < MERGE_BATCH {
            let mut list = Vec::new();
            // Shards are filled in document order, so visiting them in
            // shard order concatenates this term's postings already
            // doc-ascending.
            for reader in &mut readers {
                reader.take_term(term, &mut list)?;
            }
            num_postings += list.len() as u64;
            batch.push(list);
            term += 1;
        }

        type EncodedTerm = (Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);
        let encoded_batch: Vec<EncodedTerm> = batch
            .par_iter()
            .map(|list| {
                let (max_ids, max_tfs, min_lens) =
                    postings::build_blocks(list, &doc_lens, block_size);
                let mut encoded = Vec::new();
                let mut offsets = Vec::with_capacity(max_ids.len());
                for chunk in list.chunks(block_size) {
                    offsets.push(encoded.len() as u32);
                    compress::encode_block(chunk, &mut encoded);
                }
                (encoded, offsets, max_ids, max_tfs, min_lens)
            })
            .collect();

        for (list, (encoded, offsets, max_ids, max_tfs, min_lens)) in
            batch.iter().zip(&encoded_batch)
        {
            postings_writer
                .write_all(encoded)
                .context("failed to write postings")?;
            term_dfs.push(list.len() as u32);
            region_offsets.push(region_offsets.last().unwrap() + encoded.len() as u64);
            block_rows.push(block_rows.last().unwrap() + max_ids.len() as u32);
            term_max_tfs.push(max_tfs.iter().copied().max().unwrap_or(1));
            term_min_lens.push(min_lens.iter().copied().min().unwrap_or(1));
            block_byte_offsets.extend_from_slice(offsets);
            block_max_doc_ids.extend_from_slice(max_ids);
            block_max_tfs.extend_from_slice(max_tfs);
            block_min_lens.extend_from_slice(min_lens);
        }
    }
    postings_writer
        .flush()
        .context("failed to flush postings")?;
    drop(postings_writer);
    drop(readers);
    fs::remove_dir_all(&shard_dir).ok();

    // Dictionary: front-coded in name order, with a rank -> slot
    // indirection (slots are first-seen term order).
    let mut ranks: Vec<u32> = (0..num_terms as u32).collect();
    ranks.sort_unstable_by(|&a, &b| names[a as usize].cmp(&names[b as usize]));
    let (dict_groups, dict_bytes) = storage::front_code_dict(
        ranks.iter().map(|&slot| names[slot as usize].as_str()),
        num_terms,
    );

    let meta = MetaSections {
        avg_doc_len,
        remove_stopwords,
        block_size: block_size as u32,
        doc_lens,
        doc_offsets,
        term_dfs,
        region_offsets,
        block_rows,
        term_max_tfs,
        term_min_lens,
        name_to_slot: ranks,
        block_max_doc_ids,
        block_max_tfs,
        block_min_lens,
        block_byte_offsets,
    };
    let meta_size = storage::write_meta(&meta, (&dict_groups, &dict_bytes), out_dir)?;
    let index_bytes =
        meta_size + fs::metadata(&postings_path)?.len() + fs::metadata(&docs_path)?.len();

    Ok(ExternalStats {
        num_docs,
        num_terms,
        num_postings,
        num_shards,
        index_bytes,
        stream_secs,
        merge_secs: merge_start.elapsed().as_secs_f64(),
    })
}

/// Sequential reader over one sorted shard file.
struct ShardReader {
    reader: BufReader<File>,
    /// Header of the next run: (term, count), if any.
    next: Option<(u32, u32)>,
}

impl ShardReader {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let mut reader = Self {
            reader: BufReader::with_capacity(1 << 20, File::open(path)?),
            next: None,
        };
        reader.advance()?;
        Ok(reader)
    }

    fn read_u32(&mut self) -> anyhow::Result<Option<u32>> {
        let mut buf = [0u8; 4];
        match self.reader.read_exact(&mut buf) {
            Ok(()) => Ok(Some(u32::from_le_bytes(buf))),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn advance(&mut self) -> anyhow::Result<()> {
        self.next = match self.read_u32()? {
            Some(term) => {
                let count = self.read_u32()?.context("truncated shard")?;
                Some((term, count))
            }
            None => None,
        };
        Ok(())
    }

    /// If this shard's next run is for `term`, append its postings.
    fn take_term(&mut self, term: u32, out: &mut Vec<Posting>) -> anyhow::Result<()> {
        if self.next.map(|(t, _)| t) != Some(term) {
            return Ok(());
        }
        let (_, count) = self.next.expect("checked above");
        for _ in 0..count {
            let doc_id = self.read_u32()?.context("truncated shard")?;
            let tf = self.read_u32()?.context("truncated shard")?;
            out.push(Posting { doc_id, tf });
        }
        self.advance()
    }
}
