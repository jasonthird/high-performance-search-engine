//! Index persistence: a compressed, memory-mapped, paged on-disk format.
//!
//! Three files inside the index directory, **all memory-mapped**:
//!
//! - `meta.bin` (format v3): a small fixed header plus raw little-endian
//!   array sections — document lengths/offsets, per-term statistics,
//!   flattened block skip-tables, and a front-coded term dictionary.
//!   Nothing is deserialized at load: sections are cast in place
//!   (`bytemuck`) and paged in by the OS on first touch, so opening an
//!   index is O(1) regardless of vocabulary size.
//!
//! - `postings.bin`: every posting list as a sequence of delta-encoded,
//!   bit-packed blocks (see [`crate::compress`]). Only the blocks a query
//!   actually enters are ever paged in.
//!
//! - `docs.bin`: the document store (external id, title, snippet per doc).
//!   Scoring never touches it — only the top-k hits of a query are
//!   resolved.
//!
//! The dictionary is **front-coded**: sorted terms are grouped in blocks of
//! 16, each storing its first term in full and the rest as (shared-prefix
//! length, suffix). Lookup binary-searches group heads, then decodes at
//! most 15 suffixes. Sorted terms share long prefixes, so this roughly
//! halves dictionary bytes.
//!
//! Everything derivable is precomputed at *write* time (region offsets,
//! block rows, per-term max scores) and idf — one `ln` — is computed per
//! query term, so loading does no work at all.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Context;
use memmap2::Mmap;

use crate::bm25;
use crate::compress;
use crate::indexer::{DocSummary, Index, SearchableIndex};
use crate::postings::{BlockSource, TermPostings};

const META_FILE: &str = "meta.bin";
pub(crate) const POSTINGS_FILE: &str = "postings.bin";
pub(crate) const DOCS_FILE: &str = "docs.bin";
/// Sidecar mapping external-id hashes to doc_ids (sorted by hash), used to
/// resolve deletes/updates without scanning the doc store.
pub(crate) const IDS_FILE: &str = "ids.bin";
/// Sidecar of per-document content hashes (u64 x num_docs), used for
/// change detection in upserts.
pub(crate) const HASHES_FILE: &str = "hashes.bin";

/// Write the content-hash sidecar.
pub(crate) fn write_hashes_file(
    dir: &Path,
    hashes: impl Iterator<Item = u64>,
) -> anyhow::Result<()> {
    let path = dir.join(HASHES_FILE);
    let mut w = BufWriter::new(
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    for h in hashes {
        w.write_all(&h.to_le_bytes())?;
    }
    w.flush().context("failed to flush hashes file")?;
    Ok(())
}

/// Stable hash for external ids (FNV-1a 64).
pub(crate) fn id_hash(id: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in id.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Write the (hash, doc_id) sidecar, sorted by hash.
pub(crate) fn write_ids_file(dir: &Path, ids: impl Iterator<Item = u64>) -> anyhow::Result<()> {
    let mut entries: Vec<(u64, u32)> = ids.enumerate().map(|(d, h)| (h, d as u32)).collect();
    entries.sort_unstable();
    let path = dir.join(IDS_FILE);
    let mut w = BufWriter::new(
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    for (h, d) in entries {
        w.write_all(&h.to_le_bytes())?;
        w.write_all(&(d as u64).to_le_bytes())?;
    }
    w.flush().context("failed to flush ids file")?;
    Ok(())
}

const MAGIC: &[u8; 8] = b"MVPSRCH4";
/// Terms per front-coded dictionary group.
const DICT_GROUP: usize = 16;

// Section indices (fixed order in the file).
const S_DOC_LENS: usize = 0; // u32 x num_docs
const S_DOC_OFFSETS: usize = 1; // u64 x num_docs+1
const S_TERM_DFS: usize = 2; // u32 x num_terms   (slot-indexed)
const S_REGION_OFFSETS: usize = 3; // u64 x num_terms+1 (slot-indexed)
const S_BLOCK_ROWS: usize = 4; // u32 x num_terms+1 (slot-indexed)
const S_TERM_MAX_TFS: usize = 5; // u32 x num_terms   (slot-indexed)
const S_TERM_MIN_LENS: usize = 6; // u32 x num_terms   (slot-indexed)
const S_NAME_TO_SLOT: usize = 7; // u32 x num_terms   (rank-indexed)
const S_BLOCK_MAX_DOC_IDS: usize = 8; // u32 x num_blocks
const S_BLOCK_MAX_TFS: usize = 9; // u32 x num_blocks
const S_BLOCK_MIN_LENS: usize = 10; // u32 x num_blocks
const S_BLOCK_BYTE_OFFSETS: usize = 11; // u32 x num_blocks
const S_DICT_GROUPS: usize = 12; // u32 x num_groups+1
const S_DICT_BYTES: usize = 13; // u8  x ...
const NUM_SECTIONS: usize = 14;

/// Builder-side metadata, written verbatim as mmap-able sections.
/// "Slot" is a term's position in the per-term arrays and postings file;
/// "rank" is its position in the sorted dictionary. `name_to_slot` maps
/// rank -> slot (identity for the in-memory builder).
pub(crate) struct MetaSections {
    pub avg_doc_len: f32,
    pub remove_stopwords: bool,
    pub block_size: u32,
    pub doc_lens: Vec<u32>,
    pub doc_offsets: Vec<u64>,
    pub term_dfs: Vec<u32>,
    pub region_offsets: Vec<u64>,
    pub block_rows: Vec<u32>,
    pub term_max_tfs: Vec<u32>,
    pub term_min_lens: Vec<u32>,
    pub name_to_slot: Vec<u32>,
    pub block_max_doc_ids: Vec<u32>,
    pub block_max_tfs: Vec<u32>,
    pub block_min_lens: Vec<u32>,
    pub block_byte_offsets: Vec<u32>,
}

pub(crate) fn write_str(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len().min(u16::MAX as usize);
    out.extend_from_slice(&(len as u16).to_le_bytes());
    out.extend_from_slice(&bytes[..len]);
}

/// Front-code already-sorted terms into (group_offsets, bytes).
pub(crate) fn front_code_dict<'a>(
    terms: impl Iterator<Item = &'a str>,
    num_terms: usize,
) -> (Vec<u32>, Vec<u8>) {
    let mut group_offsets = Vec::with_capacity(num_terms.div_ceil(DICT_GROUP) + 1);
    let mut bytes: Vec<u8> = Vec::new();
    let mut prev: Vec<u8> = Vec::new();
    for (i, term) in terms.enumerate() {
        let t = term.as_bytes();
        if i % DICT_GROUP == 0 {
            group_offsets.push(bytes.len() as u32);
            bytes.extend_from_slice(&(t.len().min(u16::MAX as usize) as u16).to_le_bytes());
            bytes.extend_from_slice(t);
        } else {
            let lcp = prev
                .iter()
                .zip(t)
                .take(255)
                .take_while(|(a, b)| a == b)
                .count();
            let suffix = &t[lcp..];
            bytes.push(lcp as u8);
            bytes.extend_from_slice(&(suffix.len().min(u16::MAX as usize) as u16).to_le_bytes());
            bytes.extend_from_slice(suffix);
        }
        prev.clear();
        prev.extend_from_slice(t);
    }
    group_offsets.push(bytes.len() as u32);
    (group_offsets, bytes)
}

/// Write meta.bin v3: header + section table + 8-aligned raw sections.
/// Returns its size in bytes.
pub(crate) fn write_meta(
    meta: &MetaSections,
    dict: (&[u32], &[u8]),
    dir: &Path,
) -> anyhow::Result<u64> {
    let (dict_groups, dict_bytes) = dict;
    let sections: [&[u8]; NUM_SECTIONS] = [
        bytemuck::cast_slice(&meta.doc_lens),
        bytemuck::cast_slice(&meta.doc_offsets),
        bytemuck::cast_slice(&meta.term_dfs),
        bytemuck::cast_slice(&meta.region_offsets),
        bytemuck::cast_slice(&meta.block_rows),
        bytemuck::cast_slice(&meta.term_max_tfs),
        bytemuck::cast_slice(&meta.term_min_lens),
        bytemuck::cast_slice(&meta.name_to_slot),
        bytemuck::cast_slice(&meta.block_max_doc_ids),
        bytemuck::cast_slice(&meta.block_max_tfs),
        bytemuck::cast_slice(&meta.block_min_lens),
        bytemuck::cast_slice(&meta.block_byte_offsets),
        bytemuck::cast_slice(dict_groups),
        dict_bytes,
    ];

    // Header: magic, then u64 fields, then the section table.
    let header_len = 8 + 8 * 6 + NUM_SECTIONS * 16;
    let mut offset = (header_len as u64).next_multiple_of(8);
    let mut table = Vec::with_capacity(NUM_SECTIONS);
    for s in &sections {
        table.push((offset, s.len() as u64));
        offset = (offset + s.len() as u64).next_multiple_of(8);
    }

    let path = dir.join(META_FILE);
    let mut w = BufWriter::with_capacity(
        1 << 20,
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    w.write_all(MAGIC)?;
    w.write_all(&4u64.to_le_bytes())?; // version
    w.write_all(&(meta.doc_lens.len() as u64).to_le_bytes())?;
    w.write_all(&(meta.term_dfs.len() as u64).to_le_bytes())?;
    w.write_all(&(meta.block_size as u64).to_le_bytes())?;
    w.write_all(&(u64::from(meta.remove_stopwords)).to_le_bytes())?;
    w.write_all(&(meta.avg_doc_len.to_bits() as u64).to_le_bytes())?;
    for (off, len) in &table {
        w.write_all(&off.to_le_bytes())?;
        w.write_all(&len.to_le_bytes())?;
    }
    let mut written = header_len as u64;
    for (s, (off, _)) in sections.iter().zip(&table) {
        while written < *off {
            w.write_all(&[0u8])?;
            written += 1;
        }
        w.write_all(s)?;
        written += s.len() as u64;
    }
    w.flush().context("failed to flush metadata")?;
    drop(w);
    Ok(fs::metadata(&path)?.len())
}

pub(crate) fn mmap_file(path: &Path) -> anyhow::Result<(Mmap, u64)> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let size = file.metadata()?.len();
    // SAFETY: the mapping is invalidated if another process truncates the
    // file while we read it. The index is immutable by design, so we accept
    // the standard mmap-inherent assumption (same as Lucene's MMapDirectory).
    let map = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap {}", path.display()))?;
    Ok((map, size))
}

/// Read-only index: header in RAM, everything else memory-mapped and paged
/// in on demand.
pub struct DiskIndex {
    meta: Mmap,
    postings: Mmap,
    docs: Mmap,
    /// Optional (hash, doc_id) sidecar for external-id resolution.
    ids: Option<Mmap>,
    /// Optional per-doc content hashes for upsert change detection.
    hashes: Option<Mmap>,
    size_bytes: u64,
    num_docs: usize,
    num_terms: usize,
    block_size: usize,
    remove_stopwords: bool,
    avg_doc_len: f32,
    sections: [(u64, u64); NUM_SECTIONS],
}

/// Serialize an in-memory index into `<dir>/{meta,postings,docs}.bin`.
/// Returns the total size in bytes.
pub fn save_index(index: &Index, dir: &Path) -> anyhow::Result<u64> {
    fs::create_dir_all(dir)
        .with_context(|| format!("failed to create index directory {}", dir.display()))?;

    // --- docs.bin: the document store -----------------------------------
    let docs_path = dir.join(DOCS_FILE);
    let mut doc_offsets: Vec<u64> = Vec::with_capacity(index.docs().len() + 1);
    {
        let file = File::create(&docs_path)
            .with_context(|| format!("failed to create {}", docs_path.display()))?;
        let mut writer = BufWriter::new(file);
        let mut record = Vec::new();
        let mut offset = 0u64;
        doc_offsets.push(0);
        for meta in index.docs() {
            record.clear();
            write_str(&mut record, &meta.external_id);
            write_str(&mut record, &meta.title);
            write_str(&mut record, &meta.snippet);
            writer.write_all(&record).context("failed to write docs")?;
            offset += record.len() as u64;
            doc_offsets.push(offset);
        }
        writer.flush().context("failed to flush docs file")?;
    }
    write_ids_file(dir, index.docs().iter().map(|d| id_hash(&d.external_id)))?;
    write_hashes_file(dir, index.docs().iter().map(|d| d.content_hash))?;

    // --- postings.bin + per-term/block tables ----------------------------
    let postings_path = dir.join(POSTINGS_FILE);
    let file = File::create(&postings_path)
        .with_context(|| format!("failed to create {}", postings_path.display()))?;
    let mut writer = BufWriter::new(file);

    let num_terms = index.num_terms();
    let block_size = index.block_size();
    let mut term_dfs = Vec::with_capacity(num_terms);
    let mut region_offsets: Vec<u64> = Vec::with_capacity(num_terms + 1);
    let mut block_rows: Vec<u32> = Vec::with_capacity(num_terms + 1);
    let mut term_max_tfs = Vec::with_capacity(num_terms);
    let mut term_min_lens = Vec::with_capacity(num_terms);
    let mut block_max_doc_ids = Vec::new();
    let mut block_max_tfs = Vec::new();
    let mut block_min_lens = Vec::new();
    let mut block_byte_offsets = Vec::new();
    let mut encoded = Vec::new();
    region_offsets.push(0);
    block_rows.push(0);
    for term_id in 0..num_terms as u32 {
        let list = index.posting_list(term_id);
        encoded.clear();
        for chunk in list.postings.chunks(block_size) {
            block_byte_offsets.push(encoded.len() as u32);
            compress::encode_block(chunk, &mut encoded);
        }
        writer
            .write_all(&encoded)
            .context("failed to write postings")?;
        term_dfs.push(list.df() as u32);
        region_offsets.push(region_offsets.last().unwrap() + encoded.len() as u64);
        block_rows.push(block_rows.last().unwrap() + list.block_max_doc_ids.len() as u32);
        term_max_tfs.push(list.term_max_tf);
        term_min_lens.push(list.term_min_len);
        block_max_doc_ids.extend_from_slice(&list.block_max_doc_ids);
        block_max_tfs.extend_from_slice(&list.block_max_tfs);
        block_min_lens.extend_from_slice(&list.block_min_lens);
    }
    writer.flush().context("failed to flush postings file")?;
    drop(writer);

    // --- meta.bin --------------------------------------------------------
    // In-memory builds assign term ids in sorted-name order, so slot order
    // already is rank order.
    let mut names: Vec<(&str, u32)> = index
        .term_dict()
        .iter()
        .map(|(t, &id)| (t.as_str(), id))
        .collect();
    names.sort_unstable_by_key(|&(_, id)| id);
    let (dict_groups, dict_bytes) = front_code_dict(names.iter().map(|&(t, _)| t), num_terms);

    let meta = MetaSections {
        avg_doc_len: index.avg_doc_len(),
        remove_stopwords: index.remove_stopwords(),
        block_size: block_size as u32,
        doc_lens: index.docs().iter().map(|d| d.doc_len).collect(),
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
    let meta_size = write_meta(&meta, (&dict_groups, &dict_bytes), dir)?;
    Ok(meta_size + fs::metadata(&postings_path)?.len() + fs::metadata(&docs_path)?.len())
}

/// Open an index. No deserialization happens: the three files are
/// memory-mapped and only the small header is parsed.
pub fn load_index(dir: &Path) -> anyhow::Result<DiskIndex> {
    let (meta, meta_size) = mmap_file(&dir.join(META_FILE))?;
    let (postings, postings_size) = mmap_file(&dir.join(POSTINGS_FILE))?;
    let (docs, docs_size) = mmap_file(&dir.join(DOCS_FILE))?;
    let ids = if dir.join(IDS_FILE).exists() {
        Some(mmap_file(&dir.join(IDS_FILE))?.0)
    } else {
        None
    };
    let hashes = if dir.join(HASHES_FILE).exists() {
        Some(mmap_file(&dir.join(HASHES_FILE))?.0)
    } else {
        None
    };

    anyhow::ensure!(
        meta.len() >= 8 + 8 * 6 + NUM_SECTIONS * 16 && &meta[..8] == MAGIC,
        "not an high-performance-search-engine v3 index"
    );
    let u64_at = |pos: usize| -> u64 {
        u64::from_le_bytes(meta[pos..pos + 8].try_into().expect("header read"))
    };
    anyhow::ensure!(u64_at(8) == 4, "unsupported index version");
    let num_docs = u64_at(16) as usize;
    let num_terms = u64_at(24) as usize;
    let block_size = u64_at(32) as usize;
    let remove_stopwords = u64_at(40) != 0;
    let avg_doc_len = f32::from_bits(u64_at(48) as u32);
    let mut sections = [(0u64, 0u64); NUM_SECTIONS];
    for (i, s) in sections.iter_mut().enumerate() {
        *s = (u64_at(56 + i * 16), u64_at(64 + i * 16));
    }

    Ok(DiskIndex {
        meta,
        postings,
        docs,
        ids,
        hashes,
        size_bytes: meta_size + postings_size + docs_size,
        num_docs,
        num_terms,
        block_size,
        remove_stopwords,
        avg_doc_len,
        sections,
    })
}

impl DiskIndex {
    /// Total on-disk size (meta + postings + docs) in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    fn section(&self, i: usize) -> &[u8] {
        let (off, len) = self.sections[i];
        &self.meta[off as usize..(off + len) as usize]
    }

    fn u32s(&self, i: usize) -> &[u32] {
        bytemuck::cast_slice(self.section(i))
    }

    fn u64s(&self, i: usize) -> &[u64] {
        bytemuck::cast_slice(self.section(i))
    }

    /// First term of a dictionary group, as raw bytes.
    fn group_head<'a>(&self, dict: &'a [u8], groups: &[u32], g: usize) -> &'a [u8] {
        let pos = groups[g] as usize;
        let len = u16::from_le_bytes([dict[pos], dict[pos + 1]]) as usize;
        &dict[pos + 2..pos + 2 + len]
    }

    /// Binary search the front-coded dictionary: term -> rank.
    fn dict_lookup(&self, term: &str) -> Option<usize> {
        let groups = self.u32s(S_DICT_GROUPS);
        let dict = self.section(S_DICT_BYTES);
        let num_groups = groups.len() - 1;
        if num_groups == 0 {
            return None;
        }
        let target = term.as_bytes();

        // Rightmost group whose head <= target.
        let mut lo = 0usize;
        let mut hi = num_groups;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.group_head(dict, groups, mid) <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            return None;
        }
        let g = lo - 1;

        // Walk the group, reconstructing terms from (lcp, suffix) pairs.
        let head = self.group_head(dict, groups, g);
        if head == target {
            return Some(g * DICT_GROUP);
        }
        let mut current: Vec<u8> = head.to_vec();
        let mut pos = groups[g] as usize + 2 + head.len();
        let end = groups[g + 1] as usize;
        let mut idx = 0usize;
        while pos < end {
            idx += 1;
            let lcp = dict[pos] as usize;
            let suffix_len = u16::from_le_bytes([dict[pos + 1], dict[pos + 2]]) as usize;
            let suffix = &dict[pos + 3..pos + 3 + suffix_len];
            pos += 3 + suffix_len;
            current.truncate(lcp);
            current.extend_from_slice(suffix);
            match current.as_slice().cmp(target) {
                std::cmp::Ordering::Equal => return Some(g * DICT_GROUP + idx),
                std::cmp::Ordering::Greater => return None,
                std::cmp::Ordering::Less => {}
            }
        }
        None
    }
}

impl DiskIndex {
    /// Reconstruct the term at dictionary rank `rank` (front-coded decode).
    pub(crate) fn term_at(&self, rank: usize) -> String {
        let groups = self.u32s(S_DICT_GROUPS);
        let dict = self.section(S_DICT_BYTES);
        let g = rank / DICT_GROUP;
        let head = self.group_head(dict, groups, g);
        let mut current: Vec<u8> = head.to_vec();
        let mut pos = groups[g] as usize + 2 + head.len();
        for _ in 0..(rank % DICT_GROUP) {
            let lcp = dict[pos] as usize;
            let suffix_len = u16::from_le_bytes([dict[pos + 1], dict[pos + 2]]) as usize;
            let suffix = &dict[pos + 3..pos + 3 + suffix_len];
            pos += 3 + suffix_len;
            current.truncate(lcp);
            current.extend_from_slice(suffix);
        }
        String::from_utf8_lossy(&current).into_owned()
    }

    /// Resolve an external id to its doc_id via the hash sidecar
    /// (verifying against the doc store to rule out collisions).
    pub fn find_by_external_id(&self, id: &str) -> Option<u32> {
        let ids = self.ids.as_ref()?;
        let entries: &[u64] = bytemuck::cast_slice(&ids[..]);
        let target = id_hash(id);
        // entries = [hash, doc_id, hash, doc_id, ...] sorted by hash.
        let n = entries.len() / 2;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if entries[mid * 2] < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        while lo < n && entries[lo * 2] == target {
            let doc_id = entries[lo * 2 + 1] as u32;
            if self.doc_summary(doc_id).id == id {
                return Some(doc_id);
            }
            lo += 1;
        }
        None
    }

    /// Stored content hash of a document (u64::MAX if the sidecar is
    /// missing, which never matches a real hash in practice).
    pub fn content_hash(&self, doc_id: u32) -> u64 {
        self.hashes
            .as_ref()
            .map(|m| {
                let s: &[u64] = bytemuck::cast_slice(&m[..]);
                s[doc_id as usize]
            })
            .unwrap_or(u64::MAX)
    }

    /// Raw docs.bin record for one document (for segment merging).
    pub(crate) fn doc_record_bytes(&self, doc_id: u32) -> &[u8] {
        let offsets = self.u64s(S_DOC_OFFSETS);
        &self.docs[offsets[doc_id as usize] as usize..offsets[doc_id as usize + 1] as usize]
    }

    /// Document frequency of a term in this index alone (0 if absent).
    pub fn term_df(&self, term: &str) -> u32 {
        self.dict_lookup(term)
            .map(|rank| {
                let slot = self.u32s(S_NAME_TO_SLOT)[rank] as usize;
                self.u32s(S_TERM_DFS)[slot]
            })
            .unwrap_or(0)
    }

    /// Like [`SearchableIndex::term_postings`] but scored under
    /// caller-supplied statistics (a segment serving global stats).
    pub fn term_postings_with(
        &self,
        term: &str,
        idf: f32,
        avg_doc_len: f32,
    ) -> Option<TermPostings<'_>> {
        let rank = self.dict_lookup(term)?;
        let slot = self.u32s(S_NAME_TO_SLOT)[rank] as usize;
        let df = self.u32s(S_TERM_DFS)[slot] as usize;
        if df == 0 {
            return None;
        }
        let region_offsets = self.u64s(S_REGION_OFFSETS);
        let block_rows = self.u32s(S_BLOCK_ROWS);
        let region = region_offsets[slot] as usize..region_offsets[slot + 1] as usize;
        let rows = block_rows[slot] as usize..block_rows[slot + 1] as usize;
        Some(TermPostings {
            idf,
            avg_doc_len,
            df,
            block_size: self.block_size,
            block_max_doc_ids: &self.u32s(S_BLOCK_MAX_DOC_IDS)[rows.clone()],
            block_max_tfs: &self.u32s(S_BLOCK_MAX_TFS)[rows.clone()],
            block_min_lens: &self.u32s(S_BLOCK_MIN_LENS)[rows.clone()],
            term_max_tf: self.u32s(S_TERM_MAX_TFS)[slot],
            term_min_len: self.u32s(S_TERM_MIN_LENS)[slot],
            source: BlockSource::Compressed {
                bytes: &self.postings[region],
                block_offsets: &self.u32s(S_BLOCK_BYTE_OFFSETS)[rows],
            },
        })
    }
}

impl SearchableIndex for DiskIndex {
    fn num_docs(&self) -> usize {
        self.num_docs
    }

    fn num_terms(&self) -> usize {
        self.num_terms
    }

    fn avg_doc_len(&self) -> f32 {
        self.avg_doc_len
    }

    fn remove_stopwords(&self) -> bool {
        self.remove_stopwords
    }

    fn doc_len(&self, doc_id: u32) -> u32 {
        self.u32s(S_DOC_LENS)[doc_id as usize]
    }

    fn doc_summary(&self, doc_id: u32) -> DocSummary {
        let mut pos = self.u64s(S_DOC_OFFSETS)[doc_id as usize] as usize;
        let bytes = &self.docs[..];
        let read_str = |pos: &mut usize| -> String {
            let len = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]) as usize;
            let start = *pos + 2;
            *pos = start + len;
            String::from_utf8_lossy(&bytes[start..start + len]).into_owned()
        };
        let id = read_str(&mut pos);
        let title = read_str(&mut pos);
        DocSummary { id, title }
    }

    fn total_postings(&self) -> u64 {
        self.u32s(S_TERM_DFS).iter().map(|&df| df as u64).sum()
    }

    fn term_postings(&self, term: &str) -> Option<TermPostings<'_>> {
        let df = self.term_df(term);
        if df == 0 {
            return None;
        }
        self.term_postings_with(
            term,
            bm25::idf(self.num_docs, df as usize),
            self.avg_doc_len,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::{build_index, InputDoc};
    use crate::postings::DEFAULT_BLOCK_SIZE;
    use crate::reorder::ReorderStrategy;

    #[test]
    fn front_coding_round_trips_via_lookup() {
        // Built through the real save/load path below; here check the
        // encoder's group structure directly.
        let terms = ["alpha", "alphabet", "alphabetical", "beta", "betamax"];
        let (groups, bytes) = front_code_dict(terms.iter().copied(), terms.len());
        assert_eq!(groups.len(), 2); // one group + end sentinel
        assert!(bytes.len() < terms.iter().map(|t| t.len() + 3).sum::<usize>());
    }

    #[test]
    fn round_trips_an_index() {
        let docs = vec![
            InputDoc {
                id: "a".into(),
                title: "hello".into(),
                body: "world world alpha alphabet alphabetic".into(),
            },
            InputDoc {
                id: "b".into(),
                title: "goodbye".into(),
                body: "world beta".into(),
            },
        ];
        let index = build_index(&docs, false, DEFAULT_BLOCK_SIZE, ReorderStrategy::None);
        let dir = std::env::temp_dir().join(format!("high-performance-search-engine-test-{}", std::process::id()));
        let size = save_index(&index, &dir).unwrap();
        assert!(size > 0);

        let loaded = load_index(&dir).unwrap();
        assert_eq!(loaded.size_bytes(), size);
        assert_eq!(loaded.num_docs(), index.num_docs());
        assert_eq!(loaded.num_terms(), index.num_terms());
        assert_eq!(loaded.total_postings(), index.total_postings());
        assert!((loaded.avg_doc_len() - index.avg_doc_len()).abs() < 1e-6);
        assert_eq!(loaded.doc_len(0), index.doc_len(0));
        assert_eq!(loaded.doc_summary(0).id, index.doc_summary(0).id);
        assert_eq!(loaded.doc_summary(1).title, index.doc_summary(1).title);

        // Every term resolves through the front-coded dictionary with stats
        // matching the in-memory index; absent terms miss cleanly.
        for term in [
            "hello",
            "world",
            "alpha",
            "alphabet",
            "alphabetic",
            "beta",
            "goodbye",
        ] {
            let original = index.posting_list_for(term).expect(term);
            let loaded_term = loaded.term_postings(term).expect(term);
            assert_eq!(loaded_term.df, original.df(), "{term}");
            assert!((loaded_term.idf - original.idf).abs() < 1e-6, "{term}");
            assert_eq!(loaded_term.term_max_tf, original.term_max_tf, "{term}");
            assert_eq!(loaded_term.term_min_len, original.term_min_len, "{term}");
            let mut decoded = Vec::new();
            loaded_term.decode_block(0, &mut decoded);
            assert_eq!(decoded, original.postings, "{term}");
        }
        for missing in ["", "aaaa", "worlds", "zzzz", "alphabets"] {
            assert!(loaded.term_postings(missing).is_none(), "{missing}");
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
