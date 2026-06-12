//! One-shot migration of v3 indexes (precomputed block-max scores) to v4
//! (impact pairs). Only `meta.bin` is rewritten — postings and the doc
//! store are untouched; the impact pairs (max tf, min doc length per
//! block) are recomputed by decoding every posting block once. The id-hash
//! sidecar (`ids.bin`), introduced alongside v4, is also generated.

use std::path::Path;

use anyhow::Context;
use rayon::prelude::*;

use crate::compress;
use crate::storage::{self, MetaSections};

// v3 header: magic(8) + 6 u64 fields + 12 x (offset, len) u64 pairs.
const V3_MAGIC: &[u8; 8] = b"MVPSRCH3";
const V3_NUM_SECTIONS: usize = 12;
const V3_DOC_LENS: usize = 0;
const V3_DOC_OFFSETS: usize = 1;
const V3_TERM_DFS: usize = 2;
const V3_REGION_OFFSETS: usize = 3;
const V3_BLOCK_ROWS: usize = 4;
// 5: term_max_scores (f32) — dropped, recomputed as impact pairs
const V3_NAME_TO_SLOT: usize = 6;
const V3_BLOCK_MAX_DOC_IDS: usize = 7;
// 8: block_max_scores (f32) — dropped
const V3_BLOCK_BYTE_OFFSETS: usize = 9;
const V3_DICT_GROUPS: usize = 10;
const V3_DICT_BYTES: usize = 11;

/// Migrate `<dir>` from format v3 to v4 in place. Returns the number of
/// terms processed.
pub fn migrate_v3_to_v4(dir: &Path) -> anyhow::Result<usize> {
    let (meta, _) = storage::mmap_file(&dir.join("meta.bin"))?;
    let (postings, _) = storage::mmap_file(&dir.join("postings.bin"))?;
    let (docs, _) = storage::mmap_file(&dir.join("docs.bin"))?;

    anyhow::ensure!(
        meta.len() > 8 + 8 * 6 + V3_NUM_SECTIONS * 16 && &meta[..8] == V3_MAGIC,
        "not a v3 index (already migrated?)"
    );
    let u64_at = |pos: usize| -> u64 {
        u64::from_le_bytes(meta[pos..pos + 8].try_into().expect("header read"))
    };
    anyhow::ensure!(u64_at(8) == 3, "unexpected version field");
    let block_size = u64_at(32) as usize;
    let remove_stopwords = u64_at(40) != 0;
    let avg_doc_len = f32::from_bits(u64_at(48) as u32);

    let section = |i: usize| -> &[u8] {
        let off = u64_at(56 + i * 16) as usize;
        let len = u64_at(64 + i * 16) as usize;
        &meta[off..off + len]
    };
    let u32s = |i: usize| -> &[u32] { bytemuck::cast_slice(section(i)) };
    let u64s = |i: usize| -> &[u64] { bytemuck::cast_slice(section(i)) };

    let doc_lens: Vec<u32> = u32s(V3_DOC_LENS).to_vec();
    let doc_offsets: Vec<u64> = u64s(V3_DOC_OFFSETS).to_vec();
    let term_dfs: Vec<u32> = u32s(V3_TERM_DFS).to_vec();
    let region_offsets: Vec<u64> = u64s(V3_REGION_OFFSETS).to_vec();
    let block_rows: Vec<u32> = u32s(V3_BLOCK_ROWS).to_vec();
    let name_to_slot: Vec<u32> = u32s(V3_NAME_TO_SLOT).to_vec();
    let block_max_doc_ids: Vec<u32> = u32s(V3_BLOCK_MAX_DOC_IDS).to_vec();
    let block_byte_offsets: Vec<u32> = u32s(V3_BLOCK_BYTE_OFFSETS).to_vec();
    let dict_groups: Vec<u32> = u32s(V3_DICT_GROUPS).to_vec();
    let dict_bytes: Vec<u8> = section(V3_DICT_BYTES).to_vec();

    let num_terms = term_dfs.len();
    let num_blocks = block_max_doc_ids.len();

    // Recompute impact pairs by decoding every block once, in parallel
    // over terms.
    let mut block_max_tfs = vec![0u32; num_blocks];
    let mut block_min_lens = vec![0u32; num_blocks];
    let mut term_max_tfs = vec![0u32; num_terms];
    let mut term_min_lens = vec![0u32; num_terms];
    {
        // Disjoint per-term block ranges allow parallel writes.
        struct SendPtr<T>(*mut T);
        unsafe impl<T> Send for SendPtr<T> {}
        unsafe impl<T> Sync for SendPtr<T> {}
        impl<T> SendPtr<T> {
            /// # Safety
            /// `idx` must be in bounds and not written concurrently.
            unsafe fn write(&self, idx: usize, value: T) {
                unsafe { self.0.add(idx).write(value) };
            }
        }
        let p_btf = SendPtr(block_max_tfs.as_mut_ptr());
        let p_bml = SendPtr(block_min_lens.as_mut_ptr());
        let p_ttf = SendPtr(term_max_tfs.as_mut_ptr());
        let p_tml = SendPtr(term_min_lens.as_mut_ptr());

        (0..num_terms).into_par_iter().for_each(|slot| {
            let region =
                &postings[region_offsets[slot] as usize..region_offsets[slot + 1] as usize];
            let rows = block_rows[slot] as usize..block_rows[slot + 1] as usize;
            let mut decoded = Vec::new();
            let mut t_max_tf = 1u32;
            let mut t_min_len = u32::MAX;
            for (bi, row) in rows.clone().enumerate() {
                let offset = block_byte_offsets[row] as usize;
                compress::decode_block(&region[offset..], &mut decoded);
                let max_tf = decoded.iter().map(|p| p.tf).max().unwrap_or(1);
                let min_len = decoded
                    .iter()
                    .map(|p| doc_lens[p.doc_id as usize])
                    .min()
                    .unwrap_or(1);
                // SAFETY: each term writes only its own block rows / slot.
                unsafe {
                    p_btf.write(row, max_tf);
                    p_bml.write(row, min_len);
                }
                t_max_tf = t_max_tf.max(max_tf);
                t_min_len = t_min_len.min(min_len);
                let _ = bi;
            }
            // SAFETY: as above.
            unsafe {
                p_ttf.write(slot, t_max_tf);
                p_tml.write(slot, t_min_len.clamp(1, u32::MAX - 1));
            }
        });
    }

    // ids.bin from the doc store.
    let num_docs = doc_lens.len();
    let mut hashes = Vec::with_capacity(num_docs);
    for &offset in doc_offsets.iter().take(num_docs) {
        let pos = offset as usize;
        let len = u16::from_le_bytes([docs[pos], docs[pos + 1]]) as usize;
        let id = std::str::from_utf8(&docs[pos + 2..pos + 2 + len]).unwrap_or("");
        hashes.push(storage::id_hash(id));
    }
    storage::write_ids_file(dir, hashes.into_iter())?;

    let sections = MetaSections {
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
        name_to_slot,
        block_max_doc_ids,
        block_max_tfs,
        block_min_lens,
        block_byte_offsets,
    };
    drop(meta); // release the mapping before overwriting the file
    storage::write_meta(&sections, (&dict_groups, &dict_bytes), dir)
        .context("failed to write v4 metadata")?;
    Ok(num_terms)
}
