//! Posting lists and per-block metadata for Block-Max WAND.
//!
//! Block metadata is deliberately minimal: blocks have a fixed size, so a
//! block's posting range is derivable from its index (`start = i * B`,
//! `end = min(df, (i+1) * B)`), and the stored per-block facts are
//! `max_doc_id` (for skipping past blocks) and the **impact pair**
//! `(max_tf, min_doc_len)` — the dominating coordinates from which a safe
//! BM25 upper bound is computed *at query time* with whatever idf and
//! average document length are current. (BM25's contribution is monotone
//! increasing in tf and decreasing in doc_len, so the pair dominates every
//! posting in the block.) Storing coordinates instead of precomputed scores
//! is what keeps skipping provably exact when corpus statistics change —
//! the prerequisite for incremental, multi-segment indexes. This mirrors
//! Lucene's "impacts".

use serde::{Deserialize, Serialize};

use crate::bm25;

/// Default number of postings per block.
pub const DEFAULT_BLOCK_SIZE: usize = 128;

/// One (document, term-frequency) entry in a posting list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Posting {
    pub doc_id: u32,
    pub tf: u32,
}

/// A posting list sorted by `doc_id`, with per-block skip metadata and the
/// term's idf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingList {
    pub postings: Vec<Posting>,
    /// Last doc_id of each block (sorted ascending across blocks).
    pub block_max_doc_ids: Vec<u32>,
    /// Impact pairs: per block, the largest tf and the smallest doc length —
    /// the coordinates of a safe query-time BM25 upper bound.
    pub block_max_tfs: Vec<u32>,
    pub block_min_lens: Vec<u32>,
    /// Term-level impact pair (dominates the whole list).
    pub term_max_tf: u32,
    pub term_min_len: u32,
    /// BM25 idf of the term, computed with build-time corpus stats.
    pub idf: f32,
}

impl PostingList {
    /// Build a finalized posting list: sort by doc_id, compute idf, and build
    /// block metadata. `doc_lens` maps doc_id -> length in tokens.
    pub fn build(
        mut postings: Vec<Posting>,
        num_docs: usize,
        doc_lens: &[u32],
        block_size: usize,
    ) -> Self {
        postings.sort_unstable_by_key(|p| p.doc_id);
        let idf = bm25::idf(num_docs, postings.len());
        let (block_max_doc_ids, block_max_tfs, block_min_lens) =
            build_blocks(&postings, doc_lens, block_size);
        let term_max_tf = block_max_tfs.iter().copied().max().unwrap_or(1);
        let term_min_len = block_min_lens.iter().copied().min().unwrap_or(1);
        Self {
            postings,
            block_max_doc_ids,
            block_max_tfs,
            block_min_lens,
            term_max_tf,
            term_min_len,
            idf,
        }
    }

    /// Document frequency of the term.
    pub fn df(&self) -> usize {
        self.postings.len()
    }
}

/// Read-only view of one term's posting data, independent of where the
/// postings live (in memory or compressed on an mmap'd file). This is what
/// Block-Max WAND cursors operate on.
pub struct TermPostings<'a> {
    /// BM25 idf under the stats this query is being scored with.
    pub idf: f32,
    /// Average document length under the same stats.
    pub avg_doc_len: f32,
    /// Document frequency (total number of postings).
    pub df: usize,
    /// Postings per block (fixed for the whole index).
    pub block_size: usize,
    /// Last doc_id of each block.
    pub block_max_doc_ids: &'a [u32],
    /// Impact pairs: per-block max tf and min doc length.
    pub block_max_tfs: &'a [u32],
    pub block_min_lens: &'a [u32],
    /// Term-level impact pair.
    pub term_max_tf: u32,
    pub term_min_len: u32,
    pub source: BlockSource<'a>,
}

/// Where a term's posting blocks come from.
pub enum BlockSource<'a> {
    /// Plain postings in memory (freshly built index).
    Memory(&'a [Posting]),
    /// Delta-encoded, bit-packed blocks inside a memory-mapped region.
    /// `block_offsets[i]` is the byte offset of block `i` within `bytes`.
    Compressed {
        bytes: &'a [u8],
        block_offsets: &'a [u32],
    },
}

impl TermPostings<'_> {
    pub fn num_blocks(&self) -> usize {
        self.block_max_doc_ids.len()
    }

    /// Safe upper bound on this term's contribution anywhere in the list,
    /// computed from the term impact pair under the query's stats.
    pub fn list_bound(&self) -> f32 {
        crate::bm25::term_contribution(
            self.idf,
            self.term_max_tf,
            self.term_min_len,
            self.avg_doc_len,
        )
    }

    /// Safe upper bound on this term's contribution within block `b`.
    pub fn block_bound(&self, b: usize) -> f32 {
        crate::bm25::term_contribution(
            self.idf,
            self.block_max_tfs[b],
            self.block_min_lens[b],
            self.avg_doc_len,
        )
    }

    /// Index of the first posting in block `i` (inclusive).
    pub fn block_start(&self, i: usize) -> usize {
        i * self.block_size
    }

    /// Index one past the last posting in block `i` (exclusive).
    pub fn block_end(&self, i: usize) -> usize {
        ((i + 1) * self.block_size).min(self.df)
    }

    /// Materialize the doc_ids of one block into `out`. For the compressed
    /// source this is the only place doc_ids are ever decoded — blocks that
    /// Block-Max WAND skips are never touched (and with mmap, never read
    /// from disk). Term frequencies are *not* decoded here; fetch them with
    /// [`Self::tf_at`] only for postings that actually get scored.
    pub fn decode_docs(&self, block_idx: usize, out: &mut Vec<u32>) {
        match &self.source {
            BlockSource::Memory(postings) => {
                out.clear();
                out.extend(
                    postings[self.block_start(block_idx)..self.block_end(block_idx)]
                        .iter()
                        .map(|p| p.doc_id),
                );
            }
            BlockSource::Compressed {
                bytes,
                block_offsets,
            } => {
                let offset = block_offsets[block_idx] as usize;
                crate::compress::decode_block_docs(&bytes[offset..], out);
            }
        }
    }

    /// Term frequency of the `within`-th posting of a block. O(1): in the
    /// compressed source, fixed-width packing allows random access.
    pub fn tf_at(&self, block_idx: usize, within: usize) -> u32 {
        match &self.source {
            BlockSource::Memory(postings) => postings[self.block_start(block_idx) + within].tf,
            BlockSource::Compressed {
                bytes,
                block_offsets,
            } => {
                let offset = block_offsets[block_idx] as usize;
                crate::compress::block_tf(&bytes[offset..], within)
            }
        }
    }

    /// Materialize one full block of postings into `out` (used by tests and
    /// tooling; the search path uses `decode_docs` + `tf_at`).
    pub fn decode_block(&self, block_idx: usize, out: &mut Vec<Posting>) {
        match &self.source {
            BlockSource::Memory(postings) => {
                out.clear();
                out.extend_from_slice(
                    &postings[self.block_start(block_idx)..self.block_end(block_idx)],
                );
            }
            BlockSource::Compressed {
                bytes,
                block_offsets,
            } => {
                let offset = block_offsets[block_idx] as usize;
                crate::compress::decode_block(&bytes[offset..], out);
            }
        }
    }
}

/// Split `postings` into fixed-size blocks and compute per-block skip
/// metadata: (max doc_id, max tf, min doc length) per block.
pub fn build_blocks(
    postings: &[Posting],
    doc_lens: &[u32],
    block_size: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    assert!(block_size > 0, "block size must be positive");
    let num_blocks = postings.len().div_ceil(block_size);
    let mut max_doc_ids = Vec::with_capacity(num_blocks);
    let mut max_tfs = Vec::with_capacity(num_blocks);
    let mut min_lens = Vec::with_capacity(num_blocks);
    for chunk in postings.chunks(block_size) {
        max_doc_ids.push(chunk[chunk.len() - 1].doc_id);
        max_tfs.push(chunk.iter().map(|p| p.tf).max().expect("non-empty"));
        min_lens.push(
            chunk
                .iter()
                .map(|p| doc_lens[p.doc_id as usize])
                .min()
                .expect("non-empty"),
        );
    }
    (max_doc_ids, max_tfs, min_lens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn posting(doc_id: u32, tf: u32) -> Posting {
        Posting { doc_id, tf }
    }

    #[test]
    fn build_sorts_postings_by_doc_id() {
        let doc_lens = vec![10u32; 6];
        let list = PostingList::build(
            vec![posting(5, 1), posting(1, 2), posting(3, 1)],
            6,
            &doc_lens,
            2,
        );
        let ids: Vec<u32> = list.postings.iter().map(|p| p.doc_id).collect();
        assert_eq!(ids, vec![1, 3, 5]);
        assert_eq!(list.term_max_tf, 2);
        assert_eq!(list.term_min_len, 10);
    }

    #[test]
    fn blocks_cover_all_postings_with_correct_bounds() {
        let doc_lens = vec![10u32; 300];
        let postings: Vec<Posting> = (0..250).map(|i| posting(i, 1 + i % 3)).collect();
        let (max_doc_ids, max_tfs, min_lens) = build_blocks(&postings, &doc_lens, 128);

        assert_eq!(max_doc_ids.len(), 2);
        assert_eq!(max_tfs.len(), 2);
        assert_eq!(min_lens.len(), 2);
        // Block ranges are derived from the fixed block size; the stored
        // max_doc_id must be the last doc of each derived range.
        assert_eq!(max_doc_ids[0], 127);
        assert_eq!(max_doc_ids[1], 249);
        assert_eq!(max_tfs[0], 3);
        assert_eq!(min_lens[0], 10);
    }

    #[test]
    fn block_max_score_bounds_every_contribution_in_block() {
        // Pseudo-random tfs and doc lengths; verify the invariant that makes
        // Block-Max WAND skipping safe.
        let mut state = 0x12345678u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let num_docs = 1000usize;
        let doc_lens: Vec<u32> = (0..num_docs).map(|_| 5 + next() % 200).collect();
        let avg = doc_lens.iter().map(|&l| l as f64).sum::<f64>() as f32 / num_docs as f32;
        let mut postings = Vec::new();
        for doc_id in 0..num_docs as u32 {
            if next() % 3 == 0 {
                postings.push(posting(doc_id, 1 + next() % 10));
            }
        }
        let idf = bm25::idf(num_docs, postings.len());
        let block_size = 32;
        let (max_doc_ids, max_tfs, min_lens) = build_blocks(&postings, &doc_lens, block_size);

        // The impact-pair bound must dominate every actual contribution in
        // its block — under the build stats AND under shifted stats (more
        // docs, different average), which is the property that keeps
        // skipping exact as a segmented corpus grows.
        for (extra_docs, avg_scale) in [(0usize, 1.0f32), (5000, 1.6), (0, 0.5)] {
            let n = num_docs + extra_docs;
            let idf_now = bm25::idf(n, postings.len());
            let avg_now = avg * avg_scale;
            for (b, chunk) in postings.chunks(block_size).enumerate() {
                let bound = bm25::term_contribution(idf_now, max_tfs[b], min_lens[b], avg_now);
                for p in chunk {
                    let contribution = bm25::term_contribution(
                        idf_now,
                        p.tf,
                        doc_lens[p.doc_id as usize],
                        avg_now,
                    );
                    assert!(
                        bound >= contribution,
                        "block bound {bound} < contribution {contribution} (n={n})"
                    );
                    assert!(p.doc_id <= max_doc_ids[b]);
                }
            }
        }
        let _ = idf;
    }
}
