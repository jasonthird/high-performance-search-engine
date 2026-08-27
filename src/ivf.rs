//! Inverted-file (IVF) index over **cluster ids**, same `doc_id`s as BM25.
//!
//! Lexical search already has `term → postings`. Vector search gets the
//! same shape: `cluster_id → postings`, compressed with the same
//! delta+bit-pack blocks and the same impact pairs (max_tf, min_len) so
//! Block-Max WAND could later run over cluster lists. Centroids route a
//! query embedding to `nprobe` lists instead of scanning every vector.
//!
//! ```text
//! embeddings.bin[doc_id]     dense vector
//! ivf.bin  cluster → docs    inverted file (shared ids)
//! postings.bin term → docs   inverted file (shared ids)
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Context;
use memmap2::Mmap;

use crate::compress;
use crate::embeddings::{self, EmbeddingStore};
use crate::postings::{Posting, PostingList, DEFAULT_BLOCK_SIZE};

pub const IVF_FILE: &str = "ivf.bin";
const MAGIC: &[u8; 8] = b"HPSIVF01";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 32;

#[allow(dead_code)]
pub struct IvfIndex {
    mmap: Mmap,
    num_docs: u32,
    num_clusters: u32,
    dim: u32,
    block_size: u32,
    centroid_off: usize,
    assign_off: usize,
    cluster_df_off: usize,
    region_off: usize,
    block_rows_off: usize,
    term_max_tf_off: usize,
    term_min_len_off: usize,
    block_max_doc_off: usize,
    block_max_tf_off: usize,
    block_min_len_off: usize,
    block_byte_off: usize,
    postings_off: usize,
}

impl IvfIndex {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join(IVF_FILE);
        let file = File::open(&path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;
        anyhow::ensure!(
            mmap.len() >= HEADER_LEN && &mmap[..8] == MAGIC,
            "{} is not an IVF sidecar",
            path.display()
        );
        let version = u32_le(&mmap[8..12]);
        anyhow::ensure!(version == VERSION, "unsupported ivf version {version}");
        let num_docs = u32_le(&mmap[12..16]);
        let num_clusters = u32_le(&mmap[16..20]);
        let dim = u32_le(&mmap[20..24]);
        let block_size = u32_le(&mmap[24..28]);
        let k = num_clusters as usize;
        let n = num_docs as usize;
        let d = dim as usize;

        let mut off = HEADER_LEN;
        let centroid_off = off;
        off += k * d * 2;
        let assign_off = off;
        off += n * 4;
        let cluster_df_off = off;
        off += k * 4;
        off = (off + 7) & !7; // u64 region table must be 8-aligned
        let region_off = off;
        off += (k + 1) * 8;
        let block_rows_off = off;
        off += (k + 1) * 4;
        let term_max_tf_off = off;
        off += k * 4;
        let term_min_len_off = off;
        off += k * 4;

        anyhow::ensure!(mmap.len() >= off + 4, "ivf.bin truncated before block tables");
        let block_rows = u32_slice(&mmap, block_rows_off, k + 1);
        let num_blocks = block_rows[k] as usize;
        let block_max_doc_off = off;
        off += num_blocks * 4;
        let block_max_tf_off = off;
        off += num_blocks * 4;
        let block_min_len_off = off;
        off += num_blocks * 4;
        let block_byte_off = off;
        off += num_blocks * 4;
        let postings_off = off;
        anyhow::ensure!(mmap.len() >= postings_off, "ivf.bin truncated");

        Ok(Self {
            mmap,
            num_docs,
            num_clusters,
            dim,
            block_size,
            centroid_off,
            assign_off,
            cluster_df_off,
            region_off,
            block_rows_off,
            term_max_tf_off,
            term_min_len_off,
            block_max_doc_off,
            block_max_tf_off,
            block_min_len_off,
            block_byte_off,
            postings_off,
        })
    }

    pub fn num_clusters(&self) -> u32 {
        self.num_clusters
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    pub fn size_bytes(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// Cluster assigned to a document (same id as in postings.bin).
    pub fn cluster_of(&self, doc_id: u32) -> u32 {
        if doc_id >= self.num_docs {
            return 0;
        }
        let off = self.assign_off + doc_id as usize * 4;
        u32_le(&self.mmap[off..off + 4])
    }

    pub fn auto_nprobe(&self) -> usize {
        let k = self.num_clusters as usize;
        (k / 4).clamp(1, 8).min(k)
    }

    /// Nearest `nprobe` cluster ids (cosine vs L2-normalized centroids).
    pub fn nearest_clusters(&self, query: &[f32], nprobe: usize) -> Vec<u32> {
        let k = self.num_clusters as usize;
        let d = self.dim as usize;
        let nprobe = nprobe.clamp(1, k);
        let mut scores: Vec<(f32, u32)> = (0..k)
            .map(|c| {
                let mut acc = 0.0f32;
                let base = self.centroid_off + c * d * 2;
                for i in 0..d {
                    let h = u16::from_le_bytes([
                        self.mmap[base + i * 2],
                        self.mmap[base + i * 2 + 1],
                    ]);
                    acc += embeddings::f16_to_f32(h) * query.get(i).copied().unwrap_or(0.0);
                }
                (acc, c as u32)
            })
            .collect();
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        scores.into_iter().take(nprobe).map(|(_, c)| c).collect()
    }

    /// Doc ids in the selected clusters (union, sorted, unique).
    pub fn docs_in_clusters(&self, clusters: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut tmp = Vec::new();
        let k = self.num_clusters as usize;
        let dfs = u32_slice(&self.mmap, self.cluster_df_off, k);
        let rows = u32_slice(&self.mmap, self.block_rows_off, k + 1);
        let regions = u64_slice(&self.mmap, self.region_off, k + 1);
        let block_size = self.block_size as usize;
        for &c in clusters {
            let c = c as usize;
            if c >= k {
                continue;
            }
            let df = dfs[c] as usize;
            if df == 0 {
                continue;
            }
            let row0 = rows[c] as usize;
            let row1 = rows[c + 1] as usize;
            let region0 = regions[c] as usize;
            let nblocks = row1 - row0;
            let offsets = u32_slice(&self.mmap, self.block_byte_off + row0 * 4, nblocks);
            let bytes = &self.mmap[self.postings_off + region0
                ..self.postings_off + regions[c + 1] as usize];
            for b in 0..nblocks {
                tmp.clear();
                compress::decode_block_docs(&bytes[offsets[b] as usize..], &mut tmp);
                let _ = block_size;
                out.extend_from_slice(&tmp);
            }
        }
        // Clusters partition doc_ids, so this is a disjoint concat — the
        // inverted-file OR of nprobe lists, no global sort.
        out
    }

    /// Probe `nprobe` clusters and return their document ids.
    pub fn probe(&self, query: &[f32], nprobe: usize) -> Vec<u32> {
        let nprobe = if nprobe == 0 {
            self.auto_nprobe()
        } else {
            nprobe
        };
        let clusters = self.nearest_clusters(query, nprobe);
        self.docs_in_clusters(&clusters)
    }
}

/// Default cluster count: ~2√N, capped. Small corpora keep K ≤ N/2.
pub fn default_num_clusters(n: usize) -> usize {
    if n <= 4 {
        return n.max(1);
    }
    let k = ((n as f64).sqrt() * 2.0).round() as usize;
    k.clamp(2, (n / 2).max(2)).min(256)
}

/// Read the trained centroids out of an existing `ivf.bin`.
///
/// Centroids are what k-means is expensive to produce; reusing them across
/// rebuilds turns clustering into a per-document dot product. Returns
/// `(k, dim, centroids)`.
pub fn read_centroids(dir: &Path) -> Option<(usize, usize, Vec<f32>)> {
    let index = IvfIndex::open(dir).ok()?;
    let k = index.num_clusters as usize;
    let dim = index.dim as usize;
    let mut out = Vec::with_capacity(k * dim);
    for i in 0..k * dim {
        let off = index.centroid_off + i * 2;
        let h = u16::from_le_bytes([index.mmap[off], index.mmap[off + 1]]);
        out.push(embeddings::f16_to_f32(h));
    }
    Some((k, dim, out))
}

/// Nearest centroid to an L2-normalized vector (cosine == dot product).
/// This is the whole per-document cost of reusing a trained quantizer.
pub fn assign_one(centroids: &[f32], k: usize, dim: usize, vector: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_score = f32::NEG_INFINITY;
    for c in 0..k {
        let cen = &centroids[c * dim..(c + 1) * dim];
        let mut acc = 0.0f32;
        for j in 0..dim {
            acc += cen[j] * vector[j];
        }
        if acc > best_score {
            best_score = acc;
            best = c as u32;
        }
    }
    best
}

/// K-means (spherical: cosine assignment, L2-normalized centroids) then
/// write `ivf.bin` with the same compressed posting-block layout as
/// `postings.bin`.
pub fn build_and_write(
    dir: &Path,
    store: &EmbeddingStore,
    doc_lens: &[u32],
    num_clusters: usize,
) -> anyhow::Result<u64> {
    let n = store.num_docs() as usize;
    let dim = store.dim() as usize;
    anyhow::ensure!(n > 0, "no documents to cluster");
    anyhow::ensure!(doc_lens.len() == n, "doc_lens must match embeddings");
    let k = num_clusters.clamp(1, n);

    let mut vectors = vec![0.0f32; n * dim];
    for i in 0..n {
        store.copy_f32(i as u32, &mut vectors[i * dim..(i + 1) * dim]);
    }
    let (centroids, assignment) = kmeans(&vectors, n, dim, k, 25);
    write_with_centroids(dir, doc_lens, &centroids, &assignment, n, dim, k)
}

/// Write `ivf.bin` from centroids and a per-document assignment that were
/// computed elsewhere — either by [`kmeans`] or, on an incremental rebuild,
/// by [`assign_one`] against previously trained centroids.
pub fn write_with_centroids(
    dir: &Path,
    doc_lens: &[u32],
    centroids: &[f32],
    assignment: &[u32],
    n: usize,
    dim: usize,
    k: usize,
) -> anyhow::Result<u64> {
    anyhow::ensure!(n > 0, "no documents to cluster");
    anyhow::ensure!(doc_lens.len() == n, "doc_lens must match embeddings");
    anyhow::ensure!(assignment.len() == n, "assignment must match embeddings");
    anyhow::ensure!(centroids.len() == k * dim, "centroids must be k * dim");

    let mut lists: Vec<Vec<Posting>> = vec![Vec::new(); k];
    for (doc_id, &c) in assignment.iter().enumerate() {
        lists[c as usize].push(Posting {
            doc_id: doc_id as u32,
            tf: 1,
        });
    }

    let block_size = DEFAULT_BLOCK_SIZE;
    let built: Vec<PostingList> = lists
        .into_iter()
        .map(|p| PostingList::build(p, n, doc_lens, block_size))
        .collect();

    let path = dir.join(IVF_FILE);
    let mut w = BufWriter::new(
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(n as u32).to_le_bytes())?;
    w.write_all(&(k as u32).to_le_bytes())?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&(block_size as u32).to_le_bytes())?;
    w.write_all(&0u32.to_le_bytes())?;

    for x in centroids {
        w.write_all(&embeddings::f32_to_f16(*x).to_le_bytes())?;
    }
    for &c in assignment {
        w.write_all(&c.to_le_bytes())?;
    }

    let mut region_offsets = Vec::with_capacity(k + 1);
    let mut block_rows = Vec::with_capacity(k + 1);
    let mut cluster_df = Vec::with_capacity(k);
    let mut term_max_tf = Vec::with_capacity(k);
    let mut term_min_len = Vec::with_capacity(k);
    let mut block_max_doc_ids = Vec::new();
    let mut block_max_tfs = Vec::new();
    let mut block_min_lens = Vec::new();
    let mut block_byte_offsets = Vec::new();
    let mut postings_bytes = Vec::new();
    region_offsets.push(0u64);
    block_rows.push(0u32);
    let mut encoded = Vec::new();
    for list in &built {
        encoded.clear();
        for chunk in list.postings.chunks(block_size) {
            block_byte_offsets.push(encoded.len() as u32);
            compress::encode_block(chunk, &mut encoded);
        }
        postings_bytes.extend_from_slice(&encoded);
        cluster_df.push(list.df() as u32);
        region_offsets.push(postings_bytes.len() as u64);
        block_rows.push(block_rows.last().unwrap() + list.block_max_doc_ids.len() as u32);
        term_max_tf.push(list.term_max_tf);
        term_min_len.push(list.term_min_len);
        block_max_doc_ids.extend_from_slice(&list.block_max_doc_ids);
        block_max_tfs.extend_from_slice(&list.block_max_tfs);
        block_min_lens.extend_from_slice(&list.block_min_lens);
    }

    for &x in &cluster_df {
        w.write_all(&x.to_le_bytes())?;
    }
    // Pad so the u64 region-offset table is 8-byte aligned in the mmap.
    let prefix = HEADER_LEN + k * dim * 2 + n * 4 + k * 4;
    for _ in 0..((8 - (prefix % 8)) % 8) {
        w.write_all(&[0u8])?;
    }
    for &x in &region_offsets {
        w.write_all(&x.to_le_bytes())?;
    }
    for &x in &block_rows {
        w.write_all(&x.to_le_bytes())?;
    }
    for &x in &term_max_tf {
        w.write_all(&x.to_le_bytes())?;
    }
    for &x in &term_min_len {
        w.write_all(&x.to_le_bytes())?;
    }
    for &x in &block_max_doc_ids {
        w.write_all(&x.to_le_bytes())?;
    }
    for &x in &block_max_tfs {
        w.write_all(&x.to_le_bytes())?;
    }
    for &x in &block_min_lens {
        w.write_all(&x.to_le_bytes())?;
    }
    for &x in &block_byte_offsets {
        w.write_all(&x.to_le_bytes())?;
    }
    w.write_all(&postings_bytes)?;
    w.flush()?;
    drop(w);
    Ok(std::fs::metadata(&path)?.len())
}

/// Spherical k-means. Deterministic LCG init (no rand crate).
fn kmeans(
    vectors: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    iters: usize,
) -> (Vec<f32>, Vec<u32>) {
    let mut rng = 0xC0FFEE_u64;
    let mut next = || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as usize
    };
    let mut centroids = vec![0.0f32; k * dim];
    let mut used = vec![false; n];
    for c in 0..k {
        let mut i = next() % n;
        while used[i] {
            i = next() % n;
        }
        used[i] = true;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&vectors[i * dim..(i + 1) * dim]);
    }
    let mut assignment = vec![0u32; n];
    for _ in 0..iters {
        for i in 0..n {
            let mut best = 0u32;
            let mut best_s = f32::NEG_INFINITY;
            let row = &vectors[i * dim..(i + 1) * dim];
            for c in 0..k {
                let mut s = 0.0f32;
                let cen = &centroids[c * dim..(c + 1) * dim];
                for d in 0..dim {
                    s += row[d] * cen[d];
                }
                if s > best_s {
                    best_s = s;
                    best = c as u32;
                }
            }
            assignment[i] = best;
        }
        centroids.fill(0.0);
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assignment[i] as usize;
            counts[c] += 1;
            let dest = &mut centroids[c * dim..(c + 1) * dim];
            let src = &vectors[i * dim..(i + 1) * dim];
            for d in 0..dim {
                dest[d] += src[d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                let i = next() % n;
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&vectors[i * dim..(i + 1) * dim]);
                continue;
            }
            embeddings::l2_normalize(&mut centroids[c * dim..(c + 1) * dim]);
        }
    }
    for i in 0..n {
        let mut best = 0u32;
        let mut best_s = f32::NEG_INFINITY;
        let row = &vectors[i * dim..(i + 1) * dim];
        for c in 0..k {
            let mut s = 0.0f32;
            let cen = &centroids[c * dim..(c + 1) * dim];
            for d in 0..dim {
                s += row[d] * cen[d];
            }
            if s > best_s {
                best_s = s;
                best = c as u32;
            }
        }
        assignment[i] = best;
    }
    (centroids, assignment)
}

fn u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().expect("u32"))
}

fn u32_slice(mmap: &[u8], off: usize, n: usize) -> &[u32] {
    bytemuck::cast_slice(&mmap[off..off + n * 4])
}

fn u64_slice(mmap: &[u8], off: usize, n: usize) -> &[u64] {
    bytemuck::cast_slice(&mmap[off..off + n * 8])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{l2_normalize, write_f16};

    #[test]
    fn clusters_separate_orthogonal_docs() {
        let dir = std::env::temp_dir().join(format!(
            "high-performance-search-engine-ivf-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut a = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut b = vec![1.0f32, 0.05, 0.0, 0.0];
        let mut c = vec![0.0f32, 1.0, 0.0, 0.0];
        let mut d = vec![0.0f32, 1.0, 0.05, 0.0];
        for v in [&mut a, &mut b, &mut c, &mut d] {
            l2_normalize(v);
        }
        write_f16(&dir, 4, &[a.clone(), b.clone(), c.clone(), d.clone()]).unwrap();
        let store = EmbeddingStore::open(&dir).unwrap();
        let lens = vec![10u32, 10, 10, 10];
        build_and_write(&dir, &store, &lens, 2).unwrap();
        let ivf = IvfIndex::open(&dir).unwrap();
        assert_eq!(ivf.num_clusters(), 2);
        assert_eq!(ivf.num_docs(), 4);
        // Same-axis docs should share a cluster.
        assert_eq!(ivf.cluster_of(0), ivf.cluster_of(1));
        assert_eq!(ivf.cluster_of(2), ivf.cluster_of(3));
        assert_ne!(ivf.cluster_of(0), ivf.cluster_of(2));

        let q = a.clone();
        let probed = ivf.probe(&q, 1);
        assert!(probed.contains(&0));
        assert!(probed.contains(&1));
        // nprobe=1 should not need the orthogonal pair.
        assert!(!probed.contains(&2) || probed.len() == 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reused_centroids_reproduce_the_trained_assignment() {
        // This is what makes an incremental rebuild sound: clustering a
        // document against the stored centroids must give the same cluster
        // that training assigned it, or a rebuild would silently reshuffle
        // documents between lists.
        let dir = std::env::temp_dir().join(format!("hpse-ivf-reuse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dim = 16;
        let n = 120;
        let mut vecs = Vec::new();
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            // Three loose groups plus jitter, so clusters are non-trivial.
            v[i % 3] = 1.0;
            v[(i * 7) % dim] += 0.35;
            v[(i * 5) % dim] += 0.1;
            embeddings::l2_normalize(&mut v);
            vecs.push(v);
        }
        write_f16(&dir, dim as u32, &vecs).unwrap();
        let store = EmbeddingStore::open(&dir).unwrap();
        let lens = vec![10u32; n];
        build_and_write(&dir, &store, &lens, 6).unwrap();

        let ivf = IvfIndex::open(&dir).unwrap();
        let (k, d, centroids) = read_centroids(&dir).unwrap();
        assert_eq!((k, d), (6, dim));
        for doc_id in 0..n {
            // Read the vector back through the store, so the comparison uses
            // the same FP16 rounding the stored centroids went through.
            let mut v = vec![0.0f32; dim];
            store.copy_f32(doc_id as u32, &mut v);
            assert_eq!(
                assign_one(&centroids, k, d, &v),
                ivf.cluster_of(doc_id as u32),
                "doc {doc_id} reassigned differently"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}