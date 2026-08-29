//! Product quantization of document embeddings (IVF-PQ style scoring).
//!
//! Each 768-d vector is split into `M` subspaces; each subspace is coded
//! with 256 centroids (1 byte). Query scoring is asymmetric: the query
//! stays FP32, docs are codes, distance is a sum of M table lookups.
//!
//! Lives next to `embeddings.bin` / `ivf.bin` on the same `doc_id`s.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Context;
use memmap2::Mmap;

use crate::embeddings::{self, EmbeddingStore};

pub const PQ_FILE: &str = "pq.bin";

/// Minimum candidate documents before ADC scoring is worth using.
///
/// ADC is a near-fixed cost — building the `M x 256` lookup tables — plus
/// almost nothing per document, while exact scoring is a per-document dot
/// product. So the trade is decided by *how many documents get scored*, not
/// by corpus size directly.
///
/// Measured on an M-series laptop (`cargo run --release --example
/// pq_scoring_bench`), scoring 768-d vectors, after both paths were
/// vectorized (branch-free f16 decode, multi-accumulator dots):
///
/// ```text
/// candidates   exact FP16      PQ/ADC
///        100         68 us      181 us
///      1 000        646 us      155 us
///     10 000        6.8 ms      278 us
///    200 000        132 ms      1.9 ms
/// ```
///
/// That put the break-even near 225 candidates; the NEON f16 dot moved
/// exact scoring another ~6x (see `dot_f16_bytes`), pushing break-even to
/// ~1,100 — and ADC is no longer chosen automatically at all (recall, see
/// `PqMode`). This threshold sits well above the original break-even
/// deliberately: below break-even PQ is pure loss (it costs recall
/// and saves nothing), so the safe error is to keep scoring exactly for
/// longer than strictly necessary.
pub const MIN_CANDIDATES: usize = 900;

/// How many documents a query will score, given the IVF geometry.
///
/// With no inverted file every document is scanned; otherwise a query opens
/// `nprobe` of `num_clusters` lists, which hold `num_docs / num_clusters`
/// documents each on average.
pub fn estimated_candidates(num_docs: usize, num_clusters: usize, nprobe: usize) -> usize {
    if num_clusters == 0 || nprobe == 0 {
        return num_docs;
    }
    let nprobe = nprobe.min(num_clusters);
    (num_docs.saturating_mul(nprobe)) / num_clusters
}

/// Is ADC scoring worth it for this many candidates?
pub fn worth_using(candidates: usize) -> bool {
    candidates >= MIN_CANDIDATES
}
const MAGIC: &[u8; 8] = b"HPSPQ001";
const VERSION: u32 = 2;
const HEADER_LEN: usize = 32;
const KS: usize = 256;

pub struct PqIndex {
    mmap: Mmap,
    num_docs: u32,
    m: u32,
    subdim: u32,
    /// Trained centroids per subspace. Equals `KS` unless the corpus had
    /// fewer documents than that, in which case the remaining codebook slots
    /// are unwritten and must never be selected by an encoder.
    ks_used: u32,
    codebook_off: usize,
    codes_off: usize,
}

/// Per-query lookup tables: `table[m][code] = q_sub[m] · codebook[m][code]`.
pub struct AdcTables {
    table: Vec<f32>, // m * 256
}

impl PqIndex {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join(PQ_FILE);
        let file = File::open(&path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;
        anyhow::ensure!(
            mmap.len() >= HEADER_LEN && &mmap[..8] == MAGIC,
            "{} is not a PQ sidecar",
            path.display()
        );
        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        anyhow::ensure!(version == VERSION, "unsupported pq version");
        let num_docs = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
        let m = u32::from_le_bytes(mmap[16..20].try_into().unwrap());
        let ks = u32::from_le_bytes(mmap[20..24].try_into().unwrap());
        let subdim = u32::from_le_bytes(mmap[24..28].try_into().unwrap());
        let ks_used = u32::from_le_bytes(mmap[28..32].try_into().unwrap());
        anyhow::ensure!(ks as usize == KS, "expected 256-code PQ");
        anyhow::ensure!(
            ks_used > 0 && ks_used as usize <= KS,
            "pq.bin has an invalid trained-centroid count"
        );
        let codebook_off = HEADER_LEN;
        let codes_off = codebook_off + m as usize * KS * subdim as usize * 2;
        let need = codes_off + num_docs as usize * m as usize;
        anyhow::ensure!(mmap.len() >= need, "pq.bin truncated");
        Ok(Self {
            mmap,
            num_docs,
            m,
            subdim,
            ks_used,
            codebook_off,
            codes_off,
        })
    }

    /// The stored PQ code for a document, for repopulating the rebuild
    /// cache after the codebooks are retrained.
    pub fn codes_of(&self, doc_id: u32) -> &[u8] {
        let m = self.m as usize;
        if doc_id >= self.num_docs {
            return &[];
        }
        let off = self.codes_off + doc_id as usize * m;
        &self.mmap[off..off + m]
    }

    pub fn m(&self) -> usize {
        self.m as usize
    }

    pub fn ks_used(&self) -> usize {
        self.ks_used as usize
    }

    pub fn size_bytes(&self) -> u64 {
        self.mmap.len() as u64
    }

    pub fn prepare(&self, query: &[f32]) -> AdcTables {
        let m = self.m as usize;
        let sub = self.subdim as usize;
        let mut table = vec![0.0f32; m * KS];
        // This table build is the whole fixed cost of ADC scoring, so it is
        // written to vectorize: branch-free f16 decode and four independent
        // accumulators per centroid row (FP adds are not associative; one
        // accumulator would chain them and block SIMD).
        for sm in 0..m {
            let qlen = sub.min(query.len().saturating_sub(sm * sub));
            let qsub = &query[sm * sub..sm * sub + qlen];
            for c in 0..KS {
                let base = self.codebook_off + (sm * KS + c) * sub * 2;
                let row = &self.mmap[base..base + qlen * 2];
                let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let mut lanes = row.chunks_exact(8);
                let mut qs = qsub.chunks_exact(4);
                for (lane, q) in (&mut lanes).zip(&mut qs) {
                    a0 += embeddings::f16_to_f32_fast(u16::from_le_bytes([lane[0], lane[1]])) * q[0];
                    a1 += embeddings::f16_to_f32_fast(u16::from_le_bytes([lane[2], lane[3]])) * q[1];
                    a2 += embeddings::f16_to_f32_fast(u16::from_le_bytes([lane[4], lane[5]])) * q[2];
                    a3 += embeddings::f16_to_f32_fast(u16::from_le_bytes([lane[6], lane[7]])) * q[3];
                }
                let mut acc = (a0 + a2) + (a1 + a3);
                for (i, pair) in lanes.remainder().chunks_exact(2).enumerate() {
                    let h = u16::from_le_bytes([pair[0], pair[1]]);
                    acc += embeddings::f16_to_f32_fast(h) * qs.remainder()[i];
                }
                table[sm * KS + c] = acc;
            }
        }
        AdcTables { table }
    }

    pub fn adc(&self, tables: &AdcTables, doc_id: u32) -> f32 {
        if doc_id >= self.num_docs {
            return 0.0;
        }
        let m = self.m as usize;
        let codes = &self.mmap[self.codes_off + doc_id as usize * m
            ..self.codes_off + (doc_id as usize + 1) * m];
        let mut s = 0.0f32;
        for sm in 0..m {
            s += tables.table[sm * KS + codes[sm] as usize];
        }
        s
    }
}

/// Read the trained codebooks out of an existing `pq.bin`.
///
/// Training these (M subspaces x 256 centroids of k-means) is by far the
/// most expensive part of building the sidecar. Reusing them reduces
/// encoding a document to [`encode_one`]. Returns `(m, sub, ks_used, codebooks)`.
pub fn read_codebooks(dir: &Path) -> Option<(usize, usize, usize, Vec<f32>)> {
    let index = PqIndex::open(dir).ok()?;
    let m = index.m as usize;
    let sub = index.subdim as usize;
    let mut out = Vec::with_capacity(m * KS * sub);
    for i in 0..m * KS * sub {
        let off = index.codebook_off + i * 2;
        let h = u16::from_le_bytes([index.mmap[off], index.mmap[off + 1]]);
        out.push(embeddings::f16_to_f32(h));
    }
    Some((m, sub, index.ks_used(), out))
}

/// Quantize one vector against trained codebooks: for each subspace, the
/// nearest of its `ks_used` trained centroids.
///
/// `ks_used` matters: with fewer documents than 256, training fills only
/// part of each codebook, and the untrained (all-zero) slots would otherwise
/// win the nearest-centroid search and produce codes that training never
/// assigned.
pub fn encode_one(
    codebooks: &[f32],
    m: usize,
    sub: usize,
    ks_used: usize,
    vector: &[f32],
    out: &mut [u8],
) {
    let ks_used = ks_used.clamp(1, KS);
    for sm in 0..m {
        let q = &vector[sm * sub..(sm + 1) * sub];
        let book = &codebooks[sm * KS * sub..(sm + 1) * KS * sub];
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for c in 0..ks_used {
            let cen = &book[c * sub..(c + 1) * sub];
            let mut d = 0.0f32;
            for j in 0..sub {
                let diff = q[j] - cen[j];
                d += diff * diff;
            }
            if d < best_d {
                best_d = d;
                best = c as u8;
            }
        }
        out[sm] = best;
    }
}

/// Write `pq.bin` from codebooks and codes computed elsewhere — by training
/// here, or by [`encode_one`] against previously trained codebooks.
pub fn write_with_codebooks(
    dir: &Path,
    codebooks: &[f32],
    codes: &[u8],
    n: usize,
    m: usize,
    sub: usize,
    ks_used: usize,
) -> anyhow::Result<u64> {
    anyhow::ensure!(codes.len() == n * m, "codes must be n * m");
    anyhow::ensure!(codebooks.len() == m * KS * sub, "codebooks must be m * KS * sub");
    write_pq(dir, codebooks, codes, n, m, sub, ks_used)
}

/// Train M×256 codebooks on `store` and write `pq.bin`.
pub fn build_and_write(
    dir: &Path,
    store: &EmbeddingStore,
    m: usize,
) -> anyhow::Result<u64> {
    let n = store.num_docs() as usize;
    let dim = store.dim() as usize;
    anyhow::ensure!(n > 0 && dim > 0 && dim % m == 0, "PQ needs dim divisible by M");
    let sub = dim / m;
    let mut vectors = vec![0.0f32; n * dim];
    for i in 0..n {
        store.copy_f32(i as u32, &mut vectors[i * dim..(i + 1) * dim]);
    }

    let mut codebooks = vec![0.0f32; m * KS * sub];
    let mut codes = vec![0u8; n * m];
    for sm in 0..m {
        let mut slice = vec![0.0f32; n * sub];
        for i in 0..n {
            let src = &vectors[i * dim + sm * sub..i * dim + (sm + 1) * sub];
            slice[i * sub..(i + 1) * sub].copy_from_slice(src);
        }
        let k_use = KS.min(n);
        let (cb, assign) = subspace_kmeans(&slice, n, sub, k_use, 12);
        let dest = &mut codebooks[sm * KS * sub..(sm + 1) * KS * sub];
        dest[..k_use * sub].copy_from_slice(&cb);
        for i in 0..n {
            codes[i * m + sm] = assign[i] as u8;
        }
    }

    write_pq(dir, &codebooks, &codes, n, m, sub, KS.min(n))
}

fn write_pq(
    dir: &Path,
    codebooks: &[f32],
    codes: &[u8],
    n: usize,
    m: usize,
    sub: usize,
    ks_used: usize,
) -> anyhow::Result<u64> {
    let path = dir.join(PQ_FILE);
    let mut w = BufWriter::new(
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(n as u32).to_le_bytes())?;
    w.write_all(&(m as u32).to_le_bytes())?;
    w.write_all(&(KS as u32).to_le_bytes())?;
    w.write_all(&(sub as u32).to_le_bytes())?;
    w.write_all(&(ks_used.clamp(1, KS) as u32).to_le_bytes())?;
    for x in codebooks {
        w.write_all(&embeddings::f32_to_f16(*x).to_le_bytes())?;
    }
    w.write_all(codes)?;
    w.flush()?;
    drop(w);
    Ok(std::fs::metadata(&path)?.len())
}

fn subspace_kmeans(
    vectors: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    iters: usize,
) -> (Vec<f32>, Vec<u32>) {
    let mut rng = 0xA5A5_u64.wrapping_mul(dim as u64 + 1);
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
        let mut guard = 0;
        while used[i] && guard < n {
            i = next() % n;
            guard += 1;
        }
        used[i] = true;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&vectors[i * dim..(i + 1) * dim]);
    }
    let mut assignment = vec![0u32; n];
    for _ in 0..iters {
        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];
            let mut best = 0u32;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cen = &centroids[c * dim..(c + 1) * dim];
                let mut d = 0.0f32;
                for j in 0..dim {
                    let e = row[j] - cen[j];
                    d += e * e;
                }
                if d < best_d {
                    best_d = d;
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
            for j in 0..dim {
                dest[j] += src[j];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                let i = next() % n;
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&vectors[i * dim..(i + 1) * dim]);
                continue;
            }
            let inv = 1.0 / counts[c] as f32;
            for j in 0..dim {
                centroids[c * dim + j] *= inv;
            }
        }
    }
    // Final assignment against the centroids we are about to return. Without
    // this the loop returns an assignment computed one update behind, so the
    // stored codes are not the nearest centroids of the stored codebooks —
    // every ADC score would be computed against a centroid the encoder would
    // not have chosen. (`ivf::kmeans` closes the same way.)
    for i in 0..n {
        let row = &vectors[i * dim..(i + 1) * dim];
        let mut best = 0u32;
        let mut best_d = f32::INFINITY;
        for c in 0..k {
            let cen = &centroids[c * dim..(c + 1) * dim];
            let mut d = 0.0f32;
            for j in 0..dim {
                let e = row[j] - cen[j];
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = c as u32;
            }
        }
        assignment[i] = best;
    }
    (centroids, assignment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{l2_normalize, write_f16};

    #[test]
    fn adc_ranks_similar_vectors_first() {
        let dir = std::env::temp_dir().join(format!(
            "high-performance-search-engine-pq-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut vecs = Vec::new();
        for i in 0..32 {
            let mut v = vec![0.0f32; 8];
            v[0] = 1.0;
            v[1] = (i as f32) * 0.01;
            l2_normalize(&mut v);
            vecs.push(v);
        }
        write_f16(&dir, 8, &vecs).unwrap();
        let store = EmbeddingStore::open(&dir).unwrap();
        build_and_write(&dir, &store, 2).unwrap();
        let pq = PqIndex::open(&dir).unwrap();
        let tables = pq.prepare(&vecs[0]);
        let s0 = pq.adc(&tables, 0);
        let s31 = pq.adc(&tables, 31);
        assert!(s0 >= s31 - 0.05, "self {s0} vs far {s31}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reused_codebooks_reproduce_the_trained_codes() {
        // Same contract as the IVF test: encoding against stored codebooks
        // must match what training produced, or an incremental rebuild would
        // change every document's approximate score.
        let dir = std::env::temp_dir().join(format!("hpse-pq-reuse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dim = 8;
        let m = 2;
        let n = 40;
        let mut vecs = Vec::new();
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            v[(i * 3) % dim] += 0.25;
            crate::embeddings::l2_normalize(&mut v);
            vecs.push(v);
        }
        write_f16(&dir, dim as u32, &vecs).unwrap();
        let store = EmbeddingStore::open(&dir).unwrap();
        build_and_write(&dir, &store, m).unwrap();

        let pq = PqIndex::open(&dir).unwrap();
        let (read_m, sub, ks_used, codebooks) = read_codebooks(&dir).unwrap();
        assert_eq!((read_m, sub), (m, dim / m));
        assert_eq!(ks_used, n, "only n < 256 centroids can be trained here");
        let mut code = vec![0u8; m];
        for doc_id in 0..n {
            let mut v = vec![0.0f32; dim];
            store.copy_f32(doc_id as u32, &mut v);
            encode_one(&codebooks, m, sub, ks_used, &v, &mut code);
            assert_eq!(
                code,
                pq.codes_of(doc_id as u32),
                "doc {doc_id} re-encoded differently"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn candidate_estimate_drives_the_pq_decision() {
        // Full scan: every document is a candidate.
        assert_eq!(estimated_candidates(5_000, 0, 0), 5_000);
        // IVF: a query opens nprobe of num_clusters lists.
        assert_eq!(estimated_candidates(32_000, 256, 8), 1_000);
        assert_eq!(estimated_candidates(800, 57, 8), 112);
        // nprobe cannot exceed the cluster count.
        assert_eq!(estimated_candidates(1_000, 10, 99), 1_000);

        // A small repo scores far too few documents for ADC to pay for its
        // table build; a large one clears the bar.
        assert!(!worth_using(estimated_candidates(800, 57, 8)));
        assert!(!worth_using(estimated_candidates(13_000, 228, 8)));
        assert!(worth_using(estimated_candidates(31_000, 256, 8)));
        assert!(worth_using(estimated_candidates(1_000_000, 256, 8)));
    }
}