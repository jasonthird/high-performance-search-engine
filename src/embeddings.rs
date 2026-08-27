//! Memory-mapped document embedding sidecar (`embeddings.bin`).
//!
//! This is **not** a second index. Row `i` is the vector for inverted-index
//! `doc_id == i` (the same ids in `postings.bin`). BM25/WAND never reads
//! this file; hybrid search only fetches rows for the candidate doc_ids.
//! Layout is a 32-byte header followed by row-major vectors, one per `doc_id`.
//!
//! ```text
//! magic[8] = b"HPSEMB01"
//! version u32 = 1
//! num_docs u32
//! dim u32
//! dtype u32   0 = f32, 1 = f16
//! reserved[12]
//! then num_docs * dim * width bytes, little-endian, L2-normalized
//! ```
//!
//! FP16 is the default: 768-d vectors cost 1.5 KB/doc, and cosine on
//! L2-normalized embeddings is a plain dot product.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Context;
use memmap2::Mmap;

pub const EMBEDDINGS_FILE: &str = "embeddings.bin";
pub const MAGIC: &[u8; 8] = b"HPSEMB01";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 32;
pub const DTYPE_F32: u32 = 0;
pub const DTYPE_F16: u32 = 1;
/// CodeRankEmbed output size.
pub const CODERANK_DIM: usize = 768;

pub struct EmbeddingStore {
    mmap: Mmap,
    num_docs: u32,
    dim: u32,
    dtype: u32,
}

impl EmbeddingStore {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join(EMBEDDINGS_FILE);
        let file = File::open(&path)
            .with_context(|| format!("failed to open {} (run `embed` first)", path.display()))?;
        let size = file.metadata()?.len();
        // SAFETY: the sidecar is immutable after `embed` writes it, same
        // mmap assumption as postings.bin / docs.bin.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;
        anyhow::ensure!(
            mmap.len() >= HEADER_LEN && &mmap[..8] == MAGIC,
            "{} is not an embeddings sidecar",
            path.display()
        );
        let version = u32_le(&mmap[8..12]);
        anyhow::ensure!(version == VERSION, "unsupported embeddings version {version}");
        let num_docs = u32_le(&mmap[12..16]);
        let dim = u32_le(&mmap[16..20]);
        let dtype = u32_le(&mmap[20..24]);
        anyhow::ensure!(dtype == DTYPE_F16 || dtype == DTYPE_F32, "unknown dtype {dtype}");
        let width = if dtype == DTYPE_F16 { 2usize } else { 4 };
        let expected = HEADER_LEN as u64 + num_docs as u64 * dim as u64 * width as u64;
        anyhow::ensure!(
            size >= expected,
            "embeddings.bin truncated: have {size} want {expected}"
        );
        Ok(Self {
            mmap,
            num_docs,
            dim,
            dtype,
        })
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn dtype_f16(&self) -> bool {
        self.dtype == DTYPE_F16
    }

    pub fn size_bytes(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// Cosine similarity = dot product of L2-normalized vectors.
    pub fn cosine(&self, doc_id: u32, query: &[f32]) -> f32 {
        debug_assert_eq!(query.len(), self.dim as usize);
        if doc_id >= self.num_docs {
            return 0.0;
        }
        let dim = self.dim as usize;
        if self.dtype == DTYPE_F16 {
            let start = HEADER_LEN + doc_id as usize * dim * 2;
            let bytes = &self.mmap[start..start + dim * 2];
            // Four accumulators: FP addition is not associative, so a single
            // accumulator chains every add and blocks vectorization. Split
            // lanes so LLVM can keep 4 sums in flight and SIMD the decode.
            let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let mut lanes = bytes.chunks_exact(8);
            let mut qs = query.chunks_exact(4);
            for (lane, q) in (&mut lanes).zip(&mut qs) {
                a0 += f16_to_f32_fast(u16::from_le_bytes([lane[0], lane[1]])) * q[0];
                a1 += f16_to_f32_fast(u16::from_le_bytes([lane[2], lane[3]])) * q[1];
                a2 += f16_to_f32_fast(u16::from_le_bytes([lane[4], lane[5]])) * q[2];
                a3 += f16_to_f32_fast(u16::from_le_bytes([lane[6], lane[7]])) * q[3];
            }
            let mut acc = (a0 + a2) + (a1 + a3);
            for (i, pair) in lanes.remainder().chunks_exact(2).enumerate() {
                let h = u16::from_le_bytes([pair[0], pair[1]]);
                acc += f16_to_f32_fast(h) * qs.remainder()[i];
            }
            acc
        } else {
            let start = HEADER_LEN + doc_id as usize * dim * 4;
            let bytes = &self.mmap[start..start + dim * 4];
            let mut acc = 0.0f32;
            for i in 0..dim {
                let bits = u32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
                acc += f32::from_bits(bits) * query[i];
            }
            acc
        }
    }

    /// Copy one document vector as f32.
    pub fn copy_f32(&self, doc_id: u32, out: &mut [f32]) {
        let dim = self.dim as usize;
        if doc_id >= self.num_docs || out.len() < dim {
            return;
        }
        if self.dtype == DTYPE_F16 {
            let start = HEADER_LEN + doc_id as usize * dim * 2;
            let bytes = &self.mmap[start..start + dim * 2];
            for i in 0..dim {
                let h = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                out[i] = f16_to_f32(h);
            }
        } else {
            let start = HEADER_LEN + doc_id as usize * dim * 4;
            let bytes = &self.mmap[start..start + dim * 4];
            for i in 0..dim {
                let bits = u32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
                out[i] = f32::from_bits(bits);
            }
        }
    }
}

fn u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().expect("u32"))
}

/// IEEE 754 binary16 -> f32. Enough for L2-normalized embedding coordinates.
/// Branch-free half-to-single conversion for hot loops.
///
/// Sign and magnitude are shifted into f32 field positions, then one
/// multiply by 2^112 rescales the exponent bias (15 -> 127). Exact for
/// every finite half including subnormals; Inf/NaN would come out as large
/// finite values, but L2-normalized embeddings contain neither. The branchy
/// [`f16_to_f32`] stays the general-purpose converter.
#[inline(always)]
pub fn f16_to_f32_fast(h: u16) -> f32 {
    let scale = f32::from_bits(0x7780_0000); // 2^112
    let bits = ((h as u32 & 0x8000) << 16) | ((h as u32 & 0x7fff) << 13);
    f32::from_bits(bits) * scale
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x3ff;
    let bits = if exp == 0 {
        if frac == 0 {
            (sign as u32) << 31
        } else {
            // subnormal
            let mut f = frac as u32;
            let mut e = 127 - 15 + 1;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            (sign as u32) << 31 | (e as u32) << 23 | f << 13
        }
    } else if exp == 31 {
        (sign as u32) << 31 | 0xff << 23 | (frac as u32) << 13
    } else {
        (sign as u32) << 31 | ((exp as u32 + (127 - 15)) << 23) | (frac as u32) << 13
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;
    if exp == 255 {
        return sign | 0x7c00 | (if frac != 0 { 0x200 } else { 0 });
    }
    let exp = exp - 127 + 15;
    if exp >= 31 {
        sign | 0x7c00
    } else if exp <= 0 {
        if exp < -10 {
            sign
        } else {
            let frac = (frac | 0x800000) >> (1 - exp);
            sign | ((frac + 0x1000) >> 13) as u16
        }
    } else {
        sign | (exp as u16) << 10 | ((frac + 0x1000) >> 13) as u16
    }
}

/// L2-normalize in place. Zero vector is left as zeros.
pub fn l2_normalize(v: &mut [f32]) {
    let mut ss = 0.0f32;
    for &x in v.iter() {
        ss += x * x;
    }
    if ss <= 0.0 {
        return;
    }
    let inv = ss.sqrt().recip();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Write `vectors` (already L2-normalized, one per doc_id) as FP16.
pub fn write_f16(dir: &Path, dim: u32, vectors: &[Vec<f32>]) -> anyhow::Result<u64> {
    anyhow::ensure!(
        vectors.iter().all(|v| v.len() == dim as usize),
        "embedding dim mismatch"
    );
    let path = dir.join(EMBEDDINGS_FILE);
    let mut w = BufWriter::new(
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    let num_docs = vectors.len() as u32;
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&num_docs.to_le_bytes())?;
    w.write_all(&dim.to_le_bytes())?;
    w.write_all(&DTYPE_F16.to_le_bytes())?;
    w.write_all(&[0u8; 8])?; // reserved, header is 8+4*4+8 = 32
    for v in vectors {
        for &x in v {
            w.write_all(&f32_to_f16(x).to_le_bytes())?;
        }
    }
    w.flush()?;
    drop(w);
    Ok(std::fs::metadata(&path)?.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_decode_matches_exact_for_all_finite_halfs() {
        for h in 0..=u16::MAX {
            let exp = (h >> 10) & 0x1f;
            if exp == 31 {
                continue; // Inf/NaN: fast path deliberately diverges
            }
            let exact = f16_to_f32(h);
            let fast = f16_to_f32_fast(h);
            assert!(
                exact == fast || (exact.is_nan() && fast.is_nan()),
                "h={h:#06x}: exact {exact} vs fast {fast}"
            );
        }
    }

    #[test]
    fn f16_roundtrip_unit_range() {
        for x in [-1.0, -0.5, -0.036, 0.0, 0.036, 0.5, 1.0] {
            let back = f16_to_f32(f32_to_f16(x));
            assert!((back - x).abs() < 0.001, "{x} -> {back}");
        }
    }

    #[test]
    fn sidecar_roundtrip_cosine() {
        let dir = std::env::temp_dir().join(format!(
            "high-performance-search-engine-emb-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut a = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut b = vec![0.6, 0.8, 0.0, 0.0];
        l2_normalize(&mut a);
        l2_normalize(&mut b);
        write_f16(&dir, 4, &[a.clone(), b.clone()]).unwrap();
        let store = EmbeddingStore::open(&dir).unwrap();
        assert_eq!(store.num_docs(), 2);
        assert!(store.dtype_f16());
        let q = a.clone();
        let s0 = store.cosine(0, &q);
        let s1 = store.cosine(1, &q);
        assert!((s0 - 1.0).abs() < 0.002, "self cosine {s0}");
        assert!(s1 < s0, "orthogonal-ish {s1} vs {s0}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
