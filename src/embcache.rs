//! Content-addressed embedding + quantization cache (`embcache.bin`).
//!
//! Hybrid search needs `embeddings.bin` row `i` to be the vector for
//! inverted-index `doc_id == i`, and doc_ids are reassigned on every rebuild.
//! A file-watcher rebuild would therefore redo the whole tree for a one-line
//! edit. This cache breaks that: everything expensive is keyed by a hash of
//! the chunk text, so a rebuild only pays for chunks whose *content* changed.
//!
//! Three things are cached per chunk, all derived purely from its text:
//!
//! - the CodeRankEmbed **vector** (~65 ms of inference each);
//! - its **IVF cluster**, and its **PQ code**.
//!
//! The last two depend on trained centroids and codebooks, so they are
//! tagged with a `quantizer` id — a hash of those parameters. If the
//! quantizer is retrained the ids stop matching and the codes are recomputed
//! (cheaply, from the cached vectors); the vectors themselves stay valid,
//! because they do not depend on the quantizer at all.
//!
//! ```text
//! magic[8] = b"HPSEMC02"
//! version u32 = 2
//! dim u32
//! count u32
//! m u32          PQ subvectors per code (0 = no codes cached)
//! quantizer u64  id of the centroids/codebooks the codes were built with
//! then count * (u64 key + dim*u16 vector + u32 cluster + m*u8 code)
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use anyhow::Context;

use crate::embeddings::{f16_to_f32, f32_to_f16};

pub const CACHE_FILE: &str = "embcache.bin";
const MAGIC: &[u8; 8] = b"HPSEMC02";
const VERSION: u32 = 2;
const HEADER_LEN: usize = 32;
/// Sentinel for "this entry has no quantization cached yet".
const NO_CLUSTER: u32 = u32::MAX;

/// Cache key for one chunk: FNV-1a over the exact text handed to the encoder,
/// mixed with its length so that same-prefix truncations cannot collide.
pub fn key_for(text: &str) -> u64 {
    crate::hash::fnv1a(text.as_bytes()) ^ (text.len() as u64).wrapping_mul(0x9e3779b97f4a7c15)
}

/// Identify a trained quantizer by its parameters, so cached codes are only
/// reused against the exact centroids and codebooks that produced them.
pub fn quantizer_id(centroids: &[f32], codebooks: &[f32]) -> u64 {
    let mut h = crate::hash::Fnv1a::new();
    let mut feed = |v: &[f32]| {
        for &x in v {
            // Hash the stored FP16 form: that is what a reload will see.
            h.write(&f32_to_f16(x).to_le_bytes());
        }
    };
    feed(centroids);
    feed(codebooks);
    // Keep the length in the id so a truncated read can never collide.
    h.finish() ^ ((centroids.len() as u64) << 32 | codebooks.len() as u64)
}

struct Entry {
    lanes: Vec<u16>,
    cluster: u32,
    codes: Vec<u8>,
}

/// Hash -> (FP16 vector, IVF cluster, PQ code), loaded whole.
pub struct EmbedCache {
    dim: usize,
    m: usize,
    /// Quantizer the stored clusters/codes belong to.
    quantizer: u64,
    entries: HashMap<u64, Entry>,
}

impl EmbedCache {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            m: 0,
            quantizer: 0,
            entries: HashMap::new(),
        }
    }

    /// Load the cache from `dir`, or start empty when it is missing,
    /// unreadable, or written for a different dimension.
    pub fn load(dir: &Path, dim: usize) -> Self {
        match Self::try_load(dir, dim) {
            Ok(Some(cache)) => cache,
            _ => Self::new(dim),
        }
    }

    fn try_load(dir: &Path, dim: usize) -> anyhow::Result<Option<Self>> {
        let path = dir.join(CACHE_FILE);
        let Ok(mut file) = File::open(&path) else {
            return Ok(None);
        };
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        if buf.len() < HEADER_LEN || &buf[..8] != MAGIC {
            return Ok(None);
        }
        let read_u32 = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
        if read_u32(8) != VERSION {
            return Ok(None);
        }
        if read_u32(12) as usize != dim {
            return Ok(None);
        }
        let count = read_u32(16) as usize;
        let m = read_u32(20) as usize;
        let quantizer = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let stride = 8 + dim * 2 + 4 + m;
        if buf.len() < HEADER_LEN + count * stride {
            return Ok(None); // truncated (interrupted write): rebuild it
        }
        let mut entries = HashMap::with_capacity(count);
        for i in 0..count {
            let at = HEADER_LEN + i * stride;
            let key = u64::from_le_bytes(buf[at..at + 8].try_into().unwrap());
            let mut lanes = Vec::with_capacity(dim);
            for j in 0..dim {
                let off = at + 8 + j * 2;
                lanes.push(u16::from_le_bytes(buf[off..off + 2].try_into().unwrap()));
            }
            let coff = at + 8 + dim * 2;
            let cluster = u32::from_le_bytes(buf[coff..coff + 4].try_into().unwrap());
            let codes = buf[coff + 4..coff + 4 + m].to_vec();
            entries.insert(
                key,
                Entry {
                    lanes,
                    cluster,
                    codes,
                },
            );
        }
        Ok(Some(Self {
            dim,
            m,
            quantizer,
            entries,
        }))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The quantizer the cached clusters and codes were computed under.
    pub fn quantizer(&self) -> u64 {
        self.quantizer
    }

    /// Adopt a quantizer. Switching to a different one invalidates every
    /// cached cluster and code — but not the vectors, which are independent
    /// of the quantizer.
    pub fn set_quantizer(&mut self, id: u64, m: usize) {
        if self.quantizer != id || self.m != m {
            for entry in self.entries.values_mut() {
                entry.cluster = NO_CLUSTER;
                entry.codes.clear();
            }
            self.quantizer = id;
            self.m = m;
        }
    }

    /// Fetch a cached vector as f32.
    pub fn get(&self, key: u64) -> Option<Vec<f32>> {
        let entry = self.entries.get(&key)?;
        Some(entry.lanes.iter().copied().map(f16_to_f32).collect())
    }

    pub fn contains(&self, key: u64) -> bool {
        self.entries.contains_key(&key)
    }

    /// Cached (cluster, PQ code) for the current quantizer, if present.
    pub fn get_quant(&self, key: u64) -> Option<(u32, &[u8])> {
        let entry = self.entries.get(&key)?;
        if entry.cluster == NO_CLUSTER || entry.codes.len() != self.m {
            return None;
        }
        Some((entry.cluster, &entry.codes))
    }

    /// Record the quantization of a chunk under the current quantizer.
    pub fn set_quant(&mut self, key: u64, cluster: u32, codes: &[u8]) {
        if codes.len() != self.m {
            return;
        }
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.cluster = cluster;
            entry.codes = codes.to_vec();
        }
    }

    /// Insert a freshly encoded vector. Vectors are stored exactly as they
    /// will be written to `embeddings.bin`, so a cache hit is bit-identical
    /// to having re-encoded.
    pub fn insert(&mut self, key: u64, vector: &[f32]) {
        if vector.len() != self.dim {
            return;
        }
        self.entries.insert(
            key,
            Entry {
                lanes: vector.iter().copied().map(f32_to_f16).collect(),
                cluster: NO_CLUSTER,
                codes: Vec::new(),
            },
        );
    }

    /// Drop everything not in `live`, so the cache tracks the working tree
    /// instead of growing without bound across rebuilds.
    pub fn retain_keys(&mut self, live: &HashSet<u64>) {
        self.entries.retain(|k, _| live.contains(k));
    }

    /// Write the cache next to the index, atomically via a temp file.
    pub fn save(&self, dir: &Path) -> anyhow::Result<u64> {
        let path = dir.join(CACHE_FILE);
        let tmp = dir.join(format!("{CACHE_FILE}.tmp"));
        {
            let mut w = BufWriter::new(
                File::create(&tmp)
                    .with_context(|| format!("failed to create {}", tmp.display()))?,
            );
            w.write_all(MAGIC)?;
            w.write_all(&VERSION.to_le_bytes())?;
            w.write_all(&(self.dim as u32).to_le_bytes())?;
            w.write_all(&(self.entries.len() as u32).to_le_bytes())?;
            w.write_all(&(self.m as u32).to_le_bytes())?;
            w.write_all(&self.quantizer.to_le_bytes())?;
            let blank = vec![0u8; self.m];
            for (key, entry) in &self.entries {
                w.write_all(&key.to_le_bytes())?;
                for &lane in &entry.lanes {
                    w.write_all(&lane.to_le_bytes())?;
                }
                w.write_all(&entry.cluster.to_le_bytes())?;
                // Entries without cached codes still occupy a fixed stride,
                // so the file stays a flat array.
                if entry.codes.len() == self.m {
                    w.write_all(&entry.codes)?;
                } else {
                    w.write_all(&blank)?;
                }
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("failed to install {}", path.display()))?;
        Ok(std::fs::metadata(&path)?.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_vectors_and_codes_through_disk() {
        let dir = std::env::temp_dir().join(format!("hpse-embcache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut cache = EmbedCache::new(4);
        cache.set_quantizer(0xABCD, 2);
        let k = key_for("fn alpha() {}");
        cache.insert(k, &[0.5, -0.25, 0.0, 1.0]);
        cache.set_quant(k, 7, &[3, 9]);
        cache.save(&dir).unwrap();

        let back = EmbedCache::load(&dir, 4);
        assert_eq!(back.len(), 1);
        assert_eq!(back.quantizer(), 0xABCD);
        let v = back.get(k).unwrap();
        assert!((v[0] - 0.5).abs() < 0.001);
        let (cluster, codes) = back.get_quant(k).unwrap();
        assert_eq!((cluster, codes), (7, &[3u8, 9][..]));

        // A different dim must not be reinterpreted.
        assert!(EmbedCache::load(&dir, 8).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn retraining_invalidates_codes_but_keeps_vectors() {
        let mut cache = EmbedCache::new(4);
        cache.set_quantizer(1, 2);
        let k = key_for("x");
        cache.insert(k, &[1.0, 0.0, 0.0, 0.0]);
        cache.set_quant(k, 5, &[1, 2]);
        assert!(cache.get_quant(k).is_some());

        cache.set_quantizer(2, 2);
        assert!(cache.get_quant(k).is_none(), "codes must not survive");
        assert!(cache.get(k).is_some(), "vector must survive");
    }

    #[test]
    fn quantizer_id_tracks_parameters() {
        let a = quantizer_id(&[1.0, 0.0], &[0.5]);
        assert_eq!(a, quantizer_id(&[1.0, 0.0], &[0.5]));
        assert_ne!(a, quantizer_id(&[1.0, 0.1], &[0.5]));
        assert_ne!(a, quantizer_id(&[1.0, 0.0], &[0.25]));
    }

    #[test]
    fn keys_separate_by_length() {
        assert_ne!(key_for("abc"), key_for("abcd"));
        assert_eq!(key_for("abc"), key_for("abc"));
    }
}
