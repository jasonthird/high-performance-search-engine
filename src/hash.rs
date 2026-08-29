//! FNV-1a 64: the one shared implementation of the hash primitive behind
//! on-disk keys (external-id hashes, content hashes, embedding-cache keys,
//! quantizer ids, tree fingerprints, index-dir names).
//!
//! These hashes are persisted, so every call site must agree byte-for-byte;
//! keeping the loop in one place makes drift impossible.

/// Streaming FNV-1a 64 hasher.
pub struct Fnv1a(u64);

impl Fnv1a {
    pub fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}

impl Default for Fnv1a {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot FNV-1a 64 of a byte slice.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = Fnv1a::new();
    h.write(bytes);
    h.finish()
}
