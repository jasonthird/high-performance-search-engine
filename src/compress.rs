//! Posting-block compression: delta encoding + binary bit-packing.
//!
//! Within a block, doc_ids are strictly increasing, so instead of storing
//! absolute 32-bit ids we store the *gaps* between consecutive ids
//! (`delta - 1`, since gaps are at least 1), packed at the smallest bit width
//! that fits the largest gap in the block (Frame-of-Reference / binary
//! packing). Term frequencies are almost always tiny, so `tf - 1` is packed
//! the same way. Dense posting lists compress extremely well: gaps of 1
//! need 0 bits.
//!
//! Block layout (little-endian):
//!   u16  count            number of postings (1..=block size)
//!   u8   doc_bits         bit width of each packed gap
//!   u8   tf_bits          bit width of each packed tf
//!   u32  first_doc_id     stored raw
//!   [count-1 gaps][count tfs]   LSB-first bit stream, padded to a byte
//!
//! This trades a little decode work for a large size reduction; blocks are
//! decoded lazily, only when Block-Max WAND actually enters them.

use crate::postings::Posting;

/// Bits needed to represent `value` (0 needs 0 bits).
fn bits_needed(value: u32) -> u8 {
    (32 - value.leading_zeros()) as u8
}

/// LSB-first bit writer with a 64-bit accumulator: whole bytes are flushed
/// as they fill instead of setting bits one at a time. A value is at most
/// 32 bits and the accumulator holds < 8 residual bits before each write,
/// so nothing ever overflows.
struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    acc_bits: u32,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            acc: 0,
            acc_bits: 0,
        }
    }

    fn write(&mut self, value: u32, bits: u8) {
        debug_assert!(bits == 32 || value < (1u64 << bits) as u32);
        self.acc |= (value as u64) << self.acc_bits;
        self.acc_bits += bits as u32;
        while self.acc_bits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.acc_bits -= 8;
        }
    }

    /// Flush any residual bits (zero-padded to a byte).
    fn finish(self) {
        if self.acc_bits > 0 {
            self.out.push(self.acc as u8);
        }
    }
}

/// Load up to 8 bytes at `byte_idx` as a little-endian u64, zero-padded past
/// the end of the slice. Lets the reader extract any ≤32-bit value (at any
/// bit offset within a byte: shift ≤ 7, so shift + bits ≤ 39 < 64) with one
/// word load instead of a per-bit loop.
fn read_word(bytes: &[u8], byte_idx: usize) -> u64 {
    if let Some(chunk) = bytes.get(byte_idx..byte_idx + 8) {
        u64::from_le_bytes(chunk.try_into().expect("slice of length 8"))
    } else {
        let mut buf = [0u8; 8];
        if byte_idx < bytes.len() {
            let tail = &bytes[byte_idx..];
            buf[..tail.len()].copy_from_slice(tail);
        }
        u64::from_le_bytes(buf)
    }
}

fn width_mask(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

/// LSB-first bit reader (word-at-a-time).
struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    fn at(bytes: &'a [u8], bit_pos: usize) -> Self {
        Self { bytes, bit_pos }
    }

    fn read(&mut self, bits: u8) -> u32 {
        if bits == 0 {
            return 0;
        }
        let word = read_word(self.bytes, self.bit_pos / 8);
        let shift = self.bit_pos % 8;
        self.bit_pos += bits as usize;
        ((word >> shift) as u32) & width_mask(bits)
    }
}

const HEADER_BYTES: usize = 8;

struct BlockHeader {
    count: usize,
    doc_bits: u8,
    tf_bits: u8,
    first_doc_id: u32,
}

fn read_header(bytes: &[u8]) -> BlockHeader {
    BlockHeader {
        count: u16::from_le_bytes([bytes[0], bytes[1]]) as usize,
        doc_bits: bytes[2],
        tf_bits: bytes[3],
        first_doc_id: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    }
}

/// Append one encoded block of postings (sorted by doc_id) to `out`.
pub fn encode_block(postings: &[Posting], out: &mut Vec<u8>) {
    assert!(!postings.is_empty() && postings.len() <= u16::MAX as usize);

    let mut max_gap = 0u32;
    for pair in postings.windows(2) {
        debug_assert!(
            pair[1].doc_id > pair[0].doc_id,
            "doc_ids must be strictly increasing"
        );
        max_gap = max_gap.max(pair[1].doc_id - pair[0].doc_id - 1);
    }
    let max_tf = postings.iter().map(|p| p.tf - 1).max().unwrap();
    let doc_bits = bits_needed(max_gap);
    let tf_bits = bits_needed(max_tf);

    out.extend_from_slice(&(postings.len() as u16).to_le_bytes());
    out.push(doc_bits);
    out.push(tf_bits);
    out.extend_from_slice(&postings[0].doc_id.to_le_bytes());

    let mut writer = BitWriter::new(out);
    for pair in postings.windows(2) {
        writer.write(pair[1].doc_id - pair[0].doc_id - 1, doc_bits);
    }
    for p in postings {
        writer.write(p.tf - 1, tf_bits);
    }
    writer.finish();
}

/// Decode only the doc_ids of a block into `out`. This is the hot path:
/// Block-Max WAND needs doc_ids to align cursors, but term frequencies only
/// for the small fraction of postings it actually scores — those are fetched
/// individually with [`block_tf`].
pub fn decode_block_docs(bytes: &[u8], out: &mut Vec<u32>) {
    let header = read_header(bytes);
    out.clear();
    out.reserve(header.count);
    let mut doc_id = header.first_doc_id;
    out.push(doc_id);

    // Dense block: every gap is exactly 1, the payload is empty. This is
    // the common case for very frequent terms — precisely the ones whose
    // long posting lists dominate decode time.
    if header.doc_bits == 0 {
        for _ in 1..header.count {
            doc_id += 1;
            out.push(doc_id);
        }
        return;
    }

    let payload = &bytes[HEADER_BYTES..];
    let db = header.doc_bits as usize;
    let mask = width_mask(header.doc_bits);
    let n = header.count - 1;
    let mut bit_pos = 0usize;
    let mut i = 0usize;
    // Wide zone: one unaligned 64-bit load yields every gap whose bits fit
    // in the window regardless of starting alignment (shift <= 7, so
    // (64 - 7) / db values per load — 2 for rare terms, dozens for dense
    // ones). std::simd is still nightly; this is the stable equivalent for
    // an LSB bit-stream, and it is load-bound no more.
    let per_load = ((u64::BITS as usize - 7) / db).max(1);
    while i + per_load <= n {
        let byte = bit_pos >> 3;
        let Some(window) = payload.get(byte..byte + 8) else {
            break;
        };
        let word = u64::from_le_bytes(window.try_into().expect("slice of length 8"));
        let mut shift = bit_pos & 7;
        for _ in 0..per_load {
            doc_id += (((word >> shift) as u32) & mask) + 1;
            out.push(doc_id);
            shift += db;
        }
        bit_pos += per_load * db;
        i += per_load;
    }
    // Fast zone: remaining gaps with a full window, one load each.
    while i < n {
        let byte = bit_pos >> 3;
        let Some(window) = payload.get(byte..byte + 8) else {
            break;
        };
        let word = u64::from_le_bytes(window.try_into().expect("slice of length 8"));
        doc_id += (((word >> (bit_pos & 7)) as u32) & mask) + 1;
        out.push(doc_id);
        bit_pos += db;
        i += 1;
    }
    // Tail: the last few gaps near the end of the payload use padded loads.
    while i < n {
        let word = read_word(payload, bit_pos >> 3);
        doc_id += (((word >> (bit_pos & 7)) as u32) & mask) + 1;
        out.push(doc_id);
        bit_pos += db;
        i += 1;
    }
}

/// Random-access read of the `idx`-th term frequency in a block. Possible in
/// O(1) because every tf in the block is packed at the same fixed width.
pub fn block_tf(bytes: &[u8], idx: usize) -> u32 {
    let header = read_header(bytes);
    debug_assert!(idx < header.count);
    if header.tf_bits == 0 {
        return 1; // all tfs in the block are 1
    }
    let bit_pos = (header.count - 1) * header.doc_bits as usize + idx * header.tf_bits as usize;
    BitReader::at(&bytes[HEADER_BYTES..], bit_pos).read(header.tf_bits) + 1
}

/// Decode one full block (doc_ids and tfs) starting at the beginning of
/// `bytes` into `out`. Returns the number of bytes consumed.
pub fn decode_block(bytes: &[u8], out: &mut Vec<Posting>) -> usize {
    let header = read_header(bytes);
    let payload_bits =
        (header.count - 1) * header.doc_bits as usize + header.count * header.tf_bits as usize;
    let payload_bytes = payload_bits.div_ceil(8);

    let mut reader = BitReader::new(&bytes[HEADER_BYTES..HEADER_BYTES + payload_bytes]);
    out.clear();
    out.reserve(header.count);
    let mut doc_id = header.first_doc_id;
    out.push(Posting { doc_id, tf: 0 });
    for _ in 1..header.count {
        doc_id += reader.read(header.doc_bits) + 1;
        out.push(Posting { doc_id, tf: 0 });
    }
    for posting in out.iter_mut() {
        posting.tf = reader.read(header.tf_bits) + 1;
    }

    HEADER_BYTES + payload_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_needed_basics() {
        assert_eq!(bits_needed(0), 0);
        assert_eq!(bits_needed(1), 1);
        assert_eq!(bits_needed(2), 2);
        assert_eq!(bits_needed(255), 8);
        assert_eq!(bits_needed(256), 9);
        assert_eq!(bits_needed(u32::MAX), 32);
    }

    #[test]
    fn bit_writer_reader_round_trip() {
        let values: Vec<(u32, u8)> = vec![
            (0, 0),
            (1, 1),
            (5, 3),
            (255, 8),
            (12345, 14),
            (0, 0),
            (u32::MAX, 32),
            (7, 5),
        ];
        let mut buf = Vec::new();
        let mut writer = BitWriter::new(&mut buf);
        for &(v, bits) in &values {
            writer.write(v, bits);
        }
        writer.finish();
        let mut reader = BitReader::new(&buf);
        for &(v, bits) in &values {
            assert_eq!(reader.read(bits), v, "value {v} at {bits} bits");
        }
    }

    fn round_trip(postings: &[Posting]) {
        let mut bytes = Vec::new();
        encode_block(postings, &mut bytes);
        let mut decoded = Vec::new();
        let consumed = decode_block(&bytes, &mut decoded);
        assert_eq!(consumed, bytes.len());
        assert_eq!(decoded, postings);

        // The docs-only and random-access tf paths must agree with the full
        // decode (they are what the search cursors actually use).
        let mut docs = Vec::new();
        decode_block_docs(&bytes, &mut docs);
        let expected_docs: Vec<u32> = postings.iter().map(|p| p.doc_id).collect();
        assert_eq!(docs, expected_docs);
        for (i, p) in postings.iter().enumerate() {
            assert_eq!(block_tf(&bytes, i), p.tf, "tf at index {i}");
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        round_trip(&[Posting { doc_id: 0, tf: 1 }]);
        round_trip(&[Posting {
            doc_id: 4_000_000_000,
            tf: 9999,
        }]);
        round_trip(&[
            Posting { doc_id: 3, tf: 1 },
            Posting { doc_id: 4, tf: 2 },
            Posting { doc_id: 5, tf: 1 },
        ]);
        // Dense run: gaps of 1, tf of 1 -> 0-bit payload for both streams.
        let dense: Vec<Posting> = (100..228).map(|d| Posting { doc_id: d, tf: 1 }).collect();
        let mut bytes = Vec::new();
        encode_block(&dense, &mut bytes);
        assert_eq!(bytes.len(), 8, "dense block should be header-only");
        round_trip(&dense);
    }

    #[test]
    fn encode_decode_round_trip_random() {
        let mut state = 0xabcdef99u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..200 {
            let count = 1 + (next() as usize % 128);
            let mut doc_id = next() % 1000;
            let mut postings = Vec::with_capacity(count);
            for _ in 0..count {
                postings.push(Posting {
                    doc_id,
                    tf: 1 + next() % 50,
                });
                doc_id += 1 + next() % 5000;
            }
            round_trip(&postings);
        }
    }

    #[test]
    fn consecutive_blocks_decode_independently() {
        let a: Vec<Posting> = (0..50)
            .map(|d| Posting {
                doc_id: d * 7,
                tf: 1 + d % 3,
            })
            .collect();
        let b: Vec<Posting> = (1000..1100).map(|d| Posting { doc_id: d, tf: 2 }).collect();
        let mut bytes = Vec::new();
        encode_block(&a, &mut bytes);
        let offset = bytes.len();
        encode_block(&b, &mut bytes);

        let mut decoded = Vec::new();
        decode_block(&bytes, &mut decoded);
        assert_eq!(decoded, a);
        decode_block(&bytes[offset..], &mut decoded);
        assert_eq!(decoded, b);
    }
}
