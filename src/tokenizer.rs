//! A small hand-rolled tokenizer: lowercase, split on non-alphanumeric
//! characters (Unicode-aware), drop empty tokens, and optionally remove a
//! tiny hardcoded English stopword list.

/// Small hardcoded English stopword list (kept deliberately short for the MVP).
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

#[derive(Debug, Clone)]
pub struct Tokenizer {
    remove_stopwords: bool,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new(true)
    }
}

impl Tokenizer {
    pub fn new(remove_stopwords: bool) -> Self {
        Self { remove_stopwords }
    }

    /// Tokenize a text into lowercase alphanumeric tokens.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        self.for_each_token(text, |t| tokens.push(t.to_owned()));
        tokens
    }

    /// Stream tokens through a callback without allocating per token: the
    /// callback receives a reusable scratch buffer. On aarch64, all-ASCII
    /// 16-byte chunks are classified and lowercased with NEON (one vector op
    /// sequence instead of 16 byte-loops); non-ASCII characters and other
    /// architectures take the scalar path.
    pub fn for_each_token(&self, text: &str, mut f: impl FnMut(&str)) {
        let mut token = String::with_capacity(32);
        let bytes = text.as_bytes();
        let mut i = 0;

        #[cfg(target_arch = "aarch64")]
        while i + 16 <= bytes.len() {
            let chunk: &[u8; 16] = bytes[i..i + 16].try_into().expect("16-byte chunk");
            let Some((lowered, mask)) = neon::classify16(chunk) else {
                // Non-ASCII in the window: handle one char scalar, retry.
                let ch = text[i..].chars().next().expect("char boundary");
                if ch.is_alphanumeric() {
                    token.extend(ch.to_lowercase());
                } else {
                    self.flush(&mut token, &mut f);
                }
                i += ch.len_utf8();
                continue;
            };
            // mask has nibble j = 0xF iff byte j is alphanumeric; walk runs
            // and append whole lowercased slices.
            let mut j = 0usize;
            while j < 16 {
                if (mask >> (4 * j)) & 1 == 1 {
                    let start = j;
                    while j < 16 && (mask >> (4 * j)) & 1 == 1 {
                        j += 1;
                    }
                    token.push_str(std::str::from_utf8(&lowered[start..j]).expect("ascii slice"));
                } else {
                    self.flush(&mut token, &mut f);
                    j += 1;
                }
            }
            i += 16;
        }

        while i < bytes.len() {
            let b = bytes[i];
            if b < 0x80 {
                if b.is_ascii_alphanumeric() {
                    token.push(b.to_ascii_lowercase() as char);
                } else {
                    self.flush(&mut token, &mut f);
                }
                i += 1;
            } else {
                // SAFETY of indexing: i is on a char boundary (ASCII bytes
                // advance by 1; multi-byte chars advance by their length).
                let ch = text[i..].chars().next().expect("char boundary");
                if ch.is_alphanumeric() {
                    token.extend(ch.to_lowercase());
                } else {
                    self.flush(&mut token, &mut f);
                }
                i += ch.len_utf8();
            }
        }
        self.flush(&mut token, &mut f);
    }

    fn flush(&self, token: &mut String, f: &mut impl FnMut(&str)) {
        let drop_token =
            token.is_empty() || (self.remove_stopwords && STOPWORDS.contains(&token.as_str()));
        if !drop_token {
            f(token);
        }
        token.clear();
    }
}

/// NEON classification of one all-ASCII 16-byte chunk.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// For an all-ASCII chunk, returns the lowercased bytes and a nibble
    /// mask (nibble j = 0xF iff byte j is ASCII alphanumeric). Returns None
    /// if the chunk contains any non-ASCII byte.
    #[inline]
    pub fn classify16(chunk: &[u8; 16]) -> Option<([u8; 16], u64)> {
        // SAFETY: NEON is baseline on aarch64 (guaranteed by the cfg);
        // loads/stores use 16-byte buffers of exactly the right size.
        unsafe {
            let v = vld1q_u8(chunk.as_ptr());
            if vmaxvq_u8(v) >= 0x80 {
                return None;
            }
            let lower = vorrq_u8(v, vdupq_n_u8(0x20));
            let alpha = vandq_u8(
                vcgeq_u8(lower, vdupq_n_u8(b'a')),
                vcleq_u8(lower, vdupq_n_u8(b'z')),
            );
            let digit = vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'0')), vcleq_u8(v, vdupq_n_u8(b'9')));
            let alnum = vorrq_u8(alpha, digit);
            // Lowercase only the letters; digits and punctuation unchanged.
            let lowered_v = vbslq_u8(alpha, lower, v);
            let mut lowered = [0u8; 16];
            vst1q_u8(lowered.as_mut_ptr(), lowered_v);
            // Narrowing shift folds each byte's 0xFF/0x00 into a nibble of
            // a single u64 mask (the classic aarch64 movemask substitute).
            let nibbles = vshrn_n_u16::<4>(vreinterpretq_u16_u8(alnum));
            let mask = vget_lane_u64::<0>(vreinterpret_u64_u8(nibbles));
            Some((lowered, mask))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_and_splits_on_non_alphanumeric() {
        let tok = Tokenizer::new(false);
        assert_eq!(
            tok.tokenize("Cheap PIZZA, in Montreal!"),
            vec!["cheap", "pizza", "in", "montreal"]
        );
    }

    #[test]
    fn removes_empty_tokens() {
        let tok = Tokenizer::new(false);
        assert_eq!(
            tok.tokenize("  --  hello   world -- "),
            vec!["hello", "world"]
        );
        assert!(tok.tokenize("...!!!").is_empty());
        assert!(tok.tokenize("").is_empty());
    }

    #[test]
    fn keeps_digits() {
        let tok = Tokenizer::new(false);
        assert_eq!(tok.tokenize("route 66 rocks"), vec!["route", "66", "rocks"]);
    }

    #[test]
    fn unicode_aware() {
        let tok = Tokenizer::new(false);
        assert_eq!(
            tok.tokenize("Crème Brûlée à Montréal"),
            vec!["crème", "brûlée", "à", "montréal"]
        );
    }

    /// The NEON fast path must produce byte-identical token streams to a
    /// pure scalar reference, across ASCII, Unicode, and boundary cases.
    #[test]
    fn vectorized_path_matches_scalar_reference() {
        let scalar_tokenize = |text: &str| -> Vec<String> {
            // Reference implementation: per-char, no chunking.
            let mut out = Vec::new();
            let mut token = String::new();
            for ch in text.chars() {
                if ch.is_alphanumeric() {
                    if ch.is_ascii() {
                        token.push(ch.to_ascii_lowercase());
                    } else {
                        token.extend(ch.to_lowercase());
                    }
                } else if !token.is_empty() {
                    out.push(std::mem::take(&mut token));
                }
            }
            if !token.is_empty() {
                out.push(token);
            }
            out
        };

        let tok = Tokenizer::new(false);
        let mut state = 0xfeedu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let alphabet: Vec<char> =
            "aZ9 .,-_\u{e9}\u{4e2d}\u{1f600}\t\nABCxyz0123!?'\"\u{130}\u{df}|/"
                .chars()
                .collect();
        for _ in 0..500 {
            let len = (next() % 64) as usize;
            let text: String = (0..len)
                .map(|_| alphabet[next() as usize % alphabet.len()])
                .collect();
            assert_eq!(
                tok.tokenize(&text),
                scalar_tokenize(&text),
                "text: {text:?}"
            );
        }
        // Long pure-ASCII and mixed strings exercise full 16-byte chunks.
        let long = "The Quick Brown Fox Jumps Over 13 Lazy Dogs! ".repeat(20);
        assert_eq!(tok.tokenize(&long), scalar_tokenize(&long));
        let mixed =
            "caf\u{e9} r\u{e9}sum\u{e9}s 12345 ABCDEFGHIJKLMNOPQRSTUVWXYZ \u{4e2d}\u{6587} test"
                .repeat(8);
        assert_eq!(tok.tokenize(&mixed), scalar_tokenize(&mixed));
    }

    #[test]
    fn removes_stopwords_when_enabled() {
        let tok = Tokenizer::new(true);
        assert_eq!(
            tok.tokenize("the best pizza in the city"),
            vec!["best", "pizza", "city"]
        );
    }
}
