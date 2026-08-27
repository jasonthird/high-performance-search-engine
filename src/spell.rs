//! Query-time typo correction using the Symmetric Delete algorithm
//! (Garbe's "SymSpell"), reimplemented here — no external crate.
//!
//! The dictionary is the index's own vocabulary weighted by document
//! frequency, so every correction points at a term that actually retrieves
//! something and the on-disk index is never touched.
//!
//! Precompute: for every vocabulary term, insert the hashes of all
//! delete-variants (up to [`MAX_ED`] deletions of its first [`PREFIX_LEN`]
//! characters) into a map pointing back at the term. Lookup: generate the
//! same delete-variants of the misspelled word, union the candidate buckets,
//! verify each candidate with true Damerau-Levenshtein (OSA) distance, and
//! keep the closest, breaking ties by document frequency.
//!
//! Map keys are u64 hashes of the delete strings; hash collisions are
//! harmless because every candidate is verified by edit distance anyway.

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

/// Maximum edit distance for long words. Words of 3-5 chars cap at 1
/// (correcting "cat" at distance 2 matches half the dictionary); words under
/// 3 chars are never corrected. Same length scaling as Elasticsearch AUTO.
const MAX_ED: usize = 2;

/// Delete-variants are generated from the first `PREFIX_LEN` chars only,
/// which bounds the precomputed map size (SymSpell's prefix optimization).
/// Prefix 7 with max distance 2 is the standard SymSpell pairing.
const PREFIX_LEN: usize = 7;

/// Terms must appear in at least this many documents to be correction
/// targets: corpus typos and OCR junk are overwhelmingly rare terms, and
/// skipping them keeps the deletes map small on large corpora.
pub const MIN_DF: u32 = 3;

pub struct SpellCorrector {
    /// Correction targets: (term, document frequency).
    terms: Vec<(String, u32)>,
    /// hash(delete-variant) -> indices into `terms`. On a full-Wikipedia
    /// vocabulary this is order 1-2 GB even after the df filter; an fst plus
    /// Levenshtein automaton is the upgrade path if that ever matters.
    deletes: HashMap<u64, Vec<u32>>,
}

impl SpellCorrector {
    /// Build from (term, document frequency) pairs; terms with df below
    /// [`MIN_DF`] are ignored.
    pub fn build(vocab: impl IntoIterator<Item = (String, u32)>) -> Self {
        let mut terms = Vec::new();
        let mut deletes: HashMap<u64, Vec<u32>> = HashMap::new();
        for (term, df) in vocab {
            if df < MIN_DF {
                continue;
            }
            let id = terms.len() as u32;
            for variant in delete_variants(&term, MAX_ED) {
                deletes.entry(hash_str(&variant)).or_default().push(id);
            }
            terms.push((term, df));
        }
        Self { terms, deletes }
    }

    /// Best correction for `word`, or None if nothing is within the
    /// length-scaled edit-distance budget. The caller should already have
    /// checked that `word` itself matches nothing in the index.
    pub fn correct(&self, word: &str) -> Option<&str> {
        let word_chars: Vec<char> = word.chars().collect();
        if word_chars.len() < 3 {
            return None;
        }
        let max_ed = if word_chars.len() <= 5 { 1 } else { MAX_ED };

        let mut seen: HashSet<u32> = HashSet::new();
        // Best = smallest distance, then largest df.
        let mut best: Option<(usize, u32, u32)> = None; // (ed, df, id)
        for variant in delete_variants(word, max_ed) {
            let Some(ids) = self.deletes.get(&hash_str(&variant)) else {
                continue;
            };
            for &id in ids {
                if !seen.insert(id) {
                    continue;
                }
                let (term, df) = &self.terms[id as usize];
                let term_chars: Vec<char> = term.chars().collect();
                if term_chars.len().abs_diff(word_chars.len()) > max_ed {
                    continue;
                }
                let ed = osa_distance(&word_chars, &term_chars);
                if ed == 0 || ed > max_ed {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some((bed, bdf, _)) => ed < bed || (ed == bed && *df > bdf),
                };
                if better {
                    best = Some((ed, *df, id));
                }
            }
        }
        best.map(|(_, _, id)| self.terms[id as usize].0.as_str())
    }
}

/// All strings reachable from the first [`PREFIX_LEN`] chars of `word` by
/// deleting up to `max_ed` chars (the prefix itself included).
fn delete_variants(word: &str, max_ed: usize) -> HashSet<String> {
    let prefix: String = word.chars().take(PREFIX_LEN).collect();
    let mut out = HashSet::from([prefix.clone()]);
    let mut frontier = vec![prefix];
    for _ in 0..max_ed {
        let mut next = Vec::new();
        for w in &frontier {
            let chars: Vec<char> = w.chars().collect();
            for i in 0..chars.len() {
                let deleted: String = chars[..i].iter().chain(&chars[i + 1..]).collect();
                if out.insert(deleted.clone()) {
                    next.push(deleted);
                }
            }
        }
        frontier = next;
    }
    out
}

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Damerau-Levenshtein distance, optimal string alignment (OSA) variant:
/// insertions, deletions, substitutions, and adjacent transpositions. Uses
/// three rolling rows since the transposition term needs row i-2.
fn osa_distance(a: &[char], b: &[char]) -> usize {
    let n = b.len();
    let mut prev2: Vec<usize> = vec![0; n + 1];
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut cur: Vec<usize> = vec![0; n + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=n {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                cur[j] = cur[j].min(prev2[j - 2] + 1);
            }
        }
        std::mem::swap(&mut prev2, &mut prev);
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corrector() -> SpellCorrector {
        SpellCorrector::build(
            [
                ("pizza", 10),
                ("pirates", 8),
                ("montreal", 5),
                ("cart", 9),
                ("care", 4),
                ("rare", 1), // below MIN_DF: never a target
            ]
            .map(|(t, df)| (t.to_string(), df)),
        )
    }

    fn ed(a: &str, b: &str) -> usize {
        osa_distance(&a.chars().collect::<Vec<_>>(), &b.chars().collect::<Vec<_>>())
    }

    #[test]
    fn osa_distance_counts_edits_and_transpositions() {
        assert_eq!(ed("pizza", "pizza"), 0);
        assert_eq!(ed("pizzza", "pizza"), 1); // deletion
        assert_eq!(ed("piza", "pizza"), 1); // insertion
        assert_eq!(ed("pizca", "pizza"), 1); // substitution
        assert_eq!(ed("piarte", "pirate"), 1); // adjacent transposition
        assert_eq!(ed("", "abc"), 3);
        assert_eq!(ed("kitten", "sitting"), 3);
    }

    #[test]
    fn corrects_single_and_double_typos() {
        let c = corrector();
        assert_eq!(c.correct("pizzza"), Some("pizza")); // ed 1
        assert_eq!(c.correct("piraets"), Some("pirates")); // transposition
        assert_eq!(c.correct("montrael"), Some("montreal")); // transposition
        assert_eq!(c.correct("montraeal"), Some("montreal")); // ed 2, long word
    }

    #[test]
    fn respects_length_scaled_budget() {
        let c = corrector();
        assert_eq!(c.correct("ca"), None); // too short to correct at all
        assert_eq!(c.correct("czrt"), Some("cart")); // ed 1 on a 4-char word: ok
        assert_eq!(c.correct("czzt"), None); // ed 2 on a 4-char word: refused
        assert_eq!(c.correct("zzzzzz"), None); // nothing close
    }

    #[test]
    fn ties_break_by_document_frequency() {
        let c = corrector();
        // "carx" is ed 1 from both "cart" (df 9) and "care" (df 4).
        assert_eq!(c.correct("carx"), Some("cart"));
    }

    #[test]
    fn low_df_terms_are_not_correction_targets() {
        // "rare" (df 1) is excluded, so a typo of it finds nothing.
        let c = SpellCorrector::build([("rare".to_string(), 1)]);
        assert_eq!(c.correct("raer"), None);
    }
}
