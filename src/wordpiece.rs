//! BERT WordPiece tokenizer for CodeRankEmbed, reading its vocabulary from
//! the model's GGUF so the encoder is one self-contained file.
//!
//! Reproduces the Hugging Face pipeline the model was trained with:
//! `BertNormalizer` (clean text, isolate CJK characters, strip accents,
//! lowercase), `BertPreTokenizer` (split on whitespace, every punctuation
//! character its own word) and greedy longest-match WordPiece with the
//! `##` continuation prefix, then `[CLS] ... [SEP]`. The parity test at the
//! bottom checks it token-for-token against the `tokenizers` crate on this
//! repository's chunks whenever the original `tokenizer.json` is cached.

use std::collections::HashMap;

use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

/// Longer words are `[UNK]` outright, as in the reference implementation.
const MAX_INPUT_CHARS_PER_WORD: usize = 100;

pub struct WordPiece {
    /// Keys are word-initial pieces as-is and continuation pieces with the
    /// `##` prefix, i.e. the Hugging Face `vocab.txt` form.
    vocab: HashMap<String, u32>,
    /// Control tokens (`[CLS]`, `[SEP]`, ...) are matched verbatim in the
    /// raw text before normalization, as the reference tokenizer's added
    /// tokens are: a docstring quoting `[SEP]` yields the real id.
    specials: Vec<(String, u32)>,
    unk: u32,
    cls: u32,
    sep: u32,
}

impl WordPiece {
    /// From a Hugging Face style vocabulary (`##` marks continuation pieces).
    pub fn from_vocab(entries: impl IntoIterator<Item = (String, u32)>, unk: u32, cls: u32, sep: u32) -> Self {
        Self { vocab: entries.into_iter().collect(), specials: Vec::new(), unk, cls, sep }
    }

    /// Strings recognized verbatim in the input (the reference tokenizer's
    /// special added tokens). Longer matches win at the same position.
    pub fn with_specials(mut self, specials: impl IntoIterator<Item = (String, u32)>) -> Self {
        self.specials = specials.into_iter().filter(|(s, _)| !s.is_empty()).collect();
        self.specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        self
    }

    /// From a GGUF `tokenizer.ggml.tokens` array as llama.cpp's converter
    /// writes BERT vocabularies: word-initial pieces carry a U+2581 prefix,
    /// continuation pieces have none, and control tokens (`token_type` 3,
    /// e.g. `[CLS]`) are stored verbatim.
    pub fn from_gguf(tokens: &[String], token_types: &[i32], unk: u32, cls: u32, sep: u32) -> Self {
        let is_control = |id: usize| token_types.get(id).is_some_and(|&t| t == 3);
        let entries = tokens.iter().enumerate().map(|(id, token)| {
            let key = match token.strip_prefix('\u{2581}') {
                Some(word_initial) => word_initial.to_string(),
                None if is_control(id) => token.clone(),
                None => format!("##{token}"),
            };
            (key, id as u32)
        });
        let specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(id, _)| is_control(*id))
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        Self::from_vocab(entries, unk, cls, sep).with_specials(specials)
    }

    pub fn cls(&self) -> u32 {
        self.cls
    }

    pub fn sep(&self) -> u32 {
        self.sep
    }

    /// `[CLS] pieces [SEP]`, truncated to `max_len` tokens the way the
    /// reference tokenizer does it: the content is cut, the `[SEP]` kept.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity(64);
        ids.push(self.cls);
        let mut rest = text;
        while !rest.is_empty() {
            match self.next_special(rest) {
                Some((at, len, id)) => {
                    self.encode_segment(&rest[..at], &mut ids);
                    ids.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_segment(rest, &mut ids);
                    rest = "";
                }
            }
        }
        if ids.len() + 1 > max_len {
            ids.truncate(max_len.saturating_sub(1).max(1));
        }
        ids.push(self.sep);
        ids
    }

    /// Earliest special-token occurrence in `text` as (byte offset, byte
    /// length, id); ties at one offset go to the longest string.
    fn next_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (s, id) in &self.specials {
            if let Some(at) = text.find(s.as_str()) {
                if best.is_none_or(|(b, _, _)| at < b) {
                    best = Some((at, s.len(), *id));
                }
            }
        }
        best
    }

    /// Normalize, pre-tokenize and WordPiece one special-free span.
    fn encode_segment(&self, text: &str, ids: &mut Vec<u32>) {
        let mut word = String::new();
        let push_word = |ids: &mut Vec<u32>, word: &mut String| {
            if !word.is_empty() {
                self.word_pieces(word, ids);
                word.clear();
            }
        };
        for ch in normalize(text) {
            if is_whitespace(ch) {
                push_word(ids, &mut word);
            } else if is_punctuation(ch) {
                push_word(ids, &mut word);
                let mut single = String::new();
                single.push(ch);
                self.word_pieces(&single, ids);
            } else {
                word.push(ch);
            }
        }
        push_word(ids, &mut word);
    }

    /// Greedy longest-match over one pre-tokenized word.
    fn word_pieces(&self, word: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > MAX_INPUT_CHARS_PER_WORD {
            out.push(self.unk);
            return;
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        let mut buf = String::new();
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                buf.clear();
                if start > 0 {
                    buf.push_str("##");
                }
                buf.extend(&chars[start..end]);
                if let Some(&id) = self.vocab.get(&buf) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => pieces.push(id),
                None => {
                    out.push(self.unk);
                    return;
                }
            }
            start = end;
        }
        out.extend(pieces);
    }
}

/// `BertNormalizer` with `clean_text`, `handle_chinese_chars`, `lowercase`
/// and, because lowercase is on, accent stripping (NFD then drop
/// nonspacing marks). Returns the normalized characters.
fn normalize(text: &str) -> Vec<char> {
    let mut cleaned = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        if ch == '\0' || ch == '\u{fffd}' || is_control(ch) {
            continue;
        }
        if is_whitespace(ch) {
            cleaned.push(' ');
        } else if is_chinese_char(ch) {
            cleaned.push(' ');
            cleaned.push(ch);
            cleaned.push(' ');
        } else {
            cleaned.push(ch);
        }
    }
    cleaned
        .nfd()
        .filter(|c| !c.is_mark_nonspacing())
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r') || c.is_separator_space()
}

fn is_control(c: char) -> bool {
    if matches!(c, '\t' | '\n' | '\r') {
        return false;
    }
    c.is_other()
}

/// ASCII symbol ranges count as punctuation too, as in BERT's reference
/// `_is_punctuation`.
fn is_punctuation(c: char) -> bool {
    let cp = c as u32;
    (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
        || c.is_punctuation()
}

fn is_chinese_char(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B920..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> WordPiece {
        let vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "fn", "main", "(", ")", "{", "}", "hel", "##lo", "##p", "wor", "##ld", "ünïcode"]
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32));
        WordPiece::from_vocab(vocab, 1, 2, 3)
    }

    #[test]
    fn splits_punctuation_lowercases_and_matches_longest_pieces() {
        let tok = toy();
        assert_eq!(tok.encode("fn MAIN() { hello }", 512), vec![2, 4, 5, 6, 7, 8, 10, 11, 9, 3]);
        // No piece covers "help" fully: greedy takes "hel" then "##p".
        assert_eq!(tok.encode("help", 512), vec![2, 10, 12, 3]);
        // A word with an unknown remainder is [UNK] as a whole.
        assert_eq!(tok.encode("world hellx", 512), vec![2, 13, 14, 1, 3]);
    }

    #[test]
    fn strips_accents_and_isolates_cjk() {
        let tok = toy();
        // "ünïcode" normalizes to "unicode", absent from the toy vocab => [UNK];
        // the vocab entry itself never matches because it was never normalized.
        assert_eq!(tok.encode("ÜNÏCODE", 512), vec![2, 1, 3]);
        assert_eq!(normalize("a世b").iter().collect::<String>(), "a 世 b");
        assert_eq!(normalize("x\u{0}\u{fffd}\u{200b}y\tz").iter().collect::<String>(), "xy z");
    }

    #[test]
    fn literal_special_tokens_in_text_map_to_their_ids() {
        let tok = toy().with_specials([("[SEP]".to_string(), 3u32), ("[CLS]".to_string(), 2u32)]);
        // Matched verbatim and case-sensitively, even glued to other text.
        assert_eq!(tok.encode("fn[SEP]main (cls)", 512), vec![2, 4, 3, 5, 6, 1, 7, 3]);
    }

    #[test]
    fn unicode_punctuation_and_symbols_are_isolated() {
        let vocab = ["[UNK]", "[CLS]", "[SEP]", "a", "b", "—", "「", "$", "^", "`", "ab"]
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32));
        let tok = WordPiece::from_vocab(vocab, 0, 1, 2);
        // An em dash (Unicode Pd) and a CJK bracket (Ps) split words like ASCII punctuation.
        assert_eq!(tok.encode("a—b", 512), vec![1, 3, 5, 4, 2]);
        assert_eq!(tok.encode("「ab", 512), vec![1, 6, 10, 2]);
        // ASCII symbols outside Unicode punctuation ($ ^ `) count as punctuation for BERT.
        assert_eq!(tok.encode("a$b^ab`", 512), vec![1, 3, 7, 4, 8, 10, 9, 2]);
        assert!(is_punctuation('$') && is_punctuation('^') && is_punctuation('~') && !is_punctuation('a'));
    }

    #[test]
    fn normalization_cases() {
        // Accents stripped, then lowercased; NFD decomposition never leaks marks.
        assert_eq!(normalize("Café ÀÉÎ").iter().collect::<String>(), "cafe aei");
        // Characters that lowercase to several code points survive as a sequence.
        assert_eq!(normalize("İ").iter().collect::<String>(), "i");
        // Whitespace classes collapse to a plain space; tabs/newlines are whitespace, not control.
        assert_eq!(normalize("a\u{a0}b\tc\nd").iter().collect::<String>(), "a b c d");
        // Control and format characters vanish (zero-width space is Cf).
        assert_eq!(normalize("a\u{200b}b\u{7}c").iter().collect::<String>(), "abc");
        // Every CJK ideograph becomes its own word.
        assert_eq!(normalize("中文x").iter().collect::<String>(), " 中  文 x");
    }

    #[test]
    fn overlong_words_and_unknown_pieces() {
        let tok = toy();
        let long = "f".repeat(MAX_INPUT_CHARS_PER_WORD + 1);
        assert_eq!(tok.encode(&long, 512), vec![2, 1, 3]);
        let at_limit = "fn".repeat(MAX_INPUT_CHARS_PER_WORD / 2);
        // Exactly at the limit is still matched greedily, piece by piece.
        let ids = tok.encode(&at_limit, 512);
        assert_eq!(ids[0], 2);
        assert_eq!(*ids.last().unwrap(), 3);
        assert!(ids[1..ids.len() - 1].iter().all(|&i| i == 1), "no ##fn piece exists: {ids:?}");
        // Whitespace-only and punctuation-only inputs.
        assert_eq!(tok.encode("   \n\t ", 512), vec![2, 3]);
        assert_eq!(tok.encode("()", 512), vec![2, 6, 7, 3]);
    }

    #[test]
    fn specials_prefer_the_longest_match_and_are_case_sensitive() {
        let tok = toy().with_specials([("[SEP]".to_string(), 3u32), ("[SEP]X".to_string(), 9u32), ("[CLS]".to_string(), 2u32)]);
        // "[SEP]X" beats "[SEP]" at the same offset.
        assert_eq!(tok.encode("[SEP]X", 512), vec![2, 9, 3]);
        assert_eq!(tok.encode("[SEP][CLS]", 512), vec![2, 3, 2, 3]);
        // Lowercase "[sep]" is ordinary text: bracket, word, bracket.
        assert_eq!(tok.encode("[sep]", 512), vec![2, 1, 1, 1, 3]);
        // Empty special strings are ignored rather than matching everywhere.
        let tok = toy().with_specials([(String::new(), 7u32)]);
        assert_eq!(tok.encode("fn", 512), vec![2, 4, 3]);
    }

    #[test]
    fn output_invariants_hold_for_arbitrary_input() {
        let tok = toy().with_specials([("[SEP]".to_string(), 3u32)]);
        let long_word = "a".repeat(5000);
        let many_specials = "[SEP]".repeat(700);
        let inputs = ["", "fn", long_word.as_str(), "\u{0}\u{fffd}", many_specials.as_str(), "fn main() { hello world }"];
        for text in inputs {
            for max_len in [2usize, 3, 5, 16, 512] {
                let ids = tok.encode(text, max_len);
                assert_eq!(ids[0], tok.cls(), "{text:?} {max_len}");
                assert_eq!(*ids.last().unwrap(), tok.sep(), "{text:?} {max_len}");
                assert!(ids.len() <= max_len, "{text:?}: {} > {max_len}", ids.len());
                assert!(ids.len() >= 2);
                assert_eq!(ids, tok.encode(text, max_len), "deterministic");
            }
        }
    }

    #[test]
    fn truncation_keeps_the_separator() {
        let tok = toy();
        let ids = tok.encode("fn fn fn fn fn fn", 4);
        assert_eq!(ids, vec![2, 4, 4, 3]);
        assert_eq!(tok.encode("", 512), vec![2, 3]);
    }

    #[test]
    fn gguf_vocab_convention_round_trips() {
        let tokens: Vec<String> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "\u{2581}hel", "lo", "\u{2581}("]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let types = vec![3, 3, 3, 3, 1, 1, 1];
        let tok = WordPiece::from_gguf(&tokens, &types, 1, 2, 3);
        assert_eq!(tok.encode("hello (", 512), vec![2, 4, 5, 6, 3]);
    }

    /// Token-for-token parity with the reference tokenizer on real code.
    /// Runs whenever CodeRankEmbed's `tokenizer.json` is in the Hugging Face
    /// cache (or `HIPS_TEST_TOKENIZER_JSON` points at one).
    #[test]
    fn matches_reference_tokenizer_on_repository_chunks() {
        let Some(path) = reference_tokenizer_json() else {
            eprintln!("skipped: no tokenizer.json cached");
            return;
        };
        let reference = tokenizers::Tokenizer::from_file(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let vocab = json["model"]["vocab"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32));
        let specials = json["added_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|a| a["special"].as_bool() == Some(true))
            .map(|a| (a["content"].as_str().unwrap().to_string(), a["id"].as_u64().unwrap() as u32));
        let ours = WordPiece::from_vocab(vocab, 100, 101, 102).with_specials(specials);
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let chunks = crate::repo::collect_chunks(root).unwrap();
        let mut compared = 0;
        for chunk in chunks.iter().step_by(3) {
            let text = format!("{}\n{}", chunk.title(), chunk.body);
            let expected = reference.encode(text.as_str(), true).unwrap();
            let expected: Vec<u32> = expected.get_ids().iter().copied().take(512).collect();
            // The reference truncates at 512 through its own config; mirror that.
            let mut expected = expected;
            if expected.len() == 512 {
                *expected.last_mut().unwrap() = 102;
            }
            assert_eq!(ours.encode(&text, 512), expected, "chunk {}", chunk.id());
            compared += 1;
        }
        assert!(compared > 100, "too few chunks compared: {compared}");
    }

    fn reference_tokenizer_json() -> Option<std::path::PathBuf> {
        if let Some(p) = std::env::var_os("HIPS_TEST_TOKENIZER_JSON") {
            return Some(p.into());
        }
        let home = std::env::var_os("HOME")?;
        let hub = std::path::PathBuf::from(home).join(".cache/huggingface/hub/models--nomic-ai--CodeRankEmbed/snapshots");
        let snapshot = std::fs::read_dir(hub).ok()?.flatten().next()?.path();
        let file = snapshot.join("tokenizer.json");
        file.is_file().then_some(file)
    }
}
