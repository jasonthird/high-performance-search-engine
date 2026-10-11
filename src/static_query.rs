//! Experimental word lookup in CodeRankEmbed's original 768-dimensional space.
//! No centering or PCA: these approximate queries score the existing document rows.
use crate::{embeddings::CODERANK_DIM, tokenizer::Tokenizer};
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

pub const SCHEMA: &str = "coderank-static-query-v1";

#[derive(Serialize, Deserialize)]
pub struct WordTable {
    pub schema: String,
    pub dimensions: usize,
    pub source: serde_json::Value,
    pub vectors: BTreeMap<String, Vec<i8>>,
}

pub struct StaticQueryEncoder {
    vectors: HashMap<String, Vec<f32>>,
    source: serde_json::Value,
}

pub struct Approximation {
    pub vector: Vec<f32>,
    pub known_terms: usize,
    pub total_terms: usize,
}

impl StaticQueryEncoder {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let table: WordTable = serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?,
        ))?;
        Self::from_table(table)
    }

    pub fn from_table(table: WordTable) -> anyhow::Result<Self> {
        anyhow::ensure!(table.schema == SCHEMA, "unsupported static query schema");
        anyhow::ensure!(
            table.dimensions == CODERANK_DIM,
            "static queries must have {CODERANK_DIM} dimensions"
        );
        anyhow::ensure!(
            table.source["teacher"] == "nomic-ai/CodeRankEmbed"
                && table.source["pooling"] == "cls"
                && table.source["transform"] == "none",
            "static query table must use untransformed CodeRankEmbed CLS embeddings"
        );
        anyhow::ensure!(!table.vectors.is_empty(), "empty word table");
        let mut vectors = HashMap::with_capacity(table.vectors.len());
        for (word, row) in table.vectors {
            anyhow::ensure!(
                !word.is_empty() && word == word.to_lowercase(),
                "invalid lookup word {word:?}"
            );
            anyhow::ensure!(
                row.len() == CODERANK_DIM,
                "wrong vector length for {word:?}"
            );
            let mut row: Vec<f32> = row.into_iter().map(f32::from).collect();
            normalize(&mut row).with_context(|| format!("invalid vector for {word:?}"))?;
            vectors.insert(word, row);
        }
        Ok(Self {
            vectors,
            source: table.source,
        })
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vectors.len()
    }

    /// Sublinear term-frequency weighting, identifier splitting, then L2 normalization.
    /// Unknown terms are reported; an entirely unknown query is an explicit error.
    pub fn approximate(&self, query: &str) -> anyhow::Result<Approximation> {
        let mut counts = BTreeMap::<String, usize>::new();
        for word in Tokenizer::new(true).code(true).tokenize(query) {
            *counts.entry(word).or_default() += 1;
        }
        let total_terms = counts.len();
        let mut vector = vec![0.0; CODERANK_DIM];
        let mut known_terms = 0;
        for (word, count) in counts {
            if let Some(row) = self.vectors.get(&word) {
                known_terms += 1;
                let weight = 1.0 + (count as f32).ln();
                for (out, value) in vector.iter_mut().zip(row) {
                    *out += weight * value;
                }
            }
        }
        anyhow::ensure!(
            known_terms > 0,
            "no known terms in static query; use full CodeRankEmbed or BM25"
        );
        normalize(&mut vector)?;
        Ok(Approximation {
            vector,
            known_terms,
            total_terms,
        })
    }
}

/// A base vocabulary plus a small persistent sidecar of words learned on misses.
/// Single-writer experimental cache; the caller supplies a lazily loaded model.
pub struct GrowingQueryEncoder {
    encoder: StaticQueryEncoder,
    additions: WordTable,
    path: std::path::PathBuf,
}

pub struct Lookup {
    pub approximation: Approximation,
    pub missing_terms: usize,
}

impl GrowingQueryEncoder {
    pub fn load(base: &Path, additions: &Path) -> anyhow::Result<Self> {
        Self::with_base(StaticQueryEncoder::load(base)?, additions)
    }

    pub fn with_base(mut encoder: StaticQueryEncoder, path: &Path) -> anyhow::Result<Self> {
        let additions = if path.exists() {
            let table: WordTable =
                serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(path)?))?;
            anyhow::ensure!(
                table.source["query_prefix"].as_bool().unwrap_or(false)
                    == encoder.source["query_prefix"].as_bool().unwrap_or(false),
                "word-cache query prefix mismatch"
            );
            let loaded = StaticQueryEncoder::from_table(WordTable {
                schema: table.schema.clone(),
                dimensions: table.dimensions,
                source: table.source.clone(),
                vectors: table.vectors.clone(),
            })?;
            encoder.vectors.extend(loaded.vectors);
            table
        } else {
            WordTable {
                schema: SCHEMA.into(),
                dimensions: CODERANK_DIM,
                source: encoder.source.clone(),
                vectors: BTreeMap::new(),
            }
        };
        Ok(Self {
            encoder,
            additions,
            path: path.into(),
        })
    }

    pub fn learned_words(&self) -> usize {
        self.additions.vectors.len()
    }

    /// Encode all distinct misses once. Validate the entire batch and atomically
    /// persist it before changing resident state, so failed calls can be retried.
    pub fn lookup<F>(&mut self, query: &str, encode: F) -> anyhow::Result<Lookup>
    where
        F: FnOnce(&[String], bool) -> anyhow::Result<Vec<Vec<f32>>>,
    {
        let terms: std::collections::BTreeSet<_> = Tokenizer::new(true)
            .code(true)
            .tokenize(query)
            .into_iter()
            .collect();
        anyhow::ensure!(!terms.is_empty(), "query has no searchable terms");
        let misses: Vec<_> = terms
            .into_iter()
            .filter(|w| !self.encoder.vectors.contains_key(w))
            .collect();
        if !misses.is_empty() {
            let prefix = self.encoder.source["query_prefix"]
                .as_bool()
                .unwrap_or(false);
            let rows = encode(&misses, prefix)?;
            anyhow::ensure!(
                rows.len() == misses.len(),
                "model returned wrong number of word vectors"
            );
            let mut added = BTreeMap::new();
            let mut decoded = HashMap::new();
            for (word, mut row) in misses.iter().cloned().zip(rows) {
                anyhow::ensure!(row.len() == CODERANK_DIM, "wrong model vector dimension");
                normalize(&mut row)?;
                let quantized: Vec<i8> = row
                    .iter()
                    .map(|v| (v * 127.0).round().clamp(-127.0, 127.0) as i8)
                    .collect();
                let mut row: Vec<f32> = quantized.iter().copied().map(f32::from).collect();
                normalize(&mut row)?;
                added.insert(word.clone(), quantized);
                decoded.insert(word, row);
            }
            let mut persisted = self.additions.vectors.clone();
            persisted.extend(added);
            let table = WordTable {
                schema: SCHEMA.into(),
                dimensions: CODERANK_DIM,
                source: self.additions.source.clone(),
                vectors: persisted,
            };
            let parent = self
                .path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            std::fs::create_dir_all(parent)?;
            let mut file = tempfile::NamedTempFile::new_in(parent)?;
            {
                use std::io::Write;
                let mut writer = std::io::BufWriter::new(file.as_file_mut());
                serde_json::to_writer(&mut writer, &table)?;
                writer.flush()?;
            }
            file.as_file_mut().sync_all()?;
            file.persist(&self.path)?;
            self.additions = table;
            self.encoder.vectors.extend(decoded);
        }
        Ok(Lookup {
            approximation: self.encoder.approximate(query)?,
            missing_terms: misses.len(),
        })
    }
}

fn normalize(vector: &mut [f32]) -> anyhow::Result<()> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    anyhow::ensure!(
        norm.is_finite() && norm > 1e-8,
        "zero or non-finite embedding"
    );
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn table() -> WordTable {
        let mut row = vec![0; CODERANK_DIM];
        row[0] = 127;
        WordTable {
            schema: SCHEMA.into(),
            dimensions: CODERANK_DIM,
            source: serde_json::json!({"teacher":"nomic-ai/CodeRankEmbed","pooling":"cls","transform":"none"}),
            vectors: BTreeMap::from([("user".into(), row)]),
        }
    }
    #[test]
    fn growing_cache_persists_misses_and_reuses_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("learned.json");
        let base = || StaticQueryEncoder::from_table(table()).unwrap();
        let mut encoder = GrowingQueryEncoder::with_base(base(), &path).unwrap();
        let first = encoder
            .lookup("user qzx qzx", |words, prefix| {
                assert_eq!(words, &["qzx"]);
                assert!(!prefix);
                let mut row = vec![0.0; CODERANK_DIM];
                row[1] = 1.0;
                Ok(vec![row])
            })
            .unwrap();
        assert_eq!(first.missing_terms, 1);
        assert_eq!(
            first.approximation.known_terms,
            first.approximation.total_terms
        );
        assert_eq!(encoder.learned_words(), 1);
        let hit = encoder
            .lookup("user qzx qzx", |_, _| panic!("hit called model"))
            .unwrap();
        assert_eq!(hit.missing_terms, 0);
        assert_eq!(hit.approximation.vector, first.approximation.vector);
        let mut restarted = GrowingQueryEncoder::with_base(base(), &path).unwrap();
        let hit = restarted
            .lookup("user qzx qzx", |_, _| panic!("reload called model"))
            .unwrap();
        assert_eq!(hit.approximation.vector, first.approximation.vector);
        assert!(!restarted
            .lookup("newword", |_, _| anyhow::bail!("model unavailable"))
            .is_ok());
        assert_eq!(restarted.learned_words(), 1);
        assert!(restarted
            .lookup("badword", |_, _| Ok(vec![vec![f32::NAN; CODERANK_DIM]]))
            .is_err());
        assert_eq!(restarted.learned_words(), 1);
        assert!(restarted
            .lookup("", |_, _| panic!("empty called model"))
            .is_err());
        assert_eq!(
            GrowingQueryEncoder::with_base(base(), &path)
                .unwrap()
                .learned_words(),
            1
        );
    }
    #[test]
    fn growing_cache_rejects_partial_batches_and_failed_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("learned.json");
        let base = || StaticQueryEncoder::from_table(table()).unwrap();
        let mut encoder = GrowingQueryEncoder::with_base(base(), &path).unwrap();
        let valid = || {
            let mut row = vec![0.0; CODERANK_DIM];
            row[1] = 1.0;
            row
        };
        assert!(encoder
            .lookup("qzx abcxyz", |_, _| Ok(vec![
                valid(),
                vec![0.0; CODERANK_DIM]
            ]))
            .is_err());
        assert_eq!(encoder.learned_words(), 0);
        assert!(!path.exists());
        assert!(encoder.lookup("qzx", |_, _| Ok(Vec::new())).is_err());
        // Force the atomic replacement to fail after successful model inference.
        std::fs::create_dir(&path).unwrap();
        assert!(encoder.lookup("qzx", |_, _| Ok(vec![valid()])).is_err());
        assert_eq!(encoder.learned_words(), 0);
        std::fs::remove_dir(&path).unwrap();
        let retry = encoder
            .lookup("qzx", |words, _| {
                assert_eq!(words, &["qzx"]);
                Ok(vec![valid()])
            })
            .unwrap();
        assert_eq!(retry.missing_terms, 1);
        let mut sidecar: WordTable =
            serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
        sidecar.source["query_prefix"] = true.into();
        serde_json::to_writer(std::fs::File::create(&path).unwrap(), &sidecar).unwrap();
        assert!(GrowingQueryEncoder::with_base(base(), &path).is_err());
    }
    #[test]
    fn splits_unseen_identifiers_and_reports_coverage() {
        let encoder = StaticQueryEncoder::from_table(table()).unwrap();
        let result = encoder.approximate("getUser qzx").unwrap();
        assert_eq!(result.known_terms, 1);
        assert!(result.total_terms > result.known_terms);
        assert_eq!(result.vector.len(), CODERANK_DIM);
        assert!((result.vector.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(encoder.approximate("qzx").is_err());
        assert!(encoder.approximate("").is_err());
    }
    #[test]
    fn rejects_incompatible_and_invalid_tables() {
        let mut t = table();
        t.dimensions = 128;
        assert!(StaticQueryEncoder::from_table(t).is_err());
        let mut t = table();
        t.source["transform"] = "pca".into();
        assert!(StaticQueryEncoder::from_table(t).is_err());
        let mut t = table();
        t.vectors.get_mut("user").unwrap().fill(0);
        assert!(StaticQueryEncoder::from_table(t).is_err());
        let mut t = table();
        t.vectors.get_mut("user").unwrap().pop();
        assert!(StaticQueryEncoder::from_table(t).is_err());
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn searches_existing_document_vectors_without_loading_a_model() {
        use crate::{
            indexer::{self, InputDoc},
            query::{self, FusionArg, PqMode, RankMode, SearchOpts},
            reorder::ReorderStrategy,
            searcher::AnyIndex,
            storage,
        };
        let docs = vec![
            InputDoc {
                id: "relevant".into(),
                title: "user.rs".into(),
                body: "user lookup".into(),
            },
            InputDoc {
                id: "other".into(),
                title: "other.rs".into(),
                body: "other topic".into(),
            },
        ];
        let index = indexer::build_index(&docs, true, 128, ReorderStrategy::None);
        let dir = tempfile::tempdir().unwrap();
        storage::save_index(&index, dir.path()).unwrap();
        let mut rows = vec![vec![0.0; CODERANK_DIM]; 2];
        rows[0][0] = 1.0;
        rows[1][1] = 1.0;
        crate::embeddings::write_f16(dir.path(), CODERANK_DIM as u32, &rows).unwrap();
        let index = AnyIndex::open(dir.path()).unwrap();
        let encoder = StaticQueryEncoder::from_table(table()).unwrap();
        let opts = SearchOpts {
            mode: RankMode::Semantic,
            semantic_candidates: 20,
            fusion: FusionArg::Weighted,
            alpha: 0.15,
            rrf_k: 60.0,
            nprobe: 0,
            pq: PqMode::Off,
            ann_ef: 128,
            exact_vectors: true,
        };
        let run = query::run_ranked_with(&index, &encoder, "user", 1, &opts).unwrap();
        assert_eq!(run.results[0].id, "relevant");
        // Approximate lookup must not claim to contain a cached contextual prediction.
        assert!(!query::QueryEncoder::query_cached(&encoder, "user"));
        assert!(query::run_ranked_with(&index, &encoder, "qzx", 1, &opts).is_err());
    }
}
