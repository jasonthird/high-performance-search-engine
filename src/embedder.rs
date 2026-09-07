//! Native CodeRankEmbed inference via Candle's NomicBert.
//!
//! Gated behind `--features semantic` so the default binary stays a lean
//! lexical engine. Why Candle rather than ORT / fastembed-rs:
//!
//! - `nomic-ai/CodeRankEmbed` is a custom `nomic_bert` (RoPE + SwiGLU, no
//!   official ONNX export). fastembed-rs ships `nomic-embed-text-v1.5`,
//!   which is a *different* model (mean-pooled text, not CLS-pooled code).
//! - Candle already implements NomicBert; we load the CodeRankEmbed
//!   safetensors, CLS-pool, L2-normalize, and apply the query prefix.
//!
//! Documents are encoded as raw code. Queries are prefixed with
//! `Represent this query for searching relevant code: `.

use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{l2_normalize, Config, NomicBertModel};

use tokenizers::{Encoding, Tokenizer, TruncationParams};

use crate::embeddings::CODERANK_DIM;

pub const MODEL_ID: &str = "nomic-ai/CodeRankEmbed";
pub const QUERY_PREFIX: &str = "Represent this query for searching relevant code: ";
/// Tokenizer.json truncates at 512 by default; plenty for this prototype
/// and much cheaper than the 8192 trained limit.
pub const MAX_SEQ: usize = 512;
/// Queries plus the instruction prefix are a few dozen tokens. A shorter
/// RoPE table means less work per layer when the query embedder is on CPU.
pub const MAX_QUERY_SEQ: usize = 64;
const DEFAULT_BATCH: usize = 4;

/// How the embedder will be used. Queries cap RoPE at [`MAX_QUERY_SEQ`];
/// document indexing keeps [`MAX_SEQ`]. Device is Metal F16 on macOS for
/// both (override `HPS_EMBED_DEVICE=cpu|metal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedUse {
    Query,
    Index,
}

pub struct Embedder {
    /// Candle model + device; `None` when CoreML serves every request
    /// this instance can receive.
    model: Option<NomicBertModel>,
    tokenizer: Tokenizer,
    device: Option<Device>,
    /// CoreML/ANE backend, preferred on macOS when compiled models exist
    /// (see `crate::coreml`). `HIPS_ENCODER=candle` disables it.
    #[cfg(target_os = "macos")]
    coreml: Option<crate::coreml::CoreMlEncoder>,
    /// Query-vector LRU: a repeated query costs a map lookup instead of a
    /// ~7-15 ms forward pass. Agents re-issue queries (retries, refinement
    /// loops, "the same search with a different top_k") often enough that
    /// this is the cheapest latency win in the whole retrieval path.
    /// Document encoding is NOT cached here — that is `embcache`'s job,
    /// keyed by content and persisted.
    query_cache: std::sync::Mutex<QueryCache>,
}

/// Tiny string-keyed LRU. At 256 entries x 768 f32 this is ~0.8 MB.
struct QueryCache {
    map: std::collections::HashMap<String, Vec<f32>>,
    order: std::collections::VecDeque<String>,
}

const QUERY_CACHE_CAP: usize = 256;

impl QueryCache {
    fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<f32>> {
        let hit = self.map.get(key).cloned()?;
        // Move to most-recent; O(n) over 256 keys is nothing next to the
        // forward pass this avoids.
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos).expect("position just found");
            self.order.push_back(k);
        }
        Some(hit)
    }

    fn put(&mut self, key: String, value: Vec<f32>) {
        if self.map.len() >= QUERY_CACHE_CAP && !self.map.contains_key(&key) {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        if self.map.insert(key.clone(), value).is_none() {
            self.order.push_back(key);
        }
    }
}

impl Embedder {
    /// Download (first run) and load CodeRankEmbed for **query** encoding.
    pub fn load() -> anyhow::Result<Self> {
        Self::load_for(EmbedUse::Query)
    }

    pub fn load_for(kind: EmbedUse) -> anyhow::Result<Self> {
        // Say something only when it matters: a first-run download is slow
        // and network-bound, and its absence offline explains a failure.
        // A cached load is routine and stays silent unless `-v`.
        // `Cache::from_env` honours HF_HOME; `Api::new` does not (it always
        // uses ~/.cache/huggingface). `ApiBuilder::from_env` uses the same
        // cache (and HF_ENDPOINT), so the presence check and the download
        // agree on one directory.
        let cache = hf_hub::Cache::from_env();
        let model_cache = cache.model(MODEL_ID.to_string());
        let cached = ["config.json", "tokenizer.json", "model.safetensors"]
            .iter()
            .all(|f| model_cache.get(f).is_some());
        if !cached {
            eprintln!(
                "downloading {MODEL_ID} (~550 MB, once) into {}...",
                cache.path().display()
            );
        } else if crate::verbosity::verbose() {
            eprintln!("loading {MODEL_ID} from cache");
        }
        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .build()
            .context("huggingface hub client")?;
        let repo = api.model(MODEL_ID.to_string());
        let config_path = repo.get("config.json").context("download config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("download tokenizer.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("download model.safetensors")?;
        Self::load_from_files(&config_path, &tokenizer_path, &weights_path, kind)
    }

    pub fn load_from_files(
        config_path: &Path,
        tokenizer_path: &Path,
        weights_path: &Path,
        kind: EmbedUse,
    ) -> anyhow::Result<Self> {
        let mut config: Config = serde_json::from_str(
            &std::fs::read_to_string(config_path).context("read config.json")?,
        )
        .context("parse CodeRankEmbed config.json")?;
        // RoPE tables are sized by n_positions. Don't precompute the 8192
        // trained window; queries can go shorter still.
        // Attention is quadratic in sequence length, so this cap is the
        // single biggest lever on indexing throughput. `HPS_EMBED_MAX_SEQ`
        // overrides it for documents (see `examples/embed_bench.rs`).
        let cap = match kind {
            EmbedUse::Query => MAX_QUERY_SEQ,
            EmbedUse::Index => std::env::var("HPS_EMBED_MAX_SEQ")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&c: &usize| c > 0)
                .unwrap_or(MAX_SEQ),
        };
        config.n_positions = config.n_positions.min(cap);

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        // Tokenize once without padding; the consumer pads only its actual batch.
        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length: cap,
            ..Default::default()
        }));

        #[cfg(target_os = "macos")]
        let coreml = if std::env::var("HIPS_ENCODER").as_deref() == Ok("candle") {
            None
        } else {
            crate::coreml::CoreMlEncoder::load().filter(|cm| match kind {
                EmbedUse::Index => cm.supports_documents(cap),
                EmbedUse::Query => cm.supports_queries(cap),
            })
        };
        // Candle stays the fallback: load it unless CoreML can serve every
        // request this instance will get (queries need the batch-1 model).
        #[cfg(target_os = "macos")]
        let candle_needed = match (&coreml, kind) {
            (None, _) => true,
            (Some(cm), EmbedUse::Query) => !cm.has_query_model(),
            (Some(_), EmbedUse::Index) => false,
        };
        #[cfg(not(target_os = "macos"))]
        let candle_needed = true;

        let (model, device) = if candle_needed {
            // CANDLE_METAL_COMPUTE_PER_BUFFER is set in main() before any
            // thread exists (calling setenv here would race concurrent
            // getenv on the rayon/watcher threads — undefined behavior).
            let (device, dtype) = pick_device(kind);
            // SAFETY: weights file is not mutated while mapped.
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device) }
                    .context("mmap CodeRankEmbed safetensors")?;
            let model = load_nomic(vb, &config)
                .context("load NomicBert weights (tried prefixes '', 'bert', 'nomic_bert')")?;
            (Some(model), Some(device))
        } else {
            if crate::verbosity::verbose() {
                eprintln!("CodeRankEmbed device: ANE (CoreML)");
            }
            (None, None)
        };

        let embedder = Self {
            model,
            tokenizer,
            device,
            #[cfg(target_os = "macos")]
            coreml,
            query_cache: std::sync::Mutex::new(QueryCache::new()),
        };
        // Compile Metal pipelines (first forward is hundreds of ms).
        let _ = embedder.embed_query("warmup");
        Ok(embedder)
    }

    /// Encode documents in inverted-index `doc_id` order and write
    /// `embeddings.bin` next to `meta.bin` / `postings.bin`.
    pub fn embed_index_docs(&self, dir: &Path, texts: &[String]) -> anyhow::Result<u64> {
        anyhow::ensure!(!texts.is_empty(), "no documents to embed");
        let n = texts.len();
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (i, chunk) in texts.chunks(4096).enumerate() {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            vectors.extend(self.embed_docs(&refs)?);
            let done = vectors.len();
            if done % 256 == 0 || done == n || i == 0 {
                eprintln!("  embedded {done}/{n}");
            }
        }
        crate::embeddings::write_f16(dir, CODERANK_DIM as u32, &vectors)
    }

    pub fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        if let Ok(mut cache) = self.query_cache.lock() {
            if let Some(vector) = cache.get(query) {
                return Ok(vector);
            }
        }
        let prefixed = format!("{QUERY_PREFIX}{query}");
        let mut batch = self.embed_batch(&[prefixed.as_str()])?;
        let vector = batch.pop().expect("one query");
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.put(query.to_string(), vector.clone());
        }
        Ok(vector)
    }

    pub fn embed_docs(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_batch(texts)
    }

    /// GPU batch size. Larger batches amortize kernel launches, which
    /// dominate short BERT sequences on Metal. `HPS_EMBED_BATCH` overrides.
    pub fn batch_size() -> usize {
        std::env::var("HPS_EMBED_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&b: &usize| b > 0)
            .unwrap_or(DEFAULT_BATCH)
    }

    /// The batch size for this instance's active backend: the CoreML
    /// document models' compiled batch when CoreML is serving, else the
    /// measured candle batch.
    pub fn effective_batch(&self) -> usize {
        #[cfg(target_os = "macos")]
        if let Some(cm) = &self.coreml {
            return cm.doc_batch();
        }
        Self::batch_size()
    }

    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        // Token-length bucketing with bounded CPU lookahead. Preserve the
        // measured backend batch size; larger is not always faster on Metal.
        self.embed_docs_with(texts, self.effective_batch(), true)
    }

    /// Encode with an explicit batch size and optional length bucketing.
    ///
    /// Padding is `BatchLongest`, so a batch costs the length of its longest
    /// member times its size. Sorting by length before batching groups
    /// similar-length texts together and stops one 512-token chunk from
    /// inflating everything batched with it. Results are scattered back into
    /// the caller's order, so bucketing is invisible from outside.
    pub fn embed_docs_with(
        &self,
        texts: &[&str],
        batch: usize,
        bucket_by_length: bool,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batch = batch.max(1);
        if texts.len() <= batch {
            // Queries and tiny edits should not pay for a worker thread.
            let rows = self
                .tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
            return self.embed_encodings(&rows);
        }
        // Keep the accelerator/model on the calling thread (CoreML is not Sync).
        // Only the tokenizer and borrowed input text cross the thread boundary.
        let tokenizer = self.tokenizer.clone();
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::sync_channel(2);
            let producer = scope.spawn(move || {
                for (window, texts) in texts.chunks(1024).enumerate() {
                    let rows =
                        match prepare_window(&tokenizer, texts, window * 1024, bucket_by_length) {
                            Ok(rows) => rows,
                            Err(e) => {
                                let _ = tx.send(Err(e));
                                return;
                            }
                        };
                    let mut rows = rows.into_iter();
                    loop {
                        let group: Vec<_> = rows.by_ref().take(batch).collect();
                        if group.is_empty() {
                            break;
                        }
                        if tx.send(Ok(group)).is_err() {
                            return;
                        }
                    }
                }
            });
            let result = (|| {
                let mut out = vec![Vec::new(); texts.len()];
                // IntoIter owns the receiver: on inference error it is dropped
                // before join, releasing a producer blocked on the bounded queue.
                for group in rx {
                    let (ids, encodings): (Vec<_>, Vec<_>) = group?.into_iter().unzip();
                    let vectors = self.embed_encodings(&encodings)?;
                    anyhow::ensure!(vectors.len() == ids.len(), "embedding batch size mismatch");
                    for (id, vector) in ids.into_iter().zip(vectors) {
                        out[id] = vector;
                    }
                }
                Ok(out)
            })();
            producer
                .join()
                .map_err(|_| anyhow::anyhow!("tokenizer worker panicked"))?;
            result
        })
    }

    fn embed_encodings(&self, encodings: &[Encoding]) -> anyhow::Result<Vec<Vec<f32>>> {
        let t0 = Instant::now();

        #[cfg(target_os = "macos")]
        if let Some(cm) = &self.coreml {
            // Unpadded rows (mask == 1); CoreML pads to its compiled shape.
            let rows: Vec<Vec<u32>> = encodings
                .iter()
                .map(|e| {
                    e.get_ids()
                        .iter()
                        .zip(e.get_attention_mask())
                        .take_while(|(_, &m)| m == 1)
                        .map(|(&t, _)| t)
                        .collect()
                })
                .collect();
            if rows.len() == 1 {
                if let Some(result) = cm.encode_query(&rows[0]) {
                    return Ok(vec![result?]);
                }
            }
            let mut out = Vec::with_capacity(rows.len());
            for group in rows.chunks(cm.doc_batch()) {
                out.extend(cm.encode_docs(group)?);
            }
            return Ok(out);
        }
        let seq = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let batch = encodings.len();
        let mut ids = vec![0u32; batch * seq];
        let mut mask = vec![0u8; batch * seq];
        for (i, enc) in encodings.iter().enumerate() {
            let t = enc.get_ids();
            let m = enc.get_attention_mask();
            ids[i * seq..i * seq + t.len()].copy_from_slice(t);
            for (j, &bit) in m.iter().enumerate() {
                mask[i * seq + j] = bit as u8;
            }
        }
        let tok_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let device = self
            .device
            .as_ref()
            .context("candle backend not loaded (CoreML-only instance)")?;
        let model = self
            .model
            .as_ref()
            .context("candle backend not loaded (CoreML-only instance)")?;
        let ids = Tensor::from_vec(ids, (batch, seq), device)?;
        // Candle's Metal backend implements where_cond for (U8, F16) but not
        // (U32, F16). NomicBert builds the attention mask with where_cond, so
        // keep the mask as U8 when running F16 on the GPU.
        let mask = Tensor::from_vec(mask, (batch, seq), device)?;
        let token_types = Tensor::zeros((batch, seq), DType::U32, device)?;
        let upload_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let t2 = Instant::now();
        let hidden = model
            .forward(&ids, Some(&token_types), Some(&mask))
            .context("nomic-bert forward")?;
        // CLS pooling: first token of each sequence. Sentence-transformers
        // 1_Pooling/config.json sets pooling_mode_cls_token = true.
        let cls = hidden.i((.., 0, ..))?;
        let normed = l2_normalize(&cls)?.to_dtype(DType::F32)?;
        let vecs = if profile() {
            device.synchronize().ok();
            let fwd_ms = t2.elapsed().as_secs_f64() * 1000.0;
            let t3 = Instant::now();
            let vecs = normed.to_vec2::<f32>()?;
            let down_ms = t3.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "  embed seq={seq} batch={batch}  pad {tok_ms:.2} ms  upload {upload_ms:.2} ms  forward {fwd_ms:.2} ms  download {down_ms:.2} ms"
            );
            vecs
        } else {
            let _ = (tok_ms, upload_ms, t2);
            normed.to_vec2::<f32>()?
        };
        anyhow::ensure!(
            vecs.iter().all(|v| v.len() == CODERANK_DIM),
            "unexpected embedding dim"
        );
        Ok(vecs)
    }
}

fn prepare_window(
    tokenizer: &Tokenizer,
    texts: &[&str],
    offset: usize,
    bucket: bool,
) -> anyhow::Result<Vec<(usize, Encoding)>> {
    let mut rows: Vec<_> = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .into_iter()
        .enumerate()
        .map(|(i, row)| (offset + i, row))
        .collect();
    if bucket {
        rows.sort_by_key(|(_, row)| row.len());
    }
    Ok(rows)
}

fn profile() -> bool {
    matches!(
        std::env::var("HPS_EMBED_PROFILE").as_deref(),
        Ok("1") | Ok("true")
    )
}

#[cfg(test)]
mod scheduling_tests {
    use super::*;

    #[test]
    fn token_bucketing_preserves_rows_ids_masks_and_truncation() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[UNK]".to_owned(), 0)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 3,
                ..Default::default()
            }))
            .unwrap();
        // Byte length gives the opposite order to actual token length.
        let texts = ["a b c d e", "long_identifier_without_spaces", "a b"];
        let plain = prepare_window(&tokenizer, &texts, 1024, false).unwrap();
        let sorted = prepare_window(&tokenizer, &texts, 1024, true).unwrap();
        assert_eq!(
            sorted.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![1025, 1026, 1024]
        );
        for (i, encoding) in sorted {
            assert_eq!(encoding.get_ids(), plain[i - 1024].1.get_ids());
            assert_eq!(
                encoding.get_attention_mask(),
                plain[i - 1024].1.get_attention_mask()
            );
            assert!(encoding.len() <= 3);
        }
    }

    #[test]
    fn inference_error_releases_prefetch_worker() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[UNK]".to_owned(), 0)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut encoder = Embedder {
            model: None,
            device: None,
            tokenizer: Tokenizer::new(model),
            #[cfg(target_os = "macos")]
            coreml: None,
            query_cache: std::sync::Mutex::new(QueryCache::new()),
        };
        assert!(encoder.embed_docs_with(&[], 4, true).unwrap().is_empty());
        // More batches than queue capacity: consumer fails on the first one.
        // Returning proves join does not wait forever on a blocked sender.
        let inputs = vec!["fixture"; 2049];
        let error = encoder.embed_docs_with(&inputs, 4, true).unwrap_err();
        assert!(error.to_string().contains("backend not loaded"));
        encoder.tokenizer = Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        let error = encoder.embed_docs_with(&inputs, 4, true).unwrap_err();
        assert!(error.to_string().contains("tokenize"));
    }
}

fn cpu_device() -> (Device, DType) {
    if crate::verbosity::verbose() {
        #[cfg(target_os = "macos")]
        eprintln!("CodeRankEmbed device: CPU (F32, Accelerate)");
        #[cfg(not(target_os = "macos"))]
        eprintln!("CodeRankEmbed device: CPU (F32)");
    }
    (Device::Cpu, DType::F32)
}

fn metal_or_cpu() -> (Device, DType) {
    #[cfg(target_os = "macos")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                if crate::verbosity::verbose() {
                    eprintln!("CodeRankEmbed device: Metal (F16)");
                }
                return (device, DType::F16);
            }
            Err(err) => eprintln!("Metal unavailable ({err}); falling back to CPU"),
        }
    }
    cpu_device()
}

fn pick_device(kind: EmbedUse) -> (Device, DType) {
    match std::env::var("HPS_EMBED_DEVICE").as_deref() {
        Ok("cpu") => return cpu_device(),
        Ok("metal") => return metal_or_cpu(),
        _ => {}
    }
    // Both paths prefer Metal when it exists. CPU+Accelerate was measured
    // slower for batch-1 queries (~22 ms vs ~10 ms): Candle's NomicBert
    // graph is hundreds of unfused ops, and Accelerate only speeds GEMMs.
    let _ = kind;
    metal_or_cpu()
}

fn load_nomic(vb: VarBuilder, config: &Config) -> anyhow::Result<NomicBertModel> {
    let mut last = None;
    for prefix in ["", "bert", "nomic_bert"] {
        let sub = if prefix.is_empty() {
            vb.clone()
        } else {
            vb.pp(prefix)
        };
        match NomicBertModel::load(sub, config) {
            Ok(m) => return Ok(m),
            Err(e) => last = Some(e),
        }
    }
    Err(anyhow::anyhow!(last.unwrap()))
}
