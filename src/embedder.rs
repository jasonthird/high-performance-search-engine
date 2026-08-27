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
use hf_hub::api::sync::Api;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

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
    model: NomicBertModel,
    tokenizer: Tokenizer,
    device: Device,
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
        eprintln!("loading {MODEL_ID} via Candle (first run downloads ~550 MB)...");
        let api = Api::new().context("huggingface hub client")?;
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
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length: cap,
            ..Default::default()
        }));

        let (device, dtype) = pick_device(kind);
        // SAFETY: weights file is not mutated while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device) }
            .context("mmap CodeRankEmbed safetensors")?;

        let model = load_nomic(vb, &config).context(
            "load NomicBert weights (tried prefixes '', 'bert', 'nomic_bert')",
        )?;

        let embedder = Self {
            model,
            tokenizer,
            device,
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
        for (i, chunk) in texts.chunks(DEFAULT_BATCH).enumerate() {
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

    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        // Length bucketing on by default: measured 73.4 -> 38.6 ms/chunk on
        // real corpus chunks (padding to the batch's longest member wasted
        // 41% of the encoder in file order), and `examples/bucket_equiv.rs`
        // shows the vectors match the unbucketed path to within F16 rerun
        // noise. Batch stays 4: larger batches measured slower even
        // bucketed. `HPS_EMBED_BATCH` overrides.
        self.embed_docs_with(texts, Self::batch_size(), true)
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
        let mut order: Vec<usize> = (0..texts.len()).collect();
        if bucket_by_length {
            // Byte length is a good enough proxy for token count here, and
            // far cheaper than tokenizing twice.
            order.sort_by_key(|&i| texts[i].len());
        }
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        for group in order.chunks(batch) {
            let sub: Vec<&str> = group.iter().map(|&i| texts[i]).collect();
            for (&i, vector) in group.iter().zip(self.embed_chunk(&sub)?) {
                out[i] = vector;
            }
        }
        Ok(out)
    }

    fn embed_chunk(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let t0 = Instant::now();
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
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
        let ids = Tensor::from_vec(ids, (batch, seq), &self.device)?;
        // Candle's Metal backend implements where_cond for (U8, F16) but not
        // (U32, F16). NomicBert builds the attention mask with where_cond, so
        // keep the mask as U8 when running F16 on the GPU.
        let mask = Tensor::from_vec(mask, (batch, seq), &self.device)?;
        let token_types = Tensor::zeros((batch, seq), DType::U32, &self.device)?;
        let upload_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let t2 = Instant::now();
        let hidden = self
            .model
            .forward(&ids, Some(&token_types), Some(&mask))
            .context("nomic-bert forward")?;
        // CLS pooling: first token of each sequence. Sentence-transformers
        // 1_Pooling/config.json sets pooling_mode_cls_token = true.
        let cls = hidden.i((.., 0, ..))?;
        let normed = l2_normalize(&cls)?.to_dtype(DType::F32)?;
        let vecs = if profile() {
            self.device.synchronize().ok();
            let fwd_ms = t2.elapsed().as_secs_f64() * 1000.0;
            let t3 = Instant::now();
            let vecs = normed.to_vec2::<f32>()?;
            let down_ms = t3.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "  embed seq={seq} batch={batch}  tok {tok_ms:.2} ms  upload {upload_ms:.2} ms  forward {fwd_ms:.2} ms  download {down_ms:.2} ms"
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

fn profile() -> bool {
    matches!(
        std::env::var("HPS_EMBED_PROFILE").as_deref(),
        Ok("1") | Ok("true")
    )
}

fn cpu_device() -> (Device, DType) {
    #[cfg(target_os = "macos")]
    eprintln!("CodeRankEmbed device: CPU (F32, Accelerate)");
    #[cfg(not(target_os = "macos"))]
    eprintln!("CodeRankEmbed device: CPU (F32)");
    (Device::Cpu, DType::F32)
}

fn metal_or_cpu() -> (Device, DType) {
    #[cfg(target_os = "macos")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                eprintln!("CodeRankEmbed device: Metal (F16)");
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
