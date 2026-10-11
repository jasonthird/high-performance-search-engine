//! CodeRankEmbed inference on hips' own ggml engine (`crate::ggml`).
//!
//! Gated behind `--features semantic` so the default binary stays a lean
//! lexical engine. The model is one GGUF file (F16 weights plus the
//! WordPiece vocabulary) fetched once from a pinned release, the way the
//! grammar libraries are; `HIPS_MODEL_PATH` points at any local copy.
//!
//! Documents are encoded as raw code. Queries are prefixed with
//! `Represent this query for searching relevant code: `. Output is the
//! `[CLS]` row, L2-normalized, validated before it can reach an index or
//! the query cache.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context;

use crate::embeddings::CODERANK_DIM;
use crate::ggml::{Engine, EngineOptions};
use crate::wordpiece::WordPiece;

pub const MODEL_ID: &str = "nomic-ai/CodeRankEmbed";
pub const QUERY_PREFIX: &str = "Represent this query for searching relevant code: ";
/// The GGUF asset: the F16 conversion of the published safetensors
/// (`convert_hf_to_gguf.py --outtype f16`), vocabulary included.
pub const MODEL_FILE: &str = "CodeRankEmbed-f16.gguf";
/// Immutable release holding [`MODEL_FILE`] and its `.sha256`; bump with
/// the file, never replace an asset in place.
const MODEL_RELEASE: &str = "models-v1";
const MODEL_RELEASE_URL: &str = "https://github.com/jasonthird/high-performance-search-engine/releases/download";
const MODEL_MAX_BYTES: u64 = 512 * 1024 * 1024;
/// Plenty for code chunks and much cheaper than the 2048 trained limit;
/// attention is quadratic in it.
pub const MAX_SEQ: usize = 512;
/// Queries plus the instruction prefix are a few dozen tokens.
pub const MAX_QUERY_SEQ: usize = 64;
/// Sequences per graph. Batches are length-bucketed, so padding stays
/// small; the token budget caps the padded batch (longest x count). The
/// GPU wants big graphs to amortize launches; the CPU wants activations
/// that stay in cache (measured: 512-1024 tokens beats 8192 by 1.4x on a
/// 96 MB L3), so the default follows the backend.
const DEFAULT_BATCH: usize = 32;
const DEFAULT_TOKEN_BUDGET_GPU: usize = 8192;
const DEFAULT_TOKEN_BUDGET_CPU: usize = 1024;

/// How the embedder will be used. Queries truncate at [`MAX_QUERY_SEQ`];
/// document indexing keeps [`MAX_SEQ`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedUse {
    Query,
    Index,
}

pub struct Embedder {
    /// The ggml engine; `None` when CoreML serves every request this
    /// instance can receive.
    engine: Option<Mutex<Engine>>,
    tokenizer: WordPiece,
    pad_id: u32,
    max_seq: usize,
    device: Option<String>,
    flash: bool,
    token_budget: usize,
    /// CoreML/ANE backend, preferred on macOS when compiled models exist
    /// (see `crate::coreml`). `HIPS_ENCODER=ggml` disables it.
    #[cfg(target_os = "macos")]
    coreml: Option<crate::coreml::CoreMlEncoder>,
    /// Query-vector LRU: a repeated query costs a map lookup instead of a
    /// forward pass. Agents re-issue queries (retries, refinement loops,
    /// "the same search with a different top_k") often enough that this is
    /// the cheapest latency win in the whole retrieval path. Document
    /// encoding is NOT cached here — that is `embcache`'s job. Exact query
    /// vectors also persist in a shared disk cache, independent of indexes.
    query_cache: Mutex<QueryCache>,
    disk_query_cache: Option<crate::query_cache::DiskQueryCache>,
    disk_query_hits: AtomicU64,
    query_model_calls: AtomicU64,
    fallbacks: Vec<String>,
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

/// Where the model lives: `HIPS_MODEL_PATH`, else the csearch cache. A
/// missing file is fetched from the pinned release (`HIPS_MODEL_URL`
/// overrides the asset URL) unless `HIPS_MODEL_OFFLINE=1`.
pub fn model_path() -> anyhow::Result<PathBuf> {
    if let Some(p) = std::env::var_os("HIPS_MODEL_PATH").filter(|p| !p.is_empty()) {
        let p = PathBuf::from(p);
        anyhow::ensure!(p.is_file(), "HIPS_MODEL_PATH {} is not a file", p.display());
        return Ok(p);
    }
    let path = crate::codeindex::cache_root().join("models").join(MODEL_FILE);
    if path.is_file() {
        return Ok(path);
    }
    if std::env::var_os("HIPS_MODEL_OFFLINE").as_deref() == Some(std::ffi::OsStr::new("1")) {
        anyhow::bail!(
            "CodeRankEmbed model {} is not cached and HIPS_MODEL_OFFLINE=1 (set HIPS_MODEL_PATH to a local GGUF)",
            path.display()
        );
    }
    let url = std::env::var("HIPS_MODEL_URL")
        .unwrap_or_else(|_| format!("{MODEL_RELEASE_URL}/{MODEL_RELEASE}/{MODEL_FILE}"));
    eprintln!("downloading CodeRankEmbed ({MODEL_FILE}, ~274 MB, once) into {}...", path.display());
    crate::fetch::download_verified(&url, &path, MODEL_MAX_BYTES, crate::ggml::looks_like_gguf)
        .with_context(|| format!("download CodeRankEmbed from {url}"))?;
    Ok(path)
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name).as_deref() {
        Ok("0") | Ok("false") | Ok("off") => false,
        Ok("1") | Ok("true") | Ok("on") => true,
        _ => default,
    }
}

impl Embedder {
    /// Load CodeRankEmbed for **query** encoding (fetching it on first run).
    pub fn load() -> anyhow::Result<Self> {
        Self::load_for(EmbedUse::Query)
    }

    pub fn load_for(kind: EmbedUse) -> anyhow::Result<Self> {
        let path = model_path()?;
        Self::load_from_file(&path, kind)
    }

    pub fn load_from_file(path: &Path, kind: EmbedUse) -> anyhow::Result<Self> {
        Self::load_with_backend(path, kind, true)
    }

    fn load_with_backend(path: &Path, kind: EmbedUse, allow_coreml: bool) -> anyhow::Result<Self> {
        #[cfg(not(target_os = "macos"))]
        let _ = allow_coreml;
        // Attention is quadratic in sequence length, so this cap is the
        // single biggest lever on indexing throughput. `HPS_EMBED_MAX_SEQ`
        // overrides it for documents.
        let max_seq = match kind {
            EmbedUse::Query => MAX_QUERY_SEQ,
            EmbedUse::Index => std::env::var("HPS_EMBED_MAX_SEQ")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&c: &usize| c > 0)
                .unwrap_or(MAX_SEQ),
        };

        #[cfg(target_os = "macos")]
        let coreml = if !allow_coreml || std::env::var("HIPS_ENCODER").as_deref() == Ok("ggml") {
            None
        } else {
            crate::coreml::CoreMlEncoder::load().filter(|cm| match kind {
                EmbedUse::Index => cm.supports_documents(max_seq),
                EmbedUse::Query => cm.supports_queries(max_seq),
            })
        };
        // ggml stays the fallback: load it unless CoreML can serve every
        // request this instance will get (queries need the batch-1 model).
        #[cfg(target_os = "macos")]
        let engine_needed = match (&coreml, kind) {
            (None, _) => true,
            (Some(cm), EmbedUse::Query) => !cm.has_query_model(),
            (Some(_), EmbedUse::Index) => false,
        };
        #[cfg(not(target_os = "macos"))]
        let engine_needed = true;

        let (engine, vocab, device, flash, token_budget) = if engine_needed {
            let opts = EngineOptions {
                gpu: pick_gpu(),
                threads: std::env::var("HPS_EMBED_THREADS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .filter(|&t: &usize| t > 0)
                    .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)),
                flash_attention: env_flag("HIPS_GGML_FLASH", true),
                cls_only_last_layer: env_flag("HIPS_GGML_CLS_ONLY", true),
                cast_kv_f16: env_flag("HIPS_GGML_CAST_KV", cfg!(target_os = "macos")),
            };
            let (engine, vocab) = Engine::load(path, &opts)
                .with_context(|| format!("load CodeRankEmbed GGUF {}", path.display()))?;
            anyhow::ensure!(
                engine.hparams().n_embd == CODERANK_DIM,
                "model width {} != expected {CODERANK_DIM}",
                engine.hparams().n_embd
            );
            anyhow::ensure!(max_seq <= engine.hparams().n_ctx, "sequence cap {max_seq} exceeds the model's {}", engine.hparams().n_ctx);
            if crate::verbosity::verbose() {
                eprintln!("CodeRankEmbed device: {} ({})", engine.device(), if engine.flash_attention() { "flash attention" } else { "naive attention" });
            }
            let device = engine.device().to_string();
            let flash = engine.flash_attention();
            let budget = if engine.on_gpu() { DEFAULT_TOKEN_BUDGET_GPU } else { DEFAULT_TOKEN_BUDGET_CPU };
            (Some(Mutex::new(engine)), vocab, Some(device), flash, budget)
        } else {
            if crate::verbosity::verbose() {
                eprintln!("CodeRankEmbed backend: CoreML");
            }
            let vocab = crate::ggml::read_vocab(path)
                .with_context(|| format!("read CodeRankEmbed vocabulary from {}", path.display()))?;
            (None, vocab, None, false, DEFAULT_TOKEN_BUDGET_CPU)
        };
        let tokenizer = WordPiece::from_gguf(&vocab.tokens, &vocab.token_types, vocab.unk, vocab.cls, vocab.sep);

        let mut embedder = Self {
            engine,
            tokenizer,
            pad_id: vocab.pad,
            max_seq,
            device,
            flash,
            token_budget,
            #[cfg(target_os = "macos")]
            coreml,
            query_cache: Mutex::new(QueryCache::new()),
            disk_query_cache: None,
            disk_query_hits: AtomicU64::new(0),
            query_model_calls: AtomicU64::new(0),
            fallbacks: Vec::new(),
        };
        // A model is ready only after a successful, validated prediction.
        // CoreML can report an E5RT exception yet return NaN output as success.
        if let Err(err) = embedder.embed_query("warmup") {
            #[cfg(target_os = "macos")]
            if embedder.coreml.is_some() {
                eprintln!("note: CoreML unavailable ({err:#}); retrying CodeRankEmbed with ggml (semantic search retained)");
                drop(embedder);
                let mut retry = Self::load_with_backend(path, kind, false)
                    .with_context(|| format!("ggml retry after CoreML initialization failure ({err:#})"))?;
                retry.fallbacks.push(format!("CoreML initialization failed: {err:#}"));
                return Ok(retry);
            }
            return Err(err.context("CodeRankEmbed warmup failed"));
        }
        // Warmup must validate actual inference, even when its text is cached.
        // Attach the disk cache only afterwards. MCP reuses its indexing
        // encoder for queries; caching is attached to embed_query, never to
        // document embedding, regardless of the instance's sequence cap.
        embedder.query_model_calls.store(0, Ordering::Relaxed);
        if env_flag("HIPS_QUERY_CACHE", true) {
            let root = std::env::var_os("HIPS_QUERY_CACHE_DIR").map(PathBuf::from)
                .unwrap_or_else(|| crate::codeindex::cache_root().join("query-vectors"));
            let cache = (|| -> anyhow::Result<_> {
                let recipe = serde_json::json!({"version":1,"model":MODEL_ID,
                    "model_sha256":crate::query_cache::model_digest(path)?,
                    "prefix":QUERY_PREFIX,"pooling":"cls","dimensions":CODERANK_DIM,
                    "max_tokens":max_seq,"backend":if embedder.coreml_active() { "coreml" } else { "ggml" },
                    "device":embedder.device,"flash":embedder.flash,
                    "cls_only":env_flag("HIPS_GGML_CLS_ONLY",true),
                    "cast_kv":env_flag("HIPS_GGML_CAST_KV",cfg!(target_os="macos"))});
                crate::query_cache::DiskQueryCache::new(&root, &recipe)
            })();
            match cache {
                Ok(cache) => embedder.disk_query_cache = Some(cache),
                Err(e) => eprintln!("note: persistent query cache unavailable ({e:#})"),
            }
        }
        Ok(embedder)
    }

    fn coreml_active(&self) -> bool {
        #[cfg(target_os = "macos")]
        { self.coreml.is_some() }
        #[cfg(not(target_os = "macos"))]
        { false }
    }

    /// Inspect the resident encoder without loading a model or running inference.
    pub fn diagnostics(&self) -> serde_json::Value {
        #[cfg(target_os = "macos")]
        let coreml = self.coreml.as_ref().map(|cm| cm.diagnostics());
        #[cfg(not(target_os = "macos"))]
        let coreml: Option<serde_json::Value> = None;
        serde_json::json!({
            "state": "ready",
            "model": MODEL_ID,
            "backend": if coreml.is_some() { "coreml" } else { "ggml" },
            "ggml_device": self.device,
            "flash_attention": self.flash,
            "coreml": coreml,
            "warmup_validated": true,
            "fallbacks": self.fallbacks,
            "query_cache_entries": self.query_cache.lock().ok().map(|c| c.map.len()),
            "query_disk_cache": self.disk_query_cache.as_ref().map(|c| c.directory()),
            "query_disk_hits": self.disk_query_hits.load(Ordering::Relaxed),
            "query_model_calls": self.query_model_calls.load(Ordering::Relaxed),
        })
    }

    pub fn query_cached(&self, query: &str) -> bool {
        self.query_cache.lock().is_ok_and(|c| c.map.contains_key(query))
            || self.disk_query_cache.as_ref().is_some_and(|c| c.get(query).is_some())
    }

    /// Token ids for one text under this instance's sequence cap
    /// (`[CLS] ... [SEP]` included).
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, self.max_seq)
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
        if let Some(vector) = self.disk_query_cache.as_ref().and_then(|c| c.get(query)) {
            self.disk_query_hits.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut cache) = self.query_cache.lock() { cache.put(query.into(), vector.clone()); }
            return Ok(vector);
        }
        self.query_model_calls.fetch_add(1, Ordering::Relaxed);
        let prefixed = format!("{QUERY_PREFIX}{query}");
        let mut batch = self.embed_batch(&[prefixed.as_str()])?;
        let vector = batch.pop().expect("one query");
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.put(query.to_string(), vector.clone());
        }
        if let Some(cache) = &self.disk_query_cache {
            if let Err(e) = cache.enqueue(query, &vector) {
                if crate::verbosity::verbose() { eprintln!("note: query cache enqueue failed ({e:#})"); }
            }
        }
        Ok(vector)
    }

    pub fn embed_docs(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_batch(texts)
    }

    /// Sequences per graph. `HPS_EMBED_BATCH` overrides.
    pub fn batch_size() -> usize {
        std::env::var("HPS_EMBED_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&b: &usize| b > 0)
            .unwrap_or(DEFAULT_BATCH)
    }

    /// Cap on padded tokens (`longest sequence x sequences`) per graph.
    /// `HPS_EMBED_TOKENS` overrides the backend default.
    fn token_budget(&self) -> usize {
        std::env::var("HPS_EMBED_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&b: &usize| b > 0)
            .unwrap_or(self.token_budget)
    }

    /// The batch size for this instance's active backend: the CoreML
    /// document models' compiled batch when CoreML is serving, else the
    /// ggml batch.
    pub fn effective_batch(&self) -> usize {
        #[cfg(target_os = "macos")]
        if let Some(cm) = &self.coreml {
            return cm.doc_batch();
        }
        Self::batch_size()
    }

    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_docs_with(texts, self.effective_batch(), true)
    }

    /// Encode with an explicit batch size and optional length bucketing.
    ///
    /// A graph costs the length of its longest sequence times its size, so
    /// sorting by token count before batching groups similar lengths and
    /// stops one 512-token chunk from padding everything batched with it.
    /// Results are scattered back into the caller's order.
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
        let t0 = std::time::Instant::now();
        // WordPiece is cheap per text but a batch of thousands adds up next
        // to a GPU that encodes 100k tokens a second: spread it over cores.
        // Only the tokenizer crosses threads: the embedder as a whole is not
        // Sync on macOS (CoreML's model handles are single-threaded).
        let tokens: Vec<Vec<u32>> = {
            use rayon::prelude::*;
            let (tokenizer, max_seq) = (&self.tokenizer, self.max_seq);
            texts.par_iter().map(|t| tokenizer.encode(t, max_seq)).collect()
        };
        if matches!(std::env::var("HPS_EMBED_PROFILE").as_deref(), Ok("1") | Ok("true")) {
            let n: usize = tokens.iter().map(Vec::len).sum();
            eprintln!("  tokenize {} texts, {n} tokens: {:.3} ms", texts.len(), t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut order: Vec<usize> = (0..texts.len()).collect();
        if bucket_by_length {
            order.sort_by_key(|&i| tokens[i].len());
        }
        let mut out = vec![Vec::new(); texts.len()];
        // Largest batch first: the scheduler's buffers grow to their final
        // size once instead of on every batch.
        for group in plan_groups(&order, |i| tokens[i].len(), batch, self.token_budget()).into_iter().rev() {
            let rows: Vec<&[u32]> = group.iter().map(|&i| tokens[i].as_slice()).collect();
            let vectors = self.embed_encodings(&rows)?;
            anyhow::ensure!(vectors.len() == group.len(), "embedding batch size mismatch");
            for (&i, vector) in group.iter().zip(vectors) {
                out[i] = vector;
            }
        }
        Ok(out)
    }

    fn embed_encodings(&self, rows: &[&[u32]]) -> anyhow::Result<Vec<Vec<f32>>> {
        #[cfg(target_os = "macos")]
        if let Some(cm) = &self.coreml {
            if rows.len() == 1 {
                if let Some(result) = cm.encode_query(rows[0]) {
                    return Ok(vec![result?]);
                }
            }
            let owned: Vec<Vec<u32>> = rows.iter().map(|r| r.to_vec()).collect();
            let mut out = Vec::with_capacity(rows.len());
            for group in owned.chunks(cm.doc_batch()) {
                out.extend(cm.encode_docs(group)?);
            }
            return Ok(out);
        }
        let engine = self
            .engine
            .as_ref()
            .context("ggml backend not loaded (CoreML-only instance)")?;
        let raw = engine
            .lock()
            .map_err(|_| anyhow::anyhow!("encoder lock poisoned"))?
            .forward(rows, self.pad_id)
            .context("CodeRankEmbed forward")?;
        raw.into_iter()
            .map(|v| {
                let v = l2_normalize(v);
                validate_vector(&v).context("invalid ggml prediction")?;
                Ok(v)
            })
            .collect()
    }
}

/// Consecutive slices of `order` with at most `batch` sequences and at most
/// `budget` padded tokens (longest x count) each.
fn plan_groups(order: &[usize], len_of: impl Fn(usize) -> usize, batch: usize, budget: usize) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut longest = 0usize;
    for &i in order {
        let len = len_of(i);
        let next_longest = longest.max(len);
        if !current.is_empty() && (current.len() >= batch || next_longest * (current.len() + 1) > budget) {
            groups.push(std::mem::take(&mut current));
            longest = 0;
        }
        longest = longest.max(len);
        current.push(i);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// `HPS_EMBED_DEVICE=cpu` forces the CPU; `gpu`, `vulkan` and `metal` ask
/// for the GPU (the default when one is available).
fn pick_gpu() -> bool {
    !matches!(std::env::var("HPS_EMBED_DEVICE").as_deref(), Ok("cpu"))
}

/// Both backends promise finite, L2-normalized CodeRankEmbed output. Reject
/// broken predictions before they enter the query cache, index, or ranking.
pub(crate) fn validate_vector(vector: &[f32]) -> anyhow::Result<()> {
    anyhow::ensure!(vector.len() == CODERANK_DIM, "unexpected embedding dimension {}", vector.len());
    anyhow::ensure!(vector.iter().all(|v| v.is_finite()), "embedding contains non-finite values (NaN or infinity)");
    let norm: f64 = vector.iter().map(|&v| (v as f64).powi(2)).sum();
    anyhow::ensure!((0.9..=1.1).contains(&norm), "embedding is not L2-normalized (squared norm {norm})");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_invalid_predictions_before_ranking_or_caching() {
        let mut vector = vec![0.0; CODERANK_DIM];
        assert!(validate_vector(&vector).is_err());
        vector[0] = 1.0;
        validate_vector(&vector).unwrap();
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0] {
            vector[0] = invalid;
            assert!(validate_vector(&vector).is_err());
        }
        assert!(validate_vector(&[1.0]).is_err());
    }

    #[test]
    fn groups_respect_batch_and_padded_token_budget() {
        let lens = [3usize, 3, 500, 510, 512, 512, 512];
        let order: Vec<usize> = (0..lens.len()).collect();
        let groups = plan_groups(&order, |i| lens[i], 4, 1100);
        // Two short ones fit together; each ~512-token one is over half the
        // budget, so at most two share a graph.
        assert_eq!(groups, vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6]]);
        let groups = plan_groups(&order, |i| lens[i], 2, usize::MAX);
        assert_eq!(groups.len(), 4);
        // A single sequence always gets a group, even over budget.
        assert_eq!(plan_groups(&[0], |_| 512, 8, 10), vec![vec![0]]);
    }

    use crate::test_util::ENV;

    struct EnvGuard(Vec<(&'static str, Option<std::ffi::OsString>)>);
    impl EnvGuard {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let saved = vars.iter().map(|(k, _)| (*k, std::env::var_os(k))).collect();
            for (k, v) in vars {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
            Self(saved)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in self.0.drain(..) {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    #[test]
    fn model_path_resolution_order() {
        let _lock = ENV.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let explicit = dir.path().join("explicit.gguf");
        std::fs::write(&explicit, b"GGUF").unwrap();
        // An explicit path wins and must exist.
        let _g = EnvGuard::set(&[
            ("HIPS_MODEL_PATH", Some(explicit.to_str().unwrap())),
            ("HIPS_MODEL_OFFLINE", Some("1")),
            ("HIPS_MODEL_URL", None),
            ("CSEARCH_CACHE_DIR", Some(cache.to_str().unwrap())),
        ]);
        assert_eq!(model_path().unwrap(), explicit);
        let _g = EnvGuard::set(&[("HIPS_MODEL_PATH", Some(dir.path().join("missing.gguf").to_str().unwrap()))]);
        assert!(model_path().unwrap_err().to_string().contains("is not a file"));
        // Empty counts as unset; offline with an empty cache is a clear error.
        let _g = EnvGuard::set(&[("HIPS_MODEL_PATH", Some(""))]);
        let err = model_path().unwrap_err().to_string();
        assert!(err.contains("HIPS_MODEL_OFFLINE=1") && err.contains(MODEL_FILE), "{err}");
        // A cached file under the csearch cache is used without any network.
        std::fs::create_dir_all(cache.join("models")).unwrap();
        std::fs::write(cache.join("models").join(MODEL_FILE), b"GGUF").unwrap();
        assert_eq!(model_path().unwrap(), cache.join("models").join(MODEL_FILE));
    }

    #[test]
    fn device_and_flag_parsing() {
        let _lock = ENV.lock().unwrap_or_else(|e| e.into_inner());
        let _g = EnvGuard::set(&[("HPS_EMBED_DEVICE", None), ("HIPS_GGML_FLASH", None)]);
        assert!(pick_gpu());
        assert!(env_flag("HIPS_GGML_FLASH", true));
        assert!(!env_flag("HIPS_GGML_FLASH", false));
        let _g = EnvGuard::set(&[("HPS_EMBED_DEVICE", Some("cpu")), ("HIPS_GGML_FLASH", Some("off"))]);
        assert!(!pick_gpu());
        assert!(!env_flag("HIPS_GGML_FLASH", true));
        let _g = EnvGuard::set(&[("HPS_EMBED_DEVICE", Some("vulkan")), ("HIPS_GGML_FLASH", Some("1"))]);
        assert!(pick_gpu());
        assert!(env_flag("HIPS_GGML_FLASH", false));
        let _g = EnvGuard::set(&[("HIPS_GGML_FLASH", Some("garbage"))]);
        assert!(env_flag("HIPS_GGML_FLASH", true), "unknown values keep the default");
    }

    #[test]
    fn group_planning_edge_cases() {
        assert!(plan_groups(&[], |_| 1, 4, 100).is_empty());
        // batch = 1 is one sequence per graph regardless of budget.
        assert_eq!(plan_groups(&[0, 1, 2], |_| 5, 1, 1000), vec![vec![0], vec![1], vec![2]]);
        // The longest sequence so far sets the padded cost of the whole group.
        let lens = [10usize, 10, 100];
        assert_eq!(plan_groups(&[0, 1, 2], |i| lens[i], 8, 150), vec![vec![0, 1], vec![2]]);
        // Every index appears exactly once, in order.
        let order: Vec<usize> = (0..57).collect();
        let groups = plan_groups(&order, |i| 1 + i % 13, 5, 40);
        let flat: Vec<usize> = groups.iter().flatten().copied().collect();
        assert_eq!(flat, order);
        for g in &groups {
            assert!(g.len() <= 5);
            let longest = g.iter().map(|&i| 1 + i % 13).max().unwrap();
            assert!(g.len() == 1 || longest * g.len() <= 40, "{g:?}");
        }
    }

    /// Against vectors from the original fp32 implementation, and the
    /// batching paths against each other. Skips without a model.
    #[test]
    fn matches_reference_vectors_and_batching_is_transparent() {
        let Some(path) = crate::test_util::model() else { return };
        let _lock = ENV.lock().unwrap_or_else(|e| e.into_inner());
        let cache = tempfile::tempdir().unwrap();
        let _g = EnvGuard::set(&[("HIPS_QUERY_CACHE_DIR", Some(cache.path().to_str().unwrap())), ("HIPS_QUERY_CACHE", Some("1")), ("HPS_EMBED_MAX_SEQ", Some("512"))]);
        let fixture: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/coderank_reference.json")).unwrap();
        let rows = |key: &str| -> Vec<(String, Vec<f32>)> {
            fixture[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| (r["text"].as_str().unwrap().to_string(), r["vec"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()))
                .collect()
        };
        let cosine = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();

        // This test measures the ggml engine: on a Mac with compiled CoreML
        // shapes installed the default loader would pick CoreML instead.
        let index = Embedder::load_with_backend(&path, EmbedUse::Index, false).unwrap();
        assert_eq!(index.diagnostics()["backend"], "ggml");
        assert_eq!(index.diagnostics()["warmup_validated"], true);
        let docs = rows("docs");
        let texts: Vec<&str> = docs.iter().map(|(t, _)| t.as_str()).collect();
        let bucketed = index.embed_docs_with(&texts, 32, true).unwrap();
        let single = index.embed_docs_with(&texts, 1, false).unwrap();
        for (i, (text, expected)) in docs.iter().enumerate() {
            validate_vector(&bucketed[i]).unwrap();
            let c = cosine(&bucketed[i], expected);
            assert!(c > 0.9995, "doc {i} ({}...): cosine {c}", &text[..30]);
            assert!(cosine(&bucketed[i], &single[i]) > 0.99999, "batching changed doc {i}");
        }
        assert_eq!(index.tokenize(&docs[0].0)[0], 101);

        let query = Embedder::load_with_backend(&path, EmbedUse::Query, false).unwrap();
        for (i, (text, expected)) in rows("queries").iter().enumerate() {
            assert!(!query.query_cached(text));
            let v = query.embed_query(text).unwrap();
            validate_vector(&v).unwrap();
            let c = cosine(&v, expected);
            assert!(c > 0.9995, "query {i}: cosine {c}");
            assert!(query.query_cached(text));
            assert_eq!(query.embed_query(text).unwrap(), v, "cache returns the same vector");
        }
        // A query longer than the query cap is truncated, not rejected.
        let long = "token ".repeat(500);
        assert!(query.tokenize(&long).len() <= MAX_QUERY_SEQ);
        validate_vector(&query.embed_query(&long).unwrap()).unwrap();
        // Empty input is a no-op, not an error.
        assert!(index.embed_docs(&[]).unwrap().is_empty());
        crate::flush_query_cache_writes();
    }

    #[test]
    fn persistent_query_lookup_misses_on_complete_issue_and_survives_reload() {
        let Some(path) = crate::test_util::model() else { return };
        let _lock = ENV.lock().unwrap_or_else(|e| e.into_inner());
        let cache = tempfile::tempdir().unwrap();
        let _g = EnvGuard::set(&[("HIPS_QUERY_CACHE_DIR", Some(cache.path().to_str().unwrap())), ("HIPS_QUERY_CACHE", Some("1")), ("HPS_EMBED_MAX_SEQ", Some("512"))]);
        let encoder = Embedder::load_with_backend(&path, EmbedUse::Query, false).unwrap();
        let short = "user";
        assert!(!encoder.query_cached(short));
        encoder.embed_query(short).unwrap();
        let issue = "user ".repeat(100);
        // Every individual word is cached. The complete issue still misses.
        assert!(!encoder.query_cached(&issue));
        let prefix = format!("{QUERY_PREFIX}{issue}");
        let expected = encoder.embed_docs(&[&prefix]).unwrap().pop().unwrap();
        let calls = encoder.query_model_calls.load(Ordering::Relaxed);
        let first = encoder.embed_query(&issue).unwrap();
        assert_eq!(first, expected);
        assert_eq!(encoder.query_model_calls.load(Ordering::Relaxed), calls + 1);
        assert_eq!(encoder.embed_query(&issue).unwrap(), expected);
        assert_eq!(encoder.query_model_calls.load(Ordering::Relaxed), calls + 1);
        crate::flush_query_cache_writes();
        drop(encoder);
        let restarted = Embedder::load_with_backend(&path, EmbedUse::Query, false).unwrap();
        assert!(restarted.query_cached(&issue));
        assert_eq!(restarted.embed_query(&issue).unwrap(), expected);
        assert_eq!(restarted.query_model_calls.load(Ordering::Relaxed), 0);
        assert_eq!(restarted.disk_query_hits.load(Ordering::Relaxed), 1);
        let index_encoder = Embedder::load_with_backend(&path, EmbedUse::Index, false).unwrap();
        let index_cache = index_encoder.disk_query_cache.as_ref().unwrap();
        assert_ne!(index_cache.directory(), restarted.disk_query_cache.as_ref().unwrap().directory());
        assert!(!index_encoder.query_cached(&issue));
        assert!(!index_cache.directory().exists());
        index_encoder.embed_docs(&["indexing documents does not populate query entries"]).unwrap();
        assert!(!index_cache.directory().exists());
        // This is MCP's shared-encoder path. Only the search call populates it.
        let shared_vector = index_encoder.embed_query(&issue).unwrap();
        crate::flush_query_cache_writes();
        drop(index_encoder);
        let shared_restart = Embedder::load_with_backend(&path, EmbedUse::Index, false).unwrap();
        assert!(shared_restart.query_cached(&issue));
        assert_eq!(shared_restart.embed_query(&issue).unwrap(), shared_vector);
        assert_eq!(shared_restart.query_model_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn normalization_and_cache_order() {
        let v = l2_normalize(vec![3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut cache = QueryCache::new();
        for i in 0..QUERY_CACHE_CAP + 1 {
            cache.put(format!("q{i}"), vec![i as f32]);
        }
        assert!(cache.get("q0").is_none(), "oldest entry evicted");
        assert_eq!(cache.get("q1"), Some(vec![1.0]));
    }
}
