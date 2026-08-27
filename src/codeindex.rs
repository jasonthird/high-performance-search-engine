//! Repository indexing: turn a source tree into a searchable index directory,
//! rebuilding cheaply when files change.
//!
//! Hybrid retrieval requires a single (non-segmented) index, because
//! `embeddings.bin` is keyed positionally by inverted-index `doc_id`. So a
//! change to the tree is handled by rebuilding the whole index rather than by
//! appending a segment. That is affordable because the expensive half —
//! CodeRankEmbed inference — is served from [`crate::embcache`], keyed by
//! chunk content: only chunks whose text actually changed are re-encoded.
//!
//! Rebuilds are atomic: the new index is assembled in a sibling temp
//! directory and swapped in, so a concurrently open index is never observed
//! half-written.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::indexer::{self, InputDoc};
use crate::postings::DEFAULT_BLOCK_SIZE;
use crate::reorder::ReorderStrategy;
use crate::{repo, storage};

/// Written next to the index so `index_status` can describe what is loaded.
pub const MANIFEST_FILE: &str = "repo.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Absolute path of the indexed source tree.
    pub root: String,
    pub num_docs: usize,
    pub num_files: usize,
    /// Whether `embeddings.bin` was built, i.e. hybrid search is available.
    pub embedded: bool,
    pub build_secs: f64,
    /// Chunks encoded on this build, and chunks served from the cache.
    pub encoded: usize,
    pub cached: usize,
    /// Fingerprint of the walked file list (path, size, mtime) the index
    /// was built from. A rebuild that walks to the same fingerprint skips
    /// tokenizing, embedding, and installing entirely.
    #[serde(default)]
    pub tree_fingerprint: u64,
}

impl Manifest {
    pub fn load(index_dir: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(index_dir.join(MANIFEST_FILE))
            .with_context(|| format!("no {MANIFEST_FILE} in {}", index_dir.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    fn save(&self, index_dir: &Path) -> anyhow::Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(index_dir.join(MANIFEST_FILE), text)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BuildOpts {
    /// Build embeddings + IVF + PQ so `--mode hybrid` works.
    pub embed: bool,
    pub title_weight: u32,
    pub reorder: ReorderStrategy,
    /// 0 = the IVF default of ~2√N.
    pub ivf_clusters: usize,
    /// Retrain the IVF/PQ quantizer even if the installed one still fits.
    pub retrain: bool,
    /// Print progress to stderr (off for the MCP server, whose stdout is the
    /// protocol channel).
    pub quiet: bool,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            embed: true,
            title_weight: 2,
            reorder: ReorderStrategy::None,
            ivf_clusters: 0,
            retrain: false,
            quiet: false,
        }
    }
}

/// Phase timing for rebuilds, printed when `CSEARCH_TIMING=1`.
///
/// Rebuild latency is the whole point of the incremental path, so it is
/// worth being able to see where a slow rebuild went without a profiler.
pub struct PhaseTimer {
    on: bool,
    last: Instant,
}

impl PhaseTimer {
    pub fn new() -> Self {
        Self {
            on: std::env::var_os("CSEARCH_TIMING").is_some_and(|v| v == "1"),
            last: Instant::now(),
        }
    }

    /// Record the time since the previous mark under `name`.
    pub fn mark(&mut self, name: &str) {
        if self.on {
            eprintln!("  [timing] {name:<12} {:.3}s", self.last.elapsed().as_secs_f64());
        }
        self.last = Instant::now();
    }
}

impl Default for PhaseTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Where a repo's index lives by default: under the user's cache directory,
/// keyed by the absolute path, so indexing a codebase never writes into it.
pub fn default_index_dir(root: &Path) -> PathBuf {
    let abs = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut h = 0xcbf29ce484222325u64;
    for &b in abs.to_string_lossy().as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let name = abs
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "repo".to_string());
    cache_root().join(format!("{name}-{h:016x}"))
}

fn cache_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("CSEARCH_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("csearch")
}

/// Trained IVF centroids and PQ codebooks read back from an existing index.
#[cfg(feature = "semantic")]
struct TrainedQuantizer {
    k: usize,
    dim: usize,
    centroids: Vec<f32>,
    m: usize,
    sub: usize,
    ks_used: usize,
    codebooks: Vec<f32>,
}

/// Builds (and rebuilds) the index for one source tree.
///
/// The encoder is loaded once and reused across rebuilds — model load and
/// Metal pipeline compilation cost far more than a typical incremental
/// rebuild does.
pub struct RepoIndexer {
    root: PathBuf,
    index_dir: PathBuf,
    opts: BuildOpts,
    /// Loaded on first use and kept for the process lifetime. Lazy because a
    /// rebuild that changed nothing needs no encoder at all — and model load
    /// plus Metal pipeline compilation costs about a second, which would
    /// otherwise dominate an incremental rebuild.
    #[cfg(feature = "semantic")]
    embedder: std::cell::OnceCell<crate::embedder::Embedder>,
}

impl RepoIndexer {
    pub fn new(root: &Path, index_dir: &Path, opts: BuildOpts) -> anyhow::Result<Self> {
        let root = root
            .canonicalize()
            .with_context(|| format!("cannot open {}", root.display()))?;
        if opts.embed {
            crate::cli::require_semantic()?;
        }
        Ok(Self {
            root,
            index_dir: index_dir.to_path_buf(),
            opts,
            #[cfg(feature = "semantic")]
            embedder: std::cell::OnceCell::new(),
        })
    }

    /// Load the encoder now rather than on the first cache miss. The MCP
    /// server calls this at startup so that the first edit-then-search does
    /// not pay for model load on top of the rebuild.
    #[cfg(feature = "semantic")]
    pub fn preload_embedder(&self) -> anyhow::Result<()> {
        if self.opts.embed {
            self.embedder()?;
        }
        Ok(())
    }

    #[cfg(not(feature = "semantic"))]
    pub fn preload_embedder(&self) -> anyhow::Result<()> {
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    /// The encoder loaded for indexing, reused for query encoding so the
    /// server does not hold two copies of the model.
    #[cfg(feature = "semantic")]
    pub fn embedder(&self) -> anyhow::Result<&crate::embedder::Embedder> {
        if let Some(loaded) = self.embedder.get() {
            return Ok(loaded);
        }
        self.log("loading CodeRankEmbed...".to_string());
        let loaded = crate::embedder::Embedder::load_for(crate::embedder::EmbedUse::Index)?;
        Ok(self.embedder.get_or_init(|| loaded))
    }

    /// Walk, chunk, index, embed, and atomically install the result.
    pub fn build(&self) -> anyhow::Result<Manifest> {
        self.build_with(self.opts.retrain)
    }

    /// As [`Self::build`], but forcing the IVF/PQ quantizer to be retrained.
    /// Worth doing after the tree has changed substantially; a normal
    /// rebuild reuses it.
    pub fn build_with(&self, retrain: bool) -> anyhow::Result<Manifest> {
        let start = Instant::now();
        let mut timer = PhaseTimer::new();
        let files = repo::walk(&self.root)?;
        timer.mark("walk");
        // Byte-identical tree: nothing downstream can change, so the only
        // cost of an up-to-date `index-repo` is the walk itself.
        let tree_fingerprint = repo::fingerprint(&files);
        if !retrain {
            if let Ok(existing) = Manifest::load(&self.index_dir) {
                if existing.tree_fingerprint == tree_fingerprint
                    && existing.tree_fingerprint != 0
                    && existing.embedded == self.opts.embed
                {
                    self.log(format!(
                        "index up to date ({} chunks, fingerprint unchanged)",
                        existing.num_docs
                    ));
                    return Ok(existing);
                }
            }
        }
        let docs = repo::docs_from_chunks(repo::chunk_files(&files))?;
        timer.mark("chunk");
        anyhow::ensure!(
            !docs.is_empty(),
            "no indexable source files under {} (checked {} extensions)",
            self.root.display(),
            repo::SOURCE_EXTS.len()
        );
        self.log(format!(
            "indexing {} chunks from {} files in {}",
            docs.len(),
            files.len(),
            self.root.display()
        ));

        let staging = self.staging_dir();
        if staging.exists() {
            std::fs::remove_dir_all(&staging).ok();
        }
        std::fs::create_dir_all(&staging)
            .with_context(|| format!("cannot create {}", staging.display()))?;

        let (mut index, embed_texts) = self.build_lexical(&docs)?;
        timer.mark("tokenize");
        storage::save_index(&index, &staging)?;
        timer.mark("save");
        // `save_index` needs the built index; doc_lens are read back for IVF.
        let doc_lens: Vec<u32> = index.docs().iter().map(|d| d.doc_len).collect();
        let _ = &mut index;

        let (encoded, cached) = match embed_texts {
            Some(texts) => self.build_embeddings(&staging, &texts, &doc_lens, retrain)?,
            None => (0, 0),
        };
        timer.mark("embed");

        let manifest = Manifest {
            root: self.root.to_string_lossy().to_string(),
            num_docs: docs.len(),
            num_files: files.len(),
            embedded: self.opts.embed,
            build_secs: start.elapsed().as_secs_f64(),
            encoded,
            cached,
            tree_fingerprint,
        };
        manifest.save(&staging)?;
        self.install(&staging)?;
        timer.mark("install");
        self.log(format!(
            "index ready in {:.2}s ({} chunks, {} encoded, {} from cache)",
            manifest.build_secs, manifest.num_docs, encoded, cached
        ));
        Ok(manifest)
    }

    fn build_lexical(
        &self,
        docs: &[InputDoc],
    ) -> anyhow::Result<(indexer::Index, Option<Vec<String>>)> {
        let mut builder = indexer::IndexBuilder::new(
            true,
            self.opts.title_weight,
            DEFAULT_BLOCK_SIZE,
            self.opts.reorder,
        );
        builder.set_code_mode(true);
        builder.set_keep_embed_text(self.opts.embed);
        // Chunked so tokenization parallelizes without holding every
        // intermediate token vector at once.
        for chunk in docs.chunks(4096) {
            builder.add_documents(chunk);
        }
        let mut index = builder.finish();
        let texts = index.take_embed_texts();
        Ok((index, texts))
    }

    /// Encode only cache misses, then write `embeddings.bin` in doc_id order
    /// plus the IVF and PQ sidecars. Returns (encoded, served-from-cache).
    ///
    /// The sidecars are the reason a naive rebuild is slow: k-means for IVF
    /// and 16 x 256-centroid k-means for PQ cost ~1 s even on a small repo,
    /// and neither depends on which chunk changed. So a rebuild **reuses the
    /// trained quantizer** and only quantizes chunks whose text changed —
    /// everything else comes back from the cache. Training happens on the
    /// first build, and again only when the corpus has drifted far enough
    /// that the old centroids no longer describe it.
    #[cfg(feature = "semantic")]
    fn build_embeddings(
        &self,
        staging: &Path,
        texts: &[String],
        doc_lens: &[u32],
        retrain: bool,
    ) -> anyhow::Result<(usize, usize)> {
        use std::collections::HashSet;

        use crate::embcache::{key_for, EmbedCache};
        use crate::embeddings::CODERANK_DIM;

        // The cache lives with the installed index, not the staging dir, so
        // it survives across rebuilds.
        let mut cache = EmbedCache::load(&self.index_dir, CODERANK_DIM);

        let keys: Vec<u64> = texts.iter().map(|t| key_for(t)).collect();
        let misses: Vec<usize> = (0..texts.len())
            .filter(|&i| !cache.contains(keys[i]))
            .collect();
        let encoded = misses.len();
        let cached = texts.len() - encoded;
        if encoded > 0 {
            self.log(format!(
                "encoding {encoded} changed chunks ({cached} unchanged, from cache)"
            ));
            let embedder = self.embedder()?;
            let mut done = 0usize;
            for batch in misses.chunks(64) {
                let refs: Vec<&str> = batch.iter().map(|&i| texts[i].as_str()).collect();
                let vectors = embedder.embed_docs(&refs)?;
                for (&i, vector) in batch.iter().zip(vectors.iter()) {
                    cache.insert(keys[i], vector);
                }
                done += batch.len();
                if !self.opts.quiet && (done.is_multiple_of(512) || done == encoded) {
                    eprintln!("  encoded {done}/{encoded}");
                }
            }
        }

        let n = texts.len();
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (i, key) in keys.iter().enumerate() {
            let vector = cache
                .get(*key)
                .with_context(|| format!("embedding missing for chunk {i}"))?;
            vectors.push(vector);
        }
        let mut timer = PhaseTimer::new();
        crate::embeddings::write_f16(staging, CODERANK_DIM as u32, &vectors)?;
        let store = crate::embeddings::EmbeddingStore::open(staging)?;
        timer.mark("write_vecs");

        match self.reusable_quantizer(n, retrain) {
            Some(q) => self.quantize_incrementally(staging, doc_lens, &keys, &vectors, &mut cache, q)?,
            None => self.train_quantizer(staging, &store, doc_lens, &keys, &mut cache)?,
        }

        timer.mark("quantize");

        // Trim vectors for chunks that no longer exist, then stage the cache
        // so it is installed together with the index it describes.
        let live: HashSet<u64> = keys.into_iter().collect();
        cache.retain_keys(&live);
        cache.save(staging)?;
        timer.mark("save_cache");
        Ok((encoded, cached))
    }

    /// The trained quantizer from the installed index, if it is still a
    /// reasonable fit for a corpus of `n` chunks.
    ///
    /// Centroids stay usable as the tree is edited — this is the same
    /// train-once/add-many contract a vector index normally offers. They are
    /// only discarded when the corpus size has moved far enough that the
    /// cluster count itself is wrong.
    #[cfg(feature = "semantic")]
    fn reusable_quantizer(&self, n: usize, retrain: bool) -> Option<TrainedQuantizer> {
        if retrain {
            return None;
        }
        let (k, dim, centroids) = crate::ivf::read_centroids(&self.index_dir)?;
        let (m, sub, ks_used, codebooks) = crate::pq::read_codebooks(&self.index_dir)?;
        if dim != crate::embeddings::CODERANK_DIM || k == 0 || k > n {
            return None;
        }
        // Retrain once the ideal cluster count has drifted by more than 2x
        // in either direction: past that, list lengths are badly unbalanced.
        let ideal = crate::ivf::default_num_clusters(n);
        if ideal > k.saturating_mul(2) || k > ideal.saturating_mul(2) {
            return None;
        }
        Some(TrainedQuantizer {
            k,
            dim,
            centroids,
            m,
            sub,
            ks_used,
            codebooks,
        })
    }

    /// Reuse trained centroids and codebooks: quantize only the chunks whose
    /// quantization is not already cached, then rewrite the sidecars.
    #[cfg(feature = "semantic")]
    fn quantize_incrementally(
        &self,
        staging: &Path,
        doc_lens: &[u32],
        keys: &[u64],
        vectors: &[Vec<f32>],
        cache: &mut crate::embcache::EmbedCache,
        q: TrainedQuantizer,
    ) -> anyhow::Result<()> {
        use rayon::prelude::*;

        let n = keys.len();
        cache.set_quantizer(crate::embcache::quantizer_id(&q.centroids, &q.codebooks), q.m);

        let mut assignment = vec![0u32; n];
        let mut codes = vec![0u8; n * q.m];
        let mut fresh: Vec<usize> = Vec::new();
        for i in 0..n {
            match cache.get_quant(keys[i]) {
                Some((cluster, cached)) => {
                    assignment[i] = cluster;
                    codes[i * q.m..(i + 1) * q.m].copy_from_slice(cached);
                }
                None => fresh.push(i),
            }
        }

        // Each fresh chunk costs one pass over the centroids plus one over
        // the codebooks — microseconds, against ~1 s to retrain both.
        let computed: Vec<(usize, u32, Vec<u8>)> = fresh
            .par_iter()
            .map(|&i| {
                let vector = &vectors[i];
                let cluster = crate::ivf::assign_one(&q.centroids, q.k, q.dim, vector);
                let mut code = vec![0u8; q.m];
                crate::pq::encode_one(&q.codebooks, q.m, q.sub, q.ks_used, vector, &mut code);
                (i, cluster, code)
            })
            .collect();
        for (i, cluster, code) in computed {
            assignment[i] = cluster;
            codes[i * q.m..(i + 1) * q.m].copy_from_slice(&code);
            cache.set_quant(keys[i], cluster, &code);
        }

        crate::ivf::write_with_centroids(
            staging,
            doc_lens,
            &q.centroids,
            &assignment,
            n,
            q.dim,
            q.k,
        )?;
        crate::pq::write_with_codebooks(staging, &q.codebooks, &codes, n, q.m, q.sub, q.ks_used)?;
        self.log(format!(
            "quantizer reused: {} of {n} chunks needed clustering",
            fresh.len()
        ));
        Ok(())
    }

    /// Train IVF centroids and PQ codebooks from scratch, then seed the
    /// cache with the quantization they produced so the next rebuild can
    /// take the incremental path.
    #[cfg(feature = "semantic")]
    fn train_quantizer(
        &self,
        staging: &Path,
        store: &crate::embeddings::EmbeddingStore,
        doc_lens: &[u32],
        keys: &[u64],
        cache: &mut crate::embcache::EmbedCache,
    ) -> anyhow::Result<()> {
        let n = keys.len();
        let k = if self.opts.ivf_clusters == 0 {
            crate::ivf::default_num_clusters(n)
        } else {
            self.opts.ivf_clusters
        };
        self.log(format!("training quantizer over {n} chunks ({k} clusters)"));
        crate::ivf::build_and_write(staging, store, doc_lens, k)?;
        crate::pq::build_and_write(staging, store, 16)?;

        // Read back what was just written rather than recomputing it.
        let ivf = crate::ivf::IvfIndex::open(staging)?;
        let pq = crate::pq::PqIndex::open(staging)?;
        let (_, _, centroids) =
            crate::ivf::read_centroids(staging).context("centroids missing after training")?;
        let (m, _, _, codebooks) =
            crate::pq::read_codebooks(staging).context("codebooks missing after training")?;
        cache.set_quantizer(crate::embcache::quantizer_id(&centroids, &codebooks), m);
        for (doc_id, &key) in keys.iter().enumerate() {
            cache.set_quant(key, ivf.cluster_of(doc_id as u32), pq.codes_of(doc_id as u32));
        }
        Ok(())
    }

    #[cfg(not(feature = "semantic"))]
    fn build_embeddings(
        &self,
        _staging: &Path,
        _texts: &[String],
        _doc_lens: &[u32],
        _retrain: bool,
    ) -> anyhow::Result<(usize, usize)> {
        anyhow::bail!("this binary was built without CodeRankEmbed")
    }

    /// Swap the staged directory into place. The old index is moved aside
    /// first so the window where no index exists is a single rename.
    fn install(&self, staging: &Path) -> anyhow::Result<()> {
        if let Some(parent) = self.index_dir.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let retired = self.retired_dir();
        std::fs::remove_dir_all(&retired).ok();
        let had_old = self.index_dir.exists();
        if had_old {
            std::fs::rename(&self.index_dir, &retired).with_context(|| {
                format!("cannot retire old index at {}", self.index_dir.display())
            })?;
        }
        match std::fs::rename(staging, &self.index_dir) {
            Ok(()) => {
                std::fs::remove_dir_all(&retired).ok();
                Ok(())
            }
            Err(e) => {
                // Put the old index back rather than leaving nothing behind.
                if had_old {
                    std::fs::rename(&retired, &self.index_dir).ok();
                }
                Err(e).with_context(|| format!("cannot install {}", self.index_dir.display()))
            }
        }
    }

    fn staging_dir(&self) -> PathBuf {
        sibling(&self.index_dir, ".building")
    }

    fn retired_dir(&self) -> PathBuf {
        sibling(&self.index_dir, ".old")
    }

    fn log(&self, msg: String) {
        if !self.opts.quiet {
            eprintln!("{msg}");
        }
    }
}

fn sibling(dir: &Path, suffix: &str) -> PathBuf {
    let mut name = dir.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    match dir.parent() {
        Some(parent) => parent.join(name),
        None => PathBuf::from(name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_dir_is_stable_and_outside_the_repo() {
        let root = Path::new(".");
        let a = default_index_dir(root);
        let b = default_index_dir(root);
        assert_eq!(a, b);
        let abs = root.canonicalize().unwrap();
        assert!(!a.starts_with(&abs), "{} is inside the repo", a.display());
    }

    #[test]
    fn lexical_build_indexes_this_repo_src() {
        let dir = std::env::temp_dir().join(format!("hpse-repoidx-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let opts = BuildOpts {
            embed: false,
            quiet: true,
            ..Default::default()
        };
        let indexer = RepoIndexer::new(Path::new("src"), &dir, opts).unwrap();
        let manifest = indexer.build().unwrap();
        assert!(manifest.num_files > 5, "{manifest:?}");
        assert!(manifest.num_docs > manifest.num_files, "{manifest:?}");
        assert!(!manifest.embedded);

        let index = crate::searcher::AnyIndex::open(&dir).unwrap();
        let hits = index.search("block max wand pivot", 5);
        assert!(!hits.results.is_empty());
        // Ids must be openable locations.
        let (path, start, end) = repo::parse_id(&hits.results[0].id).unwrap();
        assert!(path.ends_with(".rs"), "{path}");
        assert!(start <= end);
        std::fs::remove_dir_all(&dir).ok();
    }
}
