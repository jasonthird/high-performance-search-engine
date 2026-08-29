//! The retrieval engine facade: ranked (hybrid/semantic) query
//! execution over either index layout.
//!
//! This module owns the query-time orchestration — `SearchOpts`,
//! `RankMode`, and the `run_ranked*` pipeline — so the CLI, the MCP
//! server, and the HTTP API all call the same engine instead of each
//! other. `cli` re-exports these names for backward compatibility.

#[cfg(feature = "semantic")]
use std::time::Instant;

#[cfg(feature = "semantic")]
use anyhow::Context;
use clap::ValueEnum;

use crate::searcher;
#[cfg(feature = "semantic")]
use crate::storage;

// Fusion defaults (weighted, alpha 0.15) come from two eval sets over 31k
// real code chunks (examples in `eval-gen`): R@10 on natural-language
// doc-comment queries / on identifier queries —
//   bm25 .255/.976   semantic .716/.974   rrf hybrid .569/.975
//   weighted alpha=.15 hybrid: .696/.993 — best on identifiers, within 3%
// of pure semantic on NL, and keeps a lexical anchor for text the encoder
// cannot see (string literals, config files, tails of >512-token chunks).

/// Whether ADC scoring is used. `Auto` defers to [`crate::pq::worth_using`]:
/// below the break-even candidate count PQ costs recall and saves nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PqMode {
    Auto,
    Off,
    /// Use ADC regardless of candidate count (benchmarking).
    Force,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum FusionArg {
    Weighted,
    Rrf,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum RankMode {
    Bm25,
    /// Encoder retrieves over all vectors; BM25 is a helper on the same doc_ids.
    Hybrid,
    /// Old retrieve-then-rerank: cosine only on BM25 candidates.
    Rerank,
    Semantic,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct SearchOpts {
    pub mode: RankMode,
    pub semantic_candidates: usize,
    pub fusion: FusionArg,
    pub alpha: f32,
    pub rrf_k: f32,
    pub nprobe: usize,
    pub pq: PqMode,
}

/// Pick the scoring path: ADC only when enough documents will be scored to
/// amortize its table build. See [`crate::pq::MIN_CANDIDATES`] for the
/// measurements behind the threshold.
#[cfg(feature = "semantic")]
fn choose_pq<'a>(
    index: &'a searcher::AnyIndex,
    opts: &SearchOpts,
    top_k: usize,
) -> Option<&'a crate::pq::PqIndex> {
    let pq = index.pq()?;
    match opts.pq {
        PqMode::Off => None,
        PqMode::Force => Some(pq),
        // Measured on 31k chunks of real code (examples/pq_tradeoff.rs):
        // IVF-only semantic recall@10 was 0.677, IVF+PQ collapsed it to
        // 0.143 — this PQ quantizes raw vectors (not residuals), and the
        // reconstruction error swamps the score differences that matter.
        // Meanwhile exact scoring of all 31k vectors costs ~16 ms with the
        // vectorized f16 path. So ADC is never chosen automatically; it
        // remains available via Force for benchmarks and future
        // residual-quantization work.
        PqMode::Auto => {
            let _ = top_k;
            None
        }
    }
}

#[cfg(feature = "semantic")]
fn fusion_from(opts: &SearchOpts) -> crate::hybrid::Fusion {
    match opts.fusion {
        FusionArg::Weighted => crate::hybrid::Fusion::weighted(opts.alpha),
        FusionArg::Rrf => crate::hybrid::Fusion::rrf(opts.rrf_k),
    }
}

pub struct RankedHit {
    pub id: String,
    pub title: String,
    pub score: f32,
    pub bm25: f32,
    pub semantic: f32,
}

pub struct RankedRun {
    pub results: Vec<RankedHit>,
    pub stats: crate::block_max_wand::SearchStats,
    pub bm25_ms: f64,
    pub embed_ms: f64,
    pub score_ms: f64,
    pub total_ms: f64,
}

pub fn require_semantic() -> anyhow::Result<()> {
    if cfg!(feature = "semantic") {
        Ok(())
    } else {
        anyhow::bail!(
            "this binary was built without CodeRankEmbed; rebuild with `cargo build --release --features semantic`"
        )
    }
}

pub fn run_ranked(
    index: &searcher::AnyIndex,
    query: &str,
    top_k: usize,
    opts: &SearchOpts,
) -> anyhow::Result<RankedRun> {
    require_semantic()?;
    #[cfg(feature = "semantic")]
    {
        let embedder = crate::embedder::Embedder::load()?;
        run_ranked_with(index, &embedder, query, top_k, opts)
    }
    #[cfg(not(feature = "semantic"))]
    {
        let _ = (index, query, top_k, opts);
        unreachable!("require_semantic already returned")
    }
}

/// Query embed and BM25 do not depend on each other. Encode on this
/// thread (Metal stays where the model was warmed up) and run WAND on a
/// helper thread; IVF/fusion still wait for the vector.
#[cfg(feature = "semantic")]
fn embed_and_bm25(
    disk: &storage::DiskIndex,
    embedder: &crate::embedder::Embedder,
    query: &str,
    k: usize,
) -> anyhow::Result<(
    Vec<f32>,
    Vec<crate::block_max_wand::SearchHit>,
    crate::block_max_wand::SearchStats,
    f64,
    f64,
)> {
    std::thread::scope(|s| {
        let bm25 = s.spawn(|| searcher::search_hits(disk, query, k));
        let t_embed = Instant::now();
        let qvec = embedder.embed_query(query)?;
        let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
        let (hits, stats, bm25_ms) = bm25
            .join()
            .map_err(|_| anyhow::anyhow!("BM25 helper thread panicked"))?;
        Ok((qvec, hits, stats, bm25_ms, embed_ms))
    })
}

#[cfg(feature = "semantic")]
pub fn run_ranked_with(
    index: &searcher::AnyIndex,
    embedder: &crate::embedder::Embedder,
    query: &str,
    top_k: usize,
    opts: &SearchOpts,
) -> anyhow::Result<RankedRun> {
    use crate::hybrid;
    use crate::indexer::SearchableIndex as _;
    if index.as_single().is_none() {
        return run_ranked_segmented(index, embedder, query, top_k, opts);
    }
    let embeddings = index
        .embeddings()
        .context("no embeddings.bin — run `embed --index ... --input ...` first")?;
    let disk = index
        .as_single()
        .context("hybrid/semantic modes currently support a single (non-segmented) index")?;

    let total = Instant::now();

    let (hits, stats, bm25_ms, embed_ms, score_ms) = match opts.mode {
        RankMode::Bm25 => unreachable!(),
        RankMode::Hybrid => {
            let pool = opts.semantic_candidates.max(top_k);
            let (qvec, bm25_hits, stats, bm25_ms, embed_ms) =
                embed_and_bm25(disk, embedder, query, pool)?;
            let t_score = Instant::now();
            let fused = hybrid::merge_retrieve_ivf(
                &bm25_hits,
                embeddings,
                &qvec,
                fusion_from(opts),
                top_k,
                pool,
                index.ivf(),
                opts.nprobe,
                choose_pq(index, opts, top_k),
            );
            let score_ms = t_score.elapsed().as_secs_f64() * 1000.0;
            (fused, stats, bm25_ms, embed_ms, score_ms)
        }
        RankMode::Rerank => {
            let candidates = opts.semantic_candidates.max(top_k);
            let (qvec, bm25_hits, stats, bm25_ms, embed_ms) =
                embed_and_bm25(disk, embedder, query, candidates)?;
            let t_score = Instant::now();
            let fused = hybrid::rerank(&bm25_hits, embeddings, &qvec, fusion_from(opts), top_k);
            let score_ms = t_score.elapsed().as_secs_f64() * 1000.0;
            (fused, stats, bm25_ms, embed_ms, score_ms)
        }
        RankMode::Semantic => {
            let t_embed = Instant::now();
            let qvec = embedder.embed_query(query)?;
            let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
            let t_score = Instant::now();
            let fused = hybrid::semantic_pool(
                embeddings,
                &qvec,
                top_k,
                index.ivf(),
                opts.nprobe,
                choose_pq(index, opts, top_k),
            );
            let score_ms = t_score.elapsed().as_secs_f64() * 1000.0;
            (
                fused,
                crate::block_max_wand::SearchStats {
                    num_docs_total: disk.num_docs(),
                    ..Default::default()
                },
                0.0,
                embed_ms,
                score_ms,
            )
        }
    };
    let results = hits
        .into_iter()
        .map(|h| {
            let s = disk.doc_summary(h.doc_id);
            RankedHit {
                id: s.id,
                title: s.title,
                score: h.score,
                bm25: h.bm25,
                semantic: h.semantic,
            }
        })
        .collect();
    Ok(RankedRun {
        results,
        stats,
        bm25_ms,
        embed_ms,
        score_ms,
        total_ms: total.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Ranked retrieval over a segmented index: BM25 across segments plus an
/// exact per-segment brute-force semantic pool, fused like the single-index
/// path. No IVF/PQ — segments stay small between merges, and exact scoring
/// measured better on both recall and simplicity at repo scale.
#[cfg(feature = "semantic")]
fn run_ranked_segmented(
    index: &searcher::AnyIndex,
    embedder: &crate::embedder::Embedder,
    query: &str,
    top_k: usize,
    opts: &SearchOpts,
) -> anyhow::Result<RankedRun> {
    use crate::hybrid;

    let seg = match index.kind() {
        searcher::IndexKind::Segmented(seg) => seg,
        searcher::IndexKind::Single(_) => unreachable!("caller checked"),
    };
    let stores = index
        .segment_stores()
        .context("segmented index has no embeddings; rebuild with --segmented (not --lexical)")?;

    let total = Instant::now();
    let pool = opts.semantic_candidates.max(top_k);

    // Query embed and BM25 in parallel, mirroring the single-index path.
    let (qvec, bm25_hits, embed_ms, bm25_ms) = std::thread::scope(|scope| {
        let bm25 = scope.spawn(|| {
            let t = Instant::now();
            let hits = if opts.mode == RankMode::Semantic {
                Vec::new()
            } else {
                seg.search_hits_raw(query, pool)
            };
            (hits, t.elapsed().as_secs_f64() * 1000.0)
        });
        let t = Instant::now();
        let qvec = embedder.embed_query(query)?;
        let embed_ms = t.elapsed().as_secs_f64() * 1000.0;
        let (hits, bm25_ms) = bm25
            .join()
            .map_err(|_| anyhow::anyhow!("BM25 helper thread panicked"))?;
        Ok::<_, anyhow::Error>((qvec, hits, embed_ms, bm25_ms))
    })?;

    let t_score = Instant::now();
    let live = |si: usize, doc: u32| seg.is_live(si, doc);
    let hits = match opts.mode {
        RankMode::Semantic => {
            let mut sem = hybrid::segmented_semantic_pool(stores, &live, &qvec, top_k);
            sem.truncate(top_k);
            sem
        }
        RankMode::Rerank => {
            // Cosine only on the BM25 candidates.
            hybrid::segmented_fuse(&bm25_hits, Vec::new(), stores, &qvec, fusion_from(opts), top_k)
        }
        _ => {
            let sem = hybrid::segmented_semantic_pool(stores, &live, &qvec, pool);
            hybrid::segmented_fuse(&bm25_hits, sem, stores, &qvec, fusion_from(opts), top_k)
        }
    };
    let score_ms = t_score.elapsed().as_secs_f64() * 1000.0;

    let results = hits
        .into_iter()
        .map(|h| {
            let summary = seg.doc_summary_in(h.segment, h.doc_id);
            RankedHit {
                id: summary.id,
                title: summary.title,
                score: h.score,
                bm25: h.bm25,
                semantic: h.semantic,
            }
        })
        .collect();
    Ok(RankedRun {
        results,
        stats: crate::block_max_wand::SearchStats {
            num_docs_total: index.num_docs() as usize,
            ..Default::default()
        },
        bm25_ms,
        embed_ms,
        score_ms,
        total_ms: total.elapsed().as_secs_f64() * 1000.0,
    })
}

/// The retrieval knobs shared by every searching subcommand; flattened
/// into each with `#[command(flatten)]` so adding a knob is one edit.
#[derive(clap::Args, Clone, Copy)]
pub struct FusionArgs {
    /// Pool size: BM25 hits kept as helper, and encoder neighbors kept,
    /// before fusion (`hybrid` / `rerank`).
    #[arg(long, default_value_t = 200)]
    pub semantic_candidates: usize,
    /// Score fusion: min-max weighted mix, or reciprocal rank fusion.
    #[arg(long, value_enum, default_value_t = FusionArg::Weighted)]
    pub fusion: FusionArg,
    /// BM25 weight for `--fusion weighted` (semantic weight is 1-alpha).
    #[arg(long, default_value_t = 0.15)]
    pub alpha: f32,
    /// RRF constant k (Cormack et al.).
    #[arg(long, default_value_t = 60.0)]
    pub rrf_k: f32,
    /// IVF lists to probe (0 = auto). Encoder retrieval uses the same
    /// inverted-file cluster postings as stored in `ivf.bin`.
    #[arg(long, default_value_t = 0)]
    pub nprobe: usize,
    /// Kept for compatibility: PQ scoring is never auto-selected (measured
    /// recall collapse — see `choose_pq`), so search behaves the same with
    /// or without this flag. `PqMode::Force` remains for benchmarks.
    #[arg(long)]
    pub no_pq: bool,
}

impl Default for SearchOpts {
    /// The measured defaults: hybrid retrieval with weighted fusion,
    /// alpha 0.15 — same values the CLI flags default to.
    fn default() -> Self {
        Self {
            mode: RankMode::Hybrid,
            semantic_candidates: 200,
            fusion: FusionArg::Weighted,
            alpha: 0.15,
            rrf_k: 60.0,
            nprobe: 0,
            pq: PqMode::Auto,
        }
    }
}

impl FusionArgs {
    pub fn to_opts(&self, mode: RankMode) -> SearchOpts {
        SearchOpts {
            mode,
            semantic_candidates: self.semantic_candidates,
            fusion: self.fusion,
            alpha: self.alpha,
            rrf_k: self.rrf_k,
            nprobe: self.nprobe,
            pq: if self.no_pq { PqMode::Off } else { PqMode::Auto },
        }
    }
}
