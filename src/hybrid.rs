//! Shared inverted-index `doc_id`s: BM25 postings and CodeRankEmbed rows.
//!
//! Two query modes, no ANN:
//!
//! - [`rerank`]: lexical first-stage, cosine only on those hits.
//! - [`merge_retrieve`]: cosine over **every** stored vector (embedding is
//!   the retriever), with BM25 as a helper signal on the same doc_ids.

use crate::block_max_wand::SearchHit;
use crate::embeddings::EmbeddingStore;
use crate::ivf::IvfIndex;
use crate::pq::PqIndex;

#[derive(Debug, Clone, Copy)]
pub enum Fusion {
    /// `alpha * norm(bm25) + (1 - alpha) * cosine`.
    Weighted { alpha: f32 },
    /// Reciprocal rank fusion with constant `k` (Cormack et al., 2009).
    Rrf { k: f32 },
}

impl Fusion {
    pub fn weighted(alpha: f32) -> Self {
        Self::Weighted {
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    pub fn rrf(k: f32) -> Self {
        Self::Rrf { k: k.max(0.0) }
    }
}

#[derive(Debug, Clone)]
pub struct HybridHit {
    pub doc_id: u32,
    pub score: f32,
    pub bm25: f32,
    pub semantic: f32,
}

/// Rerank BM25 hits with cosine against precomputed document embeddings.
///
/// `query` must be L2-normalized and of length `embeddings.dim()`.
pub fn rerank(
    bm25_hits: &[SearchHit],
    embeddings: &EmbeddingStore,
    query: &[f32],
    fusion: Fusion,
    k: usize,
) -> Vec<HybridHit> {
    if bm25_hits.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut rows: Vec<HybridHit> = bm25_hits
        .iter()
        .map(|h| HybridHit {
            doc_id: h.doc_id,
            score: 0.0,
            bm25: h.score,
            semantic: embeddings.cosine(h.doc_id, query),
        })
        .collect();
    fuse_and_truncate(&mut rows, fusion, k)
}

/// Embedding-first retrieval with BM25 as a helper, same `doc_id`s.
///
/// Takes the top `pool` encoder neighbors over **all** stored vectors
/// (brute-force mmap scan, no ANN) and unions them with the BM25 hit
/// list. Docs the encoder found that WAND never saw still compete.
/// BM25 is a bonus, not a gate.
pub fn merge_retrieve(
    bm25_hits: &[SearchHit],
    embeddings: &EmbeddingStore,
    query: &[f32],
    fusion: Fusion,
    k: usize,
    pool: usize,
) -> Vec<HybridHit> {
    merge_retrieve_ivf(bm25_hits, embeddings, query, fusion, k, pool, None, 0, None)
}

/// Like [`merge_retrieve`], but the encoder side probes IVF cluster lists
/// (the same inverted-file shape as BM25 postings) instead of scanning
/// every vector. `nprobe == 0` uses the index default.
pub fn merge_retrieve_ivf(
    bm25_hits: &[SearchHit],
    embeddings: &EmbeddingStore,
    query: &[f32],
    fusion: Fusion,
    k: usize,
    pool: usize,
    ivf: Option<&IvfIndex>,
    nprobe: usize,
    pq: Option<&PqIndex>,
) -> Vec<HybridHit> {
    if k == 0 {
        return Vec::new();
    }
    let pool = pool.max(k);
    let sem_hits = semantic_pool(embeddings, query, pool, ivf, nprobe, pq);

    // Union keyed by doc_id. Ranks are positions in each original list
    // (0 = not on that list), which is what RRF needs.
    let mut by_id: std::collections::HashMap<u32, (HybridHit, usize, usize)> =
        std::collections::HashMap::new();
    for (rank, h) in sem_hits.into_iter().enumerate() {
        by_id.insert(h.doc_id, (h, 0, rank + 1));
    }
    for (rank, h) in bm25_hits.iter().enumerate() {
        by_id
            .entry(h.doc_id)
            .and_modify(|(row, br, _)| {
                row.bm25 = h.score;
                *br = rank + 1;
            })
            .or_insert_with(|| {
                (
                    HybridHit {
                        doc_id: h.doc_id,
                        score: 0.0,
                        bm25: h.score,
                        semantic: embeddings.cosine(h.doc_id, query),
                    },
                    rank + 1,
                    0,
                )
            });
    }

    let mut rows: Vec<HybridHit> = Vec::with_capacity(by_id.len());
    let mut bm25_rank: Vec<usize> = Vec::with_capacity(by_id.len());
    let mut sem_rank: Vec<usize> = Vec::with_capacity(by_id.len());
    for (_, (hit, br, sr)) in by_id {
        rows.push(hit);
        bm25_rank.push(br);
        sem_rank.push(sr);
    }
    fuse_union(&mut rows, &bm25_rank, &sem_rank, fusion, k)
}

fn fuse_and_truncate(rows: &mut Vec<HybridHit>, fusion: Fusion, k: usize) -> Vec<HybridHit> {
    let n = rows.len();
    let mut bm25_rank = vec![0usize; n];
    let mut sem_rank = vec![0usize; n];
    match fusion {
        Fusion::Weighted { .. } => {}
        Fusion::Rrf { .. } => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                rows[b]
                    .bm25
                    .total_cmp(&rows[a].bm25)
                    .then(rows[a].doc_id.cmp(&rows[b].doc_id))
            });
            for (rank, &i) in order.iter().enumerate() {
                if rows[i].bm25 > 0.0 {
                    bm25_rank[i] = rank + 1;
                }
            }
            order.sort_by(|&a, &b| {
                rows[b]
                    .semantic
                    .total_cmp(&rows[a].semantic)
                    .then(rows[a].doc_id.cmp(&rows[b].doc_id))
            });
            for (rank, &i) in order.iter().enumerate() {
                sem_rank[i] = rank + 1;
            }
        }
    }
    fuse_union(rows, &bm25_rank, &sem_rank, fusion, k)
}

fn fuse_union(
    rows: &mut Vec<HybridHit>,
    bm25_rank: &[usize],
    sem_rank: &[usize],
    fusion: Fusion,
    k: usize,
) -> Vec<HybridHit> {
    if rows.is_empty() {
        return Vec::new();
    }
    match fusion {
        Fusion::Weighted { alpha } => {
            let (lo, hi) = min_max(rows.iter().filter(|r| r.bm25 > 0.0).map(|r| r.bm25));
            let span = hi - lo;
            for r in rows.iter_mut() {
                let n = if r.bm25 <= 0.0 {
                    0.0
                } else if span <= f32::EPSILON {
                    // All matched docs share one score (or there is a single
                    // match): a lexical hit still outranks a non-match.
                    1.0
                } else {
                    (r.bm25 - lo) / span
                };
                r.score = alpha * n + (1.0 - alpha) * r.semantic;
            }
        }
        Fusion::Rrf { k: rrf_k } => {
            for (i, r) in rows.iter_mut().enumerate() {
                let mut s = 0.0;
                if sem_rank[i] > 0 {
                    s += 1.0 / (rrf_k + sem_rank[i] as f32);
                }
                if bm25_rank[i] > 0 {
                    s += 1.0 / (rrf_k + bm25_rank[i] as f32);
                }
                r.score = s;
            }
        }
    }

    rows.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.doc_id.cmp(&b.doc_id))
    });
    rows.truncate(k);
    std::mem::take(rows)
}

/// Encoder-side candidate pool: IVF cluster lists when present, else a
/// full scan of `embeddings.bin`.
pub fn semantic_pool(
    embeddings: &EmbeddingStore,
    query: &[f32],
    pool: usize,
    ivf: Option<&IvfIndex>,
    nprobe: usize,
    pq: Option<&PqIndex>,
) -> Vec<HybridHit> {
    let docs: Vec<u32> = if let Some(ivf) = ivf {
        ivf.probe(query, nprobe)
    } else {
        (0..embeddings.num_docs()).collect()
    };
    let adc = pq.map(|p| p.prepare(query));
    let mut rows: Vec<HybridHit> = docs
        .into_iter()
        .map(|doc_id| {
            let sem = match (pq, adc.as_ref()) {
                (Some(p), Some(t)) => p.adc(t, doc_id),
                _ => embeddings.cosine(doc_id, query),
            };
            HybridHit {
                doc_id,
                score: sem,
                bm25: 0.0,
                semantic: sem,
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.semantic
            .total_cmp(&a.semantic)
            .then(a.doc_id.cmp(&b.doc_id))
    });
    rows.truncate(pool.max(1));
    rows
}

/// Brute-force cosine over every stored embedding (mmap scan, no ANN).
pub fn brute_force_semantic(
    embeddings: &EmbeddingStore,
    query: &[f32],
    k: usize,
) -> Vec<HybridHit> {
    let n = embeddings.num_docs() as usize;
    let mut rows: Vec<HybridHit> = (0..n)
        .map(|i| {
            let doc_id = i as u32;
            let sem = embeddings.cosine(doc_id, query);
            HybridHit {
                doc_id,
                score: sem,
                bm25: 0.0,
                semantic: sem,
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.doc_id.cmp(&b.doc_id))
    });
    rows.truncate(k);
    rows
}

fn min_max(vals: impl Iterator<Item = f32>) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for v in vals {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !lo.is_finite() {
        (0.0, 1.0)
    } else {
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{l2_normalize, write_f16, EmbeddingStore};
    use crate::block_max_wand::SearchHit;

    fn store() -> (std::path::PathBuf, EmbeddingStore) {
        store_named("hyb")
    }

    fn store_named(tag: &str) -> (std::path::PathBuf, EmbeddingStore) {
        let dir = std::env::temp_dir().join(format!(
            "high-performance-search-engine-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // doc 0 aligned with query, doc 1 orthogonal, doc 2 in between
        let mut d0 = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut d1 = vec![0.0, 1.0, 0.0, 0.0];
        let mut d2 = vec![0.6, 0.8, 0.0, 0.0];
        l2_normalize(&mut d0);
        l2_normalize(&mut d1);
        l2_normalize(&mut d2);
        write_f16(&dir, 4, &[d0, d1, d2]).unwrap();
        let store = EmbeddingStore::open(&dir).unwrap();
        (dir, store)
    }

    #[test]
    fn weighted_fusion_single_bm25_hit_keeps_lexical_signal() {
        // With one BM25 hit the normalization span is zero; the hit used
        // to normalize to 0.0 and lose to pure-semantic docs even at
        // alpha = 0.9.
        let (dir, store) = store_named("hyb-single");
        let query = {
            let mut q = vec![1.0f32, 0.0, 0.0, 0.0];
            l2_normalize(&mut q);
            q
        };
        // Doc 1 is orthogonal to the query but is the only lexical match.
        let bm25 = [SearchHit {
            doc_id: 1,
            score: 5.0,
        }];
        let hits = merge_retrieve(&bm25, &store, &query, Fusion::weighted(0.9), 3, 3);
        assert_eq!(hits[0].doc_id, 1, "lexical match must win at alpha=0.9");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn weighted_fusion_promotes_semantic_match() {
        let (dir, store) = store();
        let query = {
            let mut q = vec![1.0f32, 0.0, 0.0, 0.0];
            l2_normalize(&mut q);
            q
        };
        // BM25 ranked doc1 first (wrong), doc0 second (right).
        let bm25 = [
            SearchHit {
                doc_id: 1,
                score: 10.0,
            },
            SearchHit {
                doc_id: 0,
                score: 1.0,
            },
        ];
        let bm25_only = rerank(&bm25, &store, &query, Fusion::weighted(1.0), 2);
        assert_eq!(bm25_only[0].doc_id, 1);

        let hybrid = rerank(&bm25, &store, &query, Fusion::weighted(0.3), 2);
        assert_eq!(hybrid[0].doc_id, 0, "semantic should override weak BM25");

        let rrf = rerank(&bm25, &store, &query, Fusion::rrf(60.0), 2);
        assert_eq!(rrf[0].doc_id, 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn merge_recovers_encoder_hit_that_bm25_missed() {
        let (dir, store) = store_named("hyb-merge");
        let query = {
            let mut q = vec![1.0f32, 0.0, 0.0, 0.0];
            l2_normalize(&mut q);
            q
        };
        // Lexical retriever never saw doc 0; the encoder should still.
        let bm25 = [SearchHit {
            doc_id: 1,
            score: 10.0,
        }];
        let merged = merge_retrieve(&bm25, &store, &query, Fusion::weighted(0.2), 2, 3);
        assert_eq!(merged[0].doc_id, 0);
        assert!(merged[0].bm25 == 0.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}

// ---------------------------------------------------------------------------
// Hybrid retrieval over a segmented index
// ---------------------------------------------------------------------------

/// A hybrid hit addressed by (segment, segment-local doc_id).
#[derive(Debug, Clone)]
pub struct SegHybridHit {
    pub segment: usize,
    pub doc_id: u32,
    pub score: f32,
    pub bm25: f32,
    pub semantic: f32,
}

/// Per-segment vector stores for a segmented index, in segment order.
/// `None` for a segment whose embeddings sidecar is missing (it then
/// contributes only BM25 signal).
pub struct SegmentStores {
    pub stores: Vec<Option<EmbeddingStore>>,
}

impl SegmentStores {
    fn cosine(&self, segment: usize, doc_id: u32, query: &[f32]) -> f32 {
        match self.stores.get(segment).and_then(|s| s.as_ref()) {
            Some(store) => store.cosine(doc_id, query),
            None => 0.0,
        }
    }
}

/// Semantic top-`pool` over every live document of every segment: an exact
/// brute-force scan, in parallel across segments.
///
/// No IVF, deliberately. Segments stay small between merges, and the
/// measured recall/latency trade at repo scale favors exact scoring — see
/// the nprobe/PQ measurements in the README. IVF over the merged base
/// segment can be layered back in when corpora outgrow brute force.
pub fn segmented_semantic_pool(
    stores: &SegmentStores,
    live: &(dyn Fn(usize, u32) -> bool + Sync),
    query: &[f32],
    pool: usize,
) -> Vec<SegHybridHit> {
    use rayon::prelude::*;

    // Parallelism is over fixed-size doc ranges, not segments: after a
    // compaction merge the index is one big segment, and per-segment
    // parallelism would put the entire scan on one core exactly when it
    // is largest.
    const SCAN_CHUNK: u32 = 4096;
    let mut ranges: Vec<(usize, u32, u32)> = Vec::new();
    for (si, store) in stores.stores.iter().enumerate() {
        let n = store.as_ref().map(|s| s.num_docs()).unwrap_or(0);
        let mut at = 0u32;
        while at < n {
            let end = (at + SCAN_CHUNK).min(n);
            ranges.push((si, at, end));
            at = end;
        }
    }
    let mut rows: Vec<SegHybridHit> = ranges
        .par_iter()
        .flat_map_iter(|&(si, start, end)| {
            let store = stores.stores[si]
                .as_ref()
                .expect("ranges only cover segments with stores");
            let mut chunk_rows = Vec::new();
            for doc_id in start..end {
                if !live(si, doc_id) {
                    continue;
                }
                let sem = store.cosine(doc_id, query);
                chunk_rows.push(SegHybridHit {
                    segment: si,
                    doc_id,
                    score: sem,
                    bm25: 0.0,
                    semantic: sem,
                });
            }
            // Keep only this range's best `pool`; the global merge below
            // cannot need more than that from one range.
            chunk_rows.sort_by(|a, b| b.score.total_cmp(&a.score));
            chunk_rows.truncate(pool);
            chunk_rows
        })
        .collect();
    rows.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.segment.cmp(&b.segment))
            .then(a.doc_id.cmp(&b.doc_id))
    });
    rows.truncate(pool);
    rows
}

/// Fuse BM25 hits (as `(segment, doc_id, score)`) with the semantic pool.
/// Same union/normalization semantics as [`merge_retrieve`], addressed by
/// (segment, doc_id) instead of a single index's doc_id.
pub fn segmented_fuse(
    bm25_hits: &[(usize, u32, f32)],
    sem_hits: Vec<SegHybridHit>,
    stores: &SegmentStores,
    query: &[f32],
    fusion: Fusion,
    k: usize,
) -> Vec<SegHybridHit> {
    if k == 0 {
        return Vec::new();
    }
    let key = |si: usize, doc: u32| ((si as u64) << 32) | doc as u64;
    let mut by_id: std::collections::HashMap<u64, (SegHybridHit, usize, usize)> =
        std::collections::HashMap::new();
    for (rank, h) in sem_hits.into_iter().enumerate() {
        by_id.insert(key(h.segment, h.doc_id), (h, 0, rank + 1));
    }
    for (rank, &(si, doc_id, score)) in bm25_hits.iter().enumerate() {
        by_id
            .entry(key(si, doc_id))
            .and_modify(|(row, br, _)| {
                row.bm25 = score;
                *br = rank + 1;
            })
            .or_insert_with(|| {
                (
                    SegHybridHit {
                        segment: si,
                        doc_id,
                        score: 0.0,
                        bm25: score,
                        semantic: stores.cosine(si, doc_id, query),
                    },
                    rank + 1,
                    0,
                )
            });
    }

    let mut rows: Vec<SegHybridHit> = Vec::with_capacity(by_id.len());
    let mut bm25_rank: Vec<usize> = Vec::with_capacity(by_id.len());
    let mut sem_rank: Vec<usize> = Vec::with_capacity(by_id.len());
    for (_, (hit, br, sr)) in by_id {
        rows.push(hit);
        bm25_rank.push(br);
        sem_rank.push(sr);
    }
    if rows.is_empty() {
        return Vec::new();
    }
    match fusion {
        Fusion::Weighted { alpha } => {
            let (lo, hi) = min_max(rows.iter().filter(|r| r.bm25 > 0.0).map(|r| r.bm25));
            let span = hi - lo;
            for r in rows.iter_mut() {
                let n = if r.bm25 <= 0.0 {
                    0.0
                } else if span <= f32::EPSILON {
                    // All matched docs share one score (or there is a single
                    // match): a lexical hit still outranks a non-match.
                    1.0
                } else {
                    (r.bm25 - lo) / span
                };
                r.score = alpha * n + (1.0 - alpha) * r.semantic;
            }
        }
        Fusion::Rrf { k: rrf_k } => {
            for (i, r) in rows.iter_mut().enumerate() {
                let mut score = 0.0;
                if sem_rank[i] > 0 {
                    score += 1.0 / (rrf_k + sem_rank[i] as f32);
                }
                if bm25_rank[i] > 0 {
                    score += 1.0 / (rrf_k + bm25_rank[i] as f32);
                }
                r.score = score;
            }
        }
    }
    rows.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.segment.cmp(&b.segment))
            .then(a.doc_id.cmp(&b.doc_id))
    });
    rows.truncate(k);
    rows
}
