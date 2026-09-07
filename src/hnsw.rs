//! Hierarchical navigable small-world routing over the existing FP16 store.
//! Based on Malkov & Yashunin (https://arxiv.org/abs/1603.09320), with immutable
//! per-segment topology, deterministic construction, and exact candidate scores.
//! This is approximate retrieval, not a worst-case sublinear guarantee.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::embeddings::EmbeddingStore;
use anyhow::{ensure, Context, Result};

pub const FILE: &str = "hnsw.bin";
pub const MIN_DOCS: u32 = 8192;
pub const DEFAULT_EF: usize = 128;
const M: usize = 16;
const EF_BUILD: usize = 128;
const MAX_LEVEL: usize = 16;
const MAGIC: &[u8; 8] = b"HPSHNS01";

#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub doc_id: u32,
    pub score: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Neighbor {}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(other.doc_id.cmp(&self.doc_id))
    }
}

pub struct SearchResult {
    pub hits: Vec<Neighbor>,
    /// Dot products, including upper-layer routing and any exact fallback.
    pub distance_computations: usize,
    pub exact_fallback: bool,
}

pub struct HnswIndex {
    // Per-node, per-level bounded adjacency. Vectors are never duplicated here.
    links: Vec<Vec<Vec<u32>>>,
    entry: u32,
    dim: u32,
    generation: (u64, u128),
}

impl HnswIndex {
    pub fn build(store: &EmbeddingStore) -> Result<Self> {
        ensure!(
            store.num_docs() > 0 && store.dim() > 0,
            "cannot build an empty HNSW index"
        );
        let mut graph = Self {
            links: Vec::with_capacity(store.num_docs() as usize),
            entry: 0,
            dim: store.dim(),
            generation: store.generation(),
        };
        let mut random = 0x9e3779b97f4a7c15u64;
        let mut query = vec![0.0; store.dim() as usize];
        for doc in 0..store.num_docs() {
            store.copy_f32(doc, &mut query);
            ensure!(
                query.iter().all(|x| x.is_finite()),
                "nonfinite vector at {doc}"
            );
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let level = (random.trailing_zeros() as usize / 4).min(MAX_LEVEL);
            graph.links.push(vec![Vec::new(); level + 1]);
            if doc == 0 {
                continue;
            }
            let top = graph.links[graph.entry as usize].len() - 1;
            let mut count = 0;
            let mut ep = graph.score(store, &query, graph.entry, &mut count);
            for layer in ((level + 1)..=top).rev() {
                ep = graph.greedy(store, &query, ep, layer, &mut count);
            }
            let mut entries = vec![ep.doc_id];
            for layer in (0..=level.min(top)).rev() {
                let found = graph.layer(
                    store,
                    &query,
                    &entries,
                    EF_BUILD,
                    layer,
                    &mut count,
                    &|_| true,
                );
                entries = found.iter().map(|n| n.doc_id).collect();
                let chosen = Self::select_neighbors(store, &found, M);
                graph.links[doc as usize][layer] = chosen.clone();
                for neighbor in chosen {
                    let cap = if layer == 0 { 2 * M } else { M };
                    graph.links[neighbor as usize][layer].push(doc);
                    if graph.links[neighbor as usize][layer].len() > cap {
                        let mut center = vec![0.0; store.dim() as usize];
                        store.copy_f32(neighbor, &mut center);
                        let mut candidates: Vec<_> = graph.links[neighbor as usize][layer]
                            .iter()
                            .map(|&id| Neighbor {
                                doc_id: id,
                                score: store.cosine(id, &center),
                            })
                            .collect();
                        candidates.sort_unstable_by(|a, b| b.cmp(a));
                        graph.links[neighbor as usize][layer] =
                            Self::select_neighbors(store, &candidates, cap);
                    }
                }
            }
            if level > top {
                graph.entry = doc;
            }
        }
        Ok(graph)
    }

    // Diversify edges: a candidate already closer to a selected neighbor than
    // to the center is redundant. Refill unused slots from discarded candidates.
    fn select_neighbors(store: &EmbeddingStore, candidates: &[Neighbor], cap: usize) -> Vec<u32> {
        let mut chosen = Vec::with_capacity(cap);
        let mut rejected = Vec::new();
        let mut vector = vec![0.0; store.dim() as usize];
        for candidate in candidates {
            store.copy_f32(candidate.doc_id, &mut vector);
            if chosen
                .iter()
                .any(|&id| store.cosine(id, &vector) > candidate.score)
            {
                rejected.push(candidate.doc_id);
            } else {
                chosen.push(candidate.doc_id);
                if chosen.len() == cap {
                    break;
                }
            }
        }
        chosen.extend(rejected.into_iter().take(cap - chosen.len()));
        chosen
    }

    fn score(
        &self,
        store: &EmbeddingStore,
        query: &[f32],
        doc_id: u32,
        count: &mut usize,
    ) -> Neighbor {
        *count += 1;
        Neighbor {
            doc_id,
            score: store.cosine(doc_id, query),
        }
    }

    fn greedy(
        &self,
        store: &EmbeddingStore,
        query: &[f32],
        mut best: Neighbor,
        level: usize,
        count: &mut usize,
    ) -> Neighbor {
        loop {
            let old = best.doc_id;
            for &id in &self.links[old as usize][level] {
                let candidate = self.score(store, query, id, count);
                if candidate > best {
                    best = candidate;
                }
            }
            if best.doc_id == old {
                return best;
            }
        }
    }

    fn layer(
        &self,
        store: &EmbeddingStore,
        query: &[f32],
        entries: &[u32],
        ef: usize,
        level: usize,
        count: &mut usize,
        live: &dyn Fn(u32) -> bool,
    ) -> Vec<Neighbor> {
        // A sparse visited set avoids an O(N) clear/allocation on every query.
        let mut seen = HashSet::new();
        let mut frontier = BinaryHeap::new();
        let mut best = BinaryHeap::new();
        for &id in entries {
            if seen.insert(id) {
                let n = self.score(store, query, id, count);
                frontier.push(n);
                if live(id) {
                    best.push(Reverse(n));
                }
            }
        }
        while let Some(candidate) = frontier.pop() {
            if best.len() >= ef && candidate < best.peek().unwrap().0 {
                break;
            }
            for &id in &self.links[candidate.doc_id as usize][level] {
                if !seen.insert(id) {
                    continue;
                }
                let n = self.score(store, query, id, count);
                if best.len() < ef || n > best.peek().unwrap().0 {
                    frontier.push(n);
                    if live(id) {
                        best.push(Reverse(n));
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
        }
        let mut rows: Vec<_> = best.into_iter().map(|r| r.0).collect();
        rows.sort_unstable_by(|a, b| b.cmp(a));
        rows
    }

    pub fn search(
        &self,
        store: &EmbeddingStore,
        query: &[f32],
        k: usize,
        ef: usize,
        live: &dyn Fn(u32) -> bool,
    ) -> SearchResult {
        let mut result = SearchResult {
            hits: Vec::new(),
            distance_computations: 0,
            exact_fallback: false,
        };
        if k == 0 {
            return result;
        }
        assert_eq!(query.len(), self.dim as usize);
        assert!(
            self.matches(store),
            "HNSW used with a different vector generation"
        );
        let mut ep = self.score(store, query, self.entry, &mut result.distance_computations);
        for level in (1..self.links[self.entry as usize].len()).rev() {
            ep = self.greedy(store, query, ep, level, &mut result.distance_computations);
        }
        let n = store.num_docs() as usize;
        let width = ef.max(k).max(1).min(n);
        // Tombstoned nodes remain traversable. If filtering underfills the pool,
        // use the exact path rather than returning too few live results or doing
        // repeated increasingly expensive graph walks.
        result.hits = self
            .layer(
                store,
                query,
                &[ep.doc_id],
                width,
                0,
                &mut result.distance_computations,
                live,
            )
            .into_iter()
            .take(k)
            .collect();
        if result.hits.len() < k.min(n) {
            result.exact_fallback = true;
            let mut best = BinaryHeap::new();
            for id in 0..store.num_docs() {
                if live(id) {
                    best.push(Reverse(self.score(
                        store,
                        query,
                        id,
                        &mut result.distance_computations,
                    )));
                    if best.len() > k {
                        best.pop();
                    }
                }
            }
            result.hits = best.into_iter().map(|r| r.0).collect();
            result.hits.sort_unstable_by(|a, b| b.cmp(a));
        }
        result
    }

    pub fn matches(&self, store: &EmbeddingStore) -> bool {
        self.links.len() == store.num_docs() as usize
            && self.dim == store.dim()
            && self.generation == store.generation()
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        let mut temp = tempfile::NamedTempFile::new_in(dir)?;
        {
            let mut out = BufWriter::new(temp.as_file_mut());
            out.write_all(MAGIC)?;
            out.write_all(&(self.links.len() as u32).to_le_bytes())?;
            out.write_all(&self.dim.to_le_bytes())?;
            out.write_all(&self.generation.0.to_le_bytes())?;
            out.write_all(&self.generation.1.to_le_bytes())?;
            out.write_all(&self.entry.to_le_bytes())?;
            for levels in &self.links {
                out.write_all(&[levels.len() as u8])?;
                for links in levels {
                    out.write_all(&[links.len() as u8])?;
                    for id in links {
                        out.write_all(&id.to_le_bytes())?;
                    }
                }
            }
            out.flush()?;
        }
        temp.as_file().sync_all()?;
        temp.persist(dir.join(FILE)).map_err(|e| e.error)?;
        Ok(())
    }

    pub fn open(dir: &Path, store: &EmbeddingStore) -> Result<Self> {
        let file = std::fs::File::open(dir.join(FILE))?;
        let size = file.metadata()?.len();
        let mut input = BufReader::new(file);
        fn bytes<const N: usize>(r: &mut impl Read) -> Result<[u8; N]> {
            let mut b = [0; N];
            r.read_exact(&mut b)?;
            Ok(b)
        }
        ensure!(&bytes::<8>(&mut input)? == MAGIC, "invalid HNSW format");
        let n = u32::from_le_bytes(bytes(&mut input)?);
        let dim = u32::from_le_bytes(bytes(&mut input)?);
        let generation = (
            u64::from_le_bytes(bytes(&mut input)?),
            u128::from_le_bytes(bytes(&mut input)?),
        );
        let entry = u32::from_le_bytes(bytes(&mut input)?);
        ensure!(
            n == store.num_docs() && dim == store.dim() && generation == store.generation(),
            "stale HNSW vector generation"
        );
        ensure!(
            entry < n && size >= 44 + 2 * n as u64,
            "invalid HNSW node count/entry"
        );
        let mut links = Vec::with_capacity(n as usize);
        for node in 0..n {
            let levels = bytes::<1>(&mut input)?[0] as usize;
            ensure!(
                levels > 0 && levels <= MAX_LEVEL + 1,
                "invalid HNSW level count"
            );
            let mut layers = Vec::with_capacity(levels);
            for level in 0..levels {
                let degree = bytes::<1>(&mut input)?[0] as usize;
                ensure!(
                    degree <= if level == 0 { 2 * M } else { M },
                    "invalid HNSW degree"
                );
                let mut edges = Vec::with_capacity(degree);
                for _ in 0..degree {
                    let id = u32::from_le_bytes(bytes(&mut input)?);
                    ensure!(
                        id < n && id != node && !edges.contains(&id),
                        "invalid HNSW edge"
                    );
                    edges.push(id);
                }
                layers.push(edges);
            }
            links.push(layers);
        }
        ensure!(input.read(&mut [0u8; 1])? == 0, "trailing HNSW data");
        let top = links[entry as usize].len();
        for layers in &links {
            ensure!(layers.len() <= top, "invalid HNSW entry level");
            for (level, edges) in layers.iter().enumerate() {
                ensure!(
                    edges.iter().all(|&id| links[id as usize].len() > level),
                    "HNSW edge to absent level"
                );
            }
        }
        Ok(Self {
            links,
            entry,
            dim,
            generation,
        })
    }
}

/// Called by segment creation/compaction, never by the query path.
pub fn build_if_large(dir: &Path) -> Result<bool> {
    let store = EmbeddingStore::open(dir)?;
    if store.num_docs() < MIN_DOCS {
        return Ok(false);
    }
    if HnswIndex::open(dir, &store).is_ok() {
        return Ok(false);
    }
    HnswIndex::build(&store)
        .context("build HNSW routing graph")?
        .save(dir)?;
    Ok(true)
}

/// Retrofit existing immutable segments under the same lock used by writers.
pub fn build_segments(dir: &Path) -> Result<usize> {
    ensure!(
        crate::segments::is_segmented(dir),
        "build-ann needs a segmented repository index"
    );
    let lock = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("writer.lock"))?;
    lock.try_lock().context("another index writer is active")?;
    let index = crate::segments::SegmentedIndex::open(dir)?;
    let mut built = 0;
    for name in index.segment_names() {
        let segment = dir.join(&name);
        if !segment.join(crate::embeddings::EMBEDDINGS_FILE).is_file() {
            continue;
        }
        let store = EmbeddingStore::open(&segment)?;
        if store.num_docs() >= MIN_DOCS && HnswIndex::open(&segment, &store).is_err() {
            eprintln!("building HNSW for {name} ({} vectors)", store.num_docs());
            HnswIndex::build(&store)?.save(&segment)?;
            built += 1;
        }
    }
    Ok(built)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{l2_normalize, write_f16};

    fn vectors(n: usize) -> Vec<Vec<f32>> {
        let mut random = 17u64;
        (0..n)
            .map(|_| {
                let mut v: Vec<_> = (0..32)
                    .map(|_| {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        (random >> 40) as f32 / 8388608.0 - 1.0
                    })
                    .collect();
                l2_normalize(&mut v);
                v
            })
            .collect()
    }

    #[test]
    fn roundtrip_recall_and_tombstone_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let data = vectors(2064);
        write_f16(dir.path(), 32, &data[..2048]).unwrap();
        let store = EmbeddingStore::open(dir.path()).unwrap();
        let graph = HnswIndex::build(&store).unwrap();
        graph.save(dir.path()).unwrap();
        let loaded = HnswIndex::open(dir.path(), &store).unwrap();
        let mut recovered = 0;
        for query in &data[2048..] {
            let mut truth: Vec<_> = (0..2048)
                .map(|id| Neighbor {
                    doc_id: id,
                    score: store.cosine(id, query),
                })
                .collect();
            truth.sort_by(|a, b| b.cmp(a));
            let found = loaded.search(&store, query, 10, 128, &|_| true);
            assert_eq!(
                found.hits,
                graph.search(&store, query, 10, 128, &|_| true).hits
            );
            assert!(!found.exact_fallback);
            recovered += found
                .hits
                .iter()
                .filter(|h| truth[..10].contains(h))
                .count();
        }
        assert!(recovered >= 152, "held-out recall@10: {recovered}/160");
        let only_two = loaded.search(&store, &data[2048], 10, 32, &|id| id >= 2046);
        assert!(only_two.exact_fallback);
        assert_eq!(only_two.hits.len(), 2);
        assert!(only_two.hits.iter().all(|h| h.doc_id >= 2046));
        let all_dead = loaded.search(&store, &data[0], 10, 32, &|_| false);
        assert!(all_dead.hits.is_empty());
        assert!(loaded
            .search(&store, &data[0], 0, 32, &|_| true)
            .hits
            .is_empty());
        // Large ef uses the same exact scoring and deterministic tie order.
        let exhaustive = loaded.search(&store, &data[2048], 2048, 2048, &|_| true);
        assert_eq!(exhaustive.hits.len(), 2048);
    }

    #[test]
    fn rejects_corrupt_and_stale_graphs() {
        let dir = tempfile::tempdir().unwrap();
        write_f16(dir.path(), 32, &vectors(32)).unwrap();
        let store = EmbeddingStore::open(dir.path()).unwrap();
        let graph = HnswIndex::build(&store).unwrap();
        graph.save(dir.path()).unwrap();
        let bytes = std::fs::read(dir.path().join(FILE)).unwrap();
        for len in [0, 8, 43, bytes.len() - 1] {
            std::fs::write(dir.path().join(FILE), &bytes[..len]).unwrap();
            assert!(HnswIndex::open(dir.path(), &store).is_err());
        }
        let mut corrupt = bytes.clone();
        corrupt[44] = 255; // first node's layer count
        std::fs::write(dir.path().join(FILE), corrupt).unwrap();
        assert!(HnswIndex::open(dir.path(), &store).is_err());
        // Invalid topology must leave the public retrieval path usable.
        let fallback = EmbeddingStore::open(dir.path()).unwrap();
        assert!(fallback.hnsw().is_none());
        std::fs::write(dir.path().join(FILE), bytes).unwrap();
        // Same dimensions/count but a different immutable generation must fail.
        // Build another file rather than mutate a file while an mmap is alive.
        let other = tempfile::tempdir().unwrap();
        write_f16(other.path(), 32, &vectors(32)).unwrap();
        std::fs::File::options()
            .write(true)
            .open(other.path().join(crate::embeddings::EMBEDDINGS_FILE))
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1))
            .unwrap();
        let changed = EmbeddingStore::open(other.path()).unwrap();
        assert!(!graph.matches(&changed));
        assert!(HnswIndex::open(dir.path(), &changed).is_err());
    }

    #[test]
    fn retrofit_obeys_writer_lock_and_is_idempotent() {
        use crate::indexer::InputDoc;
        use crate::segments::SegmentedWriter;
        let dir = tempfile::tempdir().unwrap();
        let mut writer = SegmentedWriter::open_or_create(dir.path(), false, 1).unwrap();
        let docs: Vec<_> = (0..MIN_DOCS)
            .map(|i| InputDoc {
                id: i.to_string(),
                title: String::new(),
                body: "graph fixture".into(),
            })
            .collect();
        let name = writer.add_documents(&docs).unwrap();
        let segment = dir.path().join(name);
        write_f16(&segment, 32, &vectors(MIN_DOCS as usize)).unwrap();
        assert!(build_segments(dir.path()).is_err());
        assert!(!segment.join(FILE).exists());
        drop(writer);
        assert_eq!(build_segments(dir.path()).unwrap(), 1);
        assert_eq!(build_segments(dir.path()).unwrap(), 0);
        assert!(!build_if_large(&segment).unwrap());
        assert!(EmbeddingStore::open(&segment).unwrap().hnsw().is_some());
    }

    #[test]
    fn routed_pool_mixes_graph_exact_and_missing_stores() {
        use crate::hybrid::{
            segmented_semantic_pool, segmented_semantic_pool_with_options, SegmentStores,
        };
        let graph_dir = tempfile::tempdir().unwrap();
        let exact_dir = tempfile::tempdir().unwrap();
        let data = vectors(128);
        write_f16(graph_dir.path(), 32, &data).unwrap();
        write_f16(exact_dir.path(), 32, &data[..32]).unwrap();
        HnswIndex::build(&EmbeddingStore::open(graph_dir.path()).unwrap())
            .unwrap()
            .save(graph_dir.path())
            .unwrap();
        let stores = SegmentStores {
            stores: vec![
                Some(EmbeddingStore::open(graph_dir.path()).unwrap()),
                None,
                Some(EmbeddingStore::open(exact_dir.path()).unwrap()),
            ],
        };
        let live = |si, id| si != 0 || id % 5 != 0;
        let expected = segmented_semantic_pool(&stores, &live, &data[127], 10);
        let (actual, stats) =
            segmented_semantic_pool_with_options(&stores, &live, &data[127], 10, 128, false);
        assert_eq!(stats.graph_segments, 1);
        assert_eq!(stats.exact_segments, 1);
        assert_eq!(
            actual
                .iter()
                .map(|h| (h.segment, h.doc_id, h.score))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|h| (h.segment, h.doc_id, h.score))
                .collect::<Vec<_>>()
        );
        let (_, exact) =
            segmented_semantic_pool_with_options(&stores, &live, &data[127], 10, 32, true);
        assert_eq!(exact.graph_segments, 0);
        assert_eq!(exact.distance_computations, 134);
    }
}
