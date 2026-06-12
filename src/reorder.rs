//! Document reordering for better index compression.
//!
//! Posting lists store doc_id *gaps*. If documents that share terms get
//! adjacent doc_ids, gaps shrink and bit-packing needs fewer bits — and
//! block max scores cluster, which also helps Block-Max WAND skip more.
//! Reordering is purely a renumbering: it cannot change search results,
//! only the internal ids (a fact the tests verify).
//!
//! Two strategies:
//!
//! - `Path`: sort documents by external id. For corpora whose ids are file
//!   paths or URLs, lexicographic order clusters similar documents (same
//!   directory/site). A classic, nearly free heuristic.
//!
//! - `Bp`: recursive graph bisection ("BP") from Dhulipala et al.,
//!   "Compressing Graphs and Indexes with Recursive Graph Bisection",
//!   KDD 2016 — the state-of-the-art doc-id assignment used by modern
//!   engines. Documents are recursively split into two halves; within each
//!   split, documents are greedily swapped between halves to minimize an
//!   estimate of the compressed inverted-index size (the log-gap cost).

use rayon::prelude::*;

#[cfg(feature = "gpu")]
pub mod gpu;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReorderStrategy {
    /// Keep input order.
    None,
    /// Sort by external document id (path/URL clustering).
    Path,
    /// Recursive graph bisection (minimize estimated log-gap cost).
    Bp,
    /// Recursive graph bisection with the per-iteration gain computation
    /// running on the GPU (requires the `gpu` feature).
    BpGpu,
}

/// Stop recursing below this partition size; tiny partitions don't affect
/// the cost model enough to matter.
const BP_MIN_PARTITION: usize = 32;
/// Maximum swap iterations per bisection level (the paper uses ~20; gains
/// flatten quickly).
const BP_MAX_ITERS: usize = 12;
/// Safety cap on recursion depth (2^40 docs is unreachable anyway).
const BP_MAX_DEPTH: usize = 40;

/// Compute a BP ordering. `doc_terms[d]` lists the distinct term ids of
/// document `d`. Returns a permutation: `order[new_pos] = old_doc_index`.
pub fn bp_order(doc_terms: &[Vec<u32>]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..doc_terms.len() as u32).collect();
    bisect(&mut order, doc_terms, 0);
    order
}

/// Estimated cost (in bits, up to constants) of a term that occurs in `deg`
/// documents of a partition holding `n` documents: each of its `deg` gaps
/// costs about log2(n / (deg + 1)) bits.
fn cost(deg: u32, n: usize) -> f32 {
    if deg == 0 {
        return 0.0;
    }
    deg as f32 * (n as f32 / (deg as f32 + 1.0)).log2()
}

/// Gain (cost reduction) of moving one document containing term `t` from a
/// partition where `t` has degree `from_deg` to one where it has `to_deg`.
fn move_gain(from_deg: u32, to_deg: u32, n_from: usize, n_to: usize) -> f32 {
    cost(from_deg, n_from) + cost(to_deg, n_to)
        - cost(from_deg - 1, n_from)
        - cost(to_deg + 1, n_to)
}

pub(crate) fn bisect(order: &mut [u32], doc_terms: &[Vec<u32>], depth: usize) {
    let n = order.len();
    if n <= BP_MIN_PARTITION || depth >= BP_MAX_DEPTH {
        return;
    }
    let mid = n / 2;

    // Remap this partition's terms to a dense local id space so degree
    // counters are plain vectors.
    let mut local_ids = std::collections::HashMap::new();
    let local_docs: Vec<Vec<u32>> = order
        .iter()
        .map(|&doc| {
            doc_terms[doc as usize]
                .iter()
                .map(|&t| {
                    let next_id = local_ids.len() as u32;
                    *local_ids.entry(t).or_insert(next_id)
                })
                .collect()
        })
        .collect();
    let num_local_terms = local_ids.len();
    drop(local_ids);

    // side[i]: false = left half, true = right half (i indexes `order`).
    let mut side: Vec<bool> = (0..n).map(|i| i >= mid).collect();
    let (n_left, n_right) = (mid, n - mid);

    for _ in 0..BP_MAX_ITERS {
        // Term degrees per side.
        let mut deg_left = vec![0u32; num_local_terms];
        let mut deg_right = vec![0u32; num_local_terms];
        for (i, terms) in local_docs.iter().enumerate() {
            let deg = if side[i] {
                &mut deg_right
            } else {
                &mut deg_left
            };
            for &t in terms {
                deg[t as usize] += 1;
            }
        }

        // Move gain for every document, computed in parallel.
        let gains: Vec<f32> = local_docs
            .par_iter()
            .enumerate()
            .map(|(i, terms)| {
                let mut gain = 0.0;
                for &t in terms {
                    let (dl, dr) = (deg_left[t as usize], deg_right[t as usize]);
                    gain += if side[i] {
                        move_gain(dr, dl, n_right, n_left)
                    } else {
                        move_gain(dl, dr, n_left, n_right)
                    };
                }
                gain
            })
            .collect();

        // Candidates from each side, best gain first (deterministic ties).
        let by_gain_desc = |a: &usize, b: &usize| {
            gains[*b]
                .partial_cmp(&gains[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        };
        let mut left: Vec<usize> = (0..n).filter(|&i| !side[i]).collect();
        let mut right: Vec<usize> = (0..n).filter(|&i| side[i]).collect();
        left.sort_unstable_by(by_gain_desc);
        right.sort_unstable_by(by_gain_desc);

        // Swap the best pairs while the combined gain is positive.
        let mut swapped = 0;
        for (&l, &r) in left.iter().zip(&right) {
            if gains[l] + gains[r] <= 0.0 {
                break;
            }
            side[l] = true;
            side[r] = false;
            swapped += 1;
        }
        if swapped == 0 {
            break; // converged
        }
    }

    // Stable-partition `order` by final side assignment.
    let reordered: Vec<u32> = (0..n)
        .filter(|&i| !side[i])
        .chain((0..n).filter(|&i| side[i]))
        .map(|i| order[i])
        .collect();
    order.copy_from_slice(&reordered);

    // Recurse on both halves in parallel; they are disjoint.
    let (left_half, right_half) = order.split_at_mut(mid);
    rayon::join(
        || bisect(left_half, doc_terms, depth + 1),
        || bisect(right_half, doc_terms, depth + 1),
    );
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn is_permutation(order: &[u32], n: usize) -> bool {
        let mut seen = vec![false; n];
        for &d in order {
            if seen[d as usize] {
                return false;
            }
            seen[d as usize] = true;
        }
        order.len() == n
    }

    /// Total log-gap cost of an ordering: for every term, the sum of
    /// log2(gap) over consecutive occurrences. Lower = more compressible.
    /// Also used by the GPU BP tests.
    pub(crate) fn log_gap_cost(order: &[u32], doc_terms: &[Vec<u32>]) -> f64 {
        let num_terms = doc_terms.iter().flatten().max().map_or(0, |&t| t + 1) as usize;
        let mut new_pos = vec![0u32; doc_terms.len()];
        for (pos, &doc) in order.iter().enumerate() {
            new_pos[doc as usize] = pos as u32;
        }
        let mut term_positions: Vec<Vec<u32>> = vec![Vec::new(); num_terms];
        for (doc, terms) in doc_terms.iter().enumerate() {
            for &t in terms {
                term_positions[t as usize].push(new_pos[doc]);
            }
        }
        let mut total = 0.0f64;
        for positions in &mut term_positions {
            positions.sort_unstable();
            for pair in positions.windows(2) {
                total += f64::from(pair[1] - pair[0]).log2();
            }
        }
        total
    }

    /// Two "topics" with disjoint vocabularies, documents interleaved —
    /// the worst case for gap compression.
    fn interleaved_corpus(n: usize) -> Vec<Vec<u32>> {
        (0..n)
            .map(|i| {
                let base = if i % 2 == 0 { 0u32 } else { 50 };
                (0..10).map(|j| base + (i as u32 * 3 + j) % 50).collect()
            })
            .collect()
    }

    #[test]
    fn bp_returns_a_permutation() {
        let docs = interleaved_corpus(500);
        let order = bp_order(&docs);
        assert!(is_permutation(&order, docs.len()));
    }

    #[test]
    fn bp_reduces_log_gap_cost_on_clustered_corpus() {
        let docs = interleaved_corpus(2000);
        let identity: Vec<u32> = (0..docs.len() as u32).collect();
        let order = bp_order(&docs);

        let before = log_gap_cost(&identity, &docs);
        let after = log_gap_cost(&order, &docs);
        assert!(
            after < before * 0.8,
            "BP should cut log-gap cost noticeably: before={before:.0}, after={after:.0}"
        );
    }

    #[test]
    fn bp_handles_tiny_and_empty_inputs() {
        assert!(bp_order(&[]).is_empty());
        let one = vec![vec![1u32, 2]];
        assert_eq!(bp_order(&one), vec![0]);
        let few: Vec<Vec<u32>> = (0..5).map(|i| vec![i as u32]).collect();
        assert!(is_permutation(&bp_order(&few), 5));
    }
}
