//! Retrieval-quality metrics for the code-search experiment.

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub mrr: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub ndcg_at_10: f64,
    pub n: usize,
}

pub fn evaluate(ranked_ids: &[Vec<String>], qrels: &[Vec<(String, u32)>]) -> Metrics {
    assert_eq!(ranked_ids.len(), qrels.len());
    let n = ranked_ids.len().max(1) as f64;
    let mut mrr = 0.0;
    let mut r5 = 0.0;
    let mut r10 = 0.0;
    let mut ndcg = 0.0;
    for (ranked, rels) in ranked_ids.iter().zip(qrels.iter()) {
        if rels.is_empty() {
            continue;
        }
        mrr += reciprocal_rank(ranked, rels);
        r5 += recall_at(ranked, rels, 5);
        r10 += recall_at(ranked, rels, 10);
        ndcg += ndcg_at(ranked, rels, 10);
    }
    Metrics {
        mrr: mrr / n,
        recall_at_5: r5 / n,
        recall_at_10: r10 / n,
        ndcg_at_10: ndcg / n,
        n: ranked_ids.len(),
    }
}

fn is_relevant(rels: &[(String, u32)], id: &str) -> bool {
    rels.iter().any(|(d, r)| d == id && *r > 0)
}

fn gain(rels: &[(String, u32)], id: &str) -> f64 {
    rels.iter()
        .find(|(d, _)| d == id)
        .map(|(_, r)| 2f64.powi(*r as i32) - 1.0)
        .unwrap_or(0.0)
}

fn reciprocal_rank(ranked: &[String], rels: &[(String, u32)]) -> f64 {
    ranked
        .iter()
        .position(|id| is_relevant(rels, id))
        .map(|i| 1.0 / (i + 1) as f64)
        .unwrap_or(0.0)
}

fn recall_at(ranked: &[String], rels: &[(String, u32)], k: usize) -> f64 {
    let relevant: Vec<&str> = rels
        .iter()
        .filter(|(_, r)| *r > 0)
        .map(|(d, _)| d.as_str())
        .collect();
    if relevant.is_empty() {
        return 0.0;
    }
    let hit = ranked
        .iter()
        .take(k)
        .filter(|id| relevant.contains(&id.as_str()))
        .count();
    hit as f64 / relevant.len() as f64
}

fn ndcg_at(ranked: &[String], rels: &[(String, u32)], k: usize) -> f64 {
    let dcg = dcg_at(ranked, rels, k);
    let mut ideal: Vec<String> = rels
        .iter()
        .filter(|(_, r)| *r > 0)
        .map(|(d, r)| (d.clone(), *r))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    // Sort by gain descending for IDCG.
    ideal.sort_by(|a, b| {
        gain(rels, b)
            .partial_cmp(&gain(rels, a))
            .unwrap()
    });
    let idcg = dcg_at(&ideal, rels, k);
    if idcg <= 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn dcg_at(ranked: &[String], rels: &[(String, u32)], k: usize) -> f64 {
    ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| gain(rels, id) / (i as f64 + 2.0).log2())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_ranking() {
        let ranked = vec![vec!["a".into(), "b".into(), "c".into()]];
        let qrels = vec![vec![("a".into(), 3), ("b".into(), 1)]];
        let m = evaluate(&ranked, &qrels);
        assert!((m.mrr - 1.0).abs() < 1e-9);
        assert!((m.recall_at_5 - 1.0).abs() < 1e-9);
        assert!((m.ndcg_at_10 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn miss_and_late_hit() {
        let ranked = vec![vec!["x".into(), "a".into()]];
        let qrels = vec![vec![("a".into(), 2)]];
        let m = evaluate(&ranked, &qrels);
        assert!((m.mrr - 0.5).abs() < 1e-9);
        assert!((m.recall_at_5 - 1.0).abs() < 1e-9);
    }
}
