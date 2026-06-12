//! Query benchmark: runs a list of queries through the production
//! Block-Max WAND path and reports latency percentiles plus the debug
//! counters that show how little of the corpus each query touches.

use crate::indexer::SearchableIndex;
use crate::searcher;

#[derive(Debug)]
pub struct BenchReport {
    pub num_queries: usize,
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub avg_postings_visited: f64,
    pub avg_docs_scored: f64,
    pub avg_blocks_visited: f64,
    pub avg_blocks_skipped: f64,
    /// Average over queries of docs_scored / total docs in the corpus.
    pub avg_docs_scored_ratio: f64,
    /// Average over queries of postings_visited / total postings of the
    /// query's terms.
    pub avg_postings_visited_ratio: f64,
}

fn percentile(sorted_ms: &[f64], pct: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let rank = (pct / 100.0 * sorted_ms.len() as f64).ceil() as usize;
    sorted_ms[rank.clamp(1, sorted_ms.len()) - 1]
}

/// Run every query once and aggregate latency and skipping statistics.
pub fn run<I: SearchableIndex + ?Sized>(index: &I, queries: &[String], k: usize) -> BenchReport {
    let num_docs = index.num_docs().max(1);
    let mut latencies = Vec::with_capacity(queries.len());
    let mut postings_visited = 0u64;
    let mut docs_scored = 0u64;
    let mut blocks_visited = 0u64;
    let mut blocks_skipped = 0u64;
    let mut docs_scored_ratio = 0.0f64;
    let mut postings_ratio_sum = 0.0f64;
    let mut postings_ratio_count = 0usize;

    for query in queries {
        let outcome = searcher::search(index, query, k);
        latencies.push(outcome.took_ms);
        postings_visited += outcome.stats.num_postings_visited as u64;
        docs_scored += outcome.stats.num_docs_scored as u64;
        blocks_visited += outcome.stats.num_blocks_visited as u64;
        blocks_skipped += outcome.stats.num_blocks_skipped as u64;
        docs_scored_ratio += outcome.stats.num_docs_scored as f64 / num_docs as f64;

        // Total postings across this query's terms = the work a term-at-a-time
        // exhaustive evaluation would do.
        let term_postings: u64 = searcher::query_terms(index, query)
            .iter()
            .filter_map(|t| index.term_postings(t))
            .map(|t| t.df as u64)
            .sum();
        if term_postings > 0 {
            postings_ratio_sum += outcome.stats.num_postings_visited as f64 / term_postings as f64;
            postings_ratio_count += 1;
        }
    }

    let n = queries.len().max(1) as f64;
    let mut sorted = latencies.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    BenchReport {
        num_queries: queries.len(),
        avg_ms: latencies.iter().sum::<f64>() / n,
        p50_ms: percentile(&sorted, 50.0),
        p95_ms: percentile(&sorted, 95.0),
        p99_ms: percentile(&sorted, 99.0),
        avg_postings_visited: postings_visited as f64 / n,
        avg_docs_scored: docs_scored as f64 / n,
        avg_blocks_visited: blocks_visited as f64 / n,
        avg_blocks_skipped: blocks_skipped as f64 / n,
        avg_docs_scored_ratio: docs_scored_ratio / n,
        avg_postings_visited_ratio: if postings_ratio_count > 0 {
            postings_ratio_sum / postings_ratio_count as f64
        } else {
            0.0
        },
    }
}

pub fn print_report<I: SearchableIndex + ?Sized>(report: &BenchReport, index: &I) {
    println!("queries:                 {}", report.num_queries);
    println!("avg latency:             {:.3} ms", report.avg_ms);
    println!("p50 latency:             {:.3} ms", report.p50_ms);
    println!("p95 latency:             {:.3} ms", report.p95_ms);
    println!("p99 latency:             {:.3} ms", report.p99_ms);
    println!(
        "avg postings visited:    {:.1}",
        report.avg_postings_visited
    );
    println!("avg docs scored:         {:.1}", report.avg_docs_scored);
    println!("avg blocks visited:      {:.1}", report.avg_blocks_visited);
    println!("avg blocks skipped:      {:.1}", report.avg_blocks_skipped);
    println!(
        "avg docs scored / corpus ({} docs):            {:.4}%",
        index.num_docs(),
        report.avg_docs_scored_ratio * 100.0
    );
    println!(
        "avg postings visited / query-term postings:   {:.4}%",
        report.avg_postings_visited_ratio * 100.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&data, 50.0), 5.0);
        assert_eq!(percentile(&data, 95.0), 10.0);
        assert_eq!(percentile(&data, 100.0), 10.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
        assert_eq!(percentile(&[3.5], 95.0), 3.5);
    }
}
