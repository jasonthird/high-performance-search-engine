//! BM25 scoring implemented from scratch.
//!
//! score(D, Q) = sum over query terms q of:
//!     idf(q) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))

pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

/// idf = ln(1 + (N - df + 0.5) / (df + 0.5))
///
/// Always positive, even when a term appears in every document.
pub fn idf(num_docs: usize, df: usize) -> f32 {
    let n = num_docs as f32;
    let df = df as f32;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// BM25 contribution of a single term occurrence profile within one document.
pub fn term_contribution(idf: f32, tf: u32, doc_len: u32, avg_doc_len: f32) -> f32 {
    let tf = tf as f32;
    let norm = K1 * (1.0 - B + B * doc_len as f32 / avg_doc_len);
    idf * (tf * (K1 + 1.0)) / (tf + norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idf_matches_formula() {
        // N = 100, df = 10 -> ln(1 + 90.5 / 10.5)
        let expected = (1.0f32 + 90.5 / 10.5).ln();
        assert!((idf(100, 10) - expected).abs() < 1e-6);
    }

    #[test]
    fn idf_positive_even_for_ubiquitous_terms() {
        assert!(idf(100, 100) > 0.0);
        assert!(idf(1, 1) > 0.0);
    }

    #[test]
    fn idf_decreases_with_document_frequency() {
        assert!(idf(1000, 1) > idf(1000, 10));
        assert!(idf(1000, 10) > idf(1000, 500));
    }

    #[test]
    fn term_contribution_matches_formula() {
        let idf = 2.0;
        let tf = 3u32;
        let doc_len = 120u32;
        let avg = 100.0f32;
        let expected = idf * (3.0 * (K1 + 1.0)) / (3.0 + K1 * (1.0 - B + B * 120.0 / 100.0));
        assert!((term_contribution(idf, tf, doc_len, avg) - expected).abs() < 1e-6);
    }

    #[test]
    fn term_contribution_increases_with_tf_and_saturates() {
        let c1 = term_contribution(1.5, 1, 100, 100.0);
        let c2 = term_contribution(1.5, 2, 100, 100.0);
        let c100 = term_contribution(1.5, 100, 100, 100.0);
        assert!(c2 > c1);
        // Saturation: contribution is bounded by idf * (k1 + 1).
        assert!(c100 < 1.5 * (K1 + 1.0));
    }

    #[test]
    fn term_contribution_penalizes_long_docs() {
        let short = term_contribution(1.5, 2, 50, 100.0);
        let long = term_contribution(1.5, 2, 400, 100.0);
        assert!(short > long);
    }
}
