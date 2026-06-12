//! Concurrent HTTP search API.
//!
//! The index is loaded once, wrapped in an `Arc`, and shared read-only across
//! all request handlers; searches never mutate it, so any number of queries
//! can run concurrently.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use axum::extract::{Query, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::searcher::AnyIndex;

struct AppState {
    index: AnyIndex,
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    k: Option<usize>,
}

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    took_ms: f64,
    num_docs_total: usize,
    num_query_terms: usize,
    num_postings_visited: usize,
    num_docs_scored: usize,
    num_blocks_visited: usize,
    num_blocks_skipped: usize,
    results: Vec<ResultEntry>,
}

#[derive(Serialize)]
struct ResultEntry {
    id: String,
    score: f32,
    title: String,
}

#[derive(Serialize)]
struct StatsResponse {
    num_docs: usize,
    num_terms: usize,
    avg_doc_len: f32,
    index_size_bytes: u64,
}

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Json<SearchResponse> {
    let k = params.k.unwrap_or(10);
    let query = params.q.clone();
    // Searching is CPU-bound; run it on the blocking pool so it does not
    // stall the async executor under concurrent load.
    let outcome = tokio::task::spawn_blocking({
        let state = Arc::clone(&state);
        let query = query.clone();
        move || state.index.search(&query, k)
    })
    .await
    .expect("search task panicked");

    Json(SearchResponse {
        query,
        took_ms: outcome.took_ms,
        num_docs_total: outcome.stats.num_docs_total,
        num_query_terms: outcome.stats.num_query_terms,
        num_postings_visited: outcome.stats.num_postings_visited,
        num_docs_scored: outcome.stats.num_docs_scored,
        num_blocks_visited: outcome.stats.num_blocks_visited,
        num_blocks_skipped: outcome.stats.num_blocks_skipped,
        results: outcome
            .results
            .into_iter()
            .map(|r| ResultEntry {
                id: r.id,
                score: r.score,
                title: r.title,
            })
            .collect(),
    })
}

async fn stats_handler(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let (num_terms, avg_doc_len) = match &state.index {
        AnyIndex::Single(index) => {
            use crate::indexer::SearchableIndex as _;
            (index.num_terms(), index.avg_doc_len())
        }
        AnyIndex::Segmented(_) => (0, 0.0),
    };
    Json(StatsResponse {
        num_docs: state.index.num_docs() as usize,
        num_terms,
        avg_doc_len,
        index_size_bytes: state.index.size_bytes(),
    })
}

/// Serve the search API until interrupted.
pub async fn serve(index: AnyIndex, addr: SocketAddr) -> anyhow::Result<()> {
    let state = Arc::new(AppState { index });
    let app = Router::new()
        .route("/search", get(search_handler))
        .route("/stats", get(stats_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    println!("listening on http://{addr}");
    println!("  GET /search?q=cheap+pizza&k=10");
    println!("  GET /stats");
    axum::serve(listener, app).await.context("server error")?;
    Ok(())
}
