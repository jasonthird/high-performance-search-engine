fn main() {
    // Candle's Metal backend flushes its command buffer every
    // CANDLE_METAL_COMPUTE_PER_BUFFER ops (default 50) — several
    // mid-forward flushes for a 12-layer model. 200 measured 9.2 ->
    // 6.2 ms on a batch-1 query forward, with no effect on large
    // (compute-bound) shapes. Respect an explicit user setting.
    //
    // Set here, before any thread exists: setenv while another thread
    // reads the environment is undefined behavior, and the embedder can
    // be loaded after rayon workers or the MCP watcher have started.
    if std::env::var_os("CANDLE_METAL_COMPUTE_PER_BUFFER").is_none() {
        std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "200");
    }
    if let Err(err) = high_performance_search_engine::cli::run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
