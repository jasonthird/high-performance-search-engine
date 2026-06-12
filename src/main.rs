fn main() {
    if let Err(err) = high_performance_search_engine::cli::run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
