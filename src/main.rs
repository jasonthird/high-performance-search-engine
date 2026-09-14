fn main() {
    #[cfg(all(target_os = "macos", feature = "semantic"))]
    if let Err(err) = configure_coreml_cache() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
    // Mesa's RADV prints a conformance warning on every Vulkan instance
    // creation; the encoder creates one per process. Set before any thread
    // exists (setenv while another thread reads the environment is UB).
    #[cfg(all(feature = "semantic", target_os = "linux"))]
    if std::env::var_os("MESA_VK_IGNORE_CONFORMANCE_WARNING").is_none() {
        std::env::set_var("MESA_VK_IGNORE_CONFORMANCE_WARNING", "1");
    }
    if let Err(err) = high_performance_search_engine::cli::run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

/// Must run before any threads or Foundation/CoreML calls. This opt-in
/// relocates Foundation's home for this process, not the shell's HOME or the
/// hips/Hugging Face caches. It cannot grant access to restricted OS services.
#[cfg(all(target_os = "macos", feature = "semantic"))]
fn configure_coreml_cache() -> anyhow::Result<()> {
    use anyhow::Context;
    let Some(dir) = std::env::var_os("HIPS_COREML_CACHE_DIR") else {
        return Ok(());
    };
    anyhow::ensure!(!dir.is_empty(), "HIPS_COREML_CACHE_DIR must not be empty");
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("cannot create HIPS_COREML_CACHE_DIR {}", dir.display()))?;
    let dir = dir
        .canonicalize()
        .context("resolve HIPS_COREML_CACHE_DIR")?;
    let probe = tempfile::Builder::new()
        .prefix("hips-probe-")
        .tempdir_in(&dir)
        .with_context(|| format!("HIPS_COREML_CACHE_DIR is not writable: {}", dir.display()))?;
    std::fs::write(probe.path().join("probe"), b"cache probe")
        .context("write HIPS_COREML_CACHE_DIR probe")?;
    probe.close().context("clean HIPS_COREML_CACHE_DIR probe")?;
    std::env::set_var("CFFIXED_USER_HOME", dir);
    Ok(())
}
