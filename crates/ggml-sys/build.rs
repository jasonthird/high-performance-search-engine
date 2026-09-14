use std::{env, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("../../vendor/ggml");
    let src = src.canonicalize().unwrap_or(src);
    if !src.join("CMakeLists.txt").exists() {
        panic!(
            "vendor/ggml is empty: ggml is a git submodule, run `git submodule update --init` \
             (or clone with --recurse-submodules)"
        );
    }
    println!("cargo:rerun-if-changed={}", src.join("include").display());
    println!("cargo:rerun-if-changed={}", src.join("src").display());
    for var in ["GGML_NATIVE", "HIPS_GGML_GPU", "CMAKE_PREFIX_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let gpu = env::var("HIPS_GGML_GPU").map(|v| v != "0").unwrap_or(true);

    let mut cfg = cmake::Config::new(&src);
    // Always optimise the C library: a debug ggml makes `cargo test` crawl and
    // debugs nothing of ours.
    cfg.profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_BUILD_TESTS", "OFF")
        .define("GGML_BUILD_EXAMPLES", "OFF")
        // The tinyBLAS GEMM kernels: llama.cpp turns them on by default, the
        // standalone ggml tree does not, and without them the F16-weight
        // matmuls fall back to the generic dot product (about 2x slower on
        // x86 for this model).
        .define("GGML_LLAMAFILE", "ON");
    if let Ok(native) = env::var("GGML_NATIVE") {
        cfg.define("GGML_NATIVE", native);
    }
    match target_os.as_str() {
        "linux" => {
            cfg.define("GGML_OPENMP", "ON");
            cfg.define("GGML_VULKAN", if gpu { "ON" } else { "OFF" });
        }
        "macos" => {
            cfg.define("GGML_OPENMP", "OFF");
            cfg.define("GGML_METAL", if gpu { "ON" } else { "OFF" });
            cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
            cfg.define("GGML_BLAS", "ON");
            cfg.define("GGML_BLAS_VENDOR", "Apple");
        }
        _ => {
            cfg.define("GGML_OPENMP", "OFF");
        }
    }
    let dst = cfg.build();

    // ggml installs to lib or lib64 depending on the platform's GNUInstallDirs;
    // the backend archives sit next to the core ones.
    for dir in walk_archive_dirs(&dst) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    // Static link order: libggml's backend registry references the backends,
    // the backends reference ggml-base.
    println!("cargo:rustc-link-lib=static=ggml");
    match target_os.as_str() {
        "linux" => {
            if gpu {
                println!("cargo:rustc-link-lib=static=ggml-vulkan");
                println!("cargo:rustc-link-lib=dylib=vulkan");
            }
        }
        "macos" => {
            if gpu {
                println!("cargo:rustc-link-lib=static=ggml-metal");
            }
            println!("cargo:rustc-link-lib=static=ggml-blas");
        }
        _ => {}
    }
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            for framework in ["Foundation", "Metal", "MetalKit", "Accelerate"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
        }
        _ => {}
    }
}

/// Every directory under `root` that holds a static archive.
fn walk_archive_dirs(root: &std::path::Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        let mut has_archive = false;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "a") {
                has_archive = true;
            }
        }
        if has_archive {
            dirs.push(dir);
        }
    }
    dirs
}
