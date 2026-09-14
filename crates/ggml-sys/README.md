# ggml-sys

Builds the vendored ggml (`../../vendor/ggml`, a git submodule pinned by
commit) as static libraries and links them into hips:

- Linux: CPU backend (OpenMP) plus the Vulkan backend. Needs `cmake`, a C/C++
  compiler, `glslc` (shaderc), the Vulkan headers and loader, and
  SPIRV-Headers (Arch: `cmake shaderc vulkan-headers vulkan-icd-loader
  spirv-headers`; Debian/Ubuntu: `cmake glslc libvulkan-dev spirv-headers`).
  `HIPS_GGML_GPU=0` builds CPU-only.
- macOS: CPU, Metal (embedded shader library, compiled at first use) and the
  Accelerate BLAS backend. Needs Xcode command line tools and `cmake`.

`GGML_NATIVE=OFF` disables `-march=native` for distributable binaries. CMake
honours `CMAKE_PREFIX_PATH` for a SPIRV-Headers install outside the system
prefix.

`src/bindings.rs` is generated with bindgen 0.72 from `ggml.h`,
`ggml-alloc.h`, `ggml-backend.h`, `ggml-cpu.h` and `gguf.h` (allowlist
`(ggml|gguf)_.*` / `(GGML|GGUF)_.*`, `--no-prepend-enum-name`,
`--with-derive-default`, `--no-layout-tests`, `--use-core`) and committed, so
users need no libclang. Regenerate it when the submodule commit changes.
