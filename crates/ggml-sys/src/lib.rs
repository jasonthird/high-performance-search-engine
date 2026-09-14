//! Raw bindings to the vendored ggml (`vendor/ggml`, a pinned git submodule).
//!
//! `bindings.rs` is generated once with bindgen and committed, so building
//! hips needs a C/C++ toolchain and CMake but not libclang. Regenerate it
//! whenever the submodule moves (see README.md next to this crate).
#![no_std]
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, unnecessary_transmutes, clippy::all)]

include!("bindings.rs");
