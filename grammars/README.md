# Native grammar artifacts

This is a separate Cargo workspace so grammar parse tables never link into
the hips executable. Each small cdylib wrapper exports `hips_language`, backed
by the upstream grammar's `LanguageFn`. TypeScript and TSX are separate assets.

The registry contains 41 variants. Queries remain embedded in the CLI under
`tree-sitters/`; these packages contain parser code and tables only. See
[theory](../docs/THEORY.md#15-declaration-chunks-and-loadable-grammars) for
chunking and lifetime details, and the [main README](../README.md) for usage.

Build and test locally (a C/C++ compiler and Rust are required):

```sh
cargo build --manifest-path grammars/Cargo.toml --release --locked
HIPS_GRAMMAR_DIR="$PWD/grammars/target/release" HIPS_GRAMMAR_OFFLINE=1 cargo test --locked
```

Package maintainers can include no grammars, any subset, or all of them. To
build a subset, append repeated package selections, for example
`-p hips-grammar-python -p hips-grammar-rust`, to the build command. Install
the resulting libraries wherever the package manager prefers, and set
`HIPS_GRAMMAR_DIR` in its launcher to that trusted directory.

Lookup order is the packaged directory, the user's versioned cache, then an
on-demand download. Bundled libraries are used in place, without duplicating
them in the cache. Set `HIPS_GRAMMAR_OFFLINE=1` to disable automatic downloads;
installed and cached grammars still work, and missing languages use heuristic
chunks. This supports both minimal packages and self-contained offline packages.

For example, from the repository root:

```sh
cargo build --manifest-path grammars/Cargo.toml --release --locked \
  -p hips-grammar-python -p hips-grammar-rust
HIPS_GRAMMAR_DIR="$PWD/grammars/target/release" hips index-repo --root . --lexical
```

Only the selected packages are built by this command; a reused output directory
may still contain libraries from earlier builds. Package the selected files
explicitly. `HIPS_GRAMMAR_DIR` expects unprefixed build filenames such as
`libhips_grammar_python.dylib` or `libhips_grammar_python.so`; when installing
release assets manually, remove the target prefix from the destination name.
An invalid installed library takes precedence too: it warns and uses heuristic
chunks, rather than silently downloading a replacement. Grammar offline mode
does not control Hugging Face encoder downloads.

The grammar workflow builds 41 libraries on each of macOS arm64/x86_64 and
Linux glibc arm64/x86_64, tests declaration fixtures, and uploads each library
and its SHA-256 sidecar. Tags `grammars-v*` publish a release only after all
platforms pass. Asset names are `<target>-libhips_grammar_<name>.dylib` (macOS)
or `<target>-libhips_grammar_<name>.so` (Linux). Linux artifacts require the
Ubuntu 22.04 glibc baseline or newer; prebuilt downloads do not support musl.
Native libraries can be supplied through `HIPS_GRAMMAR_DIR` on other targets,
subject to the upstream grammar and loader supporting that target.

The loader pins `grammars-v1` in `src/grammar.rs`. Publish that release before
distributing a CLI that uses it. For any grammar source or build change, use a
new tag and update that constant; never replace an existing release's assets.
Keep the workspace lockfile committed. Add new languages as a wrapper crate,
a registry entry in `src/treesit.rs`, a query, and a declaration fixture.

Normal users download only the languages they encounter. Libraries are cached
under `$XDG_CACHE_HOME/csearch/grammars/<release>/<target>` (default
`~/.cache/csearch/grammars/...`). Downloads have a timeout, size limit, checksum
verification, ABI validation, and atomic publication. Missing/offline/broken
grammars warn once per language per process and use heuristic chunks; a new
process retries. Loaded libraries stay resident alongside their language for
the process lifetime. No compiler is needed on a user's machine.

`CSEARCH_CACHE_DIR` moves repository and CoreML caches, but does not currently
move the grammar cache. Changing the available libraries also does not
invalidate an existing repository's unchanged-file fingerprint: installing a
previously missing grammar alone will not re-chunk already indexed files.

The local macOS arm64 default-feature release build measured 10,101,008 bytes
(9.6 MiB), with roughly 68 MiB of grammar libraries separately. Sizes vary with
features and platform; users need only their bundled and encountered languages.

CI builds all grammar libraries before testing. Loader tests use localhost HTTP
to exercise checksum validation, invalid-library rejection, concurrent atomic
publication, and parsing through a downloaded library. The release matrix also
runs all language declaration fixtures on each target; publication waits for
every target to pass.
