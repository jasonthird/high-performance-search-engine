//! CoreML encoder backend: the same CodeRankEmbed weights compiled for the
//! Apple Neural Engine.
//!
//! Measured on an M3 (see `scripts/ane-prototype/README.md`): 25.4k tok/s
//! for document batches vs candle/Metal's 6.3k, and 3.8 ms vs ~6 ms for a
//! batch-1 query forward — with mean cosine 0.99968 against the fp32
//! reference, tighter than the f16 sidecar's own rounding. The ANE is also
//! the low-power engine, so sustained indexing barely throttles.
//!
//! CoreML wants static shapes, so the encoder loads a small family of
//! compiled models from `<cache>/coreml/coderank_b{B}_s{S}.mlmodelc`
//! (produced by `scripts/ane-prototype/ane_convert.py` + `coremlcompiler`):
//! batch-8 document models at several sequence lengths, and a batch-1
//! short-sequence query model. Inputs are padded to the smallest model
//! that fits. When no compiled models exist the caller falls back to the
//! candle backend, which remains the portable implementation.

#![allow(clippy::too_many_arguments)]

use std::path::{Path, PathBuf};

use anyhow::Context;
use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSNumber, NSString, NSURL};

use crate::embeddings::CODERANK_DIM;

/// One compiled static-shape model, loaded on first use: an unused shape
/// costs nothing (a query session never touches the document models), and
/// the first-ever load of each shape pays a one-time ANE plan compilation
/// that the OS caches across processes.
struct ShapeModel {
    batch: usize,
    seq: usize,
    path: PathBuf,
    model: std::cell::OnceCell<Retained<MLModel>>,
    batch_valid: std::cell::OnceCell<bool>,
}

impl ShapeModel {
    /// A compiled model must preserve row identity under batch permutation.
    /// Some traced artifacts specialize masking/unpadding to example inputs.
    fn batch_stable(&self) -> anyhow::Result<bool> {
        if let Some(&valid) = self.batch_valid.get() { return Ok(valid); }
        let rows: Vec<Vec<u32>> = (0..self.batch).map(|i| {
            let len = (4 + i * 5).min(self.seq);
            let mut row = vec![2000 + i as u32; len];
            if len >= 2 { row[0] = 101; row[len - 1] = 102; }
            row
        }).collect();
        let forward: Vec<_> = rows.iter().map(Vec::as_slice).collect();
        let backward: Vec<_> = forward.iter().copied().rev().collect();
        let a = run(self, &forward)?;
        let b = run(self, &backward)?;
        let valid = a.iter().zip(b.iter().rev()).all(|(a,b)| {
            let dot: f32 = a.iter().zip(b).map(|(x,y)| x*y).sum();
            dot.is_finite() && dot > 0.995
        });
        self.batch_valid.set(valid).ok();
        if !valid { eprintln!("warning: {} failed batch-equivalence validation; skipping this CoreML shape", self.path.display()); }
        Ok(valid)
    }

    fn model(&self) -> anyhow::Result<&Retained<MLModel>> {
        if let Some(m) = self.model.get() {
            return Ok(m);
        }
        let loaded = load_model(&self.path)?;
        Ok(self.model.get_or_init(|| loaded))
    }
}

pub struct CoreMlEncoder {
    /// Document models, ascending by sequence length. Scheduling uses their
    /// minimum batch capacity (8 in the intended family).
    docs: Vec<ShapeModel>,
    /// Batch-1 short-sequence model for queries, when present.
    query: Option<ShapeModel>,
}

#[cfg(test)]
mod shape_tests {
    use super::*;

    fn shape(batch: usize, seq: usize) -> ShapeModel {
        ShapeModel { batch, seq, path: PathBuf::new(), model: std::cell::OnceCell::new(), batch_valid: std::cell::OnceCell::new() }
    }

    #[test]
    fn native_diagnostics_go_to_stderr_and_stdout_is_restored_on_error() {
        const CHILD: &str = "HIPS_TEST_NATIVE_DIAGNOSTICS_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let result = (|| -> anyhow::Result<()> {
                let _outer = NativeDiagnostics::stderr()?;
                let _inner = NativeDiagnostics::stderr()?;
                unsafe { libc::printf(c"buffered-native-diagnostic\n".as_ptr()); }
                anyhow::bail!("prediction failed")
            })();
            assert!(result.is_err());
            println!("stdout-restored");
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "coreml::shape_tests::native_diagnostics_go_to_stderr_and_stdout_is_restored_on_error", "--nocapture"])
            .env(CHILD, "1").output().unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("stdout-restored"), "{stdout}");
        assert!(!stdout.contains("buffered-native-diagnostic"), "{stdout}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("buffered-native-diagnostic"));
    }

    #[test]
    fn cache_probe_creates_and_cleans_up_in_writable_directory() {
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("new/cache");
        check_cache_writable(&cache).unwrap();
        assert_eq!(std::fs::read_dir(cache).unwrap().count(), 0);
    }

    #[test]
    fn cache_probe_reports_path_and_permission_failure() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("denied");
        std::fs::create_dir(&cache).unwrap();
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o555)).unwrap();
        let result = check_cache_writable(&cache);
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o755)).unwrap();
        if unsafe { libc::geteuid() } != 0 {
            let err = result.unwrap_err();
            let diagnostic = format!("{err:#}");
            assert!(diagnostic.contains("CoreML/E5RT cache is not writable"));
            assert!(diagnostic.contains(cache.to_str().unwrap()));
            assert!(diagnostic.contains("Permission denied"));
        }
        // File-in-the-way is deterministic even when tests run as root.
        let file = temp.path().join("file");
        std::fs::write(&file, "occupied").unwrap();
        assert!(check_cache_writable(&file.join("cache")).is_err());
    }

    #[test]
    fn partial_families_cannot_silently_shorten_documents() {
        let query_only = CoreMlEncoder { docs: vec![], query: Some(shape(1, 64)) };
        assert!(!query_only.supports_documents(512));
        assert!(query_only.supports_queries(64));
        assert!(!query_only.supports_queries(128));
        let partial = CoreMlEncoder { docs: vec![shape(8, 128)], query: None };
        assert!(!partial.supports_documents(512));
        assert!(partial.encode_docs(&[vec![1; 129]]).is_err());
        let mixed = CoreMlEncoder { docs: vec![shape(8, 64), shape(4, 512)], query: None };
        assert!(mixed.supports_documents(512));
        assert_eq!(mixed.doc_batch(), 4);
    }
}

/// Where compiled models live: `<csearch cache>/coreml`.
pub fn model_dir() -> PathBuf {
    crate::codeindex::cache_root().join("coreml")
}

/// CoreML's device-specialized cache is separate from our compiled models.
/// The public MLModelConfiguration API does not expose a relocation option.
fn runtime_cache_dir() -> anyhow::Result<PathBuf> {
    use objc2_foundation::{
        NSBundle, NSProcessInfo, NSSearchPathDirectory, NSSearchPathDomainMask,
        NSSearchPathForDirectoriesInDomains,
    };
    let dirs = NSSearchPathForDirectoriesInDomains(
        NSSearchPathDirectory::CachesDirectory,
        NSSearchPathDomainMask::UserDomainMask,
        true,
    );
    let base = dirs.firstObject().context("macOS has no user cache directory")?;
    let name = NSBundle::mainBundle().bundleIdentifier()
        .unwrap_or_else(|| NSProcessInfo::processInfo().processName());
    Ok(PathBuf::from(base.to_string()).join(name.to_string())
        .join("com.apple.e5rt.e5bundlecache"))
}

/// Probe real create/write operations: mode bits and access(2) do not account
/// for sandbox rules. This is a best-effort preflight, not a guarantee that
/// every OS-specific runtime subdirectory will be writable.
fn check_cache_writable(dir: &Path) -> anyhow::Result<()> {
    (|| -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        let probe = tempfile::Builder::new().prefix("hips-probe-").tempdir_in(dir)?;
        std::fs::write(probe.path().join("probe"), b"cache probe")?;
        probe.close()?;
        Ok(())
    })().with_context(|| format!("CoreML/E5RT cache is not writable at {}; set HIPS_COREML_CACHE_DIR to a writable directory (CLI)", dir.display()))
}

/// Some E5RT/IOSurface errors use buffered C stdout instead of NSError or
/// stderr. Keep them off CLI JSONL and MCP protocol output. Holding Rust's
/// stdout lock serializes these calls with Rust output; flush C stdio before
/// restoring the descriptor so delayed native messages cannot leak at exit.
struct NativeDiagnostics {
    saved: std::os::fd::OwnedFd,
    _stdout: std::io::StdoutLock<'static>,
}

impl NativeDiagnostics {
    fn stderr() -> anyhow::Result<Self> {
        use std::io::Write;
        use std::os::fd::FromRawFd;
        let mut stdout = std::io::stdout().lock();
        stdout.flush()?;
        unsafe {
            libc::fflush(std::ptr::null_mut());
            let saved = libc::fcntl(libc::STDOUT_FILENO, libc::F_DUPFD_CLOEXEC, 0);
            anyhow::ensure!(saved >= 0, "save stdout for CoreML diagnostics: {}", std::io::Error::last_os_error());
            let saved = std::os::fd::OwnedFd::from_raw_fd(saved);
            anyhow::ensure!(libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) >= 0,
                "redirect CoreML diagnostics: {}", std::io::Error::last_os_error());
            Ok(Self { saved, _stdout: stdout })
        }
    }
}

impl Drop for NativeDiagnostics {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        unsafe {
            libc::fflush(std::ptr::null_mut());
            libc::dup2(self.saved.as_raw_fd(), libc::STDOUT_FILENO);
        }
    }
}

fn load_model(path: &Path) -> anyhow::Result<Retained<MLModel>> {
    let _diagnostics = NativeDiagnostics::stderr()?;
    check_cache_writable(&runtime_cache_dir()?)?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    let config = unsafe { MLModelConfiguration::new() };
    let units = match std::env::var("HIPS_COREML_COMPUTE_UNITS").as_deref() {
        Ok("cpu") => MLComputeUnits::CPUOnly,
        Ok("cpu-and-ane") | Err(std::env::VarError::NotPresent) => MLComputeUnits::CPUAndNeuralEngine,
        _ => anyhow::bail!("HIPS_COREML_COMPUTE_UNITS must be cpu or cpu-and-ane"),
    };
    if crate::verbosity::verbose() {
        eprintln!("CoreML compute units: {}; runtime cache: {}",
            if units == MLComputeUnits::CPUOnly { "CPU only" } else { "CPU and Neural Engine" },
            runtime_cache_dir()?.display());
    }
    unsafe { config.setComputeUnits(units) };
    unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
        .map_err(|e| anyhow::anyhow!("CoreML load {}: {e:?}", path.display()))
}

impl CoreMlEncoder {
    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({
            "compute_units": std::env::var("HIPS_COREML_COMPUTE_UNITS").unwrap_or_else(|_| "cpu-and-ane".into()),
            // Allowed units do not prove which device executed each operation.
            "execution_device": "not_observed",
            "runtime_cache_dir": runtime_cache_dir().ok(),
            "loaded_models": self.docs.iter().chain(self.query.iter())
                .filter(|m| m.model.get().is_some()).map(|m| &m.path).collect::<Vec<_>>(),
        })
    }

    /// Load the compiled model family, or `None` when the directory holds
    /// no usable models (the candle backend takes over).
    pub fn load() -> Option<Self> {
        let dir = model_dir();
        let entries = std::fs::read_dir(&dir).ok()?;
        let mut docs = Vec::new();
        let mut query: Option<ShapeModel> = None;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(spec) = name
                .strip_prefix("coderank_b")
                .and_then(|s| s.strip_suffix(".mlmodelc"))
            else {
                continue;
            };
            let Some((b, s)) = spec.split_once("_s") else { continue };
            let (Ok(batch), Ok(seq)) = (b.parse::<usize>(), s.parse::<usize>()) else {
                continue;
            };
            if batch == 0 || seq == 0 { continue; }
            let sm = ShapeModel {
                batch,
                seq,
                path: entry.path(),
                model: std::cell::OnceCell::new(),
                batch_valid: std::cell::OnceCell::new(),
            };
            if batch == 1 {
                if query.as_ref().is_none_or(|q| q.seq < seq) { query = Some(sm); }
            } else {
                docs.push(sm);
            }
        }
        docs.sort_by_key(|m| m.seq);
        if docs.is_empty() && query.is_none() {
            return None;
        }
        Some(Self { docs, query })
    }

    pub fn has_query_model(&self) -> bool {
        self.query.is_some()
    }

    /// Only select CoreML when installed shapes preserve the requested cap.
    pub fn supports_documents(&self, cap: usize) -> bool {
        self.docs.iter().any(|m| m.seq >= cap)
    }

    pub fn supports_queries(&self, cap: usize) -> bool {
        self.query.as_ref().is_some_and(|m| m.seq >= cap)
    }

    pub fn doc_batch(&self) -> usize {
        self.docs.iter().map(|m| m.batch).min().unwrap_or(8)
    }

    /// Encode one query's token ids (already truncated to the query cap).
    /// Returns `None` when no batch-1 model is installed.
    pub fn encode_query(&self, ids: &[u32]) -> Option<anyhow::Result<Vec<f32>>> {
        let m = self.query.as_ref()?;
        if ids.len() > m.seq {
            return None; // longer than the compiled shape: caller falls back
        }
        Some(run(m, &[ids]).map(|mut v| v.pop().expect("one row")))
    }

    /// Encode up to `doc_batch()` documents, choosing the smallest fitting
    /// shape. Never silently truncate when the installed family is incomplete.
    pub fn encode_docs(&self, ids: &[Vec<u32>]) -> anyhow::Result<Vec<Vec<f32>>> {
        anyhow::ensure!(!self.docs.is_empty(), "no CoreML document models");
        let max_seq = self.docs.last().expect("non-empty").seq;
        let longest = ids.iter().map(|r| r.len()).max().unwrap_or(1);
        anyhow::ensure!(longest <= max_seq, "CoreML document exceeds installed shapes");
        let mut selected = None;
        for m in self.docs.iter().filter(|m| m.seq >= longest) {
            if m.batch_stable()? { selected = Some(m); break; }
        }
        let m = selected.context("no CoreML shape passed batch-equivalence validation; use HIPS_ENCODER=candle and regenerate the compiled family")?;
        anyhow::ensure!(
            ids.len() <= m.batch,
            "batch {} exceeds compiled batch {}",
            ids.len(),
            m.batch
        );
        let rows: Vec<&[u32]> = ids.iter().map(|r| &r[..r.len().min(m.seq)]).collect();
        let mut out = run(m, &rows)?;
        out.truncate(ids.len());
        Ok(out)
    }
}

/// Run one prediction: pad `rows` into the model's (batch, seq) int32
/// arrays, predict, and read back the normalized embeddings.
fn run(m: &ShapeModel, rows: &[&[u32]]) -> anyhow::Result<Vec<Vec<f32>>> {
    let _diagnostics = NativeDiagnostics::stderr()?;
    let (batch, seq) = (m.batch, m.seq);
    anyhow::ensure!(rows.len() <= batch, "too many rows for compiled batch");

    let shape = NSArray::from_retained_slice(&[
        NSNumber::new_usize(batch),
        NSNumber::new_usize(seq),
    ]);
    let make = || unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape,
            MLMultiArrayDataType::Int32,
        )
        .map_err(|e| anyhow::anyhow!("MLMultiArray alloc: {e:?}"))
    };
    let ids_arr = make()?;
    let mask_arr = make()?;
    unsafe {
        #[allow(deprecated)]
        let ids_ptr = ids_arr.dataPointer().as_ptr() as *mut i32;
        #[allow(deprecated)]
        let mask_ptr = mask_arr.dataPointer().as_ptr() as *mut i32;
        std::ptr::write_bytes(ids_ptr, 0, batch * seq);
        std::ptr::write_bytes(mask_ptr, 0, batch * seq);
        for (i, row) in rows.iter().enumerate() {
            for (j, &t) in row.iter().enumerate() {
                *ids_ptr.add(i * seq + j) = t as i32;
                *mask_ptr.add(i * seq + j) = 1;
            }
        }
        // Padding rows beyond `rows.len()` keep an all-zero mask; NomicBert
        // masks them out and their (discarded) outputs cost nothing extra
        // in a static-shape model.
    }

    let provider = unsafe {
        let keys = [NSString::from_str("input_ids"), NSString::from_str("attention_mask")];
        let vals = [
            MLFeatureValue::featureValueWithMultiArray(&ids_arr),
            MLFeatureValue::featureValueWithMultiArray(&mask_arr),
        ];
        let dict = objc2_foundation::NSDictionary::from_retained_objects::<NSString>(
            &[&keys[0], &keys[1]],
            &[
                Retained::into_super(Retained::into_super(vals[0].clone())),
                Retained::into_super(Retained::into_super(vals[1].clone())),
            ],
        );
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &dict,
        )
        .map_err(|e| anyhow::anyhow!("feature provider: {e:?}"))?
    };

    let provider = objc2::runtime::ProtocolObject::from_retained(provider);
    let t_pred = std::time::Instant::now();
    let model = m.model()?;
    let output = unsafe { model.predictionFromFeatures_error(&provider) }
        .map_err(|e| anyhow::anyhow!("CoreML prediction: {e:?}"))?;
    if std::env::var_os("HPS_EMBED_PROFILE").is_some() {
        eprintln!("  coreml b{}s{} rows={} predict {:.1} ms", batch, seq, rows.len(), t_pred.elapsed().as_secs_f64() * 1000.0);
    }
    let value = unsafe { output.featureValueForName(&NSString::from_str("embedding")) }
        .context("prediction has no 'embedding' output")?;
    let arr = unsafe { value.multiArrayValue() }.context("embedding is not a multiarray")?;

    let count = unsafe { arr.count() } as usize;
    anyhow::ensure!(
        count == batch * CODERANK_DIM,
        "unexpected embedding count {count}"
    );
    let dtype = unsafe { arr.dataType() };
    let mut flat = vec![0f32; count];
    unsafe {
        #[allow(deprecated)]
        let p = arr.dataPointer().as_ptr();
        if dtype == MLMultiArrayDataType::Float32 {
            std::ptr::copy_nonoverlapping(p as *const f32, flat.as_mut_ptr(), count);
        } else if dtype == MLMultiArrayDataType::Float16 {
            let h = p as *const u16;
            for (i, v) in flat.iter_mut().enumerate() {
                *v = crate::embeddings::f16_to_f32(*h.add(i));
            }
        } else {
            anyhow::bail!("unexpected embedding dtype {dtype:?}");
        }
    }
    // Padded rows may have an all-zero mask; only real rows are observable.
    let vectors: Vec<_> = flat.chunks_exact(CODERANK_DIM)
        .take(rows.len()).map(|c| c.to_vec()).collect();
    for vector in &vectors {
        crate::embedder::validate_vector(vector).context("invalid CoreML prediction")?;
    }
    Ok(vectors)
}
