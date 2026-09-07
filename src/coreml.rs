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

fn load_model(path: &Path) -> anyhow::Result<Retained<MLModel>> {
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
    unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
        .map_err(|e| anyhow::anyhow!("CoreML load {}: {e:?}", path.display()))
}

impl CoreMlEncoder {
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
    Ok(flat
        .chunks_exact(CODERANK_DIM)
        .map(|c| c.to_vec())
        .collect())
}
