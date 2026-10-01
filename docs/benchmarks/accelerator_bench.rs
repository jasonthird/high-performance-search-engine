//! Compare safe Candle/Metal FP32 matrix-vector scoring against stored FP16
//! CPU scoring. Includes query upload, score readback and top-k selection;
//! reports resident tensor preparation separately. No model downloads.
#[cfg(not(all(feature = "semantic", target_os = "macos")))]
fn main() {
    eprintln!("requires macOS and --features semantic");
}

#[cfg(all(feature = "semantic", target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use high_performance_search_engine::{
        embeddings::{self, EmbeddingStore},
        hybrid::{segmented_semantic_pool, SegmentStores},
    };
    use std::time::Instant;
    let device = Device::new_metal(0)?;
    let sizes = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "1500,10000,100000".into());
    println!("docs,prepare_ms,cpu_p50_ms,metal_p50_ms,max_score_error,top200_overlap");
    for size in sizes.split(',') {
        let n: usize = size.parse()?;
        anyhow::ensure!(n >= 200, "need at least 200 documents");
        let dim = 768;
        let mut seed = 19u64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / 8388608.0 - 1.0
        };
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| random()).collect();
                embeddings::l2_normalize(&mut v);
                v
            })
            .collect();
        let mut query: Vec<f32> = (0..dim).map(|_| random()).collect();
        embeddings::l2_normalize(&mut query);
        let dir = tempfile::tempdir()?;
        embeddings::write_f16(dir.path(), dim as u32, &vectors)?;
        drop(vectors);
        let stores = SegmentStores {
            stores: vec![Some(EmbeddingStore::open(dir.path())?)],
        };
        let bytes = std::fs::read(dir.path().join(embeddings::EMBEDDINGS_FILE))?;
        let t = Instant::now();
        let matrix = Tensor::from_raw_buffer(
            &bytes[embeddings::HEADER_LEN..],
            DType::F16,
            &[n, dim],
            &device,
        )?
        .to_dtype(DType::F32)?;
        device.synchronize()?;
        let prepare_ms = t.elapsed().as_secs_f64() * 1000.0;
        let score = || -> anyhow::Result<Vec<f32>> {
            let q = Tensor::from_slice(&query, (dim, 1), &device)?;
            Ok(matrix.matmul(&q)?.flatten_all()?.to_vec1::<f32>()?)
        };
        let scores = score()?;
        let store = stores.stores[0].as_ref().unwrap();
        let error = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (s - store.cosine(i as u32, &query)).abs())
            .fold(0.0f32, f32::max);
        anyhow::ensure!(error < 1e-5, "score error {error}");
        let select = |scores: Vec<f32>| {
            let mut hits: Vec<_> = scores.into_iter().enumerate().collect();
            let cmp = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
            hits.select_nth_unstable_by(199, cmp);
            hits.truncate(200);
            hits.sort_by(cmp);
            hits
        };
        let truth = segmented_semantic_pool(&stores, &|_, _| true, &query, 200);
        let gpu = select(scores);
        let overlap = gpu
            .iter()
            .filter(|(i, _)| truth.iter().any(|h| h.doc_id as usize == *i))
            .count();
        let mut cpu_ms = Vec::new();
        let mut metal_ms = Vec::new();
        for round in 0..16 {
            for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                let t = Instant::now();
                if mode == 0 {
                    std::hint::black_box(segmented_semantic_pool(
                        &stores,
                        &|_, _| true,
                        &query,
                        200,
                    ));
                    cpu_ms.push(t.elapsed().as_secs_f64() * 1000.0);
                } else {
                    std::hint::black_box(select(score()?));
                    metal_ms.push(t.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
        cpu_ms.sort_by(f64::total_cmp);
        metal_ms.sort_by(f64::total_cmp);
        println!(
            "{n},{prepare_ms:.3},{:.3},{:.3},{error:.9},{overlap}/200",
            cpu_ms[7], metal_ms[7]
        );
    }
    Ok(())
}
