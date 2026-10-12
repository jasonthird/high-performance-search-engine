//! CodeRankEmbed (NomicBert) on ggml, hand-built: the weights come from a
//! GGUF, each batch becomes one small compute graph, and the scheduler runs
//! it on the GPU (Vulkan on Linux, Metal on macOS) with the CPU backend as
//! the fallback for anything a device cannot do.
//!
//! What the graph does, per layer (post-norm BERT with RoPE and SwiGLU):
//!
//! ```text
//! x   = LayerNorm(tok_embd[ids] + type_embd[0])
//! qkv = x · Wqkv                                    (no bias)
//! q,k = RoPE_neox(q, pos), RoPE_neox(k, pos)         (base 1000, full head)
//! a   = softmax(q·kᵀ / √d + mask) · v                (flash attention)
//! x   = LayerNorm(x + a · Wo)
//! x   = LayerNorm(x + (silu(x·Wgate) ⊙ (x·Wup)) · Wdown)
//! out = x[CLS]                                      (one row per sequence)
//! ```
//!
//! Batches are *padded*, not packed: sequences are laid out as
//! `[L, n_seq]` with `L` the longest in the batch, and attention runs as a
//! true batched op with a per-sequence key mask. With length-bucketed
//! batches the padding is small, and attention cost stays `L²` per sequence
//! instead of the `(L·n_seq)²` of a packed token stream with a
//! block-diagonal mask, which for this model is comparable to the matmul
//! cost and is pure waste.
//!
//! Everything here is `unsafe` FFI kept behind a small safe surface: an
//! [`Engine`] owns every ggml object it creates and frees them in `Drop`.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use anyhow::{ensure, Context, Result};
use ggml_sys as g;

/// Graph nodes per batch: about 40 per layer plus inputs and pooling.
const GRAPH_SIZE: usize = 2048;

fn fa_f16_accumulation() -> bool {
    matches!(std::env::var("HIPS_GGML_FA_F16").as_deref(), Ok("1") | Ok("true"))
}

/// `HPS_EMBED_PROFILE=1` prints a per-graph timing line (build, allocation,
/// input upload, compute, readback) to stderr.
fn profiling() -> bool {
    matches!(std::env::var("HPS_EMBED_PROFILE").as_deref(), Ok("1") | Ok("true"))
}

#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
}

impl Hparams {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

/// The tokenizer's vocabulary as stored in the GGUF.
pub struct Vocab {
    pub tokens: Vec<String>,
    pub token_types: Vec<i32>,
    pub unk: u32,
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
}

#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// Use the first GPU device ggml registered; CPU-only otherwise.
    pub gpu: bool,
    pub threads: usize,
    /// `ggml_flash_attn_ext` (fused, no score matrix) instead of the naive
    /// matmul + softmax + matmul attention. Both give the same numbers.
    pub flash_attention: bool,
    /// In the last layer, run attention, the output projection and the MLP
    /// for the `[CLS]` rows only: nothing after the last attention mixes
    /// tokens, so the other rows are never read. Same numbers, ~1/12 less
    /// work per document.
    pub cls_only_last_layer: bool,
    /// Cast K and V to F16 before flash attention. Vulkan and the CPU take
    /// F32 directly (two fewer nodes per layer); Metal is kept on the safe
    /// side until measured.
    pub cast_kv_f16: bool,
}

struct Layer {
    wqkv: *mut g::ggml_tensor,
    wo: *mut g::ggml_tensor,
    ln1_w: *mut g::ggml_tensor,
    ln1_b: *mut g::ggml_tensor,
    /// `ffn_gate` and `ffn_up` fused along the output dimension (gate rows
    /// first): one matmul instead of two, then a single SwiGLU node.
    w_gate_up: *mut g::ggml_tensor,
    w_down: *mut g::ggml_tensor,
    ln2_w: *mut g::ggml_tensor,
    ln2_b: *mut g::ggml_tensor,
}

/// A built and allocated graph kept for the next call with the same shape.
/// Queries always have the same shape (one sequence, padded to a fixed
/// length), so they skip graph construction and the scheduler's allocation
/// pass entirely: about a quarter of a query's wall time.
struct CachedGraph {
    ctx: *mut g::ggml_context,
    gf: *mut g::ggml_cgraph,
    l: i64,
    ns: i64,
    cls_only: bool,
    t_ids: *mut g::ggml_tensor,
    t_pos: *mut g::ggml_tensor,
    t_mask: *mut g::ggml_tensor,
    t_cls_mask: *mut g::ggml_tensor,
    t_cls: *mut g::ggml_tensor,
    out: *mut g::ggml_tensor,
}

pub struct Engine {
    hp: Hparams,
    weights_ctx: *mut g::ggml_context,
    weights_buf: g::ggml_backend_buffer_t,
    tok_embd: *mut g::ggml_tensor,
    type_embd: *mut g::ggml_tensor,
    tok_norm_w: *mut g::ggml_tensor,
    tok_norm_b: *mut g::ggml_tensor,
    layers: Vec<Layer>,
    backends: Vec<g::ggml_backend_t>,
    sched: g::ggml_backend_sched_t,
    cache: Option<CachedGraph>,
    device: String,
    flash: bool,
    cls_only: bool,
    cast_kv: bool,
}

// One thread drives the engine at a time (the embedder wraps it in a
// Mutex); the ggml objects themselves are plain heap allocations.
unsafe impl Send for Engine {}

struct Gguf(*mut g::gguf_context);

impl Drop for Gguf {
    fn drop(&mut self) {
        unsafe { g::gguf_free(self.0) }
    }
}

impl Gguf {
    /// Metadata only (`ctx` is None) or metadata plus a ggml context holding
    /// unallocated tensor descriptors.
    fn open(path: &Path, ctx: Option<&mut *mut g::ggml_context>) -> Result<Self> {
        let cpath = CString::new(path.as_os_str().as_encoded_bytes())
            .context("model path contains a NUL byte")?;
        let params = g::gguf_init_params {
            no_alloc: true,
            ctx: ctx.map_or(ptr::null_mut(), |c| c as *mut _),
        };
        let raw = unsafe { g::gguf_init_from_file(cpath.as_ptr(), params) };
        ensure!(!raw.is_null(), "{} is not a readable GGUF file", path.display());
        Ok(Self(raw))
    }

    fn key(&self, name: &str) -> Option<i64> {
        let cname = CString::new(name).ok()?;
        let id = unsafe { g::gguf_find_key(self.0, cname.as_ptr()) };
        (id >= 0).then_some(id)
    }

    fn u32(&self, name: &str) -> Option<u32> {
        let id = self.key(name)?;
        let value = match unsafe { g::gguf_get_kv_type(self.0, id) } {
            g::GGUF_TYPE_UINT32 => unsafe { g::gguf_get_val_u32(self.0, id) },
            g::GGUF_TYPE_INT32 => (unsafe { g::gguf_get_val_i32(self.0, id) }) as u32,
            g::GGUF_TYPE_UINT64 => (unsafe { g::gguf_get_val_u64(self.0, id) }) as u32,
            _ => return None,
        };
        Some(value)
    }

    fn f32(&self, name: &str) -> Option<f32> {
        let id = self.key(name)?;
        (unsafe { g::gguf_get_kv_type(self.0, id) } == g::GGUF_TYPE_FLOAT32)
            .then(|| unsafe { g::gguf_get_val_f32(self.0, id) })
    }

    fn string(&self, name: &str) -> Option<String> {
        let id = self.key(name)?;
        if unsafe { g::gguf_get_kv_type(self.0, id) } != g::GGUF_TYPE_STRING {
            return None;
        }
        Some(unsafe { CStr::from_ptr(g::gguf_get_val_str(self.0, id)) }.to_string_lossy().into_owned())
    }

    fn strings(&self, name: &str) -> Option<Vec<String>> {
        let id = self.key(name)?;
        if unsafe { g::gguf_get_kv_type(self.0, id) } != g::GGUF_TYPE_ARRAY
            || unsafe { g::gguf_get_arr_type(self.0, id) } != g::GGUF_TYPE_STRING
        {
            return None;
        }
        let n = unsafe { g::gguf_get_arr_n(self.0, id) };
        Some(
            (0..n)
                .map(|i| unsafe { CStr::from_ptr(g::gguf_get_arr_str(self.0, id, i)) }.to_string_lossy().into_owned())
                .collect(),
        )
    }

    fn i32s(&self, name: &str) -> Option<Vec<i32>> {
        let id = self.key(name)?;
        if unsafe { g::gguf_get_kv_type(self.0, id) } != g::GGUF_TYPE_ARRAY
            || unsafe { g::gguf_get_arr_type(self.0, id) } != g::GGUF_TYPE_INT32
        {
            return None;
        }
        let n = unsafe { g::gguf_get_arr_n(self.0, id) };
        let data = unsafe { g::gguf_get_arr_data(self.0, id) } as *const i32;
        Some(unsafe { std::slice::from_raw_parts(data, n) }.to_vec())
    }

    fn vocab(&self) -> Result<Vocab> {
        let tokens = self.strings("tokenizer.ggml.tokens").context("GGUF has no tokenizer.ggml.tokens")?;
        let token_types = self.i32s("tokenizer.ggml.token_type").unwrap_or_default();
        let id = |key: &str| self.u32(key).with_context(|| format!("GGUF has no {key}"));
        Ok(Vocab {
            unk: id("tokenizer.ggml.unknown_token_id")?,
            cls: id("tokenizer.ggml.bos_token_id")?,
            sep: id("tokenizer.ggml.eos_token_id")?,
            pad: self.u32("tokenizer.ggml.padding_token_id").unwrap_or(0),
            tokens,
            token_types,
        })
    }

    fn hparams(&self) -> Result<Hparams> {
        let arch = self.string("general.architecture").unwrap_or_default();
        ensure!(arch == "nomic-bert", "expected a nomic-bert GGUF, found architecture {arch:?}");
        let u = |key: &str| self.u32(&format!("nomic-bert.{key}")).with_context(|| format!("GGUF has no nomic-bert.{key}"));
        Ok(Hparams {
            n_layer: u("block_count")? as usize,
            n_embd: u("embedding_length")? as usize,
            n_head: u("attention.head_count")? as usize,
            n_ff: u("feed_forward_length")? as usize,
            n_ctx: u("context_length")? as usize,
            rope_base: self.f32("nomic-bert.rope.freq_base").unwrap_or(1000.0),
            norm_eps: self.f32("nomic-bert.attention.layer_norm_epsilon").unwrap_or(1e-12),
        })
    }

    /// Byte range of each tensor in the file, by name.
    fn tensor_ranges(&self) -> std::collections::HashMap<String, (u64, usize)> {
        let data_offset = unsafe { g::gguf_get_data_offset(self.0) };
        let n = unsafe { g::gguf_get_n_tensors(self.0) };
        (0..n)
            .map(|i| {
                let name = unsafe { CStr::from_ptr(g::gguf_get_tensor_name(self.0, i)) }.to_string_lossy().into_owned();
                let offset = data_offset + unsafe { g::gguf_get_tensor_offset(self.0, i) };
                let size = unsafe { g::gguf_get_tensor_size(self.0, i) };
                (name, (offset as u64, size))
            })
            .collect()
    }
}

/// The vocabulary alone, without touching the weights: enough for
/// tokenizing when another backend (CoreML) does the inference.
pub fn read_vocab(path: &Path) -> Result<Vocab> {
    Gguf::open(path, None)?.vocab()
}

/// Positional read that leaves no shared cursor behind (`pread` on Unix).
#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

/// Windows' `seek_read` may return short reads and moves the cursor; callers
/// only use positional reads, so the cursor is irrelevant.
#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
        }
    }
    Ok(())
}

/// True if the file starts with the GGUF magic; the cheap check a
/// download runs before publishing the file.
pub fn looks_like_gguf(path: &Path) -> Result<()> {
    let mut magic = [0u8; 4];
    read_exact_at(&std::fs::File::open(path)?, &mut magic, 0)?;
    ensure!(&magic == b"GGUF", "{} is not a GGUF file", path.display());
    Ok(())
}

/// ggml's own log lines (device enumeration, buffer sizes) go to stderr
/// only with `-v`; warnings and errors always do. An agent reading the CLI
/// pays for every line, so the default is silence.
unsafe extern "C" fn log_callback(level: g::ggml_log_level, text: *const std::ffi::c_char, _user: *mut std::ffi::c_void) {
    if text.is_null() {
        return;
    }
    let important = matches!(level, g::GGML_LOG_LEVEL_WARN | g::GGML_LOG_LEVEL_ERROR);
    if important || crate::verbosity::verbose() {
        let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
        eprint!("{text}");
    }
}

/// Queries are padded to this many tokens so every query reuses one cached
/// graph; `MAX_QUERY_SEQ` is a multiple of it.
const QUERY_PAD: usize = 32;

impl Engine {
    pub fn load(path: &Path, opts: &EngineOptions) -> Result<(Self, Vocab)> {
        unsafe {
            g::ggml_log_set(Some(log_callback), ptr::null_mut());
            g::ggml_backend_load_all();
        }
        let t_start = std::time::Instant::now();
        // The GGUF's own context only describes the tensors; the weights are
        // laid out in a context of ours so gate and up can be fused.
        let mut meta_ctx: *mut g::ggml_context = ptr::null_mut();
        let gguf = Gguf::open(path, Some(&mut meta_ctx))?;
        ensure!(!meta_ctx.is_null(), "GGUF tensor context was not created");
        let hp = gguf.hparams()?;
        let vocab = gguf.vocab()?;

        let mut backends = Vec::new();
        let mut device = String::from("cpu");
        if opts.gpu {
            let dev = unsafe { g::ggml_backend_dev_by_type(g::GGML_BACKEND_DEVICE_TYPE_GPU) };
            if !dev.is_null() {
                let backend = unsafe { g::ggml_backend_dev_init(dev, ptr::null()) };
                if !backend.is_null() {
                    let name = unsafe { CStr::from_ptr(g::ggml_backend_dev_name(dev)) }.to_string_lossy();
                    let desc = unsafe { CStr::from_ptr(g::ggml_backend_dev_description(dev)) }.to_string_lossy();
                    device = format!("{name} ({desc})");
                    backends.push(backend);
                }
            }
        }
        let t_backend = std::time::Instant::now();
        let cpu_dev = unsafe { g::ggml_backend_dev_by_type(g::GGML_BACKEND_DEVICE_TYPE_CPU) };
        ensure!(!cpu_dev.is_null(), "ggml has no CPU backend");
        let cpu = unsafe { g::ggml_backend_dev_init(cpu_dev, ptr::null()) };
        ensure!(!cpu.is_null(), "ggml CPU backend failed to initialize");
        unsafe { g::ggml_backend_cpu_set_n_threads(cpu, opts.threads.max(1) as i32) };
        backends.push(cpu);

        let mut engine = Self {
            hp,
            weights_ctx: ptr::null_mut(),
            weights_buf: ptr::null_mut(),
            tok_embd: ptr::null_mut(),
            type_embd: ptr::null_mut(),
            tok_norm_w: ptr::null_mut(),
            tok_norm_b: ptr::null_mut(),
            layers: Vec::new(),
            backends,
            sched: ptr::null_mut(),
            cache: None,
            device,
            flash: opts.flash_attention,
            cls_only: opts.cls_only_last_layer,
            cast_kv: opts.cast_kv_f16,
        };
        // Dropping `engine` on any error below frees whatever was created.
        let t_weights = std::time::Instant::now();
        engine.create_weights(meta_ctx, path, &gguf)?;
        unsafe { g::ggml_free(meta_ctx) };
        if profiling() {
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1000.0;
            eprintln!(
                "  ggml load: metadata {:.1} ms  backend init {:.1} ms  weights upload {:.1} ms",
                ms(t_start, t_backend),
                ms(t_backend, t_weights),
                ms(t_weights, std::time::Instant::now()),
            );
        }
        engine.sched = unsafe {
            g::ggml_backend_sched_new(
                engine.backends.as_mut_ptr(),
                ptr::null_mut(),
                engine.backends.len() as i32,
                GRAPH_SIZE,
                false,
                false,
            )
        };
        ensure!(!engine.sched.is_null(), "ggml scheduler failed to initialize");
        Ok((engine, vocab))
    }

    /// Describe the weights in our own context (fusing each layer's gate
    /// and up projections), allocate them on the first backend, and copy
    /// the bytes in from the file.
    fn create_weights(&mut self, meta_ctx: *mut g::ggml_context, path: &Path, gguf: &Gguf) -> Result<()> {
        let hp = self.hp.clone();
        let ranges = gguf.tensor_ranges();
        let n_tensors = ranges.len();
        let mem = unsafe { g::ggml_tensor_overhead() } * (n_tensors + 8);
        let ctx = unsafe { g::ggml_init(g::ggml_init_params { mem_size: mem, mem_buffer: ptr::null_mut(), no_alloc: true }) };
        ensure!(!ctx.is_null(), "ggml weights context allocation failed");
        self.weights_ctx = ctx;

        let meta = |name: &str| -> Result<*mut g::ggml_tensor> {
            let cname = CString::new(name)?;
            let t = unsafe { g::ggml_get_tensor(meta_ctx, cname.as_ptr()) };
            ensure!(!t.is_null(), "CodeRankEmbed GGUF lacks tensor {name}");
            Ok(t)
        };
        // (our tensor, [file ranges to concatenate into it])
        let mut uploads: Vec<(*mut g::ggml_tensor, Vec<(u64, usize)>)> = Vec::new();
        let mut fused_uploads: Vec<(*mut g::ggml_tensor, Vec<(u64, usize)>)> = Vec::new();
        let mut plain = |name: &str| -> Result<*mut g::ggml_tensor> {
            let src = meta(name)?;
            let t = unsafe { g::ggml_dup_tensor(ctx, src) };
            let cname = CString::new(name)?;
            unsafe { g::ggml_set_name(t, cname.as_ptr()) };
            let range = *ranges.get(name).with_context(|| format!("no data range for {name}"))?;
            uploads.push((t, vec![range]));
            Ok(t)
        };
        self.tok_embd = plain("token_embd.weight")?;
        self.type_embd = plain("token_types.weight")?;
        self.tok_norm_w = plain("token_embd_norm.weight")?;
        self.tok_norm_b = plain("token_embd_norm.bias")?;
        let mut layers = Vec::with_capacity(hp.n_layer);
        for i in 0..hp.n_layer {
            let name = |suffix: &str| format!("blk.{i}.{suffix}");
            let wqkv = plain(&name("attn_qkv.weight"))?;
            let wo = plain(&name("attn_output.weight"))?;
            let ln1_w = plain(&name("attn_output_norm.weight"))?;
            let ln1_b = plain(&name("attn_output_norm.bias"))?;
            let w_down = plain(&name("ffn_down.weight"))?;
            let ln2_w = plain(&name("layer_output_norm.weight"))?;
            let ln2_b = plain(&name("layer_output_norm.bias"))?;
            // gate|up fused: [n_embd, 2*n_ff], gate rows first. Each source
            // is [n_embd, n_ff] row-major, so the bytes simply concatenate.
            let gate = meta(&name("ffn_gate.weight"))?;
            let up = meta(&name("ffn_up.weight"))?;
            let (gt, ut) = unsafe { ((*gate).type_, (*up).type_) };
            ensure!(gt == ut, "ffn_gate and ffn_up of layer {i} differ in type");
            let fused = unsafe { g::ggml_new_tensor_2d(ctx, gt, hp.n_embd as i64, 2 * hp.n_ff as i64) };
            let fused_name = CString::new(name("ffn_gate_up.weight"))?;
            unsafe { g::ggml_set_name(fused, fused_name.as_ptr()) };
            let gr = *ranges.get(&name("ffn_gate.weight")).context("no range for ffn_gate")?;
            let ur = *ranges.get(&name("ffn_up.weight")).context("no range for ffn_up")?;
            ensure!(gr.1 + ur.1 == unsafe { g::ggml_nbytes(fused) }, "fused gate/up size mismatch in layer {i}");
            fused_uploads.push((fused, vec![gr, ur]));
            layers.push(Layer { wqkv, wo, ln1_w, ln1_b, w_gate_up: fused, w_down, ln2_w, ln2_b });
        }
        drop(plain);
        uploads.extend(fused_uploads);
        self.layers = layers;
        let embd_rows = unsafe { (*self.tok_embd).ne[0] } as usize;
        ensure!(embd_rows == hp.n_embd, "token embedding width {embd_rows} != n_embd {}", hp.n_embd);

        // Weights live on the first backend (the GPU when there is one).
        self.weights_buf = unsafe { g::ggml_backend_alloc_ctx_tensors(ctx, self.backends[0]) };
        ensure!(!self.weights_buf.is_null(), "cannot allocate the CodeRankEmbed weights on {}", self.device);
        let file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        for (tensor, parts) in uploads {
            let total: usize = parts.iter().map(|(_, size)| size).sum();
            ensure!(total == unsafe { g::ggml_nbytes(tensor) }, "tensor byte size mismatch");
            buffer.clear();
            buffer.resize(total, 0);
            let mut at = 0;
            for (offset, size) in parts {
                read_exact_at(&file, &mut buffer[at..at + size], offset).context("read tensor data")?;
                at += size;
            }
            unsafe { g::ggml_backend_tensor_set(tensor, buffer.as_ptr().cast(), 0, total) };
        }
        Ok(())
    }

    pub fn hparams(&self) -> &Hparams {
        &self.hp
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn flash_attention(&self) -> bool {
        self.flash
    }

    /// Whether a GPU backend carries the weights (the CPU backend is always
    /// present as the fallback).
    pub fn on_gpu(&self) -> bool {
        self.backends.len() > 1
    }

    /// Encode a batch: one CLS vector (unnormalized, `n_embd` floats) per
    /// sequence. Each sequence already carries its `[CLS]`/`[SEP]` and is
    /// at most `n_ctx` tokens.
    pub fn forward(&mut self, seqs: &[&[u32]], pad_id: u32) -> Result<Vec<Vec<f32>>> {
        let n_seq = seqs.len();
        ensure!(n_seq > 0, "empty batch");
        let longest = seqs.iter().map(|s| s.len()).max().unwrap_or(0);
        ensure!(longest > 0 && longest <= self.hp.n_ctx, "sequence length {longest} outside 1..={}", self.hp.n_ctx);
        // A lone sequence (a query) is padded to a fixed length so that
        // consecutive calls share one graph: on the GPU that skips the
        // scheduler's allocation pass, a quarter of a query's time. The CPU
        // allocates cheaply and pays linearly for the padding, so it keeps
        // the exact length.
        let len = if n_seq == 1 && self.on_gpu() { longest.div_ceil(QUERY_PAD) * QUERY_PAD } else { longest };
        let len = len.min(self.hp.n_ctx).max(longest);
        let n_tok = len * n_seq;
        let hp = self.hp.clone();
        let (n_embd, n_head, hd) = (hp.n_embd as i64, hp.n_head as i64, hp.head_dim() as i64);
        let (l, ns) = (len as i64, n_seq as i64);
        let cls_only = self.cls_only && n_tok >= 256;
        let t0 = std::time::Instant::now();

        // Host-side inputs.
        let mut ids = vec![pad_id as i32; n_tok];
        for (s, seq) in seqs.iter().enumerate() {
            for (j, &t) in seq.iter().enumerate() {
                ids[s * len + j] = t as i32;
            }
        }
        let pos: Vec<i32> = (0..len as i32).collect();
        let neg_inf = unsafe { g::ggml_fp32_to_fp16(f32::NEG_INFINITY) };
        let zero = unsafe { g::ggml_fp32_to_fp16(0.0) };
        // mask[s][q][k]: 0 for real keys, -inf for padding keys.
        let mut mask = vec![neg_inf; len * len * n_seq];
        let mut cls_mask = vec![neg_inf; len * n_seq];
        for (s, seq) in seqs.iter().enumerate() {
            for q in 0..len {
                mask[(s * len + q) * len..(s * len + q) * len + seq.len()].fill(zero);
            }
            cls_mask[s * len..s * len + seq.len()].fill(zero);
        }
        let cls_idx: Vec<i32> = (0..n_seq).map(|s| (s * len) as i32).collect();

        let reused = self.cache.as_ref().is_some_and(|c| c.l == l && c.ns == ns && c.cls_only == cls_only);
        if !reused {
            if let Some(old) = self.cache.take() {
                unsafe { g::ggml_free(old.ctx) };
            }
            unsafe { g::ggml_backend_sched_reset(self.sched) };
            let built = unsafe { self.build_graph(l, ns, n_embd, n_head, hd, cls_only) }?;
            ensure!(unsafe { g::ggml_backend_sched_alloc_graph(self.sched, built.gf) }, "ggml could not allocate the graph");
            self.cache = Some(built);
        }
        let t_alloc = std::time::Instant::now();
        let c = self.cache.as_ref().expect("graph cached above");
        let set = |t: *mut g::ggml_tensor, data: *const u8, bytes: usize| unsafe {
            g::ggml_backend_tensor_set(t, data.cast(), 0, bytes);
        };
        set(c.t_ids, ids.as_ptr().cast(), std::mem::size_of_val(ids.as_slice()));
        set(c.t_pos, pos.as_ptr().cast(), std::mem::size_of_val(pos.as_slice()));
        set(c.t_mask, mask.as_ptr().cast(), std::mem::size_of_val(mask.as_slice()));
        if !c.t_cls_mask.is_null() {
            set(c.t_cls_mask, cls_mask.as_ptr().cast(), std::mem::size_of_val(cls_mask.as_slice()));
        }
        set(c.t_cls, cls_idx.as_ptr().cast(), std::mem::size_of_val(cls_idx.as_slice()));
        let t_set = std::time::Instant::now();
        let status = unsafe { g::ggml_backend_sched_graph_compute(self.sched, c.gf) };
        ensure!(status == g::GGML_STATUS_SUCCESS, "ggml graph compute failed with status {status}");
        let t_compute = std::time::Instant::now();
        let mut flat = vec![0f32; (n_embd * ns) as usize];
        unsafe { g::ggml_backend_tensor_get(c.out, flat.as_mut_ptr().cast(), 0, std::mem::size_of_val(flat.as_slice())) };
        if profiling() {
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1000.0;
            let n = unsafe { g::ggml_graph_n_nodes(c.gf) };
            let mut off_device = Vec::new();
            for i in 0..n {
                let node = unsafe { g::ggml_graph_node(c.gf, i) };
                let b = unsafe { g::ggml_backend_sched_get_tensor_backend(self.sched, node) };
                if !b.is_null() && b != self.backends[0] {
                    off_device.push(unsafe { CStr::from_ptr(g::ggml_op_name((*node).op)) }.to_string_lossy().into_owned());
                }
            }
            if !off_device.is_empty() {
                eprintln!("  ggml nodes not on the primary backend: {off_device:?}");
            }
            eprintln!(
                "  ggml L={l} n_seq={ns} nodes={n} {}  inputs+build+alloc {:.3} ms  set {:.3} ms  compute {:.3} ms  get {:.3} ms",
                if reused { "(cached graph)" } else { "(new graph)" },
                ms(t0, t_alloc),
                ms(t_alloc, t_set),
                ms(t_set, t_compute),
                ms(t_compute, std::time::Instant::now()),
            );
        }
        Ok(flat.chunks(hp.n_embd).map(|row| row.to_vec()).collect())
    }

    /// Build the graph for a `[L, n_seq]` batch: descriptors only, the
    /// scheduler allocates the data.
    unsafe fn build_graph(&self, l: i64, ns: i64, n_embd: i64, n_head: i64, hd: i64, cls_only: bool) -> Result<CachedGraph> {
        let n_tok = l * ns;
        let eps = self.hp.norm_eps;
        let mem = g::ggml_tensor_overhead() * (self.hp.n_layer * 48 + 48) + g::ggml_graph_overhead_custom(GRAPH_SIZE, false);
        let ctx = g::ggml_init(g::ggml_init_params { mem_size: mem, mem_buffer: ptr::null_mut(), no_alloc: true });
        ensure!(!ctx.is_null(), "ggml graph context allocation failed");
        let gf = g::ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

        let t_ids = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, n_tok);
        let t_pos = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, l);
        let t_mask = g::ggml_new_tensor_4d(ctx, g::GGML_TYPE_F16, l, l, 1, ns);
        let t_cls_mask = if cls_only { g::ggml_new_tensor_4d(ctx, g::GGML_TYPE_F16, l, 1, 1, ns) } else { ptr::null_mut() };
        let t_cls = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, ns);
        for t in [t_ids, t_pos, t_mask, t_cls_mask, t_cls] {
            if !t.is_null() {
                g::ggml_set_input(t);
            }
        }

        let layer_norm = |x: *mut g::ggml_tensor, w: *mut g::ggml_tensor, b: *mut g::ggml_tensor| {
            let x = g::ggml_norm(ctx, x, eps);
            let x = g::ggml_mul(ctx, x, w);
            g::ggml_add(ctx, x, b)
        };

        // Embeddings: token rows plus the "sentence A" type row.
        let mut x = g::ggml_get_rows(ctx, self.tok_embd, t_ids);
        let type_row = g::ggml_view_1d(ctx, self.type_embd, n_embd, 0);
        x = g::ggml_add(ctx, x, type_row);
        x = layer_norm(x, self.tok_norm_w, self.tok_norm_b);

        let scale = 1.0 / (hd as f32).sqrt();
        let elem = std::mem::size_of::<f32>() as i64;
        let n_layer = self.layers.len();
        for (li, layer) in self.layers.iter().enumerate() {
            // Last layer: only the [CLS] query rows are ever read, so attend
            // for those alone and run the rest of the layer on n_seq rows.
            let last_cls = cls_only && li + 1 == n_layer;
            let qkv = g::ggml_mul_mat(ctx, layer.wqkv, x); // [3*n_embd, n_tok]
            let nb_tok = (*qkv).nb[1];
            // Strided views straight into the projection: [hd, n_head, L, n_seq].
            let heads = |offset: i64, rows: i64| {
                g::ggml_view_4d(ctx, qkv, hd, n_head, rows, ns, (hd * elem) as usize, nb_tok, nb_tok * l as usize, (offset * elem) as usize)
            };
            let rope = |t: *mut g::ggml_tensor| {
                g::ggml_rope_ext(
                    ctx, t, t_pos, ptr::null_mut(), hd as i32, g::GGML_ROPE_TYPE_NEOX as i32,
                    self.hp.n_ctx as i32, self.hp.rope_base, 1.0, 0.0, 1.0, 32.0, 1.0,
                )
            };
            // RoPE at position 0 is the identity, so the [CLS]-only query
            // needs none.
            let (q, mask, n_q) = if last_cls {
                (heads(0, 1), t_cls_mask, 1)
            } else {
                (rope(heads(0, l)), t_mask, l)
            };
            let k = rope(heads(n_embd, l));
            let v = heads(2 * n_embd, l);
            // FA layout: [hd, rows, n_head, n_seq].
            let inner = |t: *mut g::ggml_tensor| g::ggml_permute(ctx, t, 0, 2, 1, 3);
            let (q4, mut k4, mut v4) = (inner(q), inner(k), inner(v));
            let n_rows = n_q * ns;
            let attn = if self.flash {
                if self.cast_kv {
                    k4 = g::ggml_cast(ctx, k4, g::GGML_TYPE_F16);
                    v4 = g::ggml_cast(ctx, v4, g::GGML_TYPE_F16);
                }
                let a = g::ggml_flash_attn_ext(ctx, q4, k4, v4, mask, scale, 0.0, 0.0);
                // F32 accumulation by default; `HIPS_GGML_FA_F16=1` lets the
                // backend accumulate in F16 (an A/B knob, measurably no faster
                // on Vulkan for this model).
                if !fa_f16_accumulation() {
                    g::ggml_flash_attn_ext_set_prec(a, g::GGML_PREC_F32);
                }
                g::ggml_reshape_2d(ctx, a, n_embd, n_rows) // [hd, n_head, rows, n_seq] contiguous
            } else {
                let kq = g::ggml_mul_mat(ctx, k4, q4); // [L_kv, n_q, n_head, n_seq]
                let kq = g::ggml_soft_max_ext(ctx, kq, mask, scale, 0.0);
                let vt = g::ggml_cont(ctx, g::ggml_permute(ctx, v4, 1, 0, 2, 3)); // [L_kv, hd, n_head, n_seq]
                let kqv = g::ggml_mul_mat(ctx, vt, kq); // [hd, n_q, n_head, n_seq]
                let merged = g::ggml_cont(ctx, g::ggml_permute(ctx, kqv, 0, 2, 1, 3));
                g::ggml_reshape_2d(ctx, merged, n_embd, n_rows)
            };
            if last_cls {
                x = g::ggml_get_rows(ctx, x, t_cls); // residual stream, [CLS] rows only
            }
            let o = g::ggml_mul_mat(ctx, layer.wo, attn);
            x = layer_norm(g::ggml_add(ctx, o, x), layer.ln1_w, layer.ln1_b);

            let gate_up = g::ggml_mul_mat(ctx, layer.w_gate_up, x); // [2*n_ff, rows]
            // ggml's kernel is silu(first half) * second half with `swapped`
            // false (the header comment has it backwards): gate rows first.
            let act = g::ggml_glu(ctx, gate_up, g::GGML_GLU_OP_SWIGLU, false);
            let down = g::ggml_mul_mat(ctx, layer.w_down, act);
            x = layer_norm(g::ggml_add(ctx, down, x), layer.ln2_w, layer.ln2_b);
        }
        let out = if cls_only { x } else { g::ggml_get_rows(ctx, x, t_cls) }; // [n_embd, n_seq]
        g::ggml_set_output(out);
        g::ggml_build_forward_expand(gf, out);
        Ok(CachedGraph { ctx, gf, l, ns, cls_only, t_ids, t_pos, t_mask, t_cls_mask, t_cls, out })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if let Some(c) = self.cache.take() {
                g::ggml_free(c.ctx);
            }
            if !self.sched.is_null() {
                g::ggml_backend_sched_free(self.sched);
            }
            if !self.weights_buf.is_null() {
                g::ggml_backend_buffer_free(self.weights_buf);
            }
            if !self.weights_ctx.is_null() {
                g::ggml_free(self.weights_ctx);
            }
            for backend in self.backends.drain(..) {
                g::ggml_backend_free(backend);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wordpiece::WordPiece;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    fn opts(gpu: bool, flash: bool, cls_only: bool) -> EngineOptions {
        EngineOptions { gpu, threads: 4, flash_attention: flash, cls_only_last_layer: cls_only, cast_kv_f16: false }
    }

    /// Engine plus tokenizer on the CPU, or None without a model.
    fn cpu_engine(flash: bool, cls_only: bool) -> Option<(Engine, WordPiece, u32)> {
        let path = crate::test_util::model()?;
        let (engine, vocab) = Engine::load(&path, &opts(false, flash, cls_only)).expect("load");
        let tok = WordPiece::from_gguf(&vocab.tokens, &vocab.token_types, vocab.unk, vocab.cls, vocab.sep);
        Some((engine, tok, vocab.pad))
    }

    const TEXTS: [&str; 4] = [
        "fn main() {}",
        "pub fn download_verified(url: &str, path: &Path, max_bytes: u64) -> Result<()> { let agent = ureq::AgentBuilder::new().build(); }",
        "// Recursive graph bisection: split documents into two halves and swap greedily to minimize the log-gap cost of the posting lists.",
        "SELECT name, COUNT(*) FROM widgets GROUP BY name HAVING COUNT(*) > 1 ORDER BY name;",
    ];

    #[test]
    fn gguf_magic_and_unreadable_files() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("magic.gguf");
        std::fs::write(&good, b"GGUF\x03\x00\x00\x00").unwrap();
        looks_like_gguf(&good).unwrap();
        let bad = dir.path().join("bad.gguf");
        std::fs::write(&bad, b"not a model").unwrap();
        assert!(looks_like_gguf(&bad).unwrap_err().to_string().contains("not a GGUF file"));
        let err = Gguf::open(&bad, None).err().expect("garbage must not parse");
        assert!(err.to_string().contains("not a readable GGUF file"), "{err}");
        assert!(read_vocab(&dir.path().join("missing.gguf")).is_err());
        // A valid magic with no metadata behind it is also rejected, not a crash.
        assert!(Gguf::open(&good, None).is_err());
    }

    #[test]
    fn vocab_and_hparams_come_from_the_model() {
        let Some(path) = crate::test_util::model() else { return };
        let vocab = read_vocab(&path).unwrap();
        assert_eq!(vocab.tokens.len(), 30528);
        assert_eq!((vocab.unk, vocab.cls, vocab.sep, vocab.pad), (100, 101, 102, 0));
        assert_eq!(vocab.tokens[101], "[CLS]");
        assert_eq!(vocab.token_types[101], 3);
        let (engine, _) = Engine::load(&path, &opts(false, true, true)).unwrap();
        let hp = engine.hparams();
        assert_eq!((hp.n_layer, hp.n_embd, hp.n_head, hp.n_ff, hp.n_ctx), (12, 768, 12, 3072, 2048));
        assert_eq!(hp.head_dim(), 64);
        assert!((hp.rope_base - 1000.0).abs() < 1e-3);
        assert!(!engine.on_gpu());
        assert_eq!(engine.device(), "cpu");
    }

    #[test]
    fn rejects_empty_and_overlong_batches() {
        let Some((mut engine, tok, pad)) = cpu_engine(true, true) else { return };
        assert!(engine.forward(&[], pad).unwrap_err().to_string().contains("empty batch"));
        let long: Vec<u32> = tok.encode(&"word ".repeat(3000), usize::MAX);
        assert!(long.len() > engine.hparams().n_ctx);
        let err = engine.forward(&[long.as_slice()], pad).unwrap_err().to_string();
        assert!(err.contains("outside 1..="), "{err}");
    }

    /// The padded, masked batch must give each sequence exactly what it
    /// gets alone: this is the test that would have caught a wrong stride
    /// or mask.
    #[test]
    fn padded_batch_rows_equal_single_sequence_rows() {
        let Some((mut engine, tok, pad)) = cpu_engine(true, true) else { return };
        let seqs: Vec<Vec<u32>> = TEXTS.iter().map(|t| tok.encode(t, 512)).collect();
        assert!(seqs.iter().map(Vec::len).collect::<std::collections::HashSet<_>>().len() > 2, "lengths must differ");
        let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        let batched = engine.forward(&refs, pad).unwrap();
        assert_eq!(batched.len(), seqs.len());
        for (i, seq) in seqs.iter().enumerate() {
            let alone = engine.forward(&[seq.as_slice()], pad).unwrap().remove(0);
            assert_eq!(alone.len(), 768);
            assert!(alone.iter().all(|v| v.is_finite()));
            let c = cosine(&alone, &batched[i]);
            assert!(c > 0.99999, "row {i}: cosine {c}");
        }
        // The padded batch is invariant to the order of its rows.
        let rev: Vec<&[u32]> = refs.iter().rev().copied().collect();
        let reversed = engine.forward(&rev, pad).unwrap();
        for (i, row) in reversed.iter().rev().enumerate() {
            assert!(cosine(row, &batched[i]) > 0.99999);
        }
    }

    /// Flash and naive attention, and the [CLS]-only last layer, are
    /// different graphs for the same function.
    #[test]
    fn attention_variants_agree() {
        let Some((mut flash, tok, pad)) = cpu_engine(true, false) else { return };
        let (mut naive, _) = Engine::load(&crate::test_util::model().unwrap(), &opts(false, false, false)).unwrap();
        let (mut cls_only, _) = Engine::load(&crate::test_util::model().unwrap(), &opts(false, true, true)).unwrap();
        // Eight sequences of ~70 tokens: enough tokens for the CLS-only path
        // to engage (it needs 256 in the graph).
        let seqs: Vec<Vec<u32>> = (0..8)
            .map(|i| tok.encode(&format!("{} {} {}", TEXTS[i % 4], TEXTS[(i + 1) % 4], i), 512))
            .collect();
        let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        let a = flash.forward(&refs, pad).unwrap();
        let b = naive.forward(&refs, pad).unwrap();
        let c = cls_only.forward(&refs, pad).unwrap();
        for i in 0..seqs.len() {
            assert!(cosine(&a[i], &b[i]) > 0.99999, "flash vs naive row {i}: {}", cosine(&a[i], &b[i]));
            assert!(cosine(&a[i], &c[i]) > 0.99999, "full vs cls-only row {i}: {}", cosine(&a[i], &c[i]));
        }
    }

    /// The query shape reuses one cached graph; results must not depend on
    /// whether the graph was fresh or reused, nor on what ran before it.
    #[test]
    fn cached_query_graph_is_transparent() {
        let Some((mut engine, tok, pad)) = cpu_engine(true, true) else { return };
        let q1 = tok.encode("where are auth tokens validated", 64);
        let q2 = tok.encode("open a socket", 64);
        let first = engine.forward(&[q1.as_slice()], pad).unwrap().remove(0);
        let other = engine.forward(&[q2.as_slice()], pad).unwrap().remove(0);
        let again = engine.forward(&[q1.as_slice()], pad).unwrap().remove(0);
        assert!(cosine(&first, &again) > 0.999999);
        assert!(cosine(&first, &other) < 0.9, "different queries must differ");
        // A document batch in between invalidates and rebuilds transparently.
        let docs: Vec<Vec<u32>> = TEXTS.iter().map(|t| tok.encode(t, 512)).collect();
        let refs: Vec<&[u32]> = docs.iter().map(Vec::as_slice).collect();
        engine.forward(&refs, pad).unwrap();
        let after = engine.forward(&[q1.as_slice()], pad).unwrap().remove(0);
        assert!(cosine(&first, &after) > 0.999999);
    }

    #[test]
    fn gpu_agrees_with_cpu_when_present() {
        let Some((mut cpu, tok, pad)) = cpu_engine(true, true) else { return };
        let (mut gpu, _) = Engine::load(&crate::test_util::model().unwrap(), &opts(true, true, true)).unwrap();
        if !gpu.on_gpu() {
            eprintln!("skipped: no GPU backend");
            return;
        }
        let seqs: Vec<Vec<u32>> = TEXTS.iter().map(|t| tok.encode(t, 512)).collect();
        let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        let a = cpu.forward(&refs, pad).unwrap();
        let b = gpu.forward(&refs, pad).unwrap();
        for i in 0..seqs.len() {
            let c = cosine(&a[i], &b[i]);
            assert!(c > 0.9999, "row {i}: cpu/gpu cosine {c}");
        }
        let q = tok.encode("verify a checksum", 64);
        let qa = cpu.forward(&[q.as_slice()], pad).unwrap().remove(0);
        let qb = gpu.forward(&[q.as_slice()], pad).unwrap().remove(0);
        assert!(cosine(&qa, &qb) > 0.9999);
    }
}
