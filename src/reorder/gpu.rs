//! GPU-accelerated recursive graph bisection — zero-copy Metal via objc2.
//!
//! On Apple silicon the CPU and GPU share physical memory, so this
//! implementation never copies data for the GPU's benefit. Every array the
//! kernel touches lives in page-aligned host memory wrapped with
//! `newBufferWithBytesNoCopy` (`MTLStorageModeShared`):
//!
//! - the corpus doc-term graph (`edge_term`, `doc_offsets`) is written once
//!   into shared pages and referenced by every dispatch;
//! - the `order` permutation is one shared buffer for the whole run — the
//!   CPU permutes it in place between levels and partitions address it with
//!   a byte offset, so reordering is never re-sent;
//! - per-side degree counters are `AtomicU32` in shared memory: the CPU
//!   patches them incrementally after each swap pass and the GPU reads the
//!   same cache lines on the next dispatch;
//! - per-document gains are written by the GPU and read directly by the
//!   CPU sort — no readback copy.
//!
//! Per iteration the only CPU↔GPU "transfer" is the 12-byte params struct
//! (`setBytes`). Command buffers are committed as soon as an iteration's
//! inputs are final. Sibling partitions run as local async tasks, overlapping
//! CPU work with in-flight GPU commands without sharing Metal handles across
//! threads. The CPU fallback continues to use Rayon.
//!
//! The kernel itself is the fused per-document gain pass (one thread per
//! document, gains accumulated in registers), written in Metal Shading
//! Language and compiled at startup. Below [`GPU_MIN_PARTITION`] documents, the recursion falls back
//! to the multithreaded CPU implementation in the parent module.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use rayon::prelude::*;

use super::{BP_MAX_DEPTH, BP_MAX_ITERS};

// MTLCreateSystemDefaultDevice lives in CoreGraphics.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// Below this partition size the CPU path takes over.
pub const GPU_MIN_PARTITION: usize = 8192;

/// Partition-size cutoff, overridable via `MVP_GPU_MIN_PARTITION` for
/// experimentation.
fn gpu_min_partition() -> usize {
    std::env::var("MVP_GPU_MIN_PARTITION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(GPU_MIN_PARTITION)
}

/// The fused gain kernel: one thread per partition slot resolves its
/// document through `order`, loops the document's edges, and accumulates
/// the log-gap move gain. `log()` is the natural logarithm in MSL.
const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint  n;
    float ln_left;
    float ln_right;
};

inline float move_cost(float d, float ln_n) {
    float dc = max(d, 0.0f);
    return dc * (ln_n - log(dc + 1.0f));
}

kernel void bp_gain(
    device const uint*  order       [[buffer(0)]],
    device const uint*  edge_term   [[buffer(1)]],
    device const uint*  doc_offsets [[buffer(2)]],
    device const uint*  side        [[buffer(3)]],
    device const uint*  deg_total   [[buffer(4)]],
    device const uint*  deg_right   [[buffer(5)]],
    device float*       gains       [[buffer(6)]],
    constant Params&    p           [[buffer(7)]],
    uint slot [[thread_position_in_grid]])
{
    if (slot >= p.n) return;
    uint doc = order[slot];
    uint start = doc_offsets[doc];
    uint end = doc_offsets[doc + 1];
    bool on_right = side[slot] == 1u;
    float total = 0.0f;
    for (uint e = start; e < end; e++) {
        uint t = edge_term[e];
        float dr = float(deg_right[t]);
        float dl = float(deg_total[t] - deg_right[t]);
        float gl = move_cost(dl, p.ln_left) + move_cost(dr, p.ln_right)
                 - move_cost(dl - 1.0f, p.ln_left) - move_cost(dr + 1.0f, p.ln_right);
        float gr = move_cost(dr, p.ln_right) + move_cost(dl, p.ln_left)
                 - move_cost(dr - 1.0f, p.ln_right) - move_cost(dl + 1.0f, p.ln_left);
        total += on_right ? gr : gl;
    }
    gains[slot] = total;
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n: u32,
    ln_left: f32,
    ln_right: f32,
}

/// Apple-silicon page size (16 KiB; also a multiple of the 4 KiB Intel page
/// size). `newBufferWithBytesNoCopy` requires page-aligned pointers and
/// page-multiple lengths.
const PAGE_SIZE: usize = 16384;

/// A zero-initialized, page-aligned host allocation that the GPU can wrap
/// without copying.
struct PageAligned<T> {
    ptr: NonNull<u8>,
    len: usize,
    byte_cap: usize,
    _marker: PhantomData<T>,
}

// SAFETY: PageAligned owns its allocation exclusively; sharing follows the
// same rules as &[T]/&mut [T].
unsafe impl<T: Send> Send for PageAligned<T> {}
unsafe impl<T: Sync> Sync for PageAligned<T> {}

impl<T> PageAligned<T> {
    /// Allocate `len` zeroed elements. Only used with types for which the
    /// all-zero bit pattern is valid (u32, f32, AtomicU32).
    fn zeroed(len: usize) -> Self {
        let byte_cap = (len * size_of::<T>()).max(1).next_multiple_of(PAGE_SIZE);
        let layout = Layout::from_size_align(byte_cap, PAGE_SIZE).expect("layout");
        // SAFETY: layout has non-zero size.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).expect("allocation failed");
        Self {
            ptr,
            len,
            byte_cap,
            _marker: PhantomData,
        }
    }

    fn as_slice(&self) -> &[T] {
        // SAFETY: allocation holds at least `len` valid (zero-initialized
        // or since-written) elements of T.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above, with exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.len) }
    }
}

impl<T> Drop for PageAligned<T> {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.byte_cap, PAGE_SIZE).expect("layout");
        // SAFETY: allocated with this exact layout in `zeroed`.
        unsafe { dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// Host memory plus the Metal buffer that wraps it — same physical pages,
/// no copy. The buffer is declared first so it drops (releases) before the
/// memory is freed.
struct SharedBuf<T> {
    mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
    mem: PageAligned<T>,
}

impl<T> SharedBuf<T> {
    fn zeroed(device: &ProtocolObject<dyn MTLDevice>, len: usize) -> Self {
        let mem = PageAligned::<T>::zeroed(len);
        // SAFETY: the pointer is page-aligned, the length is a page
        // multiple, and `mem` outlives `mtl` (field order + no deallocator,
        // so Metal never frees memory it does not own).
        let mtl = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                mem.ptr.cast(),
                mem.byte_cap,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .expect("newBufferWithBytesNoCopy failed");
        Self { mtl, mem }
    }
}

struct MetalCtx {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalCtx {
    /// None when the host has no Metal device (e.g. a virtualized CI
    /// runner); the caller falls back to the CPU implementation.
    fn try_new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(MSL_SOURCE), None)
            .expect("MSL compilation failed");
        let function = library
            .newFunctionWithName(&NSString::from_str("bp_gain"))
            .expect("kernel function not found");
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .expect("pipeline creation failed");
        let queue = device.newCommandQueue().expect("command queue");
        Some(Self {
            queue,
            pipeline,
            device,
        })
    }
}

/// Corpus-wide shared state: built once, addressed by every dispatch.
struct GpuCorpus {
    ctx: MetalCtx,
    /// Term ids of every document, document-contiguous, original doc order.
    edge_term: SharedBuf<u32>,
    /// Edge range of each original document (`num_docs + 1` entries).
    doc_offsets: SharedBuf<u32>,
    /// Metal handle for the permutation buffer — only the objc object, not
    /// the host memory. The CPU side of that memory is owned by a separate
    /// `PageAligned` in `bp_order_gpu` and mutated exclusively through the
    /// `&mut [u32]` threaded down the recursion; keeping the `PageAligned`
    /// here too would alias that `&mut` through `&GpuCorpus` (UB under
    /// stacked borrows).
    order_mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
    num_terms: usize,
}

struct SendPtr<T>(*mut T);
// SAFETY: used only for disjoint parallel writes below.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// # Safety
    /// `idx` must be in bounds and not written concurrently elsewhere.
    unsafe fn write(&self, idx: usize, value: T) {
        unsafe { self.0.add(idx).write(value) };
    }
}

/// Compute a BP ordering with GPU-assisted top levels. Same contract as
/// [`super::bp_order`].
pub fn bp_order_gpu(doc_terms: &[Vec<u32>]) -> Vec<u32> {
    let Some(ctx) = MetalCtx::try_new() else {
        eprintln!("no Metal device available; falling back to CPU graph bisection");
        return super::bp_order(doc_terms);
    };
    let num_docs = doc_terms.len();
    let num_terms = doc_terms
        .iter()
        .flat_map(|terms| terms.iter())
        .max()
        .map_or(0, |&t| t as usize + 1);

    // Flatten the doc-term graph directly into shared pages: written once
    // by the CPU, read by every GPU dispatch, never copied.
    let mut doc_offsets = SharedBuf::<u32>::zeroed(&ctx.device, num_docs + 1);
    {
        let offsets = doc_offsets.mem.as_mut_slice();
        for (d, terms) in doc_terms.iter().enumerate() {
            offsets[d + 1] = offsets[d] + terms.len() as u32;
        }
    }
    let num_edges = doc_offsets.mem.as_slice()[num_docs] as usize;
    let mut edge_term = SharedBuf::<u32>::zeroed(&ctx.device, num_edges.max(1));
    {
        let offsets = doc_offsets.mem.as_slice();
        let ptr = SendPtr(edge_term.mem.as_mut_slice().as_mut_ptr());
        doc_terms.par_iter().enumerate().for_each(|(d, terms)| {
            let base = offsets[d] as usize;
            for (i, &t) in terms.iter().enumerate() {
                // SAFETY: documents write disjoint ranges [offsets[d],
                // offsets[d+1]).
                unsafe { ptr.write(base + i, t) };
            }
        });
    }

    if num_docs == 0 {
        // `zeroed(num_docs.max(1))` below would otherwise fabricate a
        // 1-element "permutation" naming a document that doesn't exist.
        return Vec::new();
    }
    // The order memory stays a standalone `PageAligned`, mutated only
    // through the `&mut [u32]` below; the corpus holds just the wrapping
    // Metal handle, so no shared Rust reference to this memory exists
    // while the recursion mutates it. The GPU reads it through the Metal
    // buffer (same physical pages); disjoint partition ranges + waiting on
    // each dispatch before mutating keep that race-free.
    let mut order_mem = PageAligned::<u32>::zeroed(num_docs);
    for (i, slot) in order_mem.as_mut_slice().iter_mut().enumerate() {
        *slot = i as u32;
    }
    // SAFETY: the pointer is page-aligned, the length is a page multiple,
    // and `order_mem` outlives `corpus` (declared before it, so it drops
    // after; no deallocator, so Metal never frees memory it does not own).
    let order_mtl = unsafe {
        ctx.device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                order_mem.ptr.cast(),
                order_mem.byte_cap,
                MTLResourceOptions::StorageModeShared,
                None,
            )
    }
    .expect("newBufferWithBytesNoCopy failed");

    let corpus = GpuCorpus {
        ctx,
        edge_term,
        doc_offsets,
        order_mtl,
        num_terms,
    };

    // These futures only await Metal completion. Poll on this thread with a
    // short bounded sleep: no runtime nesting, spinning, callbacks or Send
    // assertions for Objective-C objects. Metal's bindings expose completion
    // handlers only through an unsafe raw block pointer API.
    let leaves = drive_gpu(bisect_gpu(
        order_mem.as_mut_slice(),
        0,
        doc_terms,
        &corpus,
        0,
    ));
    drop(corpus); // all GPU work is complete before CPU leaf work starts
                  // Defer CPU leaves so a long CPU recursion cannot block polling/submitting
                  // still-active GPU siblings. The returned slices are disjoint by construction.
    leaves
        .into_par_iter()
        .for_each(|(order, depth)| super::bisect(order, doc_terms, depth));
    order_mem.as_slice().to_vec()
}

fn drive_gpu<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(result) => return result,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_micros(50)),
        }
    }
}

/// A cancelled sibling (including panic unwinding) must finish its GPU reads
/// before the caller's page-backed input/output buffers can be released.
struct InFlight(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

#[cfg(test)]
thread_local! {
    static IN_FLIGHT_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

impl InFlight {
    fn new(cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        #[cfg(test)]
        IN_FLIGHT_COUNTS.with(|counts| {
            let (active, peak) = counts.get();
            counts.set((active + 1, peak.max(active + 1)));
        });
        Self(cmd)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.waitUntilCompleted();
        #[cfg(test)]
        IN_FLIGHT_COUNTS.with(|counts| {
            let (active, peak) = counts.get();
            counts.set((active - 1, peak));
        });
    }
}

/// Encode, commit, and cooperatively await one gain pass over `n` slots at
/// element `base` of the order buffer.
async fn dispatch_gains(
    corpus: &GpuCorpus,
    base: usize,
    side: &SharedBuf<u32>,
    deg_total: &SharedBuf<AtomicU32>,
    deg_right: &SharedBuf<AtomicU32>,
    gains: &SharedBuf<f32>,
    params: Params,
) -> bool {
    let ctx = &corpus.ctx;
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(&ctx.pipeline);
    // SAFETY (setBuffer/setBytes): buffers outlive the (waited-on) command
    // buffer; offsets are 4-byte aligned; params is a plain #[repr(C)]
    // struct matching the MSL layout.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&corpus.order_mtl), base * size_of::<u32>(), 0);
        enc.setBuffer_offset_atIndex(Some(&corpus.edge_term.mtl), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&corpus.doc_offsets.mtl), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&side.mtl), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&deg_total.mtl), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&deg_right.mtl), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&gains.mtl), 0, 6);
        enc.setBytes_length_atIndex(NonNull::from(&params).cast(), size_of::<Params>(), 7);
    }
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: params.n as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    let pending = InFlight::new(cmd);
    // Private to drive_gpu's periodic local executor. Returning Pending lets
    // tokio::join! poll the sibling before the next completion check.
    std::future::poll_fn(|_| match pending.0.status() {
        MTLCommandBufferStatus::Completed => std::task::Poll::Ready(true),
        MTLCommandBufferStatus::Error => {
            eprintln!(
                "GPU bisection command failed: {:?}; using CPU",
                pending.0.error()
            );
            std::task::Poll::Ready(false)
        }
        _ => std::task::Poll::Pending,
    })
    .await
}

async fn bisect_gpu<'a>(
    order: &'a mut [u32],
    base: usize,
    doc_terms: &[Vec<u32>],
    corpus: &GpuCorpus,
    depth: usize,
) -> Vec<(&'a mut [u32], usize)> {
    let n = order.len();
    if n < gpu_min_partition() || depth >= BP_MAX_DEPTH {
        return vec![(order, depth)];
    }
    let num_terms = corpus.num_terms;
    let mid = n / 2;
    let params = Params {
        n: n as u32,
        ln_left: (mid as f32).ln(),
        ln_right: ((n - mid) as f32).ln(),
    };
    let num_edges: usize = order.iter().map(|&doc| doc_terms[doc as usize].len()).sum();
    if num_edges == 0 || num_terms == 0 {
        return Vec::new();
    }

    // Partition-local shared state. The degree counters are atomics living
    // in GPU-visible pages: the CPU patches them in place and the next
    // dispatch reads the patched values — nothing is snapshotted, converted,
    // or uploaded.
    let device = &*corpus.ctx.device;
    let mut side = SharedBuf::<u32>::zeroed(device, n);
    let gains = SharedBuf::<f32>::zeroed(device, n);
    let deg_total = SharedBuf::<AtomicU32>::zeroed(device, num_terms);
    let deg_right = SharedBuf::<AtomicU32>::zeroed(device, num_terms);

    side.mem.as_mut_slice()[mid..].fill(1);
    {
        let totals = deg_total.mem.as_slice();
        let rights = deg_right.mem.as_slice();
        order.par_iter().enumerate().for_each(|(i, &doc)| {
            for &t in &doc_terms[doc as usize] {
                totals[t as usize].fetch_add(1, Ordering::Relaxed);
                if i >= mid {
                    rights[t as usize].fetch_add(1, Ordering::Relaxed);
                }
            }
        });
    }

    let debug_timing = std::env::var_os("MVP_GPU_TIMING").is_some();

    for iter in 0..BP_MAX_ITERS {
        let t_start = std::time::Instant::now();
        if !dispatch_gains(corpus, base, &side, &deg_total, &deg_right, &gains, params).await {
            return vec![(order, depth)];
        }
        let t_gpu = t_start.elapsed();

        // The GPU wrote gains into shared memory; sort candidates directly
        // from it.
        let gains = gains.mem.as_slice();
        let side_slice = side.mem.as_mut_slice();
        let by_gain_desc = |a: &usize, b: &usize| {
            gains[*b]
                .partial_cmp(&gains[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        };
        let mut left: Vec<usize> = (0..n).filter(|&i| side_slice[i] == 0).collect();
        let mut right: Vec<usize> = (0..n).filter(|&i| side_slice[i] == 1).collect();
        left.sort_unstable_by(by_gain_desc);
        right.sort_unstable_by(by_gain_desc);

        // Pair-swap scan; patch the shared degree counters incrementally
        // for the documents that moved.
        let mut moved_to_right: Vec<u32> = Vec::new();
        let mut moved_to_left: Vec<u32> = Vec::new();
        for (&l, &r) in left.iter().zip(&right) {
            if gains[l] + gains[r] <= 0.0 {
                break;
            }
            side_slice[l] = 1;
            side_slice[r] = 0;
            moved_to_right.push(order[l]);
            moved_to_left.push(order[r]);
        }
        if debug_timing {
            eprintln!(
                "[gpu-bp] n={n} E={num_edges} iter={iter}: gpu={:.1}ms cpu={:.1}ms swapped={}",
                t_gpu.as_secs_f64() * 1e3,
                (t_start.elapsed() - t_gpu).as_secs_f64() * 1e3,
                moved_to_right.len()
            );
        }
        if moved_to_right.is_empty() {
            break; // converged
        }
        let rights = deg_right.mem.as_slice();
        moved_to_right.par_iter().for_each(|&doc| {
            for &t in &doc_terms[doc as usize] {
                rights[t as usize].fetch_add(1, Ordering::Relaxed);
            }
        });
        moved_to_left.par_iter().for_each(|&doc| {
            for &t in &doc_terms[doc as usize] {
                rights[t as usize].fetch_sub(1, Ordering::Relaxed);
            }
        });
    }

    // Stable-partition `order` in place (shared memory: the next level's
    // dispatches see the permuted ids without any transfer), then overlap
    // siblings locally. Box the recursive futures to bound their stack size.
    let side_slice = side.mem.as_slice();
    let reordered: Vec<u32> = (0..n)
        .filter(|&i| side_slice[i] == 0)
        .chain((0..n).filter(|&i| side_slice[i] == 1))
        .map(|i| order[i])
        .collect();
    order.copy_from_slice(&reordered);
    // Parent counters and scratch are no longer used. Release them before
    // both children allocate their own state on the shared-memory device.
    drop(side);
    drop(gains);
    drop(deg_total);
    drop(deg_right);

    let (left_half, right_half) = order.split_at_mut(mid);
    let (mut left, right) = tokio::join!(
        Box::pin(bisect_gpu(left_half, base, doc_terms, corpus, depth + 1)),
        Box::pin(bisect_gpu(
            right_half,
            base + mid,
            doc_terms,
            corpus,
            depth + 1
        )),
    );
    left.extend(right);
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same clusterable corpus the CPU BP tests use.
    fn interleaved_corpus(n: usize) -> Vec<Vec<u32>> {
        (0..n)
            .map(|i| {
                let base = if i % 2 == 0 { 0u32 } else { 50 };
                (0..10).map(|j| base + (i as u32 * 3 + j) % 50).collect()
            })
            .collect()
    }

    #[test]
    fn gpu_bp_returns_a_permutation_and_reduces_cost() {
        let metal_available = MTLCreateSystemDefaultDevice().is_some();
        if std::env::var_os("HIPS_REQUIRE_METAL_TESTS").is_some() {
            assert!(metal_available, "Metal hardware is required for this run");
        }
        IN_FLIGHT_COUNTS.with(|counts| counts.set((0, 0)));
        // Large enough that the top levels actually run on the GPU.
        let docs = interleaved_corpus(2 * GPU_MIN_PARTITION);
        let order = bp_order_gpu(&docs);

        let mut seen = vec![false; docs.len()];
        for &d in &order {
            assert!(!seen[d as usize], "duplicate doc in permutation");
            seen[d as usize] = true;
        }
        assert_eq!(order.len(), docs.len());
        if metal_available {
            IN_FLIGHT_COUNTS.with(|counts| {
                let (active, peak) = counts.get();
                assert_eq!(active, 0, "all GPU commands must finish before returning");
                assert!(peak >= 2, "GPU siblings must overlap; peak={peak}");
            });
        }

        let identity: Vec<u32> = (0..docs.len() as u32).collect();
        let before = super::super::tests::log_gap_cost(&identity, &docs);
        let after = super::super::tests::log_gap_cost(&order, &docs);
        assert!(
            after < before * 0.8,
            "GPU BP should cut log-gap cost: before={before:.0}, after={after:.0}"
        );
    }
}
