// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Typed launchers. Each validates sizes before launching, so a safe caller cannot
//! make a kernel read or write out of bounds.

use crate::KernelRegistry;
use astral_core::{AstralError, Result};
use astral_gpu::{param, Dev, DevMut, DeviceBuffer, DevicePtr, Function, LaunchConfig, Stream};

const BASIC: &str = "p0_basic";

fn invalid(msg: String) -> AstralError {
    AstralError::Invalid(msg)
}

/// `astral_read_bw_f32x4(const float4* src, u64 n4, float* sink)`.
pub struct ReadBandwidth {
    f: Function,
}

impl ReadBandwidth {
    pub fn new(reg: &KernelRegistry) -> Result<Self> {
        Ok(Self { f: reg.function(BASIC, "astral_read_bw_f32x4")? })
    }

    /// Reads all of `src` (length must be a multiple of 4). `sink` needs >= 1 element.
    pub fn launch(&self, stream: &Stream, src: &DeviceBuffer<f32>, sink: &mut DeviceBuffer<f32>, grid: u32) -> Result<()> {
        if !src.len().is_multiple_of(4) || sink.is_empty() || grid == 0 {
            return Err(invalid(format!("read_bw: src.len {} must be a multiple of 4, sink non-empty", src.len())));
        }
        let (mut s, mut n4, mut k) = (src.device_ptr(), (src.len() / 4) as u64, sink.device_ptr());
        let mut params = [param(&mut s), param(&mut n4), param(&mut k)];
        let cfg = LaunchConfig { grid: (grid, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: params match the kernel signature; n4*4 == src.len(); sink has >= 1 element.
        unsafe { self.f.launch(cfg, stream, &mut params) }
    }
}

/// `astral_copy_bw_f32x4(const float4* src, float4* dst, u64 n4)`.
pub struct CopyBandwidth {
    f: Function,
}

impl CopyBandwidth {
    pub fn new(reg: &KernelRegistry) -> Result<Self> {
        Ok(Self { f: reg.function(BASIC, "astral_copy_bw_f32x4")? })
    }

    pub fn launch(&self, stream: &Stream, src: &DeviceBuffer<f32>, dst: &mut DeviceBuffer<f32>, grid: u32) -> Result<()> {
        if !src.len().is_multiple_of(4) || dst.len() != src.len() || grid == 0 {
            return Err(invalid("copy_bw: equal lengths, multiple of 4".into()));
        }
        let (mut s, mut d, mut n4) = (src.device_ptr(), dst.device_ptr(), (src.len() / 4) as u64);
        let mut params = [param(&mut s), param(&mut d), param(&mut n4)];
        let cfg = LaunchConfig { grid: (grid, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: params match the kernel signature; both buffers hold n4*4 floats.
        unsafe { self.f.launch(cfg, stream, &mut params) }
    }
}

/// `astral_rmsnorm_f32(const float* x, const float* w, float* y, unsigned rows, unsigned cols, float eps)`.
pub struct RmsNormF32 {
    f: Function,
}

impl RmsNormF32 {
    pub const BLOCK: u32 = 256;

    pub fn new(reg: &KernelRegistry) -> Result<Self> {
        Ok(Self { f: reg.function(BASIC, "astral_rmsnorm_f32")? })
    }

    /// `y[r, :] = x[r, :] / sqrt(mean(x[r, :]^2) + eps) * weight` (weight optional). `x` and
    /// `y` may be the same buffer only through separate calls; pass distinct ranges.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(&self, stream: &Stream, x: &impl Dev<f32>, weight: Option<&dyn Dev<f32>>, y: &mut impl DevMut<f32>, rows: u32, cols: u32, eps: f32) -> Result<()> {
        let n = rows as usize * cols as usize;
        if cols == 0 || rows == 0 {
            return Err(invalid("rmsnorm: rows and cols must be > 0".into()));
        }
        if x.dlen() < n || y.dlen() < n {
            return Err(invalid(format!("rmsnorm: x.len {} / y.len {} < rows*cols {n}", x.dlen(), y.dlen())));
        }
        if let Some(w) = weight
            && w.dlen() < cols as usize
        {
            return Err(invalid(format!("rmsnorm: weight.len {} < cols {cols}", w.dlen())));
        }
        let (mut px, mut pw, mut py) = (x.dptr(), weight.map_or(0, |w| w.dptr()), y.dptr());
        let (mut r, mut c, mut e) = (rows, cols, eps);
        let mut params = [param(&mut px), param(&mut pw), param(&mut py), param(&mut r), param(&mut c), param(&mut e)];
        // A wide block cuts the dependent-load rounds per thread. The same block for any row
        // count, so a row's result does not depend on how many rows the launch has.
        let block = (cols / 4).next_multiple_of(32).clamp(Self::BLOCK, 1024);
        let cfg = LaunchConfig { grid: (rows, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: params match the kernel signature; sizes checked above; a null weight is handled by the kernel.
        unsafe { self.f.launch(cfg, stream, &mut params) }
    }
}

/// `astral_gemv_mxfp4_f32_b{1..4}(const u8* qs, const u8* es, const float* x, float* y, unsigned N, unsigned K)`.
/// Weight-only MXFP4 GEMV for decode, batch 1..=4 (ADR-0007). One kernel per batch size.
pub struct GemvMxfp4F32 {
    f: [Function; 4],
}

impl GemvMxfp4F32 {
    pub const MAX_BATCH: u32 = 4;
    pub const ROWS_PER_BLOCK: u32 = 8;
    /// Largest E8M0 exponent the kernel decodes exactly (see `block_scale` in mxfp4.cu).
    pub const MAX_SCALE_EXP: u8 = 240;

    pub fn new(reg: &KernelRegistry) -> Result<Self> {
        let f = |b: u32| reg.function("mxfp4", &format!("astral_gemv_mxfp4_f32_b{b}"));
        Ok(Self { f: [f(1)?, f(2)?, f(3)?, f(4)?] })
    }

    /// `qs`/`es` are the repacked planes (`astral_quant::repack_mxfp4`) of an `n x k` matrix.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(&self, stream: &Stream, qs: &DeviceBuffer<u8>, es: &DeviceBuffer<u8>, x: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>, n: u32, k: u32, b: u32) -> Result<()> {
        let (nn, kk, bb) = (n as usize, k as usize, b as usize);
        if n == 0 || k == 0 || !k.is_multiple_of(32) || b == 0 || b > Self::MAX_BATCH {
            return Err(invalid(format!("gemv_mxfp4: need n > 0, k % 32 == 0, 1 <= b <= 4 (got n {n}, k {k}, b {b})")));
        }
        if qs.len() != nn * kk / 2 || es.len() != nn * kk / 32 || x.len() < bb * kk || y.len() < bb * nn {
            return Err(invalid(format!("gemv_mxfp4: buffer sizes qs {} es {} x {} y {} do not match n {n} k {k} b {b}", qs.len(), es.len(), x.len(), y.len())));
        }
        let (mut pq, mut pe, mut px, mut py) = (qs.device_ptr(), es.device_ptr(), x.device_ptr(), y.device_ptr());
        let (mut a, mut c) = (n, k);
        let mut params = [param(&mut pq), param(&mut pe), param(&mut px), param(&mut py), param(&mut a), param(&mut c)];
        let cfg = LaunchConfig { grid: (n.div_ceil(Self::ROWS_PER_BLOCK), 1, 1), block: (32 * Self::ROWS_PER_BLOCK, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: params match the kernel signature; sizes checked above; K % 32 == 0 keeps
        // every 16-byte qs and x load in bounds and aligned (allocations are 256-byte aligned).
        unsafe { self.f[(b - 1) as usize].launch(cfg, stream, &mut params) }
    }
}

/// `astral_gemm_mxfp4_mma_f32_w{W}u{U}t{T}(const u8* q, const u8* sc, const float* x, float* y, unsigned N, unsigned K, unsigned B)`.
/// MXFP4 weights x f16-rounded activations on tensor cores, batch 1..=8 (ADR-0007, v2.3).
pub struct GemmMxfp4MmaF32 {
    /// (variant, f32-activation entry, f16-activation entry)
    variants: Vec<(MmaVariant, Function, Function)>,
    /// Variant index used for f32 and for f16 activations.
    chosen_f32: usize,
    chosen_f16: usize,
    sm_count: std::sync::OnceLock<u32>,
}

/// One compiled instantiation of the tensor-core kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmaVariant {
    pub warps: u32,
    pub unroll: u32,
    pub tile_k: u32,
}

impl MmaVariant {
    pub const ALL: [MmaVariant; 6] = [
        MmaVariant { warps: 4, unroll: 8, tile_k: 1024 },
        MmaVariant { warps: 4, unroll: 16, tile_k: 1024 },
        MmaVariant { warps: 8, unroll: 4, tile_k: 1024 },
        MmaVariant { warps: 8, unroll: 8, tile_k: 1024 },
        MmaVariant { warps: 2, unroll: 8, tile_k: 1024 },
        MmaVariant { warps: 2, unroll: 16, tile_k: 512 },
    ];

    pub fn name(&self) -> String {
        format!("w{}u{}t{}", self.warps, self.unroll, self.tile_k)
    }

    pub fn rows_per_block(&self) -> u32 {
        16 * self.warps
    }
}

impl GemmMxfp4MmaF32 {
    pub const MAX_BATCH: u32 = 8;
    /// Default variants from sweeps on the H100 PCIe (2026-11-24), both at 8 blocks/SM.
    /// f32 activations (synchronous staging): w4u8t1024. f16 activations (cp.async double
    /// buffering): w8u8t1024, which keeps batch 8 within 6-8% of batch 1 per byte.
    pub const DEFAULT_F32: MmaVariant = MmaVariant { warps: 4, unroll: 8, tile_k: 1024 };
    pub const DEFAULT_F16: MmaVariant = MmaVariant { warps: 8, unroll: 8, tile_k: 1024 };

    /// Loads every variant; `ASTRAL_MMA_VARIANT=w4u8t1024` (for example) selects one.
    pub fn new(reg: &KernelRegistry) -> Result<Self> {
        let mut variants = Vec::new();
        for v in MmaVariant::ALL {
            variants.push((v, reg.function("mxfp4_mma", &format!("astral_gemm_mxfp4_mma_f32_{}", v.name()))?, reg.function("mxfp4_mma", &format!("astral_gemm_mxfp4_mma_f16x_{}", v.name()))?));
        }
        let want = std::env::var("ASTRAL_MMA_VARIANT").ok();
        let pick = |default: MmaVariant| {
            variants
                .iter()
                .position(|(v, _, _)| match &want {
                    Some(w) => v.name() == *w,
                    None => *v == default,
                })
                .ok_or_else(|| invalid(format!("unknown ASTRAL_MMA_VARIANT {want:?}")))
        };
        let (chosen_f32, chosen_f16) = (pick(Self::DEFAULT_F32)?, pick(Self::DEFAULT_F16)?);
        Ok(Self { variants, chosen_f32, chosen_f16, sm_count: std::sync::OnceLock::new() })
    }

    /// The variant used for f32 (`false`) or f16 (`true`) activations.
    pub fn variant(&self, f16_activations: bool) -> MmaVariant {
        self.variants[if f16_activations { self.chosen_f16 } else { self.chosen_f32 }].0
    }

    /// Blocks per SM to aim for when choosing the K split. At 2 the kernel was latency-bound
    /// (14% achieved occupancy, ncu 2026-11-24); for the default variant a sweep of 4..16 on
    /// the H100 PCIe was best at 8 (performance is a sawtooth in this value because the split
    /// count quantises into waves). Overridable with `ASTRAL_MMA_BLOCKS_PER_SM`.
    fn blocks_per_sm() -> u32 {
        std::env::var("ASTRAL_MMA_BLOCKS_PER_SM").ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(8)
    }

    /// Splits K so every SM holds about `blocks_per_sm` blocks. Each split stages only its own
    /// slice of the activations, so splitting adds no activation traffic.
    fn k_splits(&self, stream: &Stream, v: MmaVariant, n: u32, k: u32) -> Result<u32> {
        let sms = match self.sm_count.get() {
            Some(&v) => v,
            None => {
                let v = stream.context().device().attribute(astral_gpu::attr::MULTIPROCESSOR_COUNT)?.max(1) as u32;
                *self.sm_count.get_or_init(|| v)
            }
        };
        let blocks = n.div_ceil(v.rows_per_block());
        Ok((Self::blocks_per_sm() * sms).div_ceil(blocks).clamp(1, (k / 64).min(32)))
    }

    /// `q`/`sc` come from `astral_quant::repack_mxfp4_mma` for an `n x k` matrix.
    /// When the K dimension is split, `y` is zeroed and accumulated with atomics on `stream`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(&self, stream: &Stream, q: &DeviceBuffer<u8>, sc: &DeviceBuffer<u8>, x: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>, n: u32, k: u32, b: u32) -> Result<()> {
        self.launch_impl(stream, q, sc, x.device_ptr(), x.len(), false, y, n, k, b)
    }

    /// Same, with activations already in f16 (`x` holds f16 bit patterns).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_f16(&self, stream: &Stream, q: &DeviceBuffer<u8>, sc: &DeviceBuffer<u8>, x: &DeviceBuffer<u16>, y: &mut DeviceBuffer<f32>, n: u32, k: u32, b: u32) -> Result<()> {
        self.launch_impl(stream, q, sc, x.device_ptr(), x.len(), true, y, n, k, b)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_impl(
        &self,
        stream: &Stream,
        q: &DeviceBuffer<u8>,
        sc: &DeviceBuffer<u8>,
        x_ptr: astral_gpu::DevicePtr,
        x_len: usize,
        x_f16: bool,
        y: &mut DeviceBuffer<f32>,
        n: u32,
        k: u32,
        b: u32,
    ) -> Result<()> {
        let (nn, kk, bb) = (n as usize, k as usize, b as usize);
        if n == 0 || !n.is_multiple_of(16) || k == 0 || !k.is_multiple_of(64) || b == 0 || b > Self::MAX_BATCH {
            return Err(invalid(format!("gemm_mxfp4_mma: need n % 16 == 0, k % 64 == 0, 1 <= b <= 8 (got n {n}, k {k}, b {b})")));
        }
        if q.len() != nn * kk / 2 || sc.len() != nn * kk / 32 || x_len < bb * kk || y.len() < bb * nn {
            return Err(invalid(format!("gemm_mxfp4_mma: buffer sizes q {} sc {} x {x_len} y {} do not match n {n} k {k} b {b}", q.len(), sc.len(), y.len())));
        }
        let (v, f32_entry, f16_entry) = &self.variants[if x_f16 { self.chosen_f16 } else { self.chosen_f32 }];
        let f = if x_f16 { f16_entry } else { f32_entry };
        let splits = self.k_splits(stream, *v, n, k)?;
        if splits > 1 {
            y.zero_async(stream)?;
        }
        let (mut pq, mut ps, mut px, mut py) = (q.device_ptr(), sc.device_ptr(), x_ptr, y.device_ptr());
        let (mut a, mut c, mut d) = (n, k, b);
        let mut params = [param(&mut pq), param(&mut ps), param(&mut px), param(&mut py), param(&mut a), param(&mut c), param(&mut d)];
        let cfg = LaunchConfig { grid: (n.div_ceil(v.rows_per_block()), splits, 1), block: (32 * v.warps, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: params match the kernel signature (f32 or f16 activations, chosen together with
        // the entry point); sizes and divisibility checked above, which keeps every weight and
        // activation load in bounds and aligned (allocations are 256-byte aligned).
        unsafe { f.launch(cfg, stream, &mut params) }
    }
}

/// Weight types handled by the general GEMV (`gemv.cu`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvType {
    F32,
    Bf16,
    /// Repacked Q8_0 (`astral_quant::repack_q8_0`).
    Q8_0,
    /// Repacked NVFP4 (`astral_quant::repack_nvfp4`).
    Nvfp4,
}

impl GemvType {
    fn idx(self) -> usize {
        self as usize
    }

    /// Bytes per row in the value plane and in the auxiliary (scale) plane.
    pub fn row_bytes(self, k: u32) -> (usize, usize) {
        let k = k as usize;
        match self {
            GemvType::F32 => (4 * k, 0),
            GemvType::Bf16 => (2 * k, 0),
            GemvType::Q8_0 => (k, k / 32 * 2),
            GemvType::Nvfp4 => (k / 2, k / 16),
        }
    }

    fn k_multiple(self) -> u32 {
        match self {
            GemvType::F32 | GemvType::Bf16 => 4,
            GemvType::Q8_0 => 32,
            GemvType::Nvfp4 => 64,
        }
    }
}

/// One weight matrix, or a stack of `n_mats` equally shaped matrices (MoE experts), on the GPU.
pub struct GemvMatrix {
    pub ty: GemvType,
    pub n: u32,
    pub k: u32,
    pub n_mats: u32,
    pub qs: DeviceBuffer<u8>,
    pub aux: DeviceBuffer<u8>,
    /// Optional per-matrix output scale (NVFP4 experts' `.scale` tensors).
    pub mat_scale: Option<DeviceBuffer<f32>>,
    /// NVFP4 only: the tensor-core fragment layout (values, scales) of
    /// `astral_quant::nvfp4_frag_layout`, for the prefill GEMM [`Gemv::launch_mma5`].
    pub frag: Option<(DeviceBuffer<u8>, DeviceBuffer<u8>)>,
    /// Q8_0 only: the weights as f16 A fragments (`astral_quant::f16_frag_layout`), for the
    /// dense warpgroup-MMA prefill GEMM [`Gemv::launch_wgmma_f16`].
    pub frag16: Option<DeviceBuffer<u16>>,
    /// Q8_0 only: the weights as row-major f16 (`q * d` rounded once) with their TMA tensor map,
    /// for the prefill GEMM [`Gemv::launch_gemm_f16_tma`].
    pub w16: Option<(DeviceBuffer<u16>, astral_gpu::TensorMap)>,
    /// NVFP4 layout holding MXFP4 data: the native E8M0 scales (K/32 per row). When set, decode
    /// reads `qs` with these instead of `aux` (exact to the checkpoint, ~5.6% fewer bytes).
    pub mx_scale: Option<DeviceBuffer<u8>>,
}

impl GemvMatrix {
    /// Uploads repacked planes. `qs`/`aux` hold all `n_mats` matrices back to back.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(ctx: &std::sync::Arc<astral_gpu::Context>, ty: GemvType, n: u32, k: u32, n_mats: u32, qs: &[u8], aux: &[u8], mat_scale: Option<&[f32]>) -> Result<Self> {
        let (rq, ra) = ty.row_bytes(k);
        let (mats, rows) = (n_mats as usize, n as usize);
        if n == 0 || n_mats == 0 || k == 0 || !k.is_multiple_of(ty.k_multiple()) {
            return Err(invalid(format!("gemv matrix: k {k} must be a positive multiple of {} for {ty:?}", ty.k_multiple())));
        }
        if qs.len() != mats * rows * rq || aux.len() != mats * rows * ra || mat_scale.is_some_and(|m| m.len() != mats) {
            return Err(invalid(format!("gemv matrix {ty:?} n{n} k{k} x{n_mats}: plane sizes qs {} aux {} do not match", qs.len(), aux.len())));
        }
        Ok(Self {
            ty,
            n,
            k,
            n_mats,
            qs: DeviceBuffer::from_slice(ctx, qs)?,
            aux: DeviceBuffer::from_slice(ctx, aux)?,
            mat_scale: mat_scale.map(|m| DeviceBuffer::from_slice(ctx, m)).transpose()?,
            frag: None,
            frag16: None,
            w16: None,
            mx_scale: None,
        })
    }

    pub fn bytes(&self) -> usize {
        self.qs.bytes() + self.aux.bytes() + self.frag.as_ref().map_or(0, |(f, s)| f.bytes() + s.bytes()) + self.frag16.as_ref().map_or(0, |f| f.bytes()) + self.w16.as_ref().map_or(0, |(w, _)| w.bytes()) + self.mx_scale.as_ref().map_or(0, |b| b.bytes())
    }

    /// Adds the f16 fragment layout of a single Q8_0 matrix from its host planes (`qs`, `aux` as
    /// given to [`GemvMatrix::upload`]); needs `n % 128 == 0` and `k % 64 == 0`.
    pub fn add_frag16(&mut self, ctx: &std::sync::Arc<astral_gpu::Context>, qs: &[u8], aux: &[u8]) -> Result<()> {
        if self.ty != GemvType::Q8_0 || self.n_mats != 1 || !self.n.is_multiple_of(128) || !self.k.is_multiple_of(64) {
            return Err(invalid(format!("add_frag16: {:?} n {} k {} x{}", self.ty, self.n, self.k, self.n_mats)));
        }
        let f = astral_quant::f16_frag_layout(qs, aux, self.n as usize, self.k as usize)?;
        self.frag16 = Some(DeviceBuffer::from_slice(ctx, &f)?);
        Ok(())
    }

    /// Adds a row-major f16 copy of a Q8_0 matrix stack (from its host planes; `[n_mats * n][k]`)
    /// and its TMA tensor map; needs `n % 128 == 0` and `k % 64 == 0`.
    pub fn add_w16(&mut self, ctx: &std::sync::Arc<astral_gpu::Context>, qs: &[u8], aux: &[u8]) -> Result<()> {
        if self.ty != GemvType::Q8_0 || !self.n.is_multiple_of(128) || !self.k.is_multiple_of(64) {
            return Err(invalid(format!("add_w16: {:?} n {} k {} x{}", self.ty, self.n, self.k, self.n_mats)));
        }
        let (n, k) = (self.n as usize * self.n_mats as usize, self.k as usize);
        if qs.len() != n * k || aux.len() != n * k / 16 {
            return Err(invalid("add_w16: planes do not match the matrix".into()));
        }
        let mut h = vec![0u16; n * k];
        std::thread::scope(|sc| {
            let per = n.div_ceil(std::thread::available_parallelism().map_or(8, |v| v.get()).min(32)).max(1);
            for (ci, rows) in h.chunks_mut(per * k).enumerate() {
                sc.spawn(move || {
                    for (ri, row) in rows.chunks_mut(k).enumerate() {
                        let r = ci * per + ri;
                        for (c, out) in row.iter_mut().enumerate() {
                            let o = (r * (k / 32) + c / 32) * 2;
                            let d = astral_quant::f16_to_f32(u16::from_le_bytes([aux[o], aux[o + 1]]));
                            *out = astral_quant::f32_to_f16(qs[r * k + c] as i8 as f32 * d);
                        }
                    }
                });
            }
        });
        let buf = DeviceBuffer::from_slice(ctx, &h)?;
        let tm = astral_gpu::TensorMap::f16_2d_sw128(buf.device_ptr(), n as u64, k as u64, 128, 64)?;
        self.w16 = Some((buf, tm));
        Ok(())
    }

    /// Adds the prefill fragment layout of an NVFP4 matrix from its host planes (`qs`, `aux` as
    /// given to [`GemvMatrix::upload`]).
    pub fn add_frag(&mut self, ctx: &std::sync::Arc<astral_gpu::Context>, qs: &[u8], aux: &[u8]) -> Result<()> {
        if self.ty != GemvType::Nvfp4 {
            return Err(invalid(format!("add_frag: {:?} is not NVFP4", self.ty)));
        }
        let (f, s) = astral_quant::nvfp4_frag_layout(qs, aux, self.n as usize, self.k as usize, self.n_mats as usize)?;
        self.frag = Some((DeviceBuffer::from_slice(ctx, &f)?, DeviceBuffer::from_slice(ctx, &s)?));
        Ok(())
    }
}

/// `astral_gemv_{f32,bf16,q8_0,nvfp4}(qs, aux, x, y, N, K, mat_qs, mat_aux, x_stride, ids, mat_scale, n_mats)`.
pub struct Gemv {
    f: [Function; 4],
    /// int8-activation (dp4a) kernels for Q8_0 and NVFP4, and the activation quantiser.
    fq: [Function; 2],
    fq_r1: Function,
    /// Shared-memory NVFP4 GEMV, 1 / 2 / 4 rows per warp.
    fq_s: [Function; 3],
    fq_long: [Function; 2],
    fq_mx: Function,
    fq_mx_w: [Function; 2],
    quant: Function,
    /// Block-per-row F32 / BF16 kernels for short, wide matrices.
    rb: [Function; 2],
    multi: Function,
    /// [`Gemv::launch_q_multi_t`] for 2..=8 tokens (index T - 2).
    multi_t: Vec<Function>,
    multi_mma: Function,
    items_s: Function,
    items_q8x: Function,
    /// Batched decode on tensor cores ([`Gemv::launch_items_mma`]): Q8_0, NVFP4 fragment layout.
    items_mma: [Function; 2],
    /// [`Gemv::launch_q_dense`]: 2, 4 and 8 tiles of 8 tokens per block.
    dense_mma: [Function; 3],
    /// [`Gemv::launch_q_tiled`].
    q_tiled: Function,
    wgmma_f16: Function,
    /// TMA-fed grouped f16 GEMM ([`Gemv::launch_gemm_f16_tma`]).
    gemm_tma: [Function; 2],
    /// Warp-specialised 256-row NVFP4 expert GEMM (Hopper prefill).
    ws256: [Function; 3],
    /// Q8_0 rows kernels for 9..=16 and 17..=32 tokens ([`Gemv::launch_q_rows`]).
    rows_t: [Function; 2],
    /// Tensor-core multi-matrix Q8_0 GEMV for 9..=64 tokens (2, 4, 8 tiles of 8).
    multi_mma_n: [Function; 3],
    items: [Function; 2],
    mma: [Function; 2],
    mma2: [Function; 2],
    mma3: [Function; 2],
    mma4: Function,
    mma5: Function,
    wgmma: Function,
    wgmma2: Function,
    zero: Function,
}

/// Split K across blocks with atomic adds when a GEMM's grid is small (`ASTRAL_SPLITK=1`).
/// Off by default: atomic adds make the sums depend on block timing, so a prompt's results
/// would vary from run to run (and batched decode could not match single-sequence decode).
pub fn split_k() -> bool {
    std::env::var_os("ASTRAL_SPLITK").is_some()
}

/// Tokens per work item of [`Gemv::launch_mma`] (`MMA_TOK` in gemv.cu).
pub const MMA_TOKENS: usize = 64;

/// Work lists for [`Gemv::launch_mma`]: items `[e, start, count]` over index arrays mapping
/// each item's tokens to activation rows (`xidx`) and output rows (`yidx`).
#[derive(Debug, Default, Clone)]
pub struct MmaWork {
    pub items: Vec<u32>,
    pub xidx: Vec<u32>,
    pub yidx: Vec<u32>,
}

impl MmaWork {
    /// Adds the (activation row, output row) pairs of matrix `e`, in items of up to 64.
    pub fn push(&mut self, e: u32, rows: &[(u32, u32)]) {
        self.push_chunked(e, rows, MMA_TOKENS);
    }

    /// As [`MmaWork::push`] with items of up to `chunk` tokens (128 for [`Gemv::launch_mma4`]).
    pub fn push_chunked(&mut self, e: u32, rows: &[(u32, u32)], chunk: usize) {
        for chunk in rows.chunks(chunk) {
            self.items.extend_from_slice(&[e, self.xidx.len() as u32, chunk.len() as u32]);
            for &(x, y) in chunk {
                self.xidx.push(x);
                self.yidx.push(y);
            }
        }
    }

    pub fn n_items(&self) -> u32 {
        (self.items.len() / 3) as u32
    }
}

/// Dynamic shared memory of `astral_gemm_nvfp4_wgmma2`: 3 stages of `G7Stage` (21504 bytes).
const WGMMA2_SMEM: u32 = 5 * 21504;
/// Dynamic shared memory of `astral_gemm_nvfp4_ws256`: 5 stages of 25 KB.
const WS256_SMEM: u32 = 5 * 25600;
/// Dynamic shared memory of `astral_gemv_q8_0_tiled`: 3 stages of 19968 bytes.
const Q_TILED_SMEM: u32 = 3 * 19968;
/// Dynamic shared memory of `astral_gemm_f16_tma`: 6 stages of 32 KB (or, with 256-token
/// tiles, 4 of 48 KB).
const GEMM_TMA_SMEM: u32 = 6 * 32768;
/// Dynamic shared memory of `astral_gemm_f16_wgmma`: 3 stages of 32 KB.
const WGMMA_F16_SMEM: u32 = 3 * 32768;

/// Whether the kernels were built for sm_90a (wgmma).
/// Whether the kernels were built with an sm_90a (Hopper) target.
pub fn has_sm90a_build() -> bool {
    astral_kernels_src_has_sm90a()
}

fn astral_kernels_src_has_sm90a() -> bool {
    crate::BUILT_ARCHS.iter().any(|a| a.contains("90a"))
}

/// Activation rows per [`QItem`] (`NB` in gemv.cu).
pub const QITEM_ROWS: usize = 8;

/// One work item of [`Gemv::launch_items`]: matrix `e` times activation rows `xi[..n]`,
/// written to output rows `yi[..n]` (`struct QItem` of gemv.cu).
#[derive(Debug, Clone, Copy, Default)]
pub struct QItem {
    pub e: u32,
    pub n: u32,
    pub xi: [u32; QITEM_ROWS],
    pub yi: [u32; QITEM_ROWS],
}

impl QItem {
    pub const WORDS: usize = 2 + 2 * QITEM_ROWS;

    /// Items as the u32 words the kernel reads.
    pub fn words(items: &[QItem]) -> Vec<u32> {
        let mut w = Vec::with_capacity(items.len() * Self::WORDS);
        for it in items {
            w.push(it.e);
            w.push(it.n);
            w.extend_from_slice(&it.xi);
            w.extend_from_slice(&it.yi);
        }
        w
    }

    /// Dense items over rows `0..rows` (output row = activation row), matrix 0.
    pub fn dense(rows: u32) -> Vec<QItem> {
        (0..rows.div_ceil(QITEM_ROWS as u32))
            .map(|c| {
                let mut it = QItem { e: 0, ..Default::default() };
                for b in 0..QITEM_ROWS as u32 {
                    let r = c * QITEM_ROWS as u32 + b;
                    if r < rows {
                        it.xi[it.n as usize] = r;
                        it.yi[it.n as usize] = r;
                        it.n += 1;
                    }
                }
                it
            })
            .collect()
    }
}

/// `struct MultiQ8` of gemv.cu, passed by value.
#[repr(C)]
struct MultiQ8 {
    qs: [DevicePtr; 8],
    aux: [DevicePtr; 8],
    xq: [DevicePtr; 8],
    xd: [DevicePtr; 8],
    y: [DevicePtr; 8],
    n: [u32; 8],
    start: [u32; 9],
    count: u32,
}

/// One product of [`Gemv::launch_q_multi`]: `y = m x`.
pub struct MultiItem<'a> {
    pub m: &'a GemvMatrix,
    pub x: &'a QAct,
    pub y: &'a mut dyn DevMut<f32>,
}

/// Activations quantised to int8 per 32 values (`d = amax / 127`, llama.cpp's q8_1 scheme)
/// for [`Gemv::launch_q`]. Holds up to `max_slots` vectors of `k` values.
pub struct QAct {
    q: DeviceBuffer<i8>,
    d: DeviceBuffer<f32>,
    /// The same activations in f16 (prefill tensor-core GEMM); written only when allocated.
    h: Option<DeviceBuffer<u16>>,
    pub k: u32,
    pub max_slots: u32,
    /// Vectors held by the last [`Gemv::quantize`] (1 = shared by every slot).
    pub slots: u32,
}

impl QAct {
    /// Device address of the f16 copy of the activations.
    pub fn f16_ptr(&self) -> Result<DevicePtr> {
        self.h.as_ref().map(|h| h.device_ptr()).ok_or_else(|| invalid("activations have no f16 copy".into()))
    }

    /// Device planes for a kernel that writes `slots` quantised vectors of `k` values.
    /// Planes (int8, scales, optional f16) for a kernel that writes `slots` vectors of `k`.
    pub(crate) fn planes_for(&mut self, slots: u32) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
        if slots == 0 || slots > self.max_slots {
            return Err(invalid(format!("qact: {slots} vectors, buffer holds {}", self.max_slots)));
        }
        self.slots = slots;
        Ok((self.q.device_ptr(), self.d.device_ptr(), self.h.as_ref().map_or(0, |h| h.device_ptr())))
    }

    /// Device pointers of the int8 values and the per-32 scales.
    pub fn planes(&self) -> (DevicePtr, DevicePtr) {
        (self.q.device_ptr(), self.d.device_ptr())
    }

    /// Like [`QAct::new`], also keeping an f16 copy (for [`Gemv::launch_mma`]'s f16 kernels).
    pub fn with_f16(ctx: &std::sync::Arc<astral_gpu::Context>, k: u32, max_slots: u32) -> Result<Self> {
        let mut a = Self::new(ctx, k, max_slots)?;
        a.h = Some(DeviceBuffer::zeroed(ctx, (k * max_slots) as usize)?);
        Ok(a)
    }

    pub fn new(ctx: &std::sync::Arc<astral_gpu::Context>, k: u32, max_slots: u32) -> Result<Self> {
        if k == 0 || !k.is_multiple_of(32) || max_slots == 0 {
            return Err(invalid(format!("qact: k {k} must be a positive multiple of 32")));
        }
        let n = (k * max_slots) as usize;
        Ok(Self { q: DeviceBuffer::zeroed(ctx, n)?, d: DeviceBuffer::zeroed(ctx, n / 32)?, h: None, k, max_slots, slots: 0 })
    }
}

impl Gemv {
    const ROWS_PER_BLOCK: u32 = 8;

    pub fn new(reg: &KernelRegistry) -> Result<Self> {
        let f = |n: &str| reg.function("gemv", n);
        Ok(Self {
            f: [f("astral_gemv_f32")?, f("astral_gemv_bf16")?, f("astral_gemv_q8_0")?, f("astral_gemv_nvfp4")?],
            fq: [f("astral_gemv_q8_0_dp4a")?, f("astral_gemv_nvfp4_dp4a")?],
            fq_r1: f("astral_gemv_nvfp4_dp4a_r1")?,
            fq_s: [f("astral_gemv_nvfp4_s1")?, f("astral_gemv_nvfp4_s2")?, f("astral_gemv_nvfp4_s4")?],
            fq_long: [f("astral_gemv_nvfp4_s1u12")?, f("astral_gemv_nvfp4_s1u9")?],
            fq_mx: f("astral_gemv_mx4_s1")?,
            fq_mx_w: [f("astral_gemv_mx4_w2")?, f("astral_gemv_mx4_w4")?],
            quant: f("astral_quantize_x_q8")?,
            rb: [f("astral_gemv_f32_rowblock")?, f("astral_gemv_bf16_rowblock")?],
            multi: f("astral_gemv_q8_0_multi")?,
            multi_t: (2..=8).map(|t| f(&format!("astral_gemv_q8_0_multi_t{t}"))).collect::<Result<_>>()?,
            multi_mma: f("astral_gemv_q8_0_multi_mma")?,
            items_s: {
                let k = f("astral_gemv_nvfp4_items_s1")?;
                // Staged activations: let several blocks share an SM's shared memory.
                k.set_carveout(100)?;
                k
            },
            items_q8x: f("astral_gemv_q8_0_items_x")?,
            items_mma: {
                let q = f("astral_gemv_q8_0_mma_items")?;
                let h = f("astral_gemv_nvfp4_frag_items")?;
                [q, h]
            },
            dense_mma: [f("astral_gemv_q8_0_dense_n2")?, f("astral_gemv_q8_0_dense_n4")?, f("astral_gemv_q8_0_dense_n8")?],
            q_tiled: {
                let fw = f("astral_gemv_q8_0_tiled")?;
                let _ = fw.set_max_dynamic_shared(Q_TILED_SMEM);
                fw
            },
            ws256: {
                let a = f("astral_gemm_nvfp4_ws256")?;
                let b = f("astral_gemm_nvfp4_ws256_t64")?;
                let c = f("astral_gemm_nvfp4_ws256_t32")?;
                for k in [&a, &b, &c] {
                    let _ = k.set_max_dynamic_shared(WS256_SMEM);
                }
                [a, b, c]
            },
            gemm_tma: {
                let a = f("astral_gemm_f16_tma")?;
                let b = f("astral_gemm_f16_tma_t256")?;
                let _ = a.set_max_dynamic_shared(GEMM_TMA_SMEM);
                let _ = b.set_max_dynamic_shared(GEMM_TMA_SMEM);
                [a, b]
            },
            wgmma_f16: {
                let fw = f("astral_gemm_f16_wgmma")?;
                // 3 stages of 32 KB; fails harmlessly on GPUs without that much (Hopper only).
                let _ = fw.set_max_dynamic_shared(WGMMA_F16_SMEM);
                fw
            },
            rows_t: [f("astral_gemv_q8_0_rows_t16")?, f("astral_gemv_q8_0_rows_t32")?],

            multi_mma_n: [f("astral_gemv_q8_0_multi_mma_n2")?, f("astral_gemv_q8_0_multi_mma_n4")?, f("astral_gemv_q8_0_multi_mma_n8")?],

            items: [f("astral_gemm_q8_0_items")?, f("astral_gemm_nvfp4_items")?],
            mma: [f("astral_gemm_q8_0_mma")?, f("astral_gemm_nvfp4_mma")?],
            mma2: [f("astral_gemm_q8_0_mma2")?, f("astral_gemm_nvfp4_mma2")?],
            mma3: [f("astral_gemm_q8_0_mma3")?, f("astral_gemm_nvfp4_mma3")?],
            mma4: f("astral_gemm_nvfp4_mma4")?,
            mma5: f("astral_gemm_nvfp4_mma5")?,
            wgmma: f("astral_gemm_nvfp4_wgmma")?,
            wgmma2: {
                let fw = f("astral_gemm_nvfp4_wgmma2")?;
                // 3 stages of 21 KB (1024-aligned) of dynamic shared memory; fails harmlessly on
                // GPUs without that much (the kernel is only used on Hopper).
                let _ = fw.set_max_dynamic_shared(WGMMA2_SMEM);
                fw
            },
            zero: f("astral_zero_f32")?,
        })
    }

    /// Quantise `slots` vectors of `out.k` values (row `s` at `x[s * x_stride]`) into `out`.
    pub fn quantize(&self, stream: &Stream, x: &impl Dev<f32>, x_stride: u32, slots: u32, out: &mut QAct) -> Result<()> {
        if slots == 0 || slots > out.max_slots {
            return Err(invalid(format!("quantize: {slots} slots, buffer holds {}", out.max_slots)));
        }
        let need = (slots as usize - 1) * x_stride as usize + out.k as usize;
        if x.dlen() < need {
            return Err(invalid(format!("quantize: x has {} values, need {need}", x.dlen())));
        }
        let total = slots * out.k / 32;
        let (mut px, mut pq, mut pd, mut k, mut xs, mut tb) = (x.dptr(), out.q.device_ptr(), out.d.device_ptr(), out.k, x_stride, total);
        let mut p = [param(&mut px), param(&mut pq), param(&mut pd), param(&mut k), param(&mut xs), param(&mut tb)];
        let cfg = LaunchConfig { grid: (total.div_ceil(8), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: params match astral_quantize_x_q8; x and the output planes were size-checked.
        unsafe { self.quant.launch(cfg, stream, &mut p) }?;
        out.slots = slots;
        Ok(())
    }

    /// Grouped int8 GEMM (prefill): for each of the first `n_items` items in `items` (words from
    /// [`QItem::words`]), `y[yi[b]] = W_e x[xi[b]]`. `x` holds `x_rows` quantised rows; `y` has
    /// `y_rows` rows of `m.n`. Indices are clamped on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_items(&self, stream: &Stream, m: &GemvMatrix, x: &QAct, x_rows: u32, y: &mut dyn DevMut<f32>, y_rows: u32, items: &DeviceBuffer<u32>, n_items: u32) -> Result<()> {
        let fi = match m.ty {
            GemvType::Q8_0 => 0,
            GemvType::Nvfp4 => 1,
            t => return Err(invalid(format!("launch_items: {t:?} has no int8-activation kernel"))),
        };
        if n_items == 0 || x_rows == 0 || y_rows == 0 || m.k != x.k || x_rows > x.max_slots {
            return Err(invalid(format!("launch_items: {n_items} items, {x_rows} rows of k {} for a k {} matrix", x.k, m.k)));
        }
        if items.len() < n_items as usize * QItem::WORDS || y.dlen() < y_rows as usize * m.n as usize {
            return Err(invalid("launch_items: items or y too small".into()));
        }
        let (rq, ra) = m.ty.row_bytes(m.k);
        let (mut pq, mut pa, mut px, mut pd, mut py) = (m.qs.device_ptr(), m.aux.device_ptr(), x.q.device_ptr(), x.d.device_ptr(), y.dptr());
        let (mut n, mut k) = (m.n, m.k);
        let (mut mq, mut ma) = ((rq * m.n as usize) as u64, (ra * m.n as usize) as u64);
        let (mut pi, mut xr, mut yr) = (items.device_ptr(), x_rows, y_rows);
        let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
        let mut nm = m.n_mats;
        let mut params = [
            param(&mut pq),
            param(&mut pa),
            param(&mut px),
            param(&mut pd),
            param(&mut py),
            param(&mut n),
            param(&mut k),
            param(&mut mq),
            param(&mut ma),
            param(&mut pi),
            param(&mut xr),
            param(&mut yr),
            param(&mut psc),
            param(&mut nm),
        ];
        let rows = Self::ROWS_PER_BLOCK * Self::q_rows_per_warp(m.ty);
        let cfg = LaunchConfig { grid: (m.n.div_ceil(rows), n_items, 1), block: (32 * Self::ROWS_PER_BLOCK, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: params match GEMM_ITEMS_ENTRY in gemv.cu; planes validated at upload, the
        // item words cover n_items, y covers y_rows, and the kernel clamps every index.
        unsafe { self.items[fi].launch(cfg, stream, &mut params) }
    }

    /// Tensor-core grouped GEMM (prefill): `y[yidx[j]] = W_e x[xidx[j]]` for the tokens `j` of
    /// each of `n_items` items (device copies of an [`MmaWork`]: `items` 3 words per item,
    /// `xidx` / `yidx` of `idx_len`). Needs `m.n % 16 == 0`. Indices are clamped on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mma(
        &self,
        stream: &Stream,
        m: &GemvMatrix,
        x: &QAct,
        x_rows: u32,
        y: &mut dyn DevMut<f32>,
        y_rows: u32,
        items: &DeviceBuffer<u32>,
        n_items: u32,
        xidx: &DeviceBuffer<u32>,
        yidx: &DeviceBuffer<u32>,
        idx_len: u32,
    ) -> Result<()> {
        let fi = match m.ty {
            GemvType::Q8_0 => 0,
            GemvType::Nvfp4 => 1,
            t => return Err(invalid(format!("launch_mma: {t:?} has no tensor-core kernel"))),
        };
        if n_items == 0 || x_rows == 0 || y_rows == 0 || idx_len == 0 || m.k != x.k || x_rows > x.max_slots || !m.n.is_multiple_of(16) {
            return Err(invalid(format!("launch_mma: {n_items} items, {x_rows} rows, n {} k {} (activations k {})", m.n, m.k, x.k)));
        }
        if items.len() < 3 * n_items as usize || xidx.len() < idx_len as usize || yidx.len() < idx_len as usize || y.dlen() < y_rows as usize * m.n as usize {
            return Err(invalid("launch_mma: work lists or y too small".into()));
        }
        let (rq, ra) = m.ty.row_bytes(m.k);
        let (mut pq, mut pa, mut px, mut pd, mut py) = (m.qs.device_ptr(), m.aux.device_ptr(), x.q.device_ptr(), x.d.device_ptr(), y.dptr());
        let (mut n, mut k) = (m.n, m.k);
        let (mut mq, mut ma) = ((rq * m.n as usize) as u64, (ra * m.n as usize) as u64);
        let (mut pi, mut pxi, mut pyi, mut il, mut xr, mut yr) = (items.device_ptr(), xidx.device_ptr(), yidx.device_ptr(), idx_len, x_rows, y_rows);
        // f16 path (dequantised weights, f16 activations) when the activations carry an f16 copy.
        if let Some(h) = x.h.as_ref().filter(|_| m.n.is_multiple_of(64) && m.k.is_multiple_of(64) && std::env::var_os("ASTRAL_MMA_INT8").is_none()) {
            let mut ph = h.device_ptr();
            let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
            let mut nm = m.n_mats;
            // Split K when the grid would be too small to fill the GPU (dense layers, short
            // prompts); partial sums are added atomically into a zeroed y.
            let kt = m.k / 64;
            let blocks = (m.n / 64) * n_items;
            let splits = if blocks >= 256 || !split_k() { 1 } else { 256u32.div_ceil(blocks).min(kt).min(8) };
            let mut ktps = kt.div_ceil(splits);
            let splits = kt.div_ceil(ktps);
            if splits > 1 {
                let (mut pz, mut nz) = (y.dptr(), (y_rows as u64) * m.n as u64);
                let mut zp = [param(&mut pz), param(&mut nz)];
                let zc = LaunchConfig { grid: ((nz as u32).div_ceil(256).min(4096), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
                // SAFETY: y holds at least y_rows * n floats (checked above).
                unsafe { self.zero.launch(zc, stream, &mut zp) }?;
            }
            let mut params = [
                param(&mut pq),
                param(&mut pa),
                param(&mut ph),
                param(&mut py),
                param(&mut n),
                param(&mut k),
                param(&mut mq),
                param(&mut ma),
                param(&mut pi),
                param(&mut pxi),
                param(&mut pyi),
                param(&mut il),
                param(&mut xr),
                param(&mut yr),
                param(&mut psc),
                param(&mut nm),
                param(&mut ktps),
            ];
            let cfg = LaunchConfig { grid: (m.n / 64, n_items, splits), block: (128, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: params match GEMM_MMA3_ENTRY in gemv.cu; planes validated at upload, the
            // f16 plane holds max_slots >= x_rows rows of k, the work lists cover n_items /
            // idx_len, y covers y_rows, the kernel clamps every index; n, k % 64 checked above.
            return unsafe { self.mma3[fi].launch(cfg, stream, &mut params) };
        }
        let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
        let mut nm = m.n_mats;
        let mut params = [
            param(&mut pq),
            param(&mut pa),
            param(&mut px),
            param(&mut pd),
            param(&mut py),
            param(&mut n),
            param(&mut k),
            param(&mut mq),
            param(&mut ma),
            param(&mut pi),
            param(&mut pxi),
            param(&mut pyi),
            param(&mut il),
            param(&mut xr),
            param(&mut yr),
            param(&mut psc),
            param(&mut nm),
        ];
        // The pipelined kernel needs whole 64-row tiles and 64-wide K stages.
        let v2 = m.n.is_multiple_of(64) && m.k.is_multiple_of(64) && std::env::var_os("ASTRAL_MMA_V1").is_none();
        let (f, cfg) = if v2 {
            (&self.mma2[fi], LaunchConfig { grid: (m.n / 64, n_items, 1), block: (128, 1, 1), shared_mem_bytes: 0 })
        } else {
            (&self.mma[fi], LaunchConfig { grid: (m.n.div_ceil(128), n_items, 1), block: (256, 1, 1), shared_mem_bytes: 0 })
        };
        // SAFETY: params match GEMM_MMA_ENTRY / GEMM_MMA2_ENTRY in gemv.cu (same signature);
        // planes validated at upload, the work lists cover n_items / idx_len, y covers y_rows,
        // and the kernels clamp every index (v2 also needs n % 64 == 0 and k % 64 == 0).
        unsafe { f.launch(cfg, stream, &mut params) }
    }

    /// Big-tile tensor-core GEMM for dense NVFP4 layers (prefill): like the f16 path of
    /// [`Gemv::launch_mma`] but with items of up to 128 tokens ([`MmaWork::push_chunked`]).
    /// Needs `m.n % 128 == 0`, `m.k % 64 == 0` and activations with an f16 copy.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mma4(
        &self,
        stream: &Stream,
        m: &GemvMatrix,
        x: &QAct,
        x_rows: u32,
        y: &mut dyn DevMut<f32>,
        y_rows: u32,
        items: &DeviceBuffer<u32>,
        n_items: u32,
        xidx: &DeviceBuffer<u32>,
        yidx: &DeviceBuffer<u32>,
        idx_len: u32,
    ) -> Result<()> {
        let Some(h) = x.h.as_ref() else { return Err(invalid("launch_mma4: activations have no f16 copy".into())) };
        if m.ty != GemvType::Nvfp4 || !m.n.is_multiple_of(128) || !m.k.is_multiple_of(64) || m.k != x.k || n_items == 0 || x_rows == 0 || y_rows == 0 || idx_len == 0 || x_rows > x.max_slots {
            return Err(invalid(format!("launch_mma4: {:?} n {} k {} (activations k {})", m.ty, m.n, m.k, x.k)));
        }
        if items.len() < 3 * n_items as usize || xidx.len() < idx_len as usize || yidx.len() < idx_len as usize || y.dlen() < y_rows as usize * m.n as usize {
            return Err(invalid("launch_mma4: work lists or y too small".into()));
        }
        let (rq, ra) = m.ty.row_bytes(m.k);
        let (mut pq, mut pa, mut ph, mut py) = (m.qs.device_ptr(), m.aux.device_ptr(), h.device_ptr(), y.dptr());
        let (mut n, mut k) = (m.n, m.k);
        let (mut mq, mut ma) = ((rq * m.n as usize) as u64, (ra * m.n as usize) as u64);
        let (mut pi, mut pxi, mut pyi, mut il, mut xr, mut yr) = (items.device_ptr(), xidx.device_ptr(), yidx.device_ptr(), idx_len, x_rows, y_rows);
        let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
        let mut nm = m.n_mats;
        let mut params = [
            param(&mut pq),
            param(&mut pa),
            param(&mut ph),
            param(&mut py),
            param(&mut n),
            param(&mut k),
            param(&mut mq),
            param(&mut ma),
            param(&mut pi),
            param(&mut pxi),
            param(&mut pyi),
            param(&mut il),
            param(&mut xr),
            param(&mut yr),
            param(&mut psc),
            param(&mut nm),
        ];
        let cfg = LaunchConfig { grid: (m.n / 128, n_items, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: params match GEMM_MMA4_ENTRY in gemv.cu; planes validated at upload, the f16
        // plane holds max_slots >= x_rows rows of k, the work lists cover n_items / idx_len, y
        // covers y_rows, the kernel clamps every index; n % 128 and k % 64 checked above.
        unsafe { self.mma4.launch(cfg, stream, &mut params) }
    }

    /// The warp-specialised NVFP4 expert GEMM (Hopper, fragment layout, `m.n % 256 == 0`) with
    /// items of up to `tile` tokens (32, 64 or 128; the work lists must be chunked to match).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_ws256(
        &self,
        stream: &Stream,
        tile: u32,
        m: &GemvMatrix,
        x: &QAct,
        x_rows: u32,
        y: &mut dyn DevMut<f32>,
        y_rows: u32,
        items: &DeviceBuffer<u32>,
        n_items: u32,
        xidx: &DeviceBuffer<u32>,
        yidx: &DeviceBuffer<u32>,
        idx_len: u32,
    ) -> Result<()> {
        let Some(h) = x.h.as_ref() else { return Err(invalid("launch_ws256: activations have no f16 copy".into())) };
        let Some((fq, fs)) = m.frag.as_ref() else { return Err(invalid("launch_ws256: matrix has no fragment layout".into())) };
        if !m.n.is_multiple_of(256) || !m.k.is_multiple_of(64) || m.k != x.k || n_items == 0 || x_rows == 0 || y_rows == 0 || idx_len == 0 || x_rows > x.max_slots {
            return Err(invalid(format!("launch_ws256: n {} k {} (activations k {})", m.n, m.k, x.k)));
        }
        let plane = m.n_mats as usize * m.n as usize * m.k as usize;
        if fq.len() < plane / 2 || fs.len() < plane / 16 || items.len() < 3 * n_items as usize || xidx.len() < idx_len as usize || yidx.len() < idx_len as usize || y.dlen() < y_rows as usize * m.n as usize {
            return Err(invalid("launch_ws256: planes, work lists or y too small".into()));
        }
        let (f, stage) = match tile {
            128 => (&self.ws256[0], 25600u32),
            64 => (&self.ws256[1], 17408),
            32 => (&self.ws256[2], 13312),
            _ => return Err(invalid(format!("launch_ws256: tile {tile}"))),
        };
        let (mut pq, mut pa, mut ph, mut py) = (fq.device_ptr(), fs.device_ptr(), h.device_ptr(), y.dptr());
        let (mut n, mut k) = (m.n, m.k);
        let (mut pi, mut pxi, mut pyi, mut il, mut xr, mut yr) = (items.device_ptr(), xidx.device_ptr(), yidx.device_ptr(), idx_len, x_rows, y_rows);
        let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
        let (mut nm, mut unused) = (m.n_mats, 0u32);
        let mut params = [
            param(&mut pq),
            param(&mut pa),
            param(&mut ph),
            param(&mut py),
            param(&mut n),
            param(&mut k),
            param(&mut pi),
            param(&mut pxi),
            param(&mut pyi),
            param(&mut il),
            param(&mut xr),
            param(&mut yr),
            param(&mut psc),
            param(&mut nm),
            param(&mut unused),
        ];
        let cfg = LaunchConfig { grid: (m.n / 256, n_items, 1), block: (384, 1, 1), shared_mem_bytes: 5 * stage };
        // SAFETY: params match astral_gemm_nvfp4_ws256*; the fragment planes hold n_mats * n * k
        // values (checked), the f16 activations x_rows rows of k, the work lists cover n_items /
        // idx_len (items of <= tile tokens), y covers y_rows; the kernel clamps every index.
        unsafe { f.launch(cfg, stream, &mut params) }
    }

    /// Tensor-core GEMM on the fragment layout ([`GemvMatrix::add_frag`]) for prefill: items of
    /// up to 128 tokens ([`MmaWork::push_chunked`]); `m.n % 128 == 0`, `m.k % 64 == 0`,
    /// activations with an f16 copy.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mma5(
        &self,
        stream: &Stream,
        m: &GemvMatrix,
        x: &QAct,
        x_rows: u32,
        y: &mut dyn DevMut<f32>,
        y_rows: u32,
        items: &DeviceBuffer<u32>,
        n_items: u32,
        xidx: &DeviceBuffer<u32>,
        yidx: &DeviceBuffer<u32>,
        idx_len: u32,
    ) -> Result<()> {
        let Some(h) = x.h.as_ref() else { return Err(invalid("launch_mma5: activations have no f16 copy".into())) };
        let Some((fq, fs)) = m.frag.as_ref() else { return Err(invalid("launch_mma5: matrix has no fragment layout".into())) };
        if !m.n.is_multiple_of(128) || !m.k.is_multiple_of(64) || m.k != x.k || n_items == 0 || x_rows == 0 || y_rows == 0 || idx_len == 0 || x_rows > x.max_slots {
            return Err(invalid(format!("launch_mma5: n {} k {} (activations k {})", m.n, m.k, x.k)));
        }
        let plane = m.n_mats as usize * m.n as usize * m.k as usize;
        if fq.len() < plane / 2
            || fs.len() < plane / 16
            || items.len() < 3 * n_items as usize
            || xidx.len() < idx_len as usize
            || yidx.len() < idx_len as usize
            || y.dlen() < y_rows as usize * m.n as usize
        {
            return Err(invalid("launch_mma5: planes, work lists or y too small".into()));
        }
        let (mut pq, mut pa, mut ph, mut py) = (fq.device_ptr(), fs.device_ptr(), h.device_ptr(), y.dptr());
        let (mut n, mut k) = (m.n, m.k);
        let (mut pi, mut pxi, mut pyi, mut il, mut xr, mut yr) = (items.device_ptr(), xidx.device_ptr(), yidx.device_ptr(), idx_len, x_rows, y_rows);
        let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
        let mut nm = m.n_mats;
        // Split K when the grid is too small to fill the GPU (about 2 blocks per SM resident).
        let kc = m.k / 64;
        let blocks = (m.n / 128) * n_items;
        let splits = if blocks >= 384 || !split_k() { 1 } else { 512u32.div_ceil(blocks).min(kc).min(8) };
        let mut kcps = kc.div_ceil(splits);
        let splits = kc.div_ceil(kcps);
        if splits > 1 {
            let (mut pz, mut nz) = (y.dptr(), (y_rows as u64) * m.n as u64);
            let mut zp = [param(&mut pz), param(&mut nz)];
            let zc = LaunchConfig { grid: ((nz as u32).div_ceil(256).min(4096), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: y holds at least y_rows * n floats (checked above).
            unsafe { self.zero.launch(zc, stream, &mut zp) }?;
        }
        let mut params = [
            param(&mut pq),
            param(&mut pa),
            param(&mut ph),
            param(&mut py),
            param(&mut n),
            param(&mut k),
            param(&mut pi),
            param(&mut pxi),
            param(&mut pyi),
            param(&mut il),
            param(&mut xr),
            param(&mut yr),
            param(&mut psc),
            param(&mut nm),
            param(&mut kcps),
        ];
        // SAFETY: params match astral_gemm_nvfp4_mma5; the fragment planes hold n_mats * n * k
        // values (checked), the f16 activations max_slots >= x_rows rows of k, the work lists
        // cover n_items / idx_len, y covers y_rows; the kernel clamps every index.
        // v6 (32 x 64 warp tiles, `ASTRAL_MMA6=1`) measured slower than v5 on Qwen3.8 prefill.
        // Hopper (sm_90a build on an sm_90 device): the warpgroup-MMA kernel; else v5.
        let hopper = astral_kernels_src_has_sm90a() && stream.context().device().compute_capability().map(|(m, _)| m == 9).unwrap_or(false);
        if hopper && splits == 1 && m.n.is_multiple_of(256) && std::env::var_os("ASTRAL_NO_WGMMA").is_none() && std::env::var_os("ASTRAL_NO_WS256").is_none() {
            let cfg = LaunchConfig { grid: (m.n / 256, n_items, 1), block: (384, 1, 1), shared_mem_bytes: WS256_SMEM };
            // SAFETY: as for the kernels below (same parameters); one K split.
            return unsafe { self.ws256[0].launch(cfg, stream, &mut params) };
        }
        let (f, smem) = if hopper && std::env::var_os("ASTRAL_NO_WGMMA").is_none() {
            if std::env::var_os("ASTRAL_WGMMA1").is_some() {
                (&self.wgmma, 0)
            } else {
                (&self.wgmma2, WGMMA2_SMEM)
            }
        } else {
            (&self.mma5, 0)
        };
        let cfg = LaunchConfig { grid: (m.n / 128, n_items, splits), block: (256, 1, 1), shared_mem_bytes: smem };
        unsafe { f.launch(cfg, stream, &mut params) }
    }

    /// Weight rows per warp of the int8-activation kernels (`QShape` in gemv.cu).
    fn q_rows_per_warp(ty: GemvType) -> u32 {
        if ty == GemvType::Q8_0 {
            1
        } else {
            4
        }
    }

    /// Up to 8 single-matrix Q8_0 products with the same `k` in one launch. Falls back to one
    /// [`Gemv::launch_q`] per item when they do not all qualify.
    /// Whether [`Gemv::launch_q_multi_t`] takes these products.
    pub fn multi_t_ok(items: &[MultiItem], ntok: u32) -> bool {
        let k = items.first().map_or(0, |it| it.m.k);
        (2..=8).contains(&ntok)
            && (1..=8).contains(&items.len())
            && items.iter().all(|it| it.m.ty == GemvType::Q8_0 && it.m.n_mats == 1 && it.m.mat_scale.is_none() && it.m.k == k && it.x.k == k && it.x.slots >= ntok)
    }

    /// [`Gemv::launch_q_multi_t`] on int8 tensor cores (`mma.sync m16n8k32`), for 1..=8 tokens:
    /// 16 weight rows per block of 4 warps (K split 4 ways). Used for verify passes and the
    /// draft unless `ASTRAL_SMALL_DP4A=1`.
    pub fn launch_q_multi_mma(&self, stream: &Stream, items: &mut [MultiItem], ntok: u32) -> Result<()> {
        let k = items.first().map_or(0, |it| it.m.k);
        let ok = (1..=8).contains(&ntok)
            && (1..=8).contains(&items.len())
            && k.is_multiple_of(128)
            && items.iter().all(|it| it.m.ty == GemvType::Q8_0 && it.m.n_mats == 1 && it.m.mat_scale.is_none() && it.m.k == k && it.x.k == k && it.x.slots >= ntok);
        if !ok {
            return Err(invalid(format!("launch_q_multi_mma: {} products for {ntok} tokens do not qualify", items.len())));
        }
        let mut d = MultiQ8 { qs: [0; 8], aux: [0; 8], xq: [0; 8], xd: [0; 8], y: [0; 8], n: [0; 8], start: [0; 9], count: items.len() as u32 };
        let mut blocks = 0u32;
        for (i, it) in items.iter().enumerate() {
            if it.y.dlen() < (ntok * it.m.n) as usize {
                return Err(invalid(format!("launch_q_multi_mma: y {} has {} values, need {}", i, it.y.dlen(), ntok * it.m.n)));
            }
            d.qs[i] = it.m.qs.device_ptr();
            d.aux[i] = it.m.aux.device_ptr();
            d.xq[i] = it.x.q.device_ptr();
            d.xd[i] = it.x.d.device_ptr();
            d.y[i] = it.y.dptr();
            d.n[i] = it.m.n;
            d.start[i] = blocks;
            blocks += it.m.n.div_ceil(16);
        }
        d.start[items.len()] = blocks;
        let (mut kk, mut t) = (k, ntok);
        let mut params = [param(&mut d), param(&mut kk), param(&mut t)];
        let cfg = LaunchConfig { grid: (blocks, 1, 1), block: (128, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: MultiQ8 mirrors gemv.cu; planes validated at upload, activations hold >= ntok
        // rows of k and each y ntok * n values (checked); k % 128 == 0 for the 4-way K split.
        unsafe { self.multi_mma.launch(cfg, stream, &mut params) }
    }

    /// [`Gemv::launch_q_multi_mma`] for any number of tokens, 8 at a time. Each token's result
    /// equals its single-token launch (the tensor-core kernel computes every column alike).
    pub fn launch_q_multi_mma_rows(&self, stream: &Stream, items: &mut [MultiItem], ntok: u32) -> Result<()> {
        let mut t0 = 0;
        while t0 < ntok {
            let c = (ntok - t0).min(64);
            let f = match c {
                1..=8 => &self.multi_mma,
                9..=16 => &self.multi_mma_n[0],
                17..=32 => &self.multi_mma_n[1],
                _ => &self.multi_mma_n[2],
            };
            self.multi_q8_chunk(stream, items, t0, c, f, true)?;
            t0 += c;
        }
        Ok(())
    }


    /// `y[t] = m x[t]` for `ntok` int8 activation rows of a Q8_0 matrix, each row computed
    /// exactly as [`Gemv::launch_q`] computes one (8 rows share each weight read).
    pub fn launch_q_rows(&self, stream: &Stream, m: &GemvMatrix, x: &QAct, y: &mut dyn DevMut<f32>, ntok: u32) -> Result<()> {
        if m.ty != GemvType::Q8_0 || m.n_mats != 1 || m.mat_scale.is_some() || m.k != x.k || x.slots < ntok || ntok == 0 {
            return Err(invalid(format!("launch_q_rows: {:?} matrix k {} for {ntok} rows of k {} ({} held)", m.ty, m.k, x.k, x.slots)));
        }
        let mut items = [MultiItem { m, x, y }];
        let mut t0 = 0;
        while t0 < ntok {
            let c = (ntok - t0).min(32);
            if c > 8 {
                self.q8_rows_chunk(stream, &mut items[0], t0, c)?;
            } else {
                let f = if c == 1 { &self.multi } else { &self.multi_t[c as usize - 2] };
                self.multi_q8_chunk(stream, &mut items, t0, c, f, false)?;
            }
            t0 += c;
        }
        Ok(())
    }

    /// Tokens `t0..t0 + c` (9..=32) of one Q8_0 product on the rows kernels.
    fn q8_rows_chunk(&self, stream: &Stream, it: &mut MultiItem, t0: u32, c: u32) -> Result<()> {
        let k = it.m.k;
        if !(9..=32).contains(&c) || it.x.slots < t0 + c || it.y.dlen() < ((t0 + c) * it.m.n) as usize {
            return Err(invalid(format!("q8_rows: tokens {t0}..+{c} do not fit")));
        }
        let mut d = MultiQ8 { qs: [0; 8], aux: [0; 8], xq: [0; 8], xd: [0; 8], y: [0; 8], n: [0; 8], start: [0; 9], count: 1 };
        d.qs[0] = it.m.qs.device_ptr();
        d.aux[0] = it.m.aux.device_ptr();
        d.xq[0] = it.x.q.device_ptr() + (t0 * k) as u64;
        d.xd[0] = it.x.d.device_ptr() + 4 * (t0 * (k / 32)) as u64;
        d.y[0] = it.y.dptr() + 4 * (t0 * it.m.n) as u64;
        d.n[0] = it.m.n;
        // 8 warps x 4 rows per block.
        let blocks = it.m.n.div_ceil(4 * Self::ROWS_PER_BLOCK);
        d.start[1] = blocks;
        let (mut kk, mut cc) = (k, c);
        let mut params = [param(&mut d), param(&mut kk), param(&mut cc)];
        let cfg = LaunchConfig { grid: (blocks, 1, 1), block: (32 * Self::ROWS_PER_BLOCK, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: MultiQ8 mirrors gemv.cu; the activation planes hold >= t0 + c rows of k and y
        // (t0 + c) * n values (checked), so the offset pointers stay inside them.
        unsafe { self.rows_t[(c > 16) as usize].launch(cfg, stream, &mut params) }
    }

    /// One launch of a MultiQ8 kernel over tokens `t0..t0 + c` of every item.
    fn multi_q8_chunk(&self, stream: &Stream, items: &mut [MultiItem], t0: u32, c: u32, f: &Function, mma: bool) -> Result<()> {
        let k = items.first().map_or(0, |it| it.m.k);
        let ok = (1..=if mma { 64 } else { 8 }).contains(&c)
            && (1..=8).contains(&items.len())
            && (!mma || k.is_multiple_of(128))
            && items.iter().all(|it| it.m.ty == GemvType::Q8_0 && it.m.n_mats == 1 && it.m.mat_scale.is_none() && it.m.k == k && it.x.k == k && it.x.slots >= t0 + c);
        if !ok {
            return Err(invalid(format!("multi_q8: {} products for tokens {t0}..+{c} do not qualify", items.len())));
        }
        let mut d = MultiQ8 { qs: [0; 8], aux: [0; 8], xq: [0; 8], xd: [0; 8], y: [0; 8], n: [0; 8], start: [0; 9], count: items.len() as u32 };
        let per_block = if mma { 16 } else { Self::ROWS_PER_BLOCK };
        let mut blocks = 0u32;
        for (i, it) in items.iter().enumerate() {
            if it.y.dlen() < ((t0 + c) * it.m.n) as usize {
                return Err(invalid(format!("multi_q8: y {} has {} values, need {}", i, it.y.dlen(), (t0 + c) * it.m.n)));
            }
            d.qs[i] = it.m.qs.device_ptr();
            d.aux[i] = it.m.aux.device_ptr();
            d.xq[i] = it.x.q.device_ptr() + (t0 * k) as u64;
            d.xd[i] = it.x.d.device_ptr() + 4 * (t0 * (k / 32)) as u64;
            d.y[i] = it.y.dptr() + 4 * (t0 * it.m.n) as u64;
            d.n[i] = it.m.n;
            d.start[i] = blocks;
            blocks += it.m.n.div_ceil(per_block);
        }
        d.start[items.len()] = blocks;
        let (mut kk, mut t) = (k, c);
        let cfg = LaunchConfig { grid: (blocks, 1, 1), block: (if mma { 128 } else { 32 * Self::ROWS_PER_BLOCK }, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: MultiQ8 mirrors gemv.cu; planes validated at upload, the activation planes
        // hold >= t0 + c rows of k and each y (t0 + c) * n values (checked), so the offset
        // pointers stay inside them; the mma kernel needs k % 128 == 0 (checked).
        if mma {
            let mut params = [param(&mut d), param(&mut kk), param(&mut t)];
            unsafe { f.launch(cfg, stream, &mut params) }
        } else {
            let mut params = [param(&mut d), param(&mut kk)];
            unsafe { f.launch(cfg, stream, &mut params) }
        }
    }

    /// Grouped NVFP4 products for batched decode: item `i` of `items` (device [`QItem`] words,
    /// `n_items` of them, unused ones with n = 0) multiplies matrix `e` with activation rows
    /// `xi` into output rows `yi`. Each (row, activation) is computed exactly as
    /// [`Gemv::launch_q`] computes it for a single token.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_items_exact(&self, stream: &Stream, m: &GemvMatrix, x: &QAct, x_rows: u32, y: &mut dyn DevMut<f32>, y_rows: u32, items: &DeviceBuffer<u32>, n_items: u32) -> Result<()> {
        let q8 = m.ty == GemvType::Q8_0;
        if !(q8 || m.ty == GemvType::Nvfp4) || m.mx_scale.is_some() || m.k != x.k || x_rows == 0 || x_rows > x.max_slots || n_items == 0 || !m.k.is_multiple_of(32) {
            return Err(invalid(format!("launch_items_exact: {:?} matrix k {} for activations k {} x {x_rows}", m.ty, m.k, x.k)));
        }
        if items.len() < n_items as usize * QItem::WORDS || y.dlen() < y_rows as usize * m.n as usize {
            return Err(invalid("launch_items_exact: work items or y too small".into()));
        }
        let units = m.k / 32;
        let smem = if q8 { 0 } else { QITEM_ROWS as u32 * 16 * (2 * units + units.div_ceil(4)) };
        if smem > 48 * 1024 {
            return Err(invalid(format!("launch_items_exact: k {} needs {smem} bytes of shared memory", m.k)));
        }
        let (rq, ra) = m.ty.row_bytes(m.k);
        let (mut pq, mut pa, mut px, mut pd, mut py) = (m.qs.device_ptr(), m.aux.device_ptr(), x.q.device_ptr(), x.d.device_ptr(), y.dptr());
        let (mut n, mut k) = (m.n, m.k);
        let (mut mq, mut ma) = ((rq * m.n as usize) as u64, (ra * m.n as usize) as u64);
        let (mut pi, mut xr, mut yr) = (items.device_ptr(), x_rows, y_rows);
        let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
        let mut nm = m.n_mats;
        let mut params = [
            param(&mut pq),
            param(&mut pa),
            param(&mut px),
            param(&mut pd),
            param(&mut py),
            param(&mut n),
            param(&mut k),
            param(&mut mq),
            param(&mut ma),
            param(&mut pi),
            param(&mut xr),
            param(&mut yr),
            param(&mut psc),
            param(&mut nm),
        ];
        let cfg = LaunchConfig { grid: (m.n.div_ceil(64), n_items, 1), block: (32 * Self::ROWS_PER_BLOCK, 1, 1), shared_mem_bytes: smem };
        // SAFETY: params match astral_gemv_nvfp4_items_s1 / astral_gemv_q8_0_items_x; planes validated at upload, the item
        // words cover n_items, y covers y_rows, and the kernel clamps e, xi and yi.
        unsafe { if q8 { &self.items_q8x } else { &self.items_s }.launch(cfg, stream, &mut params) }
    }

    /// Whether [`Gemv::launch_mma5`] runs the warpgroup-MMA kernel on this device (an sm_90a
    /// build on a Hopper GPU) rather than its mma.sync form.
    pub fn has_wgmma(stream: &Stream) -> bool {
        astral_kernels_src_has_sm90a() && stream.context().device().compute_capability().map(|(m, _)| m == 9).unwrap_or(false) && std::env::var_os("ASTRAL_NO_WGMMA").is_none()
    }

    /// Grouped products for batched decode on tensor cores: item `i` of `items` (device
    /// [`QItem`] words, `n_items` of them, unused ones with n = 0) multiplies matrix `e` with
    /// activation rows `xi` into output rows `yi`. Q8_0 matrices use int8 MMA, NVFP4 matrices
    /// their fragment layout ([`GemvMatrix::add_frag`]) with the f16 activations. Results are
    /// not bit-identical to [`Gemv::launch_q`]; see [`Gemv::launch_items_exact`] for that.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_items_mma(&self, stream: &Stream, m: &GemvMatrix, x: &QAct, x_rows: u32, y: &mut dyn DevMut<f32>, y_rows: u32, items: &DeviceBuffer<u32>, n_items: u32) -> Result<()> {
        if m.k != x.k || x_rows == 0 || x_rows > x.max_slots || n_items == 0 || m.mx_scale.is_some() {
            return Err(invalid(format!("launch_items_mma: {:?} matrix k {} for activations k {} x {x_rows}", m.ty, m.k, x.k)));
        }
        if items.len() < n_items as usize * QItem::WORDS || y.dlen() < y_rows as usize * m.n as usize {
            return Err(invalid("launch_items_mma: work items or y too small".into()));
        }
        let (mut n, mut k, mut pi, mut xr, mut yr, mut nm) = (m.n, m.k, items.device_ptr(), x_rows, y_rows, m.n_mats);
        let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
        match m.ty {
            GemvType::Q8_0 => {
                if !m.k.is_multiple_of(128) {
                    return Err(invalid(format!("launch_items_mma: Q8_0 k {}", m.k)));
                }
                let (rq, ra) = m.ty.row_bytes(m.k);
                let (mut pq, mut pa, mut px, mut pd, mut py) = (m.qs.device_ptr(), m.aux.device_ptr(), x.q.device_ptr(), x.d.device_ptr(), y.dptr());
                let (mut mq, mut ma) = ((rq * m.n as usize) as u64, (ra * m.n as usize) as u64);
                let mut params = [
                    param(&mut pq),
                    param(&mut pa),
                    param(&mut px),
                    param(&mut pd),
                    param(&mut py),
                    param(&mut n),
                    param(&mut k),
                    param(&mut mq),
                    param(&mut ma),
                    param(&mut pi),
                    param(&mut xr),
                    param(&mut yr),
                    param(&mut psc),
                    param(&mut nm),
                ];
                let cfg = LaunchConfig { grid: (n_items, m.n.div_ceil(32), 1), block: (256, 1, 1), shared_mem_bytes: 0 };
                // SAFETY: params match astral_gemv_q8_0_mma_items; planes validated at upload, the
                // item words cover n_items, y covers y_rows, and the kernel clamps e, rows, xi and yi.
                unsafe { self.items_mma[0].launch(cfg, stream, &mut params) }
            }
            GemvType::Nvfp4 => {
                let Some(h) = x.h.as_ref() else { return Err(invalid("launch_items_mma: activations have no f16 copy".into())) };
                let Some((fq, fs)) = m.frag.as_ref() else { return Err(invalid("launch_items_mma: matrix has no fragment layout".into())) };
                let plane = m.n_mats as usize * m.n as usize * m.k as usize;
                if !m.n.is_multiple_of(64) || !m.k.is_multiple_of(64) || fq.len() < plane / 2 || fs.len() < plane / 16 {
                    return Err(invalid(format!("launch_items_mma: NVFP4 n {} k {}", m.n, m.k)));
                }
                let smem = QITEM_ROWS as u32 * (m.k + 8) * 2;
                if smem > 48 * 1024 {
                    return Err(invalid(format!("launch_items_mma: NVFP4 k {}", m.k)));
                }
                let (mut pq, mut pa, mut ph, mut py) = (fq.device_ptr(), fs.device_ptr(), h.device_ptr(), y.dptr());
                let mut params = [
                    param(&mut pq),
                    param(&mut pa),
                    param(&mut ph),
                    param(&mut py),
                    param(&mut n),
                    param(&mut k),
                    param(&mut pi),
                    param(&mut xr),
                    param(&mut yr),
                    param(&mut psc),
                    param(&mut nm),
                ];
                // Warps per block (16 rows each): more rows share each staged activation tile.
                let warps = std::env::var("ASTRAL_FRAG_WARPS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(8).clamp(4, 16);
                let cfg = LaunchConfig { grid: (n_items, (m.n / 16).div_ceil(warps), 1), block: (32 * warps, 1, 1), shared_mem_bytes: smem };
                // SAFETY: params match astral_gemv_nvfp4_frag_items; the fragment planes hold
                // n_mats * n * k values (checked), the f16 activations x_rows rows of k, the item
                // words cover n_items, y covers y_rows; the kernel clamps e, xi and yi.
                unsafe { self.items_mma[1].launch(cfg, stream, &mut params) }
            }
            t => Err(invalid(format!("launch_items_mma: {t:?} weights"))),
        }
    }

    /// Dense prefill GEMMs on Hopper warpgroup MMA with the f16 fragment copies
    /// ([`GemvMatrix::add_frag16`]), up to 8 matrices with the same `k` in one launch:
    /// `y_i[yidx[j]] = W_i x_i[xidx[j]]` over items of up to 128 tokens ([`MmaWork::push_chunked`]
    /// with 128), f16 activations. Check [`Gemv::has_wgmma`] first.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_wgmma_f16(
        &self,
        stream: &Stream,
        prods: &mut [MultiItem],
        x_rows: u32,
        y_rows: u32,
        items: &DeviceBuffer<u32>,
        n_items: u32,
        xidx: &DeviceBuffer<u32>,
        yidx: &DeviceBuffer<u32>,
        idx_len: u32,
    ) -> Result<()> {
        #[repr(C)]
        struct F16Multi {
            frag: [DevicePtr; 8],
            xh: [DevicePtr; 8],
            y: [DevicePtr; 8],
            n: [u32; 8],
            start: [u32; 9],
            count: u32,
        }
        let k = prods.first().map_or(0, |p| p.m.k);
        if prods.is_empty() || prods.len() > 8 || !k.is_multiple_of(64) || n_items == 0 || x_rows == 0 || y_rows == 0 || idx_len == 0 {
            return Err(invalid(format!("launch_wgmma_f16: {} products, k {k}", prods.len())));
        }
        if items.len() < 3 * n_items as usize || xidx.len() < idx_len as usize || yidx.len() < idx_len as usize {
            return Err(invalid("launch_wgmma_f16: work lists too small".into()));
        }
        let mut mm = F16Multi { frag: [0; 8], xh: [0; 8], y: [0; 8], n: [0; 8], start: [0; 9], count: prods.len() as u32 };
        let mut blocks = 0u32;
        for (i, p) in prods.iter().enumerate() {
            let m = p.m;
            let Some(h) = p.x.h.as_ref() else { return Err(invalid("launch_wgmma_f16: activations have no f16 copy".into())) };
            let Some(fr) = m.frag16.as_ref() else { return Err(invalid("launch_wgmma_f16: matrix has no f16 fragment layout".into())) };
            if !m.n.is_multiple_of(128) || m.k != k || p.x.k != k || x_rows > p.x.max_slots || fr.len() < (m.n * m.k) as usize || p.y.dlen() < y_rows as usize * m.n as usize {
                return Err(invalid(format!("launch_wgmma_f16: product {i}: n {} k {} (activations k {})", m.n, m.k, p.x.k)));
            }
            mm.frag[i] = fr.device_ptr();
            mm.xh[i] = h.device_ptr();
            mm.y[i] = p.y.dptr();
            mm.n[i] = m.n;
            mm.start[i] = blocks;
            blocks += m.n / 128;
        }
        mm.start[prods.len()] = blocks;
        let (mut kk, mut pi, mut pxi, mut pyi, mut il, mut xr, mut yr) = (k, items.device_ptr(), xidx.device_ptr(), yidx.device_ptr(), idx_len, x_rows, y_rows);
        let kc = k / 64;
        let total = blocks * n_items;
        // Split K when the grid is too small to fill the GPU.
        let splits = if total >= 264 || !split_k() { 1 } else { 400u32.div_ceil(total).min(kc / 4).clamp(1, 8) };
        let mut kcps = kc.div_ceil(splits);
        let splits = kc.div_ceil(kcps);
        if splits > 1 {
            for p in prods.iter_mut() {
                let (mut pz, mut nz) = (p.y.dptr(), (y_rows as u64) * p.m.n as u64);
                let mut zp = [param(&mut pz), param(&mut nz)];
                let zc = LaunchConfig { grid: ((nz as u32).div_ceil(256).min(4096), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
                // SAFETY: y holds at least y_rows * n floats (checked above).
                unsafe { self.zero.launch(zc, stream, &mut zp) }?;
            }
        }
        let mut params = [param(&mut mm), param(&mut kk), param(&mut pi), param(&mut pxi), param(&mut pyi), param(&mut il), param(&mut xr), param(&mut yr), param(&mut kcps)];
        let cfg = LaunchConfig { grid: (blocks, n_items, splits), block: (256, 1, 1), shared_mem_bytes: WGMMA_F16_SMEM };
        // SAFETY: F16Multi mirrors gemv.cu; every fragment plane holds n * k halves, the f16
        // activations x_rows rows of k and each y y_rows * n values (checked), the work lists
        // cover n_items / idx_len; the kernel clamps every index.
        unsafe { self.wgmma_f16.launch(cfg, stream, &mut params) }
    }

    /// Grouped f16 GEMM on Hopper with TMA-loaded weights: `y[yidx[j]] = mat_scale[e] W_e
    /// x[xidx[j]]` for the tokens `j` of each item (`items`: 3 words per item, `(e, start,
    /// count <= 128)`). `w` is the tensor map of the row-major f16 weights of all `n_mats`
    /// matrices (`[n_mats * n][k]`, [`astral_gpu::TensorMap::f16_2d_sw128`] with 128 x 64 boxes),
    /// `xh` the f16 activations (`x_rows` rows of `k`). `n % 128 == 0`, `k % 64 == 0`. `tile`
    /// is the item size the work lists were built with: 128 or 256 tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_f16_tma(
        &self,
        stream: &Stream,
        tile: u32,
        w: &astral_gpu::TensorMap,
        n: u32,
        k: u32,
        n_mats: u32,
        mat_scale: Option<&DeviceBuffer<f32>>,
        xh: DevicePtr,
        x_rows: u32,
        y: &mut dyn DevMut<f32>,
        y_rows: u32,
        items: &DeviceBuffer<u32>,
        n_items: u32,
        xidx: &DeviceBuffer<u32>,
        yidx: &DeviceBuffer<u32>,
        idx_len: u32,
    ) -> Result<()> {
        if !n.is_multiple_of(128) || !k.is_multiple_of(64) || n_items == 0 || x_rows == 0 || y_rows == 0 || idx_len == 0 || n_mats == 0 {
            return Err(invalid(format!("launch_gemm_f16_tma: n {n} k {k} x{n_mats}")));
        }
        if items.len() < 3 * n_items as usize || xidx.len() < idx_len as usize || yidx.len() < idx_len as usize || y.dlen() < y_rows as usize * n as usize {
            return Err(invalid("launch_gemm_f16_tma: work lists or y too small".into()));
        }
        let mut tm = *w;
        // Split K when the grid is too small to fill the GPU (batched decode).
        let kc = k / 64;
        let blocks = (n / 128) * n_items;
        // (Prefill-sized calls run unsplit: the zeroing launches and atomics cost more than the
        // idle SMs; ASTRAL_TMA_SPLIT=1 splits K when the grid is small.)
        let splits = if blocks >= 200 || std::env::var_os("ASTRAL_TMA_SPLIT").is_none() { 1 } else { 400u32.div_ceil(blocks).min(kc / 4).clamp(1, 8) };
        let mut kcps = kc.div_ceil(splits);
        let splits = kc.div_ceil(kcps);
        if splits > 1 {
            let (mut pz, mut nz) = (y.dptr(), (y_rows as u64) * n as u64);
            let mut zp = [param(&mut pz), param(&mut nz)];
            let zc = LaunchConfig { grid: ((nz as u32).div_ceil(256).min(4096), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: y holds at least y_rows * n floats (checked above).
            unsafe { self.zero.launch(zc, stream, &mut zp) }?;
        }
        let (mut px, mut py, mut nn, mut kk) = (xh, y.dptr(), n, k);
        let (mut pi, mut pxi, mut pyi, mut il, mut xr, mut yr) = (items.device_ptr(), xidx.device_ptr(), yidx.device_ptr(), idx_len, x_rows, y_rows);
        let (mut psc, mut nm) = (mat_scale.map_or(0, |b| b.device_ptr()), n_mats);
        let mut params = [
            param(&mut tm),
            param(&mut px),
            param(&mut py),
            param(&mut nn),
            param(&mut kk),
            param(&mut pi),
            param(&mut pxi),
            param(&mut pyi),
            param(&mut il),
            param(&mut xr),
            param(&mut yr),
            param(&mut psc),
            param(&mut nm),
            param(&mut kcps),
        ];
        let cfg = LaunchConfig { grid: (n / 128, n_items, splits), block: (384, 1, 1), shared_mem_bytes: GEMM_TMA_SMEM };
        // SAFETY: params match astral_gemm_f16_tma; the tensor map covers n_mats * n rows of k
        // (built by the caller for these weights), the activations hold x_rows rows of k
        // halves, the work lists cover n_items / idx_len, y covers y_rows (checked); the kernel
        // clamps every index.
        let f = match tile {
            128 => &self.gemm_tma[0],
            256 => &self.gemm_tma[1],
            _ => return Err(invalid(format!("launch_gemm_f16_tma: tile {tile}"))),
        };
        unsafe { f.launch(cfg, stream, &mut params) }
    }

    /// Up to 8 Q8_0 products with the same `k` for 1..=64 activation rows (batched decode),
    /// tiled through shared memory: weight and activation chunks staged with cp.async, 64
    /// weight rows per block, K split across blocks (atomic adds into zeroed outputs) when there
    /// are too few blocks. Needs `k % 128 == 0`.
    pub fn launch_q_tiled(&self, stream: &Stream, items: &mut [MultiItem], ntok: u32) -> Result<()> {
        let k = items.first().map_or(0, |it| it.m.k);
        let ok = (1..=64).contains(&ntok)
            && (1..=8).contains(&items.len())
            && k.is_multiple_of(128)
            && items.iter().all(|it| it.m.ty == GemvType::Q8_0 && it.m.n_mats == 1 && it.m.mat_scale.is_none() && it.m.k == k && it.x.k == k && it.x.slots >= ntok);
        if !ok {
            return Err(invalid(format!("launch_q_tiled: {} products for {ntok} tokens do not qualify", items.len())));
        }
        let mut d = MultiQ8 { qs: [0; 8], aux: [0; 8], xq: [0; 8], xd: [0; 8], y: [0; 8], n: [0; 8], start: [0; 9], count: items.len() as u32 };
        let mut blocks = 0u32;
        for (i, it) in items.iter().enumerate() {
            if it.y.dlen() < (ntok * it.m.n) as usize {
                return Err(invalid(format!("launch_q_tiled: y {} has {} values, need {}", i, it.y.dlen(), ntok * it.m.n)));
            }
            d.qs[i] = it.m.qs.device_ptr();
            d.aux[i] = it.m.aux.device_ptr();
            d.xq[i] = it.x.q.device_ptr();
            d.xd[i] = it.x.d.device_ptr();
            d.y[i] = it.y.dptr();
            d.n[i] = it.m.n;
            d.start[i] = blocks;
            blocks += it.m.n.div_ceil(64);
        }
        d.start[items.len()] = blocks;
        let nc = k / 128;
        let splits = if blocks >= 264 { 1 } else { 396u32.div_ceil(blocks).min(nc / 2).max(1) };
        if splits > 1 {
            for it in items.iter_mut() {
                let (mut pz, mut nz) = (it.y.dptr(), (ntok * it.m.n) as u64);
                let mut zp = [param(&mut pz), param(&mut nz)];
                let zc = LaunchConfig { grid: ((nz as u32).div_ceil(256).min(4096), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
                // SAFETY: y holds at least ntok * n floats (checked above).
                unsafe { self.zero.launch(zc, stream, &mut zp) }?;
            }
        }
        let (mut kk, mut t) = (k, ntok);
        let mut params = [param(&mut d), param(&mut kk), param(&mut t)];
        let cfg = LaunchConfig { grid: (blocks, splits, 1), block: (256, 1, 1), shared_mem_bytes: Q_TILED_SMEM };
        // SAFETY: MultiQ8 mirrors gemv.cu; planes validated at upload, the activation planes hold
        // >= ntok rows of k and each y ntok * n values (checked); k % 128 == 0 (checked); the
        // kernel clamps rows and tokens.
        unsafe { self.q_tiled.launch(cfg, stream, &mut params) }
    }

    /// Up to 8 Q8_0 products with the same `k` for `ntok` activation rows (batched decode):
    /// `y_i[t] = W_i x_i[t]`, every weight row read once per 64 tokens (or per 16 / 32 for smaller
    /// batches) on int8 tensor cores. Not bit-identical to the one-token kernels.
    pub fn launch_q_dense(&self, stream: &Stream, items: &mut [MultiItem], ntok: u32) -> Result<()> {
        let k = items.first().map_or(0, |it| it.m.k);
        let ok = ntok > 0
            && (1..=8).contains(&items.len())
            && k.is_multiple_of(128)
            && items.iter().all(|it| it.m.ty == GemvType::Q8_0 && it.m.n_mats == 1 && it.m.mat_scale.is_none() && it.m.k == k && it.x.k == k && it.x.slots >= ntok);
        if !ok {
            return Err(invalid(format!("launch_q_dense: {} products for {ntok} tokens do not qualify", items.len())));
        }
        let mut d = MultiQ8 { qs: [0; 8], aux: [0; 8], xq: [0; 8], xd: [0; 8], y: [0; 8], n: [0; 8], start: [0; 9], count: items.len() as u32 };
        let mut blocks = 0u32;
        for (i, it) in items.iter().enumerate() {
            if it.y.dlen() < (ntok * it.m.n) as usize {
                return Err(invalid(format!("launch_q_dense: y {} has {} values, need {}", i, it.y.dlen(), ntok * it.m.n)));
            }
            d.qs[i] = it.m.qs.device_ptr();
            d.aux[i] = it.m.aux.device_ptr();
            d.xq[i] = it.x.q.device_ptr();
            d.xd[i] = it.x.d.device_ptr();
            d.y[i] = it.y.dptr();
            d.n[i] = it.m.n;
            d.start[i] = blocks;
            blocks += it.m.n.div_ceil(32);
        }
        d.start[items.len()] = blocks;
        // 32-token tiles for larger batches measured best (64 rows: 24.6 vs 25.5 ms per step).
        let (f, per) = match ntok.min(32) {
            0..=16 => (&self.dense_mma[0], 16),
            17..=32 => (&self.dense_mma[1], 32),
            _ => (&self.dense_mma[2], 64),
        };
        let (mut kk, mut t) = (k, ntok);
        let mut params = [param(&mut d), param(&mut kk), param(&mut t)];
        let cfg = LaunchConfig { grid: (blocks, ntok.div_ceil(per), 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: MultiQ8 mirrors gemv.cu; planes validated at upload, the activation planes hold
        // >= ntok rows of k and each y ntok * n values (checked); k % 128 == 0 (checked).
        unsafe { f.launch(cfg, stream, &mut params) }
    }

    /// Whether the small-batch products run on tensor cores ([`Gemv::launch_q_multi_mma`]).
    pub fn small_mma() -> bool {
        std::env::var_os("ASTRAL_SMALL_DP4A").is_none()
    }

    /// Up to 8 single-matrix Q8_0 products with the same `k` for `ntok` (2..=8) tokens in one
    /// launch: `y_i[t] = W_i x_i[t]` (activation rows 0..ntok of each `x_i`, outputs token-major
    /// `[ntok][n_i]`). Each weight row is read once for all tokens. Check [`Gemv::multi_t_ok`].
    pub fn launch_q_multi_t(&self, stream: &Stream, items: &mut [MultiItem], ntok: u32) -> Result<()> {
        if !Self::multi_t_ok(items, ntok) {
            return Err(invalid(format!("launch_q_multi_t: {} products for {ntok} tokens do not qualify", items.len())));
        }
        let k = items[0].m.k;
        if Self::small_mma() && k.is_multiple_of(128) {
            return self.launch_q_multi_mma(stream, items, ntok);
        }
        let mut d = MultiQ8 { qs: [0; 8], aux: [0; 8], xq: [0; 8], xd: [0; 8], y: [0; 8], n: [0; 8], start: [0; 9], count: items.len() as u32 };
        let mut blocks = 0u32;
        for (i, it) in items.iter().enumerate() {
            if it.y.dlen() < (ntok * it.m.n) as usize {
                return Err(invalid(format!("launch_q_multi_t: y {} has {} values, need {}", i, it.y.dlen(), ntok * it.m.n)));
            }
            d.qs[i] = it.m.qs.device_ptr();
            d.aux[i] = it.m.aux.device_ptr();
            d.xq[i] = it.x.q.device_ptr();
            d.xd[i] = it.x.d.device_ptr();
            d.y[i] = it.y.dptr();
            d.n[i] = it.m.n;
            d.start[i] = blocks;
            blocks += it.m.n.div_ceil(Self::ROWS_PER_BLOCK);
        }
        d.start[items.len()] = blocks;
        let mut kk = k;
        let mut params = [param(&mut d), param(&mut kk)];
        let cfg = LaunchConfig { grid: (blocks, 1, 1), block: (32 * Self::ROWS_PER_BLOCK, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: MultiQ8 mirrors gemv.cu; planes validated at upload (n rows of k), activations
        // hold >= ntok rows of k (checked) and each y holds ntok * n values (checked above).
        unsafe { self.multi_t[ntok as usize - 2].launch(cfg, stream, &mut params) }
    }

    pub fn launch_q_multi(&self, stream: &Stream, items: &mut [MultiItem]) -> Result<()> {
        let k = items.first().map_or(0, |it| it.m.k);
        let fused = (1..=8).contains(&items.len()) && items.iter().all(|it| it.m.ty == GemvType::Q8_0 && it.m.n_mats == 1 && it.m.mat_scale.is_none() && it.m.k == k && it.x.k == k && it.x.slots == 1);
        if !fused {
            for it in items.iter_mut() {
                self.launch_q(stream, it.m, it.x, &mut *it.y, None)?;
            }
            return Ok(());
        }
        let mut d = MultiQ8 { qs: [0; 8], aux: [0; 8], xq: [0; 8], xd: [0; 8], y: [0; 8], n: [0; 8], start: [0; 9], count: items.len() as u32 };
        let mut blocks = 0u32;
        for (i, it) in items.iter().enumerate() {
            if it.y.dlen() < it.m.n as usize {
                return Err(invalid(format!("launch_q_multi: y {} has {} values, need {}", i, it.y.dlen(), it.m.n)));
            }
            d.qs[i] = it.m.qs.device_ptr();
            d.aux[i] = it.m.aux.device_ptr();
            d.xq[i] = it.x.q.device_ptr();
            d.xd[i] = it.x.d.device_ptr();
            d.y[i] = it.y.dptr();
            d.n[i] = it.m.n;
            d.start[i] = blocks;
            blocks += it.m.n.div_ceil(Self::ROWS_PER_BLOCK * Self::q_rows_per_warp(GemvType::Q8_0));
        }
        d.start[items.len()] = blocks;
        let mut kk = k;
        let mut params = [param(&mut d), param(&mut kk)];
        let cfg = LaunchConfig { grid: (blocks, 1, 1), block: (32 * Self::ROWS_PER_BLOCK, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: MultiQ8 mirrors gemv.cu; every plane was validated at upload (n rows of k),
        // activations hold k values and each y holds n values (checked above).
        unsafe { self.multi.launch(cfg, stream, &mut params) }
    }

    /// `y[s] = W_e x_s` with int8 activations (Q8_0 and NVFP4 weights only). `e = ids[s]` when
    /// ids are given (one slot per id), else `s` over all `n_mats`. A `QAct` holding one vector
    /// is shared by every slot.
    pub fn launch_q(&self, stream: &Stream, m: &GemvMatrix, x: &QAct, y: &mut (impl DevMut<f32> + ?Sized), ids: Option<&dyn Dev<i32>>) -> Result<()> {
        let fi = match m.ty {
            GemvType::Q8_0 => 0,
            GemvType::Nvfp4 => 1,
            t => return Err(invalid(format!("launch_q: {t:?} has no int8-activation kernel"))),
        };
        let slots = ids.map_or(m.n_mats, |i| i.dlen() as u32);
        if m.k != x.k || slots == 0 || (x.slots != 1 && x.slots < slots) {
            return Err(invalid(format!("launch_q: matrix k {} / {} slots vs activations k {} x {}", m.k, slots, x.k, x.slots)));
        }
        if y.dlen() < slots as usize * m.n as usize {
            return Err(invalid(format!("launch_q: y has {} values, need {}", y.dlen(), slots * m.n)));
        }
        let (rq, ra) = m.ty.row_bytes(m.k);
        // Native MXFP4 scales (exact, fewer bytes) when the matrix carries them.
        let mx = m.mx_scale.as_ref().filter(|_| std::env::var_os("ASTRAL_MX_VIA_NVFP4").is_none());
        let (aux_ptr, ra) = match mx {
            Some(e) => (e.device_ptr(), m.k as usize / 32),
            None => (m.aux.device_ptr(), ra),
        };
        let (mut pq, mut pa, mut px, mut pd, mut py) = (m.qs.device_ptr(), aux_ptr, x.q.device_ptr(), x.d.device_ptr(), y.dptr());
        let (mut n, mut k) = (m.n, m.k);
        let (mut mq, mut ma) = ((rq * m.n as usize) as u64, (ra * m.n as usize) as u64);
        let mut shared = (x.slots == 1) as u32;
        let mut pid = ids.map_or(0, |i| i.dptr());
        let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
        let mut nm = m.n_mats;
        let mut params = [
            param(&mut pq),
            param(&mut pa),
            param(&mut px),
            param(&mut pd),
            param(&mut py),
            param(&mut n),
            param(&mut k),
            param(&mut mq),
            param(&mut ma),
            param(&mut shared),
            param(&mut pid),
            param(&mut psc),
            param(&mut nm),
        ];
        // NVFP4: the shared-memory kernel (activations staged once per block, weights streamed),
        // one row per warp (ASTRAL_GEMV_R=2|4 to override).
        let (f, rows, smem) = if mx.is_some() {
            // Short, wide matrices: split each row's K over 2 / 4 warps.
            let wr = match std::env::var("ASTRAL_MX_WR").ok().and_then(|v| v.parse::<u32>().ok()) {
                Some(w @ (1 | 2 | 4)) => w,
                _ if m.n * slots <= 8192 && m.k >= 4096 => 2,
                _ => 1,
            };
            match wr {
                2 => (&self.fq_mx_w[0], Self::ROWS_PER_BLOCK / 2, m.k + m.k / 8),
                4 => (&self.fq_mx_w[1], Self::ROWS_PER_BLOCK / 4, m.k + m.k / 8),
                _ => (&self.fq_mx, Self::ROWS_PER_BLOCK, m.k + m.k / 8),
            }
        } else if m.ty == GemvType::Nvfp4 && std::env::var_os("ASTRAL_GEMV_V1").is_none() {
            let r = match std::env::var("ASTRAL_GEMV_R").ok().and_then(|v| v.parse::<u32>().ok()) {
                Some(r @ (1 | 2 | 4)) => r,
                _ => 1, // fastest on H100 for every shape measured (Qwen3.8 and Inkling decode)
            };
            let fi = match r {
                1 => 0,
                2 => 1,
                _ => 2,
            };
            // Long rows: more loads in flight per lane (K/32 units over 32 lanes).
            let long = match std::env::var("ASTRAL_GEMV_LONG_U").ok().as_deref() {
                Some("12") => Some(0),
                Some("9") => Some(1),
                Some(_) => None,
                None => None, // measured: no gain over U = 6 (U = 12 is slower)
            };
            let f = match long {
                Some(li) if r == 1 => &self.fq_long[li],
                _ => &self.fq_s[fi],
            };
            (f, Self::ROWS_PER_BLOCK * r, m.k + m.k / 8)
        } else {
            // Long NVFP4 rows: a whole warp per row (more blocks, fewer dependent rounds).
            let long = m.ty == GemvType::Nvfp4 && (m.k >= 8192 || m.n * slots <= 8192);
            let rows = Self::ROWS_PER_BLOCK * if long { 1 } else { Self::q_rows_per_warp(m.ty) };
            (if long { &self.fq_r1 } else { &self.fq[fi] }, rows, 0)
        };
        if smem > 48 * 1024 {
            return Err(invalid(format!("launch_q: k {} needs {smem} bytes of shared memory", m.k)));
        }
        let cfg = LaunchConfig { grid: (m.n.div_ceil(rows), slots, 1), block: (32 * Self::ROWS_PER_BLOCK, 1, 1), shared_mem_bytes: smem };
        // SAFETY: params match gemv.cu's GEMV_Q_ENTRY / astral_gemv_nvfp4_dp4a_r1; planes validated at upload, activations
        // and y checked above; device ids are clamped to n_mats - 1 in the kernel.
        unsafe { f.launch(cfg, stream, &mut params) }
    }

    fn check(m: &GemvMatrix, x_len: usize, x_stride: u32, y_len: usize, slots: u32) -> Result<()> {
        if slots == 0 || !x_stride.is_multiple_of(4) {
            return Err(invalid("gemv: need slots > 0 and x_stride % 4 == 0".into()));
        }
        let need_x = (slots as usize - 1) * x_stride as usize + m.k as usize;
        if x_len < need_x || y_len < slots as usize * m.n as usize {
            return Err(invalid(format!("gemv: x {x_len} (need {need_x}) / y {y_len} (need {}) too small", slots * m.n)));
        }
        Ok(())
    }

    fn raw(&self, stream: &Stream, m: &GemvMatrix, x: DevicePtr, x_stride: u32, y: DevicePtr, slots: u32, ids: DevicePtr) -> Result<()> {
        let (rq, ra) = m.ty.row_bytes(m.k);
        let (mut pq, mut pa, mut px, mut py) = (m.qs.device_ptr(), m.aux.device_ptr(), x, y);
        let (mut n, mut k) = (m.n, m.k);
        let (mut mq, mut ma) = ((rq * m.n as usize) as u64, (ra * m.n as usize) as u64);
        let mut xs = x_stride;
        let mut pid = ids;
        let mut psc = m.mat_scale.as_ref().map_or(0, |b| b.device_ptr());
        let mut nm = m.n_mats;
        let mut params = [
            param(&mut pq),
            param(&mut pa),
            param(&mut px),
            param(&mut py),
            param(&mut n),
            param(&mut k),
            param(&mut mq),
            param(&mut ma),
            param(&mut xs),
            param(&mut pid),
            param(&mut psc),
            param(&mut nm),
        ];
        // Short, wide F32 / BF16 matrices: one block per row instead of one warp. Chosen by the
        // matrix shape alone, so a row's result does not depend on how many slots share the launch.
        let rowblock = matches!(m.ty, GemvType::F32 | GemvType::Bf16) && m.n < 2048 && m.k >= 1024;
        let (f, cfg) = if rowblock {
            (&self.rb[m.ty.idx()], LaunchConfig { grid: (m.n, slots, 1), block: (256, 1, 1), shared_mem_bytes: 0 })
        } else {
            (&self.f[m.ty.idx()], LaunchConfig { grid: (m.n.div_ceil(Self::ROWS_PER_BLOCK), slots, 1), block: (32 * Self::ROWS_PER_BLOCK, 1, 1), shared_mem_bytes: 0 })
        };
        // SAFETY: params match gemv.cu (both entry kinds share one signature); plane sizes
        // validated at upload and x/y sizes by the callers via `check`; device ids are clamped
        // to n_mats - 1 inside the kernel.
        unsafe { f.launch(cfg, stream, &mut params) }
    }

    /// `y[s] = W_s x_s` for every matrix `s < m.n_mats` (`x_stride = 0` shares one `x`).
    pub fn launch(&self, stream: &Stream, m: &GemvMatrix, x: &impl Dev<f32>, x_stride: u32, y: &mut impl DevMut<f32>) -> Result<()> {
        Self::check(m, x.dlen(), x_stride, y.dlen(), m.n_mats)?;
        self.raw(stream, m, x.dptr(), x_stride, y.dptr(), m.n_mats, 0)
    }

    /// `y[s] = W_{ids[s]} x_s` for `s < ids.len()` (MoE: the selected experts). Ids outside
    /// `0..n_mats` are clamped on the device.
    pub fn launch_ids(&self, stream: &Stream, m: &GemvMatrix, x: &impl Dev<f32>, x_stride: u32, y: &mut impl DevMut<f32>, ids: &impl Dev<i32>) -> Result<()> {
        let slots = ids.dlen() as u32;
        Self::check(m, x.dlen(), x_stride, y.dlen(), slots)?;
        self.raw(stream, m, x.dptr(), x_stride, y.dptr(), slots, ids.dptr())
    }

    /// [`Gemv::launch_ids`] with host ids, range-checked then uploaded (tests).
    pub fn launch_ids_checked(&self, stream: &Stream, m: &GemvMatrix, x: &impl Dev<f32>, x_stride: u32, y: &mut impl DevMut<f32>, ids: &[i32]) -> Result<()> {
        if ids.iter().any(|&e| e < 0 || e as u32 >= m.n_mats) {
            return Err(invalid(format!("gemv: expert id out of range 0..{}", m.n_mats)));
        }
        let ctx = m.qs.context().clone();
        let dids = DeviceBuffer::from_slice(&ctx, ids)?;
        let r = self.launch_ids(stream, m, x, x_stride, y, &dids);
        stream.synchronize()?; // `dids` is freed on return; the kernel must have read it
        r
    }
}
