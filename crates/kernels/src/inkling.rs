// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Typed launchers for `cu/inkling.cu`: the Inkling 2 Mini decode operations. Every launcher
//! checks buffer sizes before launching; device-side indices (token, position) are clamped by
//! the kernels, so callers never need `unsafe`.

use crate::ops::{GemvMatrix, GemvType, QAct};
use crate::KernelRegistry;
use astral_core::{AstralError, Result};
use astral_gpu::{param, Dev, DevMut, DevicePtr, Function, LaunchConfig, Stream};

/// Maximum attention-residual states mixed in one call (`MAX_DEPTH` in inkling.cu).
pub const MAX_DEPTH: usize = 8;
/// Maximum context the decode attention kernel handles (`MAX_CTX` in inkling.cu).
pub const MAX_CTX: u32 = 8192;
/// Positions per split of the split-K attention (`ATT_CHUNK` in inkling.cu).
pub const ATT_CHUNK: u32 = 64;

/// Most work items [`InklingOps::moe_group_q`] can produce for `p` picks over `n_expert`
/// experts (each used expert, plus one per extra 8 picks of an expert).
pub fn moe_items_bound(p: u32, n_expert: u32) -> u32 {
    p.min(n_expert + p.div_ceil(crate::ops::QITEM_ROWS as u32))
}

/// Workspace floats [`InklingOps::attn_decode_split`] needs.
pub fn attn_workspace(ntok: u32, heads: u32, head_dim: u32, max_splits: u32) -> usize {
    ntok as usize * (heads * max_splits * (head_dim + 2)) as usize
}

fn invalid(m: String) -> AstralError {
    AstralError::Invalid(m)
}

fn need(what: &str, have: usize, want: usize) -> Result<()> {
    if have < want {
        return Err(invalid(format!("{what}: buffer has {have} elements, need {want}")));
    }
    Ok(())
}

fn elementwise(n: u32) -> LaunchConfig {
    LaunchConfig { grid: (n.div_ceil(256).clamp(1, 1024), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }
}

/// `struct DepthIdx { unsigned v[MAX_DEPTH]; }` passed by value.
#[repr(C)]
struct DepthIdx {
    v: [u32; MAX_DEPTH],
}

/// Dynamic shared memory of the prefill attention kernels (`FA_SMEM` in inkling.cu).
/// Tokens per thread of the parallel short-conv kernels (`CONV_T` in inkling.cu).
const CONV_T: u32 = 8;

const FA_SMEM: u32 = 2 * 64 * 136 * 2 + 4 * 16 * 80 * 4;

pub struct InklingOps {
    embed: Function,
    clip: Function,
    tau: Function,
    mix_all: Function,
    mix: Function,
    concat_rms2: Function,
    control: Function,
    conv: Function,
    kv_append: Function,
    attn: Function,
    router: Function,
    situ: Function,
    combine: Function,
    add: Function,
    mul_sigmoid: Function,
    softcap: Function,
    argmax: Function,
    rms_q8: Function,
    attn_prep: Function,
    /// Packed-prefill segment kernels: conv_par, conv_state, conv_res_par, attn_prep.
    segs: [Function; 4],
    sum_splits: Function,
    ctrl_down3_t: Function,
    ctrl_combine3_t: Function,
    /// Batched controllers for large batches (not batch-exact): 8 ranks or 256 outputs x 8 tokens per block.
    ctrl_b: [Function; 2],
    /// Tensor-core controller down projection: norms, then the K-sliced bf16 GEMM.
    ctrl_tc: [Function; 2],
    ctrl_down: Function,
    ctrl_combine: Function,
    gate_q8: Function,
    situ_q8: Function,
    conv_residual: Function,
    conv_seq: Function,
    conv_par: Function,
    conv_res_par: Function,
    conv_state: Function,
    moe_combine_t: Function,
    rms2: Function,
    /// [`InklingOps::sgemm_mode`] on bf16 tensor cores (64 x 64 tiles).
    tgemm_tn: Function,
    sgemm_tn: Function,
    ctrl_apply: Function,
    ctrl_gf_apply: Function,
    moe_combine_t16: Function,
    attn_prefill: Function,
    attn_prefill_segs: Function,
    moe_group: Function,
    moe_group_q: Function,
    attn_split: Function,
    attn_merge: Function,
    attn_split1: Function,
    /// [`InklingOps::attn_decode_split`] one-chunk splits, 3 query heads per KV head in a block.
    attn_split_gqa3: Function,
    attn_merge_q8: Function,
    mix_rms_q8: Function,
    router_b: Function,
    router_fused: Function,
    ctrl_down3: Function,
    ctrl_combine3: Function,
    rms_q8_3: Function,
}

/// Everything [`InklingOps::attn_prep`] reads and writes for one layer and token.
/// Device pointers for [`InklingOps::attn_inputs_segs`] (row 0 / sequence 0 of each buffer).
pub struct SegAttnInputs<'a> {
    pub q: DevicePtr,
    pub r: DevicePtr,
    pub k_raw: DevicePtr,
    pub v_raw: DevicePtr,
    pub k_out: DevicePtr,
    pub v_out: DevicePtr,
    /// Dynamic conv taps (`(rows, row stride)`) of full-attention layers.
    pub kdyn: Option<(DevicePtr, u32)>,
    pub vdyn: Option<(DevicePtr, u32)>,
    pub dyn_heads: u32,
    /// Conv states of all sequences (K at 0, V at 3 * width within each).
    pub conv: DevicePtr,
    pub k_conv: DevicePtr,
    pub v_conv: DevicePtr,
    pub q_scale: DevicePtr,
    pub r_scale: DevicePtr,
    pub kc: DevicePtr,
    pub vc: DevicePtr,
    /// Start positions, indexed by the table's position word.
    pub pos: DevicePtr,
    pub seg: &'a dyn Dev<i32>,
    pub n_seg: u32,
    pub max_rows: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub n_rel: u32,
    pub thr_qk: f32,
    pub thr_rel: f32,
    pub eps: f32,
    pub global: bool,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub capacity: u32,
}

pub struct AttnPrep<'a> {
    pub q: &'a mut dyn DevMut<f32>,
    pub r: &'a mut dyn DevMut<f32>,
    pub k: &'a mut dyn DevMut<f32>,
    pub v: &'a mut dyn DevMut<f32>,
    pub q_scale: &'a dyn Dev<f32>,
    pub r_scale: &'a dyn Dev<f32>,
    /// This slot's conv state from channel 0 (K at 0, V at `width`), at least `6 * width`.
    pub conv_kv: &'a mut dyn DevMut<f32>,
    pub k_conv: &'a dyn Dev<f32>,
    pub v_conv: &'a dyn Dev<f32>,
    /// Dynamic taps (`[dyn_heads * 4]` each) on full-attention layers.
    pub kdyn: Option<&'a dyn Dev<f32>>,
    pub vdyn: Option<&'a dyn Dev<f32>>,
    pub dyn_heads: u32,
    /// Tokens (prefill) and whether the K/V convs run here (decode) or ran in `conv_seq`.
    pub ntok: u32,
    pub conv: bool,
    pub kc: &'a mut dyn DevMut<u16>,
    pub vc: &'a mut dyn DevMut<u16>,
    pub pos: &'a dyn Dev<i32>,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub n_rel: u32,
    pub thr_qk: f32,
    pub thr_rel: f32,
    pub eps: f32,
    /// Full-attention layer: apply the log-length scale to Q and R.
    pub global: bool,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub capacity: u32,
    /// Batched decode rows (one sequence per row), or `None` for consecutive tokens.
    pub rows: Option<SeqRows<'a>>,
}

/// Rows of a batched decode pass: row `t` is one token of sequence `seq[t]` at position
/// `pos[t]` (the `pos` buffer then holds one position per row). Each of the `n_seqs` sequences
/// owns `capacity` positions of every KV cache buffer and `conv_stride` floats of every conv
/// state buffer, back to back.
#[derive(Clone, Copy)]
pub struct SeqRows<'a> {
    pub seq: &'a dyn Dev<i32>,
    pub n_seqs: u32,
    pub conv_stride: u32,
}

impl SeqRows<'_> {
    fn parts(r: Option<&Self>) -> (DevicePtr, u32, u32) {
        r.map_or((0, 1, 0), |r| (r.seq.dptr(), r.n_seqs.max(1), r.conv_stride))
    }
}

/// Attention geometry of one layer.
#[derive(Debug, Clone, Copy)]
pub struct AttnShape {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub n_rel: u32,
    /// Relative-bias table rows (distance >= extent uses a zero row).
    pub extent: u32,
    /// Sliding window (0 = full attention).
    pub window: u32,
    /// KV cache capacity in positions.
    pub capacity: u32,
}

impl InklingOps {
    pub fn new(reg: &KernelRegistry) -> Result<Self> {
        let f = |n: &str| reg.function("inkling", n);
        Ok(Self {
            embed: f("astral_ink_embed_q8_0")?,
            clip: f("astral_ink_clip_heads")?,
            tau: f("astral_ink_scale_tau")?,
            mix: f("astral_ink_mix")?,
            mix_all: f("astral_ink_mix_all")?,
            concat_rms2: f("astral_ink_concat_rms2")?,
            control: f("astral_ink_control_combine")?,
            conv: f("astral_ink_conv_step")?,
            kv_append: f("astral_ink_kv_append")?,
            attn: f("astral_ink_attn_decode")?,
            router: f("astral_ink_router_topk")?,
            situ: f("astral_ink_situ")?,
            combine: f("astral_ink_moe_combine")?,
            add: f("astral_ink_add")?,
            mul_sigmoid: f("astral_ink_mul_sigmoid")?,
            softcap: f("astral_ink_softcap")?,
            argmax: f("astral_ink_argmax")?,
            rms_q8: f("astral_ink_rms_q8")?,
            attn_prep: f("astral_ink_attn_prep")?,
            segs: [f("astral_ink_conv_par_segs")?, f("astral_ink_conv_state_segs")?, f("astral_ink_conv_res_par_segs")?, f("astral_ink_attn_prep_segs")?],
            sum_splits: f("astral_ink_sum_splits")?,
            ctrl_down3_t: f("astral_ink_ctrl_down3_t")?,
            ctrl_combine3_t: f("astral_ink_ctrl_combine3_t")?,
            ctrl_b: [f("astral_ink_ctrl_down3_b")?, f("astral_ink_ctrl_combine3_b")?],
            ctrl_tc: [f("astral_ink_ctrl_norms")?, f("astral_ink_ctrl_down_tc")?],
            ctrl_down: f("astral_ink_ctrl_down")?,
            ctrl_combine: f("astral_ink_ctrl_combine")?,
            gate_q8: f("astral_ink_gate_q8")?,
            situ_q8: f("astral_ink_situ_q8")?,
            conv_residual: f("astral_ink_conv_residual")?,
            conv_seq: f("astral_ink_conv_seq")?,
            conv_par: f("astral_ink_conv_par")?,
            conv_res_par: f("astral_ink_conv_res_par")?,
            conv_state: f("astral_ink_conv_state")?,
            moe_combine_t: f("astral_ink_moe_combine_t")?,
            rms2: f("astral_ink_rms2")?,
            sgemm_tn: f("astral_ink_sgemm_tn")?,
            tgemm_tn: f("astral_ink_tgemm_tn")?,
            ctrl_apply: f("astral_ink_ctrl_apply")?,
            ctrl_gf_apply: f("astral_ink_ctrl_gf_apply")?,
            moe_combine_t16: f("astral_ink_moe_combine_t16")?,
            attn_prefill: {
                let k = f("astral_ink_attn_prefill")?;
                k.set_max_dynamic_shared(FA_SMEM)?;
                k
            },
            attn_prefill_segs: {
                let k = f("astral_ink_attn_prefill_segs")?;
                k.set_max_dynamic_shared(FA_SMEM)?;
                k
            },
            moe_group: f("astral_ink_moe_group")?,
            moe_group_q: f("astral_ink_moe_group_q")?,
            attn_split: f("astral_ink_attn_split")?,
            attn_merge: f("astral_ink_attn_merge")?,
            attn_split1: f("astral_ink_attn_split1")?,
            attn_split_gqa3: f("astral_ink_attn_split_gqa3")?,
            attn_merge_q8: f("astral_ink_attn_merge_q8")?,
            mix_rms_q8: f("astral_ink_mix_rms_q8")?,
            router_fused: f("astral_ink_router_fused")?,
            router_b: f("astral_ink_router_b")?,
            ctrl_down3: f("astral_ink_ctrl_down3")?,
            ctrl_combine3: f("astral_ink_ctrl_combine3")?,
            rms_q8_3: f("astral_ink_rms_q8_3")?,
        })
    }

    fn go(f: &Function, stream: &Stream, cfg: LaunchConfig, params: &mut [*mut std::ffi::c_void]) -> Result<()> {
        // SAFETY: every caller builds `params` to match the kernel's C signature in inkling.cu
        // and checks buffer sizes first; device-side indices are clamped by the kernels.
        unsafe { f.launch(cfg, stream, params) }
    }
    /// `out[t] = dequant(row token[t])` of a Q8_0 embedding table, for `ntok` tokens.
    pub fn embed(&self, s: &Stream, table: &GemvMatrix, token: &impl Dev<i32>, out: &mut impl DevMut<f32>, ntok: u32) -> Result<()> {
        if table.ty != crate::ops::GemvType::Q8_0 || table.n_mats != 1 || ntok == 0 {
            return Err(invalid("embed: table must be a single Q8_0 matrix".into()));
        }
        need("embed token", token.dlen(), ntok as usize)?;
        need("embed out", out.dlen(), (table.k * ntok) as usize)?;
        let (mut q, mut d, mut t, mut o, mut dim, mut rows) = (table.qs.device_ptr(), table.aux.device_ptr(), token.dptr(), out.dptr(), table.k, table.n);
        let mut p = [param(&mut q), param(&mut d), param(&mut t), param(&mut o), param(&mut dim), param(&mut rows)];
        Self::go(&self.embed, s, LaunchConfig { grid: (ntok, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Per-head L2 clip with optional per-channel scale (`[heads * hd]`).
    pub fn clip_heads(&self, s: &Stream, x: &mut impl DevMut<f32>, scale: Option<&impl Dev<f32>>, heads: u32, hd: u32, thr: f32) -> Result<()> {
        let n = (heads * hd) as usize;
        need("clip x", x.dlen(), n)?;
        if let Some(sc) = scale {
            need("clip scale", sc.dlen(), n)?;
        }
        let (mut px, mut ps, mut h, mut d, mut t) = (x.dptr(), scale.map_or(0, |b| b.dptr()), heads, hd, thr);
        let mut p = [param(&mut px), param(&mut ps), param(&mut h), param(&mut d), param(&mut t)];
        Self::go(&self.clip, s, LaunchConfig { grid: (heads.div_ceil(8), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `x *= 1 + alpha * log(max(1, (pos + 1) / floor))`.
    pub fn scale_tau(&self, s: &Stream, x: &mut impl DevMut<f32>, n: u32, pos: &impl Dev<i32>, floor: f32, alpha: f32) -> Result<()> {
        need("tau x", x.dlen(), n as usize)?;
        need("tau pos", pos.dlen(), 1)?;
        let (mut px, mut nn, mut pp, mut fl, mut al) = (x.dptr(), n, pos.dptr(), floor, alpha);
        let mut p = [param(&mut px), param(&mut nn), param(&mut pp), param(&mut fl), param(&mut al)];
        Self::go(&self.tau, s, elementwise(n), &mut p)
    }
    /// Depth mix for `ntok` tokens: states = bank rows `idx` (row r of token t at
    /// `r * row_stride + t * dim`) plus `cur[t]`, `nq` queries (rows of `query`); out `[t][nq][dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn mix(
        &self,
        s: &Stream,
        bank: &impl Dev<f32>,
        idx: &[u32],
        row_stride: usize,
        cur: &impl Dev<f32>,
        query: &impl Dev<f32>,
        nq: u32,
        out: &mut impl DevMut<f32>,
        dim: u32,
        ntok: u32,
    ) -> Result<()> {
        if idx.len() >= MAX_DEPTH {
            return Err(invalid(format!("mix: {} states exceed the kernel limit {}", idx.len() + 1, MAX_DEPTH)));
        }
        let (d, nt) = (dim as usize, ntok as usize);
        if !dim.is_multiple_of(4) || ntok == 0 || nq == 0 {
            return Err(invalid(format!("mix: dim {dim} must be a multiple of 4, tokens and queries > 0")));
        }
        if idx.iter().any(|&i| i as usize * row_stride + nt * d > bank.dlen()) {
            return Err(invalid("mix: state index outside the bank".into()));
        }
        need("mix cur", cur.dlen(), nt * d)?;
        need("mix query", query.dlen(), nq as usize * d)?;
        need("mix out", out.dlen(), nt * nq as usize * d)?;
        let mut di = DepthIdx { v: [0; MAX_DEPTH] };
        di.v[..idx.len()].copy_from_slice(idx);
        let (mut pb, mut depth, mut rs, mut pc, mut pq, mut po, mut dm) = (bank.dptr(), idx.len() as u32, row_stride as u32, cur.dptr(), query.dptr(), out.dptr(), dim);
        let block = (dim / 4).next_multiple_of(32).clamp(32, 1024);
        // Several queries: one block per token reads each state once for all of them.
        if (2..=4).contains(&nq) && dim / 4 <= 1024 && std::env::var_os("ASTRAL_MIX_PER_QUERY").is_none() {
            let mut nqq = nq;
            let mut p2 = [param(&mut pb), param(&mut di), param(&mut depth), param(&mut rs), param(&mut pc), param(&mut pq), param(&mut nqq), param(&mut po), param(&mut dm)];
            return Self::go(&self.mix_all, s, LaunchConfig { grid: (ntok, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 }, &mut p2);
        }
        let mut p = [param(&mut pb), param(&mut di), param(&mut depth), param(&mut rs), param(&mut pc), param(&mut pq), param(&mut po), param(&mut dm)];
        Self::go(&self.mix, s, LaunchConfig { grid: (nq, ntok, 1), block: (block, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// [`InklingOps::mix`] fused with the RMSNorm + quantisation of each query's mixed row:
    /// `outs[j] = (weight, normalised row y [ntok][dim] (optional), activations)` for the `nq =
    /// outs.len()` queries. `out` (the mixed rows, `[t][nq][dim]`) is optional. Needs
    /// `dim % 128 == 0` and `dim <= 4096`.
    #[allow(clippy::too_many_arguments)]
    pub fn mix_rms_q8(
        &self,
        s: &Stream,
        bank: &impl Dev<f32>,
        idx: &[u32],
        row_stride: usize,
        cur: &impl Dev<f32>,
        query: &impl Dev<f32>,
        out: Option<&mut dyn DevMut<f32>>,
        dim: u32,
        ntok: u32,
        outs: &mut [(&dyn Dev<f32>, Option<&mut dyn DevMut<f32>>, &mut QAct)],
        eps: f32,
    ) -> Result<()> {
        let nq = outs.len();
        if idx.len() >= MAX_DEPTH || nq == 0 || nq > 3 || !dim.is_multiple_of(128) || dim > 4096 || ntok == 0 {
            return Err(invalid(format!("mix_rms_q8: {} states, {nq} queries, dim {dim}", idx.len() + 1)));
        }
        let (d, nt) = (dim as usize, ntok as usize);
        if idx.iter().any(|&i| i as usize * row_stride + nt * d > bank.dlen()) {
            return Err(invalid("mix_rms_q8: state index outside the bank".into()));
        }
        need("mix_rms_q8 cur", cur.dlen(), nt * d)?;
        need("mix_rms_q8 query", query.dlen(), nq * d)?;
        let po = match out {
            Some(o) => {
                need("mix_rms_q8 out", o.dlen(), nt * nq * d)?;
                o.dptr()
            }
            None => 0,
        };
        #[repr(C)]
        struct MixRmsOut {
            w: [DevicePtr; 3],
            y: [DevicePtr; 3],
            xq: [DevicePtr; 3],
            xd: [DevicePtr; 3],
            xh: [DevicePtr; 3],
        }
        let mut mo = MixRmsOut { w: [0; 3], y: [0; 3], xq: [0; 3], xd: [0; 3], xh: [0; 3] };
        for (j, (w, y, qa)) in outs.iter_mut().enumerate() {
            need("mix_rms_q8 w", w.dlen(), d)?;
            if qa.k != dim {
                return Err(invalid(format!("mix_rms_q8: activations k {} != dim {dim}", qa.k)));
            }
            mo.w[j] = w.dptr();
            if let Some(y) = y {
                need("mix_rms_q8 y", y.dlen(), nt * d)?;
                mo.y[j] = y.dptr();
            }
            let (q, dd, h) = qa.planes_for(ntok)?;
            mo.xq[j] = q;
            mo.xd[j] = dd;
            mo.xh[j] = h;
        }
        let mut di = DepthIdx { v: [0; MAX_DEPTH] };
        di.v[..idx.len()].copy_from_slice(idx);
        let (mut pb, mut depth, mut rs, mut pc, mut pq, mut po, mut dm, mut e) = (bank.dptr(), idx.len() as u32, row_stride as u32, cur.dptr(), query.dptr(), po, dim, eps);
        let mut p = [param(&mut pb), param(&mut di), param(&mut depth), param(&mut rs), param(&mut pc), param(&mut pq), param(&mut po), param(&mut dm), param(&mut mo), param(&mut e)];
        Self::go(&self.mix_rms_q8, s, LaunchConfig { grid: (nq as u32, ntok, 1), block: (dim / 4, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `out[..2*dim] = concat(rms(a), rms(b))`.
    pub fn concat_rms2(&self, s: &Stream, a: &impl Dev<f32>, b: &impl Dev<f32>, out: &mut impl DevMut<f32>, dim: u32) -> Result<()> {
        need("concat a", a.dlen(), dim as usize)?;
        need("concat b", b.dlen(), dim as usize)?;
        need("concat out", out.dlen(), 2 * dim as usize)?;
        let (mut pa, mut pb, mut po, mut d) = (a.dptr(), b.dptr(), out.dptr(), dim);
        let mut p = [param(&mut pa), param(&mut pb), param(&mut po), param(&mut d)];
        Self::go(&self.concat_rms2, s, LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `out = cur - tanh(forget) * cur + tanh(gain) * msg`.
    #[allow(clippy::too_many_arguments)]
    pub fn control_combine(&self, s: &Stream, cur: &impl Dev<f32>, msg: &impl Dev<f32>, gain: &impl Dev<f32>, forget: &impl Dev<f32>, out: &mut impl DevMut<f32>, dim: u32) -> Result<()> {
        for (w, b) in [("cur", cur.dlen()), ("msg", msg.dlen()), ("gain", gain.dlen()), ("forget", forget.dlen()), ("out", out.dlen())] {
            need(w, b, dim as usize)?;
        }
        let (mut a, mut b, mut c, mut d, mut o, mut n) = (cur.dptr(), msg.dptr(), gain.dptr(), forget.dptr(), out.dptr(), dim);
        let mut p = [param(&mut a), param(&mut b), param(&mut c), param(&mut d), param(&mut o), param(&mut n)];
        Self::go(&self.control, s, elementwise(dim), &mut p)
    }

    /// In-place 4-tap causal conv with residual over `channels`; advances `state` (`3 * channels`).
    /// `dyn_taps`: optional `[heads * 4]` input-dependent taps.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_step(&self, s: &Stream, x: &mut impl DevMut<f32>, state: &mut impl DevMut<f32>, weight: &impl Dev<f32>, dyn_taps: Option<&impl Dev<f32>>, channels: u32, heads: u32) -> Result<()> {
        let c = channels as usize;
        need("conv x", x.dlen(), c)?;
        need("conv state", state.dlen(), 3 * c)?;
        need("conv weight", weight.dlen(), 4 * c)?;
        if let Some(t) = dyn_taps {
            if heads == 0 || !channels.is_multiple_of(heads) {
                return Err(invalid(format!("conv: {channels} channels not divisible by {heads} heads")));
            }
            need("conv taps", t.dlen(), 4 * heads as usize)?;
        }
        let (mut px, mut ps, mut pw, mut pd, mut ch, mut hh) = (x.dptr(), state.dptr(), weight.dptr(), dyn_taps.map_or(0, |b| b.dptr()), channels, heads);
        let mut p = [param(&mut px), param(&mut ps), param(&mut pw), param(&mut pd), param(&mut ch), param(&mut hh)];
        Self::go(&self.conv, s, elementwise(channels), &mut p)
    }

    /// Store this token's K and V (`[width]` f32) as f16 at cache position `pos`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append(&self, s: &Stream, k: &impl Dev<f32>, v: &impl Dev<f32>, kc: &mut impl DevMut<u16>, vc: &mut impl DevMut<u16>, pos: &impl Dev<i32>, width: u32, capacity: u32) -> Result<()> {
        need("kv k", k.dlen(), width as usize)?;
        need("kv v", v.dlen(), width as usize)?;
        need("kv k cache", kc.dlen(), (width * capacity) as usize)?;
        need("kv v cache", vc.dlen(), (width * capacity) as usize)?;
        need("kv pos", pos.dlen(), 1)?;
        let (mut a, mut b, mut c, mut d, mut e, mut w, mut cap) = (k.dptr(), v.dptr(), kc.dptr(), vc.dptr(), pos.dptr(), width, capacity.max(1));
        let mut p = [param(&mut a), param(&mut b), param(&mut c), param(&mut d), param(&mut e), param(&mut w), param(&mut cap)];
        Self::go(&self.kv_append, s, LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// One-token attention with relative-position bias and sinks (see inkling.cu).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        s: &Stream,
        q: &impl Dev<f32>,
        r: &impl Dev<f32>,
        sink: &impl Dev<f32>,
        kc: &impl Dev<u16>,
        vc: &impl Dev<u16>,
        rel_bias: &impl Dev<f32>,
        out: &mut impl DevMut<f32>,
        pos: &impl Dev<i32>,
        a: AttnShape,
    ) -> Result<()> {
        if a.kv_heads == 0 || !a.heads.is_multiple_of(a.kv_heads) || a.n_rel > 64 || a.capacity == 0 || a.capacity > MAX_CTX {
            return Err(invalid(format!("attn: unsupported shape {a:?} (n_rel <= 64, capacity <= {MAX_CTX})")));
        }
        let width = (a.kv_heads * a.head_dim) as usize;
        need("attn q", q.dlen(), (a.heads * a.head_dim) as usize)?;
        need("attn r", r.dlen(), (a.heads * a.n_rel) as usize)?;
        need("attn sink", sink.dlen(), a.heads as usize)?;
        need("attn k cache", kc.dlen(), width * a.capacity as usize)?;
        need("attn v cache", vc.dlen(), width * a.capacity as usize)?;
        need("attn rel bias", rel_bias.dlen(), (a.extent * a.n_rel) as usize)?;
        need("attn out", out.dlen(), (a.heads * a.head_dim) as usize)?;
        need("attn pos", pos.dlen(), 1)?;
        let (mut pq, mut pr, mut ps, mut pk, mut pv, mut pb, mut po, mut pp): (DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr) =
            (q.dptr(), r.dptr(), sink.dptr(), kc.dptr(), vc.dptr(), rel_bias.dptr(), out.dptr(), pos.dptr());
        let (mut h, mut kh, mut hd, mut nr, mut ex, mut win) = (a.heads, a.kv_heads, a.head_dim, a.n_rel, a.extent, a.window);
        let mut scale = 1.0f32 / (a.head_dim as f32).sqrt();
        let mut cap = a.capacity;
        let mut p = [
            param(&mut pq),
            param(&mut pr),
            param(&mut ps),
            param(&mut pk),
            param(&mut pv),
            param(&mut pb),
            param(&mut po),
            param(&mut pp),
            param(&mut h),
            param(&mut kh),
            param(&mut hd),
            param(&mut nr),
            param(&mut ex),
            param(&mut win),
            param(&mut scale),
            param(&mut cap),
        ];
        Self::go(&self.attn, s, LaunchConfig { grid: (a.heads, 1, 1), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }
    /// MoE router for `ntok` tokens: sigmoid, top-k by `prob + bias`, normalised weights times
    /// `w_scale`. `logits [t][n_expert]`, `ids` / `weights [t][k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn router_topk(
        &self,
        s: &Stream,
        logits: &impl Dev<f32>,
        bias: Option<&impl Dev<f32>>,
        ids: &mut impl DevMut<i32>,
        weights: &mut impl DevMut<f32>,
        n_expert: u32,
        k: u32,
        w_scale: f32,
        ntok: u32,
    ) -> Result<()> {
        if n_expert > 256 || k == 0 || k > 32 || k > n_expert || ntok == 0 {
            return Err(invalid(format!("router: need k <= 32, k <= n_expert <= 256 (got {k}, {n_expert})")));
        }
        need("router logits", logits.dlen(), (n_expert * ntok) as usize)?;
        if let Some(b) = bias {
            need("router bias", b.dlen(), n_expert as usize)?;
        }
        need("router ids", ids.dlen(), (k * ntok) as usize)?;
        need("router weights", weights.dlen(), (k * ntok) as usize)?;
        let (mut pl, mut pb, mut pi, mut pw, mut ne, mut kk, mut ws) = (logits.dptr(), bias.map_or(0, |b| b.dptr()), ids.dptr(), weights.dptr(), n_expert, k, w_scale);
        let mut p = [param(&mut pl), param(&mut pb), param(&mut pi), param(&mut pw), param(&mut ne), param(&mut kk), param(&mut ws)];
        Self::go(&self.router, s, LaunchConfig { grid: (ntok, 1, 1), block: (n_expert.next_multiple_of(32), 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// The router GEMV (an F32 `[n_expert, k]` matrix over `ntok` rows of `x`) fused with
    /// [`InklingOps::router_topk`]. `counter` holds `ntok` zeros and is left zeroed.
    #[allow(clippy::too_many_arguments)]
    /// Router logits `[ntok][n_expert]` of many tokens (F32 weights), 8 experts x 8 tokens per
    /// block; pair with [`InklingOps::router_topk`].
    pub fn router_logits_b(&self, s: &Stream, router: &GemvMatrix, x: &dyn Dev<f32>, logits: &mut dyn DevMut<f32>, ntok: u32) -> Result<()> {
        let (ne, k) = (router.n, router.k);
        if router.ty != GemvType::F32 || router.n_mats != 1 || !k.is_multiple_of(4) || ntok == 0 {
            return Err(invalid(format!("router_logits_b: {:?} {ne} x {k}", router.ty)));
        }
        need("router x", x.dlen(), (ntok * k) as usize)?;
        need("router logits", logits.dlen(), (ne * ntok) as usize)?;
        let (mut pw, mut px, mut kk, mut pl, mut n, mut nt) = (router.qs.device_ptr(), x.dptr(), k, logits.dptr(), ne, ntok);
        let mut p = [param(&mut pw), param(&mut px), param(&mut kk), param(&mut pl), param(&mut n), param(&mut nt)];
        Self::go(&self.router_b, s, LaunchConfig { grid: (ne.div_ceil(8), ntok.div_ceil(8), 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    pub fn router_fused(
        &self,
        s: &Stream,
        router: &GemvMatrix,
        x: &dyn Dev<f32>,
        logits: &mut dyn DevMut<f32>,
        bias: Option<&dyn Dev<f32>>,
        ids: &mut dyn DevMut<i32>,
        weights: &mut dyn DevMut<f32>,
        k: u32,
        w_scale: f32,
        ntok: u32,
        counter: &mut dyn DevMut<u32>,
    ) -> Result<()> {
        let ne = router.n;
        if router.ty != GemvType::F32 || router.n_mats != 1 || ne > 256 || k == 0 || k > 32 || k > ne || ntok == 0 || !router.k.is_multiple_of(4) {
            return Err(invalid(format!("router_fused: {:?} n {ne} k {} top-{k}", router.ty, router.k)));
        }
        need("router x", x.dlen(), (router.k * ntok) as usize)?;
        need("router logits", logits.dlen(), (ne * ntok) as usize)?;
        if let Some(b) = bias {
            need("router bias", b.dlen(), ne as usize)?;
        }
        need("router ids", ids.dlen(), (k * ntok) as usize)?;
        need("router weights", weights.dlen(), (k * ntok) as usize)?;
        need("router counter", counter.dlen(), ntok as usize)?;
        let (mut pw, mut px, mut kk0, mut pl, mut pb, mut pi, mut pws) = (router.qs.device_ptr(), x.dptr(), router.k, logits.dptr(), bias.map_or(0, |b| b.dptr()), ids.dptr(), weights.dptr());
        let (mut n, mut kk, mut ws, mut pc) = (ne, k, w_scale, counter.dptr());
        let mut p = [param(&mut pw), param(&mut px), param(&mut kk0), param(&mut pl), param(&mut pb), param(&mut pi), param(&mut pws), param(&mut n), param(&mut kk), param(&mut ws), param(&mut pc)];
        Self::go(&self.router_fused, s, LaunchConfig { grid: (ne, ntok, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// SITU over `slots` fused gate_up rows (`2 * n_ff` each) into `act` (`n_ff` each).
    pub fn situ(&self, s: &Stream, gu: &impl Dev<f32>, act: &mut impl DevMut<f32>, n_ff: u32, slots: u32, beta: f32, lb: f32) -> Result<()> {
        need("situ gate_up", gu.dlen(), (2 * n_ff * slots) as usize)?;
        need("situ act", act.dlen(), (n_ff * slots) as usize)?;
        let (mut g, mut a, mut n, mut sl, mut b, mut l) = (gu.dptr(), act.dptr(), n_ff, slots, beta, lb);
        let mut p = [param(&mut g), param(&mut a), param(&mut n), param(&mut sl), param(&mut b), param(&mut l)];
        Self::go(&self.situ, s, elementwise(n_ff * slots), &mut p)
    }

    /// `out[c] = sum_s w[s] * y[s, c]`.
    pub fn moe_combine(&self, s: &Stream, y: &impl Dev<f32>, w: &impl Dev<f32>, out: &mut impl DevMut<f32>, dim: u32, slots: u32) -> Result<()> {
        need("combine y", y.dlen(), (dim * slots) as usize)?;
        need("combine w", w.dlen(), slots as usize)?;
        need("combine out", out.dlen(), dim as usize)?;
        let (mut py, mut pw, mut po, mut d, mut sl) = (y.dptr(), w.dptr(), out.dptr(), dim, slots);
        let mut p = [param(&mut py), param(&mut pw), param(&mut po), param(&mut d), param(&mut sl)];
        Self::go(&self.combine, s, elementwise(dim), &mut p)
    }

    pub fn add(&self, s: &Stream, a: &impl Dev<f32>, b: &impl Dev<f32>, out: &mut impl DevMut<f32>, n: u32) -> Result<()> {
        need("add a", a.dlen(), n as usize)?;
        need("add b", b.dlen(), n as usize)?;
        need("add out", out.dlen(), n as usize)?;
        let (mut pa, mut pb, mut po, mut nn) = (a.dptr(), b.dptr(), out.dptr(), n);
        let mut p = [param(&mut pa), param(&mut pb), param(&mut po), param(&mut nn)];
        Self::go(&self.add, s, elementwise(n), &mut p)
    }

    /// `x *= sigmoid(g)`.
    pub fn mul_sigmoid(&self, s: &Stream, x: &mut impl DevMut<f32>, g: &impl Dev<f32>, n: u32) -> Result<()> {
        need("mul_sigmoid x", x.dlen(), n as usize)?;
        need("mul_sigmoid g", g.dlen(), n as usize)?;
        let (mut px, mut pg, mut nn) = (x.dptr(), g.dptr(), n);
        let mut p = [param(&mut px), param(&mut pg), param(&mut nn)];
        Self::go(&self.mul_sigmoid, s, elementwise(n), &mut p)
    }

    pub fn softcap(&self, s: &Stream, x: &mut impl DevMut<f32>, n: u32, cap: f32) -> Result<()> {
        need("softcap", x.dlen(), n as usize)?;
        let (mut px, mut nn, mut c) = (x.dptr(), n, cap);
        let mut p = [param(&mut px), param(&mut nn), param(&mut c)];
        Self::go(&self.softcap, s, elementwise(n), &mut p)
    }

    /// Greedy pick over the first `n` logits.
    pub fn argmax(&self, s: &Stream, x: &impl Dev<f32>, n: u32, out: &mut impl DevMut<i32>) -> Result<()> {
        need("argmax x", x.dlen(), n as usize)?;
        need("argmax out", out.dlen(), 1)?;
        let (mut px, mut nn, mut po) = (x.dptr(), n, out.dptr());
        let mut p = [param(&mut px), param(&mut nn), param(&mut po)];
        Self::go(&self.argmax, s, LaunchConfig { grid: (1, 1, 1), block: (1024, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }
    /// For `ntok` rows (row t at `x[t * x_stride]`): `y = rms_norm(x) * w` (`y`, `w` optional,
    /// `y` token-major) and `y` quantised into `qa` (`ntok` vectors).
    #[allow(clippy::too_many_arguments)]
    pub fn rms_q8(&self, s: &Stream, x: &dyn Dev<f32>, x_stride: usize, w: Option<&dyn Dev<f32>>, y: Option<&mut dyn DevMut<f32>>, qa: &mut QAct, ntok: u32, eps: f32) -> Result<()> {
        let dim = qa.k;
        if ntok == 0 {
            return Err(invalid("rms_q8: no tokens".into()));
        }
        need("rms_q8 x", x.dlen(), (ntok as usize - 1) * x_stride + dim as usize)?;
        if let Some(w) = w {
            need("rms_q8 w", w.dlen(), dim as usize)?;
        }
        let py = match y {
            Some(y) => {
                need("rms_q8 y", y.dlen(), (dim * ntok) as usize)?;
                y.dptr()
            }
            None => 0,
        };
        let (mut pq, mut pd, mut ph) = qa.planes_for(ntok)?;
        let (mut px, mut xs, mut pw, mut py, mut d, mut e) = (x.dptr(), x_stride as u32, w.map_or(0, |w| w.dptr()), py, dim, eps);
        let mut p = [param(&mut px), param(&mut xs), param(&mut pw), param(&mut py), param(&mut pq), param(&mut pd), param(&mut ph), param(&mut d), param(&mut e)];
        let block = (dim / 4).next_multiple_of(32).clamp(32, 1024);
        Self::go(&self.rms_q8, s, LaunchConfig { grid: (ntok, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }
    /// Q/R clips (+ log-length scale), K/V short convs (unless `a.conv` is false), K clip, V
    /// norm and the KV append, for `a.ntok` tokens at positions `pos[0]..`.
    /// The attention inputs of every segment of a packed prefill pass, one launch per step:
    /// K and V convs (into `k_out` / `v_out`) and conv states, then clips, value norm and the KV
    /// append. Rows, caches and conv states are those of row 0 / sequence 0; the segment table
    /// (8 words per segment, see `astral_ink_conv_par_segs`) places each segment.
    pub fn attn_inputs_segs(&self, s: &Stream, a: SegAttnInputs) -> Result<()> {
        if a.n_seg == 0 || a.max_rows == 0 || a.head_dim != 128 || a.kv_heads == 0 || a.n_rel > 128 || !a.n_rel.is_multiple_of(4) {
            return Err(invalid(format!("attn_inputs_segs: {} segments of up to {} rows", a.n_seg, a.max_rows)));
        }
        need("segs table", a.seg.dlen(), 8 * a.n_seg as usize)?;
        let width = a.kv_heads * a.head_dim;
        let ct = |rows: u32| LaunchConfig { grid: (width.div_ceil(128), rows, a.n_seg), block: (128, 1, 1), shared_mem_bytes: 0 };
        for (src, dst, w, taps, off) in [(a.k_raw, a.k_out, a.k_conv, a.kdyn, 0u64), (a.v_raw, a.v_out, a.v_conv, a.vdyn, 3 * width as u64)] {
            let (mut px, mut xs, mut po, mut os, mut ps, mut pw, mut ch, mut pt) = (src, width, dst, width, a.conv + 4 * off, w, width, a.seg.dptr());
            let (mut pd, mut ds, mut dh) = match taps {
                Some((t, stride)) => (t, stride, a.dyn_heads),
                None => (0, 0, 0),
            };
            let mut p = [param(&mut px), param(&mut xs), param(&mut po), param(&mut os), param(&mut ps), param(&mut pw), param(&mut pd), param(&mut ds), param(&mut dh), param(&mut ch), param(&mut pt)];
            Self::go(&self.segs[0], s, ct(a.max_rows.div_ceil(CONV_T)), &mut p)?;
            let (mut px, mut xs, mut ps, mut ch, mut pt) = (src, width, a.conv + 4 * off, width, a.seg.dptr());
            let mut p = [param(&mut px), param(&mut xs), param(&mut ps), param(&mut ch), param(&mut pt)];
            Self::go(&self.segs[1], s, LaunchConfig { grid: (width.div_ceil(128), 1, a.n_seg), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        }
        let (mut pq, mut pr, mut pk, mut pv, mut qs, mut rs, mut cs, mut kw, mut vw) = (a.q, a.r, a.k_out, a.v_out, a.q_scale, a.r_scale, a.conv, a.k_conv, a.v_conv);
        let (mut kc, mut vc, mut pp, mut hh, mut khh, mut hdd, mut nrr) = (a.kc, a.vc, a.pos, a.heads, a.kv_heads, a.head_dim, a.n_rel);
        let (mut tq, mut tr, mut ep, mut gl, mut fl, mut al, mut cap, mut pt) = (a.thr_qk, a.thr_rel, a.eps, a.global as u32, a.log_floor, a.log_alpha, a.capacity, a.seg.dptr());
        let mut p = [
            param(&mut pq),
            param(&mut pr),
            param(&mut pk),
            param(&mut pv),
            param(&mut qs),
            param(&mut rs),
            param(&mut cs),
            param(&mut kw),
            param(&mut vw),
            param(&mut kc),
            param(&mut vc),
            param(&mut pp),
            param(&mut hh),
            param(&mut khh),
            param(&mut hdd),
            param(&mut nrr),
            param(&mut tq),
            param(&mut tr),
            param(&mut ep),
            param(&mut gl),
            param(&mut fl),
            param(&mut al),
            param(&mut cap),
            param(&mut pt),
        ];
        let jobs = 2 * a.heads + 2 * a.kv_heads;
        Self::go(&self.segs[3], s, LaunchConfig { grid: (jobs.div_ceil(4), a.max_rows, a.n_seg), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// [`InklingOps::conv_res_par`] and [`InklingOps::conv_state`] for every segment of a packed
    /// pass: `src`, `cur` and `bank_row` hold rows from row 0, `state` is sequence 0's conv state
    /// at this conv's offset.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_residual_segs(&self, s: &Stream, src: DevicePtr, state: DevicePtr, weight: DevicePtr, cur: DevicePtr, bank_row: Option<DevicePtr>, channels: u32, seg: &dyn Dev<i32>, n_seg: u32, max_rows: u32) -> Result<()> {
        need("segs table", seg.dlen(), 8 * n_seg as usize)?;
        let (mut px, mut ps, mut pw, mut pc, mut pb, mut ch, mut pt) = (src, state, weight, cur, bank_row.unwrap_or(0), channels, seg.dptr());
        let mut p = [param(&mut px), param(&mut ps), param(&mut pw), param(&mut pc), param(&mut pb), param(&mut ch), param(&mut pt)];
        Self::go(&self.segs[2], s, LaunchConfig { grid: (channels.div_ceil(128), max_rows.div_ceil(CONV_T), n_seg), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        let (mut px, mut xs, mut ps, mut ch, mut pt) = (src, channels, state, channels, seg.dptr());
        let mut p = [param(&mut px), param(&mut xs), param(&mut ps), param(&mut ch), param(&mut pt)];
        Self::go(&self.segs[1], s, LaunchConfig { grid: (channels.div_ceil(128), 1, n_seg), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    pub fn attn_prep(&self, s: &Stream, a: AttnPrep) -> Result<()> {
        let (h, kh, hd, nr, nt) = (a.heads as usize, a.kv_heads as usize, a.head_dim as usize, a.n_rel as usize, a.ntok as usize);
        let width = kh * hd;
        if hd > 256 || nr > 256 || kh == 0 || a.capacity == 0 || nt == 0 || (a.kdyn.is_some() != a.vdyn.is_some()) || (a.kdyn.is_some() && (!a.conv || (nt != 1 && a.rows.is_none()))) {
            return Err(invalid(format!("attn_prep: unsupported shape hd {hd} n_rel {nr} kv_heads {kh} tokens {nt}")));
        }
        need("prep q", a.q.dlen(), nt * h * hd)?;
        need("prep r", a.r.dlen(), nt * h * nr)?;
        need("prep k", a.k.dlen(), nt * width)?;
        need("prep v", a.v.dlen(), nt * width)?;
        need("prep q scale", a.q_scale.dlen(), h * hd)?;
        need("prep r scale", a.r_scale.dlen(), h * nr)?;
        need("prep conv state", a.conv_kv.dlen(), 6 * width)?;
        need("prep k conv", a.k_conv.dlen(), 4 * width)?;
        need("prep v conv", a.v_conv.dlen(), 4 * width)?;
        for t in [a.kdyn, a.vdyn].into_iter().flatten() {
            if a.dyn_heads == 0 || !width.is_multiple_of(a.dyn_heads as usize) {
                return Err(invalid(format!("attn_prep: {width} channels not divisible by {} tap heads", a.dyn_heads)));
            }
            need("prep taps", t.dlen(), 4 * a.dyn_heads as usize * if a.rows.is_some() { nt } else { 1 })?;
        }
        let seqs = a.rows.map_or(1, |r| r.n_seqs.max(1) as usize);
        need("prep k cache", a.kc.dlen(), seqs * width * a.capacity as usize)?;
        need("prep v cache", a.vc.dlen(), seqs * width * a.capacity as usize)?;
        need("prep pos", a.pos.dlen(), if a.rows.is_some() { nt } else { 1 })?;
        if let Some(r) = &a.rows {
            need("prep seq", r.seq.dlen(), nt)?;
            need("prep conv state", a.conv_kv.dlen(), (seqs - 1) * r.conv_stride as usize + 6 * width)?;
            if !a.conv {
                return Err(invalid("attn_prep: batched rows need the per-token conv".into()));
            }
        }
        let (mut pq, mut pr, mut pk, mut pv) = (a.q.dptr(), a.r.dptr(), a.k.dptr(), a.v.dptr());
        let (mut qs, mut rs, mut cs, mut kw, mut vw) = (a.q_scale.dptr(), a.r_scale.dptr(), a.conv_kv.dptr(), a.k_conv.dptr(), a.v_conv.dptr());
        let (mut kd, mut vd, mut dh) = (a.kdyn.map_or(0, |b| b.dptr()), a.vdyn.map_or(0, |b| b.dptr()), if a.kdyn.is_some() { a.dyn_heads } else { 0 });
        let (mut kc, mut vc, mut pp) = (a.kc.dptr(), a.vc.dptr(), a.pos.dptr());
        let (mut hh, mut khh, mut hdd, mut nrr) = (a.heads, a.kv_heads, a.head_dim, a.n_rel);
        let (mut tq, mut tr, mut ep, mut gl, mut fl, mut al, mut cap) = (a.thr_qk, a.thr_rel, a.eps, a.global as u32, a.log_floor, a.log_alpha, a.capacity);
        let mut dc = a.conv as u32;
        let (mut sq, mut ns, mut cst) = SeqRows::parts(a.rows.as_ref());
        let mut p = [
            param(&mut pq),
            param(&mut pr),
            param(&mut pk),
            param(&mut pv),
            param(&mut qs),
            param(&mut rs),
            param(&mut cs),
            param(&mut kw),
            param(&mut vw),
            param(&mut kd),
            param(&mut vd),
            param(&mut dh),
            param(&mut kc),
            param(&mut vc),
            param(&mut pp),
            param(&mut hh),
            param(&mut khh),
            param(&mut hdd),
            param(&mut nrr),
            param(&mut tq),
            param(&mut tr),
            param(&mut ep),
            param(&mut gl),
            param(&mut fl),
            param(&mut al),
            param(&mut cap),
            param(&mut dc),
            param(&mut sq),
            param(&mut ns),
            param(&mut cst),
        ];
        let jobs = 2 * (a.heads + a.kv_heads);
        Self::go(&self.attn_prep, s, LaunchConfig { grid: (jobs, a.ntok, 1), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// In-place short conv over `ntok` consecutive tokens (row t at `x[t * x_stride]`), with
    /// optional per-token dynamic taps `(taps, row stride, tap heads)`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_seq(
        &self,
        s: &Stream,
        x: &mut dyn DevMut<f32>,
        x_stride: usize,
        ntok: u32,
        state: &mut dyn DevMut<f32>,
        weight: &dyn Dev<f32>,
        taps: Option<(&dyn Dev<f32>, usize, u32)>,
        channels: u32,
    ) -> Result<()> {
        let c = channels as usize;
        if ntok == 0 || x_stride < c {
            return Err(invalid("conv_seq: need tokens > 0 and x_stride >= channels".into()));
        }
        need("conv_seq x", x.dlen(), (ntok as usize - 1) * x_stride + c)?;
        need("conv_seq state", state.dlen(), 3 * c)?;
        need("conv_seq weight", weight.dlen(), 4 * c)?;
        let (mut pd, mut ds, mut dh) = (0, 0u32, 0u32);
        if let Some((t, stride, heads)) = taps {
            if heads == 0 || !channels.is_multiple_of(heads) || stride < 4 * heads as usize {
                return Err(invalid(format!("conv_seq: {channels} channels / {heads} tap heads / stride {stride}")));
            }
            need("conv_seq taps", t.dlen(), (ntok as usize - 1) * stride + 4 * heads as usize)?;
            (pd, ds, dh) = (t.dptr(), stride as u32, heads);
        }
        let (mut px, mut xs, mut nt, mut ps, mut pw, mut ch) = (x.dptr(), x_stride as u32, ntok, state.dptr(), weight.dptr(), channels);
        let mut p = [param(&mut px), param(&mut xs), param(&mut nt), param(&mut ps), param(&mut pw), param(&mut pd), param(&mut ds), param(&mut dh), param(&mut ch)];
        Self::go(&self.conv_seq, s, LaunchConfig { grid: (channels.div_ceil(128), 1, 1), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    fn f32_mat(m: &GemvMatrix, n: u32, k: u32, what: &str) -> Result<()> {
        if m.ty != GemvType::F32 || m.n_mats != 1 || m.n != n || m.k != k {
            return Err(invalid(format!("{what}: need one F32 [{n} x {k}] matrix, got {:?} [{} x {}] x{}", m.ty, m.n, m.k, m.n_mats)));
        }
        Ok(())
    }
    /// Loop controller part 1 for `ntok` tokens: `cd[t] = W_down concat(rms(x[t]), rms(msg[t]))`.
    #[allow(clippy::too_many_arguments)]
    pub fn ctrl_down(&self, s: &Stream, wd: &GemvMatrix, x: &dyn Dev<f32>, x_stride: usize, msg: &dyn Dev<f32>, msg_stride: usize, cd: &mut dyn DevMut<f32>, dim: u32, ntok: u32) -> Result<()> {
        Self::f32_mat(wd, wd.n, 2 * dim, "ctrl_down")?;
        let (d, nt) = (dim as usize, ntok as usize);
        if nt == 0 {
            return Err(invalid("ctrl_down: no tokens".into()));
        }
        need("ctrl_down x", x.dlen(), (nt - 1) * x_stride + d)?;
        need("ctrl_down msg", msg.dlen(), (nt - 1) * msg_stride + d)?;
        need("ctrl_down out", cd.dlen(), nt * wd.n as usize)?;
        let (mut pw, mut px, mut xs, mut pm, mut ms, mut po, mut dd) = (wd.qs.device_ptr(), x.dptr(), x_stride as u32, msg.dptr(), msg_stride as u32, cd.dptr(), dim);
        let mut p = [param(&mut pw), param(&mut px), param(&mut xs), param(&mut pm), param(&mut ms), param(&mut po), param(&mut dd)];
        Self::go(&self.ctrl_down, s, LaunchConfig { grid: (wd.n, ntok, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }
    /// Both loop-controller parts for up to three independent jobs `(down, gain_forget, x, msg,
    /// cd, out)` (a looped layer's Q, K, V inputs) in two launches. All share `x_stride`,
    /// `msg_stride`, `dim` and the rank. `exact`: each token computed as by the one-token
    /// kernels (else, for more than 8 tokens, the faster tiled kernels).
    #[allow(clippy::too_many_arguments)]
    pub fn ctrl_multi(
        &self,
        s: &Stream,
        jobs: &mut [(&GemvMatrix, &GemvMatrix, &dyn Dev<f32>, &dyn Dev<f32>, &mut dyn DevMut<f32>, &mut dyn DevMut<f32>)],
        x_stride: usize,
        msg_stride: usize,
        dim: u32,
        ntok: u32,
        exact: bool,
        scratch: &mut dyn DevMut<f32>,
    ) -> Result<()> {
        #[repr(C)]
        struct Ctrl3 {
            wd: [DevicePtr; 3],
            wgf: [DevicePtr; 3],
            x: [DevicePtr; 3],
            msg: [DevicePtr; 3],
            cd: [DevicePtr; 3],
            out: [DevicePtr; 3],
        }
        let nj = jobs.len();
        if nj == 0 || nj > 3 || ntok == 0 {
            return Err(invalid(format!("ctrl_multi: {nj} jobs, {ntok} tokens")));
        }
        let (d, nt) = (dim as usize, ntok as usize);
        let rank = jobs[0].0.n;
        let mut c = Ctrl3 { wd: [0; 3], wgf: [0; 3], x: [0; 3], msg: [0; 3], cd: [0; 3], out: [0; 3] };
        for (j, (wd, wgf, x, msg, cd, out)) in jobs.iter_mut().enumerate() {
            Self::f32_mat(wd, rank, 2 * dim, "ctrl_multi down")?;
            Self::f32_mat(wgf, 2 * dim, rank, "ctrl_multi gain_forget")?;
            need("ctrl_multi x", x.dlen(), (nt - 1) * x_stride + d)?;
            need("ctrl_multi msg", msg.dlen(), (nt - 1) * msg_stride + d)?;
            need("ctrl_multi cd", cd.dlen(), nt * rank as usize)?;
            need("ctrl_multi out", out.dlen(), nt * d)?;
            c.wd[j] = wd.qs.device_ptr();
            c.wgf[j] = wgf.qs.device_ptr();
            c.x[j] = x.dptr();
            c.msg[j] = msg.dptr();
            c.cd[j] = cd.dptr();
            c.out[j] = out.dptr();
        }
        let (mut xs, mut ms, mut dd) = (x_stride as u32, msg_stride as u32, dim);
        let mut c2 = Ctrl3 { wd: c.wd, wgf: c.wgf, x: c.x, msg: c.msg, cd: c.cd, out: c.out };
        if !exact && ntok > 8 && rank <= 64 && rank.is_multiple_of(8) && dim.is_multiple_of(4) && x_stride.is_multiple_of(4) && msg_stride.is_multiple_of(4) {
            let tb = ntok.div_ceil(8);
            let tc = rank == 64 && ntok <= 64 && dim.is_multiple_of(128) && scratch.dlen() >= 2 * 3 * ntok as usize && std::env::var_os("ASTRAL_CTRL_SIMT").is_none();
            if tc {
                // Norm scales (and zeroed cd), then the down projection on bf16 tensor cores.
                let (mut nt1, mut rk, mut psc) = (ntok, rank, scratch.dptr());
                let mut p = [param(&mut c), param(&mut xs), param(&mut ms), param(&mut dd), param(&mut nt1), param(&mut rk), param(&mut psc)];
                Self::go(&self.ctrl_tc[0], s, LaunchConfig { grid: (ntok, nj as u32, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
                let mut c1 = Ctrl3 { wd: c.wd, wgf: c.wgf, x: c.x, msg: c.msg, cd: c.cd, out: c.out };
                let (mut xs1, mut ms1, mut dd1, mut nt2, mut psc2) = (x_stride as u32, msg_stride as u32, dim, ntok, scratch.dptr());
                let mut p = [param(&mut c1), param(&mut xs1), param(&mut ms1), param(&mut dd1), param(&mut nt2), param(&mut psc2)];
                Self::go(&self.ctrl_tc[1], s, LaunchConfig { grid: (2 * dim / 128, nj as u32, ntok.div_ceil(64)), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
            } else {
                let mut nt1 = ntok;
                let mut p = [param(&mut c), param(&mut xs), param(&mut ms), param(&mut dd), param(&mut nt1)];
                Self::go(&self.ctrl_b[0], s, LaunchConfig { grid: (rank / 8, tb, nj as u32), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
            }
            let (mut xs2, mut ms2, mut dd2, mut r, mut nt2) = (x_stride as u32, msg_stride as u32, dim, rank, ntok);
            let mut p2 = [param(&mut c2), param(&mut xs2), param(&mut ms2), param(&mut dd2), param(&mut r), param(&mut nt2)];
            return Self::go(&self.ctrl_b[1], s, LaunchConfig { grid: (dim.div_ceil(256), tb, nj as u32), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p2);
        }
        if ntok > 1 && rank <= 64 {
            // Several tokens: 8 per block, each weight load shared (same per-token arithmetic).
            let tb = ntok.div_ceil(8);
            let mut nt1 = ntok;
            let mut p = [param(&mut c), param(&mut xs), param(&mut ms), param(&mut dd), param(&mut nt1)];
            Self::go(&self.ctrl_down3_t, s, LaunchConfig { grid: (rank, tb, nj as u32), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
            let (mut xs2, mut ms2, mut dd2, mut r, mut nt2) = (x_stride as u32, msg_stride as u32, dim, rank, ntok);
            let mut p2 = [param(&mut c2), param(&mut xs2), param(&mut ms2), param(&mut dd2), param(&mut r), param(&mut nt2)];
            return Self::go(&self.ctrl_combine3_t, s, LaunchConfig { grid: (dim.div_ceil(8), tb, nj as u32), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p2);
        }
        let mut p = [param(&mut c), param(&mut xs), param(&mut ms), param(&mut dd)];
        Self::go(&self.ctrl_down3, s, LaunchConfig { grid: (rank, ntok, nj as u32), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        let (mut xs2, mut ms2, mut dd2, mut r) = (x_stride as u32, msg_stride as u32, dim, rank);
        let mut p2 = [param(&mut c2), param(&mut xs2), param(&mut ms2), param(&mut dd2), param(&mut r)];
        Self::go(&self.ctrl_combine3, s, LaunchConfig { grid: (dim.div_ceil(8), ntok, nj as u32), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p2)
    }

    /// Up to three [`InklingOps::rms_q8`] jobs `(x, weight, y, activations)` in one launch (rows
    /// of every `x` at `t * x_stride`, `dim = qa.k`).
    pub fn rms_q8_multi(&self, s: &Stream, jobs: &mut [(&dyn Dev<f32>, &dyn Dev<f32>, Option<&mut dyn DevMut<f32>>, &mut QAct)], x_stride: usize, ntok: u32, eps: f32) -> Result<()> {
        #[repr(C)]
        struct Rms3 {
            x: [DevicePtr; 3],
            w: [DevicePtr; 3],
            y: [DevicePtr; 3],
            xq: [DevicePtr; 3],
            xd: [DevicePtr; 3],
            xh: [DevicePtr; 3],
        }
        let nj = jobs.len();
        if nj == 0 || nj > 3 || ntok == 0 {
            return Err(invalid(format!("rms_q8_multi: {nj} jobs, {ntok} tokens")));
        }
        let dim = jobs[0].3.k;
        let mut r = Rms3 { x: [0; 3], w: [0; 3], y: [0; 3], xq: [0; 3], xd: [0; 3], xh: [0; 3] };
        for (j, (x, w, y, qa)) in jobs.iter_mut().enumerate() {
            if qa.k != dim {
                return Err(invalid("rms_q8_multi: activation widths differ".into()));
            }
            need("rms_q8_multi x", x.dlen(), (ntok as usize - 1) * x_stride + dim as usize)?;
            need("rms_q8_multi w", w.dlen(), dim as usize)?;
            if let Some(y) = y {
                need("rms_q8_multi y", y.dlen(), (dim * ntok) as usize)?;
                r.y[j] = y.dptr();
            }
            let (q, dd, h) = qa.planes_for(ntok)?;
            r.x[j] = x.dptr();
            r.w[j] = w.dptr();
            r.xq[j] = q;
            r.xd[j] = dd;
            r.xh[j] = h;
        }
        let (mut xs, mut d, mut e) = (x_stride as u32, dim, eps);
        let mut p = [param(&mut r), param(&mut xs), param(&mut d), param(&mut e)];
        let block = (dim / 4).next_multiple_of(32).clamp(32, 1024);
        Self::go(&self.rms_q8_3, s, LaunchConfig { grid: (ntok, nj as u32, 1), block: (block, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Loop controller part 2 for `ntok` tokens: gain/forget from `cd[t]`, then the combine into
    /// `out[t]` (token-major).
    #[allow(clippy::too_many_arguments)]
    pub fn ctrl_combine(
        &self,
        s: &Stream,
        wgf: &GemvMatrix,
        cd: &dyn Dev<f32>,
        x: &dyn Dev<f32>,
        x_stride: usize,
        msg: &dyn Dev<f32>,
        msg_stride: usize,
        out: &mut dyn DevMut<f32>,
        dim: u32,
        ntok: u32,
    ) -> Result<()> {
        Self::f32_mat(wgf, 2 * dim, wgf.k, "ctrl_combine")?;
        let (d, nt) = (dim as usize, ntok as usize);
        if nt == 0 {
            return Err(invalid("ctrl_combine: no tokens".into()));
        }
        need("ctrl_combine cd", cd.dlen(), nt * wgf.k as usize)?;
        need("ctrl_combine x", x.dlen(), (nt - 1) * x_stride + d)?;
        need("ctrl_combine msg", msg.dlen(), (nt - 1) * msg_stride + d)?;
        need("ctrl_combine out", out.dlen(), nt * d)?;
        let (mut pw, mut pc, mut px, mut xs, mut pm, mut ms, mut po, mut dd, mut r) =
            (wgf.qs.device_ptr(), cd.dptr(), x.dptr(), x_stride as u32, msg.dptr(), msg_stride as u32, out.dptr(), dim, wgf.k);
        let mut p = [param(&mut pw), param(&mut pc), param(&mut px), param(&mut xs), param(&mut pm), param(&mut ms), param(&mut po), param(&mut dd), param(&mut r)];
        Self::go(&self.ctrl_combine, s, LaunchConfig { grid: (dim.div_ceil(8), ntok, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }
    /// `x *= sigmoid(g)` for `ntok` rows of `qa.k`, then `x` quantised into `qa`.
    pub fn gate_q8(&self, s: &Stream, x: &mut dyn DevMut<f32>, g: &dyn Dev<f32>, qa: &mut QAct, ntok: u32) -> Result<()> {
        let n = qa.k * ntok;
        need("gate_q8 x", x.dlen(), n as usize)?;
        need("gate_q8 g", g.dlen(), n as usize)?;
        let (mut pq, mut pd, mut ph) = qa.planes_for(ntok)?;
        let (mut px, mut pg, mut nn) = (x.dptr(), g.dptr(), n);
        let mut p = [param(&mut px), param(&mut pg), param(&mut pq), param(&mut pd), param(&mut ph), param(&mut nn)];
        Self::go(&self.gate_q8, s, LaunchConfig { grid: ((n / 32).div_ceil(8), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// SITU over `slots` fused gate_up rows, quantised into `qa` (`qa.k == n_ff`).
    pub fn situ_q8(&self, s: &Stream, gu: &dyn Dev<f32>, qa: &mut QAct, slots: u32, beta: f32, lb: f32) -> Result<()> {
        let n_ff = qa.k;
        need("situ_q8 gate_up", gu.dlen(), (2 * n_ff * slots) as usize)?;
        let (mut pq, mut pd, mut ph) = qa.planes_for(slots)?;
        let (mut pg, mut n, mut sl, mut b, mut l) = (gu.dptr(), n_ff, slots, beta, lb);
        let mut p = [param(&mut pg), param(&mut pq), param(&mut pd), param(&mut ph), param(&mut n), param(&mut sl), param(&mut b), param(&mut l)];
        Self::go(&self.situ_q8, s, LaunchConfig { grid: ((slots * n_ff / 32).div_ceil(8), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }
    /// Short conv of `x` (or of the MoE combine: `moe = (weights [t][slots], slots)`, `x [t][slot][c]`)
    /// into the residual stream over `ntok` tokens: `bank_row = cur, cur = y` when `bank_row` is
    /// given, else `cur += y`. All token-major.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_residual(
        &self,
        s: &Stream,
        x: &dyn Dev<f32>,
        moe: Option<(&dyn Dev<f32>, u32)>,
        ntok: u32,
        state: &mut dyn DevMut<f32>,
        weight: &dyn Dev<f32>,
        cur: &mut dyn DevMut<f32>,
        bank_row: Option<&mut dyn DevMut<f32>>,
        channels: u32,
        rows: Option<SeqRows>,
    ) -> Result<()> {
        let (c, nt) = (channels as usize, ntok as usize);
        let slots = moe.map_or(1, |m| m.1);
        if nt == 0 || slots == 0 {
            return Err(invalid("conv_residual: no tokens".into()));
        }
        need("conv_residual x", x.dlen(), nt * c * slots as usize)?;
        if let Some((w, _)) = moe {
            need("conv_residual moe weights", w.dlen(), nt * slots as usize)?;
        }
        match &rows {
            Some(r) => {
                need("conv_residual seq", r.seq.dlen(), nt)?;
                need("conv_residual state", state.dlen(), (r.n_seqs.max(1) as usize - 1) * r.conv_stride as usize + 3 * c)?;
            }
            None => need("conv_residual state", state.dlen(), 3 * c)?,
        }
        need("conv_residual weight", weight.dlen(), 4 * c)?;
        need("conv_residual cur", cur.dlen(), nt * c)?;
        let pb = match bank_row {
            Some(b) => {
                need("conv_residual bank row", b.dlen(), nt * c)?;
                b.dptr()
            }
            None => 0,
        };
        let (mut px, mut pm, mut sl, mut nn, mut ps, mut pw, mut pc, mut pb, mut ch) = (x.dptr(), moe.map_or(0, |m| m.0.dptr()), slots, ntok, state.dptr(), weight.dptr(), cur.dptr(), pb, channels);
        let (mut sq, mut ns, mut cst) = SeqRows::parts(rows.as_ref());
        let mut p = [param(&mut px), param(&mut pm), param(&mut sl), param(&mut nn), param(&mut ps), param(&mut pw), param(&mut pc), param(&mut pb), param(&mut ch), param(&mut sq), param(&mut ns), param(&mut cst)];
        let cfg = if rows.is_some() {
            LaunchConfig { grid: (channels.div_ceil(256), ntok, 1), block: (256, 1, 1), shared_mem_bytes: 0 }
        } else if nt == 1 {
            elementwise(channels)
        } else {
            LaunchConfig { grid: (channels.div_ceil(128), 1, 1), block: (128, 1, 1), shared_mem_bytes: 0 }
        };
        Self::go(&self.conv_residual, s, cfg, &mut p)
    }
    /// Split-K (flash-decode) attention for `ntok` causal queries at positions `pos[0]..`;
    /// `head_dim` must be 128 and `n_rel <= 32`. `max_splits` must cover the longest query's
    /// span (`ceil(span / ATT_CHUNK)`); `work` holds [`attn_workspace`]`(ntok, ..)` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_split(
        &self,
        s: &Stream,
        q: &impl Dev<f32>,
        r: &impl Dev<f32>,
        sink: &impl Dev<f32>,
        kc: &impl Dev<u16>,
        vc: &impl Dev<u16>,
        rel_bias: &impl Dev<f32>,
        work: &mut impl DevMut<f32>,
        out: &mut impl DevMut<f32>,
        pos: &impl Dev<i32>,
        a: AttnShape,
        ntok: u32,
        max_splits: u32,
        chunk_loop: bool,
        gate: Option<(&dyn Dev<f32>, &mut QAct)>,
        rows: Option<SeqRows>,
    ) -> Result<()> {
        if a.head_dim != 128 || a.n_rel > 32 || a.kv_heads == 0 || !a.heads.is_multiple_of(a.kv_heads) || a.capacity == 0 || a.capacity > MAX_CTX || ntok == 0 || max_splits == 0 {
            return Err(invalid(format!("attn split: unsupported shape {a:?} (head_dim 128, n_rel <= 32, capacity <= {MAX_CTX})")));
        }
        let (width, nt) = ((a.kv_heads * a.head_dim) as usize, ntok as usize);
        need("attn q", q.dlen(), nt * (a.heads * a.head_dim) as usize)?;
        need("attn r", r.dlen(), nt * (a.heads * a.n_rel) as usize)?;
        need("attn sink", sink.dlen(), nt * a.heads as usize)?;
        let seqs = rows.as_ref().map_or(1, |r| r.n_seqs.max(1) as usize);
        need("attn k cache", kc.dlen(), seqs * width * a.capacity as usize)?;
        need("attn v cache", vc.dlen(), seqs * width * a.capacity as usize)?;
        need("attn rel bias", rel_bias.dlen(), (a.extent * a.n_rel) as usize)?;
        need("attn work", work.dlen(), attn_workspace(ntok, a.heads, a.head_dim, max_splits))?;
        need("attn out", out.dlen(), nt * (a.heads * a.head_dim) as usize)?;
        need("attn pos", pos.dlen(), if rows.is_some() { nt } else { 1 })?;
        if let Some(r) = &rows {
            need("attn seq", r.seq.dlen(), nt)?;
        }
        let (mut sq, mut ns, _) = SeqRows::parts(rows.as_ref());
        let mut sq2 = sq;
        let (mut pq, mut pr, mut ps, mut pk, mut pv, mut pb, mut pw, mut pp) = (q.dptr(), r.dptr(), sink.dptr(), kc.dptr(), vc.dptr(), rel_bias.dptr(), work.dptr(), pos.dptr());
        let (mut h, mut kh, mut nr, mut ex, mut win, mut cap, mut ms) = (a.heads, a.kv_heads, a.n_rel, a.extent, a.window, a.capacity, max_splits);
        let mut scale = 1.0f32 / (a.head_dim as f32).sqrt();
        let mut p = [
            param(&mut pq),
            param(&mut pr),
            param(&mut ps),
            param(&mut pk),
            param(&mut pv),
            param(&mut pb),
            param(&mut pw),
            param(&mut pp),
            param(&mut h),
            param(&mut kh),
            param(&mut nr),
            param(&mut ex),
            param(&mut win),
            param(&mut scale),
            param(&mut cap),
            param(&mut ms),
            param(&mut sq),
            param(&mut ns),
        ];
        // One chunk per split (decode, exact-size prefill splits) or splits looping over chunks
        // (graph-captured multi-token passes with a capped split count).
        let gqa3 = !chunk_loop && a.heads == 3 * a.kv_heads && std::env::var_os("ASTRAL_NO_GQA").is_none();
        if gqa3 {
            // A KV head's 3 query heads in one block: each K / V row read once.
            let f: u32 = std::env::var("ASTRAL_GQA_GY").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
            let gy = (max_splits * f).div_ceil(ntok).clamp(1, max_splits);
            Self::go(&self.attn_split_gqa3, s, LaunchConfig { grid: (a.kv_heads, gy, ntok), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        } else {
            let f = if chunk_loop { &self.attn_split } else { &self.attn_split1 };
            // One-chunk splits: blocks loop over splits, so many rows need fewer (mostly idle) blocks.
            let gy = if chunk_loop { max_splits } else { (max_splits * 8).div_ceil(ntok).clamp(1, max_splits) };
            Self::go(f, s, LaunchConfig { grid: (a.heads, gy, ntok), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        }
        let (mut pw2, mut po, mut pp2, mut win2, mut cap2, mut ms2) = (work.dptr(), out.dptr(), pos.dptr(), a.window, a.capacity, max_splits);
        let cfg = LaunchConfig { grid: (a.heads, ntok, 1), block: (128, 1, 1), shared_mem_bytes: 0 };
        match gate {
            // Merge + output gate + quantisation for the O projection (gate_q8) in one kernel.
            Some((g, qa)) => {
                need("attn gate", g.dlen(), nt * (a.heads * a.head_dim) as usize)?;
                if qa.k != a.heads * a.head_dim {
                    return Err(invalid(format!("attn gate: activations k {} != {}", qa.k, a.heads * a.head_dim)));
                }
                let (mut pq, mut pd, mut ph) = qa.planes_for(ntok)?;
                let mut pg = g.dptr();
                let mut p2 = [param(&mut pw2), param(&mut po), param(&mut pp2), param(&mut win2), param(&mut cap2), param(&mut ms2), param(&mut pg), param(&mut pq), param(&mut pd), param(&mut ph), param(&mut sq2)];
                Self::go(&self.attn_merge_q8, s, cfg, &mut p2)
            }
            None => {
                let mut p2 = [param(&mut pw2), param(&mut po), param(&mut pp2), param(&mut win2), param(&mut cap2), param(&mut ms2), param(&mut sq2)];
                Self::go(&self.attn_merge, s, cfg, &mut p2)
            }
        }
    }

    fn grid_ct(channels: u32, ntok: u32) -> LaunchConfig {
        LaunchConfig { grid: (channels.div_ceil(128), ntok, 1), block: (128, 1, 1), shared_mem_bytes: 0 }
    }

    /// Prefill short conv, every (token, channel) in parallel: `dst[t] = conv(src)[t]` with the
    /// saved state for the first inputs and optional per-token taps `(taps, stride, heads)`.
    /// Does not touch the state: follow with [`InklingOps::conv_state`].
    #[allow(clippy::too_many_arguments)]
    pub fn conv_par(
        &self,
        s: &Stream,
        src: &dyn Dev<f32>,
        src_stride: usize,
        dst: &mut dyn DevMut<f32>,
        dst_stride: usize,
        ntok: u32,
        state: &dyn Dev<f32>,
        weight: &dyn Dev<f32>,
        taps: Option<(&dyn Dev<f32>, usize, u32)>,
        channels: u32,
        hist: Option<&mut dyn DevMut<f32>>,
    ) -> Result<()> {
        let (c, nt) = (channels as usize, ntok as usize);
        if nt == 0 || src_stride < c || dst_stride < c {
            return Err(invalid("conv_par: need tokens > 0 and strides >= channels".into()));
        }
        need("conv_par src", src.dlen(), (nt - 1) * src_stride + c)?;
        need("conv_par dst", dst.dlen(), (nt - 1) * dst_stride + c)?;
        need("conv_par state", state.dlen(), 3 * c)?;
        need("conv_par weight", weight.dlen(), 4 * c)?;
        let (mut pd, mut ds, mut dh) = (0, 0u32, 0u32);
        if let Some((t, stride, heads)) = taps {
            if heads == 0 || !channels.is_multiple_of(heads) || stride < 4 * heads as usize {
                return Err(invalid(format!("conv_par: {channels} channels / {heads} tap heads / stride {stride}")));
            }
            need("conv_par taps", t.dlen(), (nt - 1) * stride + 4 * heads as usize)?;
            (pd, ds, dh) = (t.dptr(), stride as u32, heads);
        }
        let mut phist = match hist {
            Some(h) => {
                need("conv_par hist", h.dlen(), nt * c)?;
                h.dptr()
            }
            None => 0,
        };
        let (mut px, mut xs, mut po, mut os, mut n, mut ps, mut pw, mut ch) = (src.dptr(), src_stride as u32, dst.dptr(), dst_stride as u32, ntok, state.dptr(), weight.dptr(), channels);
        let mut p = [
            param(&mut px),
            param(&mut xs),
            param(&mut po),
            param(&mut os),
            param(&mut n),
            param(&mut ps),
            param(&mut pw),
            param(&mut pd),
            param(&mut ds),
            param(&mut dh),
            param(&mut ch),
            param(&mut phist),
        ];
        Self::go(&self.conv_par, s, Self::grid_ct(channels, ntok.div_ceil(CONV_T)), &mut p)
    }

    /// Prefill residual conv (token-major, width `channels`): `y = conv(src)`, then
    /// `bank_row = cur, cur = y` or `cur += y`. Follow with [`InklingOps::conv_state`].
    #[allow(clippy::too_many_arguments)]
    /// `hist` (verify passes): also stores the conv inputs `src` there (`[ntok][channels]`).
    #[allow(clippy::too_many_arguments)]
    pub fn conv_res_par(
        &self,
        s: &Stream,
        src: &dyn Dev<f32>,
        ntok: u32,
        state: &dyn Dev<f32>,
        weight: &dyn Dev<f32>,
        cur: &mut dyn DevMut<f32>,
        bank_row: Option<&mut dyn DevMut<f32>>,
        channels: u32,
        hist: Option<&mut dyn DevMut<f32>>,
    ) -> Result<()> {
        let n = (channels * ntok) as usize;
        need("conv_res_par src", src.dlen(), n)?;
        need("conv_res_par state", state.dlen(), 3 * channels as usize)?;
        need("conv_res_par weight", weight.dlen(), 4 * channels as usize)?;
        need("conv_res_par cur", cur.dlen(), n)?;
        let pb = match bank_row {
            Some(b) => {
                need("conv_res_par bank row", b.dlen(), n)?;
                b.dptr()
            }
            None => 0,
        };
        let mut phist = match hist {
            Some(h) => {
                need("conv_res_par hist", h.dlen(), n)?;
                h.dptr()
            }
            None => 0,
        };
        let (mut px, mut nt, mut ps, mut pw, mut pc, mut pb, mut ch) = (src.dptr(), ntok, state.dptr(), weight.dptr(), cur.dptr(), pb, channels);
        let mut p = [param(&mut px), param(&mut nt), param(&mut ps), param(&mut pw), param(&mut pc), param(&mut pb), param(&mut ch), param(&mut phist)];
        Self::go(&self.conv_res_par, s, Self::grid_ct(channels, ntok.div_ceil(CONV_T)), &mut p)
    }

    /// Conv state after `ntok` inputs of `src` (row t at `t * src_stride`).
    pub fn conv_state(&self, s: &Stream, src: &dyn Dev<f32>, src_stride: usize, ntok: u32, state: &mut dyn DevMut<f32>, channels: u32) -> Result<()> {
        let c = channels as usize;
        if ntok == 0 || src_stride < c {
            return Err(invalid("conv_state: need tokens > 0 and stride >= channels".into()));
        }
        need("conv_state src", src.dlen(), (ntok as usize - 1) * src_stride + c)?;
        need("conv_state state", state.dlen(), 3 * c)?;
        let (mut px, mut xs, mut nt, mut ps, mut ch) = (src.dptr(), src_stride as u32, ntok, state.dptr(), channels);
        let mut p = [param(&mut px), param(&mut xs), param(&mut nt), param(&mut ps), param(&mut ch)];
        Self::go(&self.conv_state, s, LaunchConfig { grid: (channels.div_ceil(128), 1, 1), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// [`InklingOps::moe_combine_t`] over f16 expert outputs `y` (pointer to `ntok * slots * dim` halves).
    pub fn moe_combine_t16(&self, s: &Stream, y: DevicePtr, w: &dyn Dev<f32>, out: &mut dyn DevMut<f32>, dim: u32, slots: u32, ntok: u32) -> Result<()> {
        need("combine w", w.dlen(), (slots * ntok) as usize)?;
        need("combine out", out.dlen(), (dim * ntok) as usize)?;
        let (mut py, mut pw, mut po, mut d, mut sl, mut nt) = (y, w.dptr(), out.dptr(), dim, slots, ntok);
        let mut p = [param(&mut py), param(&mut pw), param(&mut po), param(&mut d), param(&mut sl), param(&mut nt)];
        if !dim.is_multiple_of(8) {
            return Err(invalid(format!("moe_combine_t16: dim {dim} must be a multiple of 8")));
        }
        Self::go(&self.moe_combine_t16, s, LaunchConfig { grid: ((dim / 8).div_ceil(128), ntok, 1), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `out[t][c] = sum_s w[t][s] y[t][s][c]` over `ntok` tokens.
    pub fn moe_combine_t(&self, s: &Stream, y: &dyn Dev<f32>, w: &dyn Dev<f32>, out: &mut dyn DevMut<f32>, dim: u32, slots: u32, ntok: u32) -> Result<()> {
        need("combine y", y.dlen(), (dim * slots * ntok) as usize)?;
        need("combine w", w.dlen(), (slots * ntok) as usize)?;
        need("combine out", out.dlen(), (dim * ntok) as usize)?;
        let (mut py, mut pw, mut po, mut d, mut sl, mut nt) = (y.dptr(), w.dptr(), out.dptr(), dim, slots, ntok);
        let mut p = [param(&mut py), param(&mut pw), param(&mut po), param(&mut d), param(&mut sl), param(&mut nt)];
        Self::go(&self.moe_combine_t, s, Self::grid_ct(dim, ntok), &mut p)
    }

    /// Loop controller for `ntok` prefill tokens: `out[t] = control(x[t], msg[t])` as two f32
    /// GEMMs (down: split-K over concat(rms x, rms msg); gain/forget) and an elementwise apply.
    /// Scratch: `rms` 2 * ntok floats, `cd` ntok * rank (zeroed here), `gf` ntok * 2 * dim.
    #[allow(clippy::too_many_arguments)]
    pub fn ctrl_t(
        &self,
        s: &Stream,
        wd: &GemvMatrix,
        wgf: &GemvMatrix,
        x: &dyn Dev<f32>,
        x_stride: usize,
        msg: &dyn Dev<f32>,
        msg_stride: usize,
        rms: &mut dyn DevMut<f32>,
        cd: &mut crate::DeviceBufferF32,
        gf: &mut dyn DevMut<f32>,
        out: &mut dyn DevMut<f32>,
        dim: u32,
        ntok: u32,
    ) -> Result<()> {
        Self::f32_mat(wd, wd.n, 2 * dim, "ctrl_t down")?;
        Self::f32_mat(wgf, 2 * dim, wd.n, "ctrl_t gain/forget")?;
        let (d, nt, rank) = (dim as usize, ntok as usize, wd.n);
        if nt == 0 || x_stride < d || msg_stride < d {
            return Err(invalid("ctrl_t: need tokens > 0 and strides >= dim".into()));
        }
        need("ctrl_t x", x.dlen(), (nt - 1) * x_stride + d)?;
        need("ctrl_t msg", msg.dlen(), (nt - 1) * msg_stride + d)?;
        need("ctrl_t rms", rms.dlen(), 2 * nt)?;
        need("ctrl_t cd", cd.len(), nt * rank as usize)?;
        need("ctrl_t gf", gf.dlen(), nt * 2 * d)?;
        need("ctrl_t out", out.dlen(), nt * d)?;
        let (mut px, mut xs, mut pm, mut ms, mut pr, mut dd) = (x.dptr(), x_stride as u32, msg.dptr(), msg_stride as u32, rms.dptr(), dim);
        let mut p = [param(&mut px), param(&mut xs), param(&mut pm), param(&mut ms), param(&mut pr), param(&mut dd)];
        Self::go(&self.rms2, s, LaunchConfig { grid: (ntok, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        // Rank 64: the K-sliced bf16 tensor-core down projection (atomic partial sums into a
        // zeroed cd); ASTRAL_CTRL_SIMT=1 keeps the split GEMM below.
        if rank == 64 && dim.is_multiple_of(128) && x_stride.is_multiple_of(4) && msg_stride.is_multiple_of(4) && std::env::var_os("ASTRAL_CTRL_SIMT").is_none() {
            #[repr(C)]
            struct Ctrl3 {
                wd: [DevicePtr; 3],
                wgf: [DevicePtr; 3],
                x: [DevicePtr; 3],
                msg: [DevicePtr; 3],
                cd: [DevicePtr; 3],
                out: [DevicePtr; 3],
            }
            cd.zero_range_async(0, nt * rank as usize, s)?;
            let mut c = Ctrl3 { wd: [wd.qs.device_ptr(), 0, 0], wgf: [0; 3], x: [x.dptr(), 0, 0], msg: [msg.dptr(), 0, 0], cd: [cd.device_ptr(), 0, 0], out: [0; 3] };
            let (mut xs1, mut ms1, mut dd1, mut nt1, mut psc) = (x_stride as u32, msg_stride as u32, dim, ntok, rms.dptr());
            let mut p = [param(&mut c), param(&mut xs1), param(&mut ms1), param(&mut dd1), param(&mut nt1), param(&mut psc)];
            Self::go(&self.ctrl_tc[1], s, LaunchConfig { grid: (2 * dim / 128, 1, ntok.div_ceil(64)), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
            // Gain/forget GEMM and apply fused (ASTRAL_CTRL_UNFUSED=1: through gf).
            if dim.is_multiple_of(32) && std::env::var_os("ASTRAL_CTRL_UNFUSED").is_none() {
                let (mut pc, mut rk, mut pw, mut px, mut xs, mut pm, mut ms, mut po, mut dd, mut n) = (cd.device_ptr(), rank, wgf.qs.device_ptr(), x.dptr(), x_stride as u32, msg.dptr(), msg_stride as u32, out.dptr(), dim, ntok);
                let mut p = [param(&mut pc), param(&mut rk), param(&mut pw), param(&mut px), param(&mut xs), param(&mut pm), param(&mut ms), param(&mut po), param(&mut dd), param(&mut n)];
                return Self::go(&self.ctrl_gf_apply, s, LaunchConfig { grid: (dim / 32, ntok.div_ceil(64), 1), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p);
            }
            self.sgemm(s, cd.device_ptr(), 0, rank, rank, rank, 0, wgf.qs.device_ptr(), rank, gf.dptr(), 2 * dim, ntok, 2 * dim, rank, rank.next_multiple_of(32), false)?;
            let (mut pg, mut px, mut xs, mut pm, mut ms, mut po, mut dd, mut n) = (gf.dptr(), x.dptr(), x_stride as u32, msg.dptr(), msg_stride as u32, out.dptr(), dim, ntok);
            let mut p = [param(&mut pg), param(&mut px), param(&mut xs), param(&mut pm), param(&mut ms), param(&mut po), param(&mut dd), param(&mut n)];
            return Self::go(&self.ctrl_apply, s, LaunchConfig { grid: (dim.div_ceil(256), ntok, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p);
        }
        // cd = concat(rx x, rm msg) . W_down^T, K = 2 dim split 8 ways: per-split partials in gf
        // (free until the next GEMM), summed in a fixed order (deterministic).
        let k = 2 * dim;
        let splits = 8u32;
        let kps = k.div_ceil(splits).next_multiple_of(32);
        let splits = k.div_ceil(kps);
        need("ctrl_t split partials", gf.dlen(), (splits * ntok * rank) as usize)?;
        self.sgemm_mode(s, x.dptr(), msg.dptr(), x_stride as u32, msg_stride as u32, dim, rms.dptr(), wd.qs.device_ptr(), k, gf.dptr(), rank, ntok, rank, k, kps, 2)?;
        let n = ntok * rank;
        let (mut pp, mut po, mut nn, mut sp) = (gf.dptr(), cd.device_ptr(), n, splits);
        let mut p = [param(&mut pp), param(&mut po), param(&mut nn), param(&mut sp)];
        Self::go(&self.sum_splits, s, LaunchConfig { grid: (n.div_ceil(256).min(1024), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        // gf = cd . W_gf^T (gain rows then forget rows).
        self.sgemm(s, cd.device_ptr(), 0, rank, rank, rank, 0, wgf.qs.device_ptr(), rank, gf.dptr(), 2 * dim, ntok, 2 * dim, rank, rank.next_multiple_of(32), false)?;
        let (mut pg, mut px, mut xs, mut pm, mut ms, mut po, mut dd, mut n) = (gf.dptr(), x.dptr(), x_stride as u32, msg.dptr(), msg_stride as u32, out.dptr(), dim, ntok);
        let mut p = [param(&mut pg), param(&mut px), param(&mut xs), param(&mut pm), param(&mut ms), param(&mut po), param(&mut dd), param(&mut n)];
        Self::go(&self.ctrl_apply, s, LaunchConfig { grid: (dim.div_ceil(256), ntok, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `c[m][n] = sum_k a[m][k] * w[n][k]` in f32 (`a` rows `lda` apart, `w` an F32 matrix,
    /// `c` rows of `w.n`): small dense GEMMs (prefill projections kept in high precision).
    /// K is split 8 ways (atomic adds into `c`, which is zeroed first) to fill the GPU.
    pub fn gemm_f32(&self, s: &Stream, a: &dyn Dev<f32>, lda: usize, w: &GemvMatrix, c: &mut dyn DevMut<f32>, m: u32) -> Result<()> {
        if w.ty != GemvType::F32 || w.n_mats != 1 || m == 0 || lda < w.k as usize {
            return Err(invalid(format!("gemm_f32: need one F32 matrix, m > 0, lda >= k (got {:?}, m {m}, lda {lda})", w.ty)));
        }
        need("gemm_f32 a", a.dlen(), (m as usize - 1) * lda + w.k as usize)?;
        need("gemm_f32 c", c.dlen(), (m * w.n) as usize)?;
        // K is not split (a split would add partials atomically, in timing-dependent order).
        self.sgemm(s, a.dptr(), 0, lda as u32, lda as u32, w.k, 0, w.qs.device_ptr(), w.k, c.dptr(), w.n, m, w.n, w.k, w.k.next_multiple_of(32), false)
    }

    /// [`InklingOps::gemm_f32`] with K split across blocks (up to 8 ways, by K alone): the
    /// per-split partials go to `ws` and are summed in a fixed order (deterministic).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32_ws(&self, s: &Stream, a: &dyn Dev<f32>, lda: usize, w: &GemvMatrix, c: &mut dyn DevMut<f32>, m: u32, ws: &mut dyn DevMut<f32>) -> Result<()> {
        // The split depends on K only, so a row's sum does not depend on how many rows there are.
        let splits = 8u32.min(w.k.div_ceil(256));
        if splits == 1 || ws.dlen() < (splits * m * w.n) as usize {
            return self.gemm_f32(s, a, lda, w, c, m);
        }
        if w.ty != GemvType::F32 || w.n_mats != 1 || m == 0 || lda < w.k as usize {
            return Err(invalid(format!("gemm_f32_ws: need one F32 matrix, m > 0, lda >= k (got {:?}, m {m}, lda {lda})", w.ty)));
        }
        need("gemm_f32 a", a.dlen(), (m as usize - 1) * lda + w.k as usize)?;
        need("gemm_f32 c", c.dlen(), (m * w.n) as usize)?;
        let kps = w.k.div_ceil(splits).next_multiple_of(32);
        let splits = w.k.div_ceil(kps);
        self.sgemm_mode(s, a.dptr(), 0, lda as u32, lda as u32, w.k, 0, w.qs.device_ptr(), w.k, ws.dptr(), w.n, m, w.n, w.k, kps, 2)?;
        let n = m * w.n;
        let (mut pp, mut po, mut nn, mut sp) = (ws.dptr(), c.dptr(), n, splits);
        let mut p = [param(&mut pp), param(&mut po), param(&mut nn), param(&mut sp)];
        Self::go(&self.sum_splits, s, LaunchConfig { grid: (n.div_ceil(256).min(1024), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Raw launch of astral_ink_sgemm_tn; callers have checked every extent.
    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        &self,
        s: &Stream,
        a0: DevicePtr,
        a1: DevicePtr,
        lda: u32,
        lda1: u32,
        k_split: u32,
        row_scale: DevicePtr,
        w: DevicePtr,
        ldw: u32,
        c: DevicePtr,
        ldc: u32,
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        atomic: bool,
    ) -> Result<()> {
        self.sgemm_mode(s, a0, a1, lda, lda1, k_split, row_scale, w, ldw, c, ldc, m, n, k, k_per_split, atomic as u32)
    }

    /// [`InklingOps::sgemm`] with the kernel's split mode: 0 plain, 1 atomic adds, 2 per-split partials.
    #[allow(clippy::too_many_arguments)]
    fn sgemm_mode(
        &self,
        s: &Stream,
        a0: DevicePtr,
        a1: DevicePtr,
        lda: u32,
        lda1: u32,
        k_split: u32,
        row_scale: DevicePtr,
        w: DevicePtr,
        ldw: u32,
        c: DevicePtr,
        ldc: u32,
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        mode: u32,
    ) -> Result<()> {
        let (mut pa0, mut pa1, mut la, mut la1, mut ks, mut prs, mut pw, mut lw, mut pc, mut lc, mut mm, mut nn, mut kk, mut kps, mut at) =
            (a0, a1, lda, lda1, k_split, row_scale, w, ldw, c, ldc, m, n, k, k_per_split, mode);
        let mut p = [
            param(&mut pa0),
            param(&mut pa1),
            param(&mut la),
            param(&mut la1),
            param(&mut ks),
            param(&mut prs),
            param(&mut pw),
            param(&mut lw),
            param(&mut pc),
            param(&mut lc),
            param(&mut mm),
            param(&mut nn),
            param(&mut kk),
            param(&mut kps),
            param(&mut at),
        ];
        let splits = k.div_ceil(k_per_split);
        // Many rows (prefill): bf16 tensor cores when the operands allow 16-byte loads.
        let aligned = [lda, lda1, k_split, ldw, k_per_split].iter().all(|v| v.is_multiple_of(4)) && [a0, a1, w].iter().all(|p| p % 16 == 0);
        if m >= 64 && aligned && std::env::var_os("ASTRAL_SGEMM_F32").is_none() {
            return Self::go(&self.tgemm_tn, s, LaunchConfig { grid: (n.div_ceil(64), m.div_ceil(64), splits), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p);
        }
        Self::go(&self.sgemm_tn, s, LaunchConfig { grid: (n.div_ceil(32), m.div_ceil(32), splits), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Flash-attention prefill on tensor cores for `ntok` causal queries at positions
    /// `pos[0]..` (keys already in the cache). `head_dim` must be 128 and `n_rel <= 16`.
    #[allow(clippy::too_many_arguments)]
    /// [`InklingOps::attn_prefill`] for every segment of a packed prefill pass in one launch:
    /// `seg` holds 4 words per segment (first row, rows, KV-cache offset in halves, index of its
    /// start position in `pos`); `kc` / `vc` are the whole caches, `max_rows` the longest segment.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_segs(
        &self,
        s: &Stream,
        q: &impl Dev<f32>,
        r: &impl Dev<f32>,
        sink: &impl Dev<f32>,
        kc: &impl Dev<u16>,
        vc: &impl Dev<u16>,
        rel_bias: &impl Dev<f32>,
        out: &mut impl DevMut<f32>,
        pos: &impl Dev<i32>,
        seg: &impl Dev<i32>,
        n_seg: u32,
        max_rows: u32,
        rows: u32,
        a: AttnShape,
    ) -> Result<()> {
        if a.head_dim != 128 || a.n_rel > 16 || !a.n_rel.is_multiple_of(4) || a.kv_heads == 0 || !a.heads.is_multiple_of(a.kv_heads) || a.capacity == 0 || a.capacity > MAX_CTX || n_seg == 0 || max_rows == 0 {
            return Err(invalid(format!("attn prefill segs: unsupported shape {a:?}")));
        }
        let nt = rows as usize;
        need("attn q", q.dlen(), nt * (a.heads * a.head_dim) as usize)?;
        need("attn r", r.dlen(), nt * (a.heads * a.n_rel) as usize)?;
        need("attn sink", sink.dlen(), nt * a.heads as usize)?;
        need("attn rel bias", rel_bias.dlen(), (a.extent * a.n_rel) as usize)?;
        need("attn out", out.dlen(), nt * (a.heads * a.head_dim) as usize)?;
        need("attn seg table", seg.dlen(), 4 * n_seg as usize)?;
        let (mut pq, mut pr, mut ps, mut pk, mut pv, mut pb, mut po, mut pp, mut pt) = (q.dptr(), r.dptr(), sink.dptr(), kc.dptr(), vc.dptr(), rel_bias.dptr(), out.dptr(), pos.dptr(), seg.dptr());
        let (mut h, mut kh, mut nr, mut ex, mut win, mut cap) = (a.heads, a.kv_heads, a.n_rel, a.extent, a.window, a.capacity);
        let mut scale = 1.0f32 / (a.head_dim as f32).sqrt();
        let mut p = [
            param(&mut pq),
            param(&mut pr),
            param(&mut ps),
            param(&mut pk),
            param(&mut pv),
            param(&mut pb),
            param(&mut po),
            param(&mut pp),
            param(&mut pt),
            param(&mut h),
            param(&mut kh),
            param(&mut nr),
            param(&mut ex),
            param(&mut win),
            param(&mut scale),
            param(&mut cap),
        ];
        // The caller wrote the table for these caches (offsets + capacity * width stay inside).
        Self::go(&self.attn_prefill_segs, s, LaunchConfig { grid: (max_rows.div_ceil(64), a.heads, n_seg), block: (128, 1, 1), shared_mem_bytes: FA_SMEM }, &mut p)
    }

    pub fn attn_prefill(
        &self,
        s: &Stream,
        q: &impl Dev<f32>,
        r: &impl Dev<f32>,
        sink: &impl Dev<f32>,
        kc: &impl Dev<u16>,
        vc: &impl Dev<u16>,
        rel_bias: &impl Dev<f32>,
        out: &mut impl DevMut<f32>,
        pos: &impl Dev<i32>,
        a: AttnShape,
        ntok: u32,
    ) -> Result<()> {
        if a.head_dim != 128 || a.n_rel > 16 || !a.n_rel.is_multiple_of(4) || a.kv_heads == 0 || !a.heads.is_multiple_of(a.kv_heads) || a.capacity == 0 || a.capacity > MAX_CTX || ntok == 0 {
            return Err(invalid(format!("attn prefill: unsupported shape {a:?} (head_dim 128, n_rel <= 16 and % 4 == 0)")));
        }
        let (width, nt) = ((a.kv_heads * a.head_dim) as usize, ntok as usize);
        need("attn q", q.dlen(), nt * (a.heads * a.head_dim) as usize)?;
        need("attn r", r.dlen(), nt * (a.heads * a.n_rel) as usize)?;
        need("attn sink", sink.dlen(), nt * a.heads as usize)?;
        need("attn k cache", kc.dlen(), width * a.capacity as usize)?;
        need("attn v cache", vc.dlen(), width * a.capacity as usize)?;
        need("attn rel bias", rel_bias.dlen(), (a.extent * a.n_rel) as usize)?;
        need("attn out", out.dlen(), nt * (a.heads * a.head_dim) as usize)?;
        need("attn pos", pos.dlen(), 1)?;
        let (mut pq, mut pr, mut ps, mut pk, mut pv, mut pb, mut po, mut pp) = (q.dptr(), r.dptr(), sink.dptr(), kc.dptr(), vc.dptr(), rel_bias.dptr(), out.dptr(), pos.dptr());
        let (mut n, mut h, mut kh, mut nr, mut ex, mut win, mut cap) = (ntok, a.heads, a.kv_heads, a.n_rel, a.extent, a.window, a.capacity);
        let mut scale = 1.0f32 / (a.head_dim as f32).sqrt();
        let mut p = [
            param(&mut pq),
            param(&mut pr),
            param(&mut ps),
            param(&mut pk),
            param(&mut pv),
            param(&mut pb),
            param(&mut po),
            param(&mut pp),
            param(&mut n),
            param(&mut h),
            param(&mut kh),
            param(&mut nr),
            param(&mut ex),
            param(&mut win),
            param(&mut scale),
            param(&mut cap),
        ];
        Self::go(&self.attn_prefill, s, LaunchConfig { grid: (ntok.div_ceil(64), a.heads, 1), block: (128, 1, 1), shared_mem_bytes: FA_SMEM }, &mut p)
    }

    /// Groups the `p = ntok * k` router picks in `ids` by expert into tensor-core GEMM work
    /// lists for gate_up (`*_gu`, activation row = token) and down (`*_dn`, row = pair), on
    /// the GPU. Each items buffer holds `max_items` entries (3 words); unused ones get count 0.
    /// At most `min(p, p / 64 + n_expert)` items are ever used (each holds at least one pick).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_group(
        &self,
        s: &Stream,
        ids: &dyn Dev<i32>,
        p: u32,
        k: u32,
        n_expert: u32,
        gu: (&mut dyn DevMut<u32>, &mut dyn DevMut<u32>, &mut dyn DevMut<u32>),
        dn: (&mut dyn DevMut<u32>, &mut dyn DevMut<u32>, &mut dyn DevMut<u32>),
        max_items: u32,
        chunk: u32,
    ) -> Result<()> {
        if p == 0 || k == 0 || n_expert == 0 || n_expert > 255 || !(1..=256).contains(&chunk) || max_items < (p.div_ceil(chunk) + n_expert).min(p) {
            return Err(invalid(format!("moe_group: {p} picks, {n_expert} experts, {max_items} items")));
        }
        need("moe_group ids", ids.dlen(), p as usize)?;
        for (w, b) in [("items gu", gu.0.dlen()), ("items dn", dn.0.dlen())] {
            need(w, b, 3 * max_items as usize)?;
        }
        for (w, b) in [("xidx gu", gu.1.dlen()), ("yidx gu", gu.2.dlen()), ("xidx dn", dn.1.dlen()), ("yidx dn", dn.2.dlen())] {
            need(w, b, p as usize)?;
        }
        let (mut pi, mut pp, mut kk, mut ne) = (ids.dptr(), p, k, n_expert);
        let (mut a0, mut a1, mut a2, mut b0, mut b1, mut b2, mut mi) = (gu.0.dptr(), gu.1.dptr(), gu.2.dptr(), dn.0.dptr(), dn.1.dptr(), dn.2.dptr(), max_items);
        let mut ch = chunk;
        let mut prm = [param(&mut pi), param(&mut pp), param(&mut kk), param(&mut ne), param(&mut a0), param(&mut a1), param(&mut a2), param(&mut b0), param(&mut b1), param(&mut b2), param(&mut mi), param(&mut ch)];
        Self::go(&self.moe_group, s, LaunchConfig { grid: (n_expert, 1, 1), block: (1024, 1, 1), shared_mem_bytes: 0 }, &mut prm)
    }

    /// Groups the `p = ntok * k` router picks in `ids` by expert into [`crate::ops::QItem`]
    /// work lists for [`crate::ops::Gemv::launch_items`] (up to 8 picks per item, so a small
    /// batch reads each used expert once): gate_up items map token -> pair row, down items pair
    /// -> pair. Each list holds `max_items` items; unused ones get n = 0.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_group_q(&self, s: &Stream, ids: &dyn Dev<i32>, p: u32, k: u32, n_expert: u32, items_gu: &mut dyn DevMut<u32>, items_dn: &mut dyn DevMut<u32>, max_items: u32) -> Result<()> {
        if p == 0 || k == 0 || n_expert == 0 || n_expert > 255 || max_items < moe_items_bound(p, n_expert) {
            return Err(invalid(format!("moe_group_q: {p} picks, {n_expert} experts, {max_items} items")));
        }
        need("moe_group_q ids", ids.dlen(), p as usize)?;
        let words = max_items as usize * crate::ops::QItem::WORDS;
        need("moe_group_q items gu", items_gu.dlen(), words)?;
        need("moe_group_q items dn", items_dn.dlen(), words)?;
        let (mut pi, mut pp, mut kk, mut ne, mut a, mut b, mut mi) = (ids.dptr(), p, k, n_expert, items_gu.dptr(), items_dn.dptr(), max_items);
        let mut prm = [param(&mut pi), param(&mut pp), param(&mut kk), param(&mut ne), param(&mut a), param(&mut b), param(&mut mi)];
        Self::go(&self.moe_group_q, s, LaunchConfig { grid: (1, 1, 1), block: (1024, 1, 1), shared_mem_bytes: 0 }, &mut prm)
    }
}
