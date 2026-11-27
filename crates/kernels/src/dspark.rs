// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Typed launchers for `cu/dspark.cu`: the Inkling 2 DSpark draft model and the speculative
//! decoding bookkeeping (acceptance, short-conv rollback). Every launcher checks buffer sizes;
//! device-side indices (tokens, positions) are clamped by the kernels.

use crate::ops::QAct;
use crate::KernelRegistry;
use astral_core::{AstralError, Result};
use astral_gpu::{param, Dev, DevMut, DevicePtr, Function, LaunchConfig, Stream};

/// Head dimension the draft kernels are written for (`DSP_HD`).
pub const HEAD_DIM: u32 = 128;
/// Query heads per KV head `astral_dsp_attn` handles (`DSP_MAXG`).
pub const MAX_GROUP: u32 = 4;

fn invalid(m: String) -> AstralError {
    AstralError::Invalid(m)
}

fn need(what: &str, have: usize, want: usize) -> Result<()> {
    if have < want {
        return Err(invalid(format!("{what}: buffer has {have} elements, need {want}")));
    }
    Ok(())
}

/// Where a kernel reads a position (or row) from: `ptr[idx]` on the device.
#[derive(Clone, Copy)]
pub struct DevIndex {
    pub ptr: DevicePtr,
    pub idx: u32,
}

impl DevIndex {
    pub fn of(buf: &impl Dev<i32>, idx: u32) -> Result<Self> {
        need("device index", buf.dlen(), idx as usize + 1)?;
        Ok(Self { ptr: buf.dptr(), idx })
    }
}

pub struct DsparkOps {
    add_rms_q8: Function,
    quant: Function,
    mix4: Function,
    head_prep: Function,
    attn: Function,
    swiglu_q8: Function,
    markov_in: Function,
    markov_gate: Function,
    add_argmax: Function,
    accept: Function,
    conv_commit: Function,
    rowmax_part: Function,
    fold_part: Function,
    argmax_fin: Function,
    /// Batched draft (B blocks or rows of several sequences).
    head_prep_b: Function,
    attn_b: Function,
    mix4_b: Function,
    markov_in_b: Function,
    markov_gate_b: Function,
    add_rows: Function,
}

/// Blocks of the multi-block argmax (partials per row).
pub const ARGMAX_PARTS: u32 = 264;

impl DsparkOps {
    pub fn new(reg: &KernelRegistry) -> Result<Self> {
        let f = |n: &str| reg.function("dspark", n);
        Ok(Self {
            add_rms_q8: f("astral_dsp_add_rms_q8")?,
            quant: f("astral_dsp_quant")?,
            mix4: f("astral_dsp_mix4")?,
            head_prep: f("astral_dsp_head_prep")?,
            attn: f("astral_dsp_attn")?,
            swiglu_q8: f("astral_dsp_swiglu_q8")?,
            markov_in: f("astral_dsp_markov_in")?,
            markov_gate: f("astral_dsp_markov_gate")?,
            add_argmax: f("astral_dsp_add_argmax")?,
            accept: f("astral_dsp_accept")?,
            conv_commit: f("astral_dsp_conv_commit")?,
            rowmax_part: f("astral_dsp_rowmax_part")?,
            fold_part: f("astral_dsp_fold_part")?,
            argmax_fin: f("astral_dsp_argmax_fin")?,
            head_prep_b: f("astral_dsp_head_prep_b")?,
            attn_b: f("astral_dsp_attn_b")?,
            mix4_b: f("astral_dsp_mix4_b")?,
            markov_in_b: f("astral_dsp_markov_in_b")?,
            markov_gate_b: f("astral_dsp_markov_gate_b")?,
            add_rows: f("astral_dsp_add_rows")?,
        })
    }

    /// [`DsparkOps::head_prep`] over `rows` rows in blocks of `rps`: row t is row t % rps of
    /// block b = t / rps, at position `pos[b] + t % rps`; `to_cache` writes row
    /// `seq[b] * capacity + position` of `dst` (a cache of `capacity` rows per sequence).
    #[allow(clippy::too_many_arguments)]
    pub fn head_prep_b(&self, s: &Stream, src: DevicePtr, src_stride: u32, dst: DevicePtr, dst_stride: u32, w: Option<&dyn Dev<f32>>, rope: bool, pos: &dyn Dev<i32>, seq: &dyn Dev<i32>, rps: u32, to_cache: bool, capacity: u32, rows: u32, heads: u32, theta: f32, eps: f32) -> Result<()> {
        if rows == 0 || rps == 0 || heads == 0 || capacity == 0 || heads * HEAD_DIM > dst_stride {
            return Err(invalid(format!("head_prep_b: {rows} rows x {heads} heads")));
        }
        need("head_prep_b pos", pos.dlen().min(seq.dlen()), rows.div_ceil(rps) as usize)?;
        let (mut ps, mut ss, mut pd, mut ds, mut pw, mut r) = (src, src_stride, dst, dst_stride, w.map_or(0, |w| w.dptr()), rope as u32);
        let (mut pp, mut sq, mut rp, mut tc, mut cap, mut th, mut e) = (pos.dptr(), seq.dptr(), rps, to_cache as u32, capacity, theta, eps);
        let mut p = [param(&mut ps), param(&mut ss), param(&mut pd), param(&mut ds), param(&mut pw), param(&mut r), param(&mut pp), param(&mut sq), param(&mut rp), param(&mut tc), param(&mut cap), param(&mut th), param(&mut e)];
        Self::go(&self.head_prep_b, s, LaunchConfig { grid: (rows, heads, 1), block: (64, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// [`DsparkOps::attn`] for `blocks` blocks of `nblk` queries (block b over sequence
    /// `seq[b]`'s cache, `capacity` rows per sequence, ending at `pos[b] + nblk`).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_b(&self, s: &Stream, q: DevicePtr, q_stride: u32, kc: DevicePtr, vc: DevicePtr, out: DevicePtr, pos: &dyn Dev<i32>, seq: &dyn Dev<i32>, blocks: u32, nblk: u32, heads: u32, kv_heads: u32, window: u32, capacity: u32) -> Result<()> {
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) || heads / kv_heads > MAX_GROUP || nblk == 0 || blocks == 0 {
            return Err(invalid(format!("dspark attn_b: {heads} heads over {kv_heads} KV heads")));
        }
        need("attn_b pos", pos.dlen().min(seq.dlen()), blocks as usize)?;
        let (mut pq, mut qs, mut pk, mut pv, mut po, mut pp, mut pi) = (q, q_stride, kc, vc, out, pos.dptr(), 0u32);
        let (mut nb, mut h, mut kh, mut win, mut sc, mut cap, mut sq) = (nblk, heads, kv_heads, window, 1.0f32 / (HEAD_DIM as f32).sqrt(), capacity, seq.dptr());
        let mut p = [param(&mut pq), param(&mut qs), param(&mut pk), param(&mut pv), param(&mut po), param(&mut pp), param(&mut pi), param(&mut nb), param(&mut h), param(&mut kh), param(&mut win), param(&mut sc), param(&mut cap), param(&mut sq)];
        Self::go(&self.attn_b, s, LaunchConfig { grid: (nblk, kv_heads, blocks), block: (32 * 16, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `out[t] = sum alpha[i] * feat[i][rows[t]]` (4 taps `tap_stride` floats apart).
    #[allow(clippy::too_many_arguments)]
    pub fn mix4_b(&self, s: &Stream, feat: &dyn Dev<f32>, tap_stride: usize, rows_idx: &dyn Dev<i32>, alpha: [f32; 4], out: &mut dyn DevMut<f32>, rows: u32, dim: u32) -> Result<()> {
        need("mix4_b out", out.dlen(), (rows * dim) as usize)?;
        need("mix4_b rows", rows_idx.dlen(), rows as usize)?;
        let (mut pf, mut ts, mut pr, mut po, mut d) = (feat.dptr(), tap_stride as u64, rows_idx.dptr(), out.dptr(), dim);
        let [mut a0, mut a1, mut a2, mut a3] = alpha;
        let mut p = [param(&mut pf), param(&mut ts), param(&mut pr), param(&mut a0), param(&mut a1), param(&mut a2), param(&mut a3), param(&mut po), param(&mut d)];
        Self::go(&self.mix4_b, s, LaunchConfig { grid: (dim.div_ceil(256), rows, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// [`DsparkOps::markov_in`] for `b` sequences (state rows, tokens `prev[0..b]`, x rows of 2 rank).
    #[allow(clippy::too_many_arguments)]
    pub fn markov_in_b(&self, s: &Stream, state: DevicePtr, w1: &dyn Dev<f32>, prev: DevicePtr, vocab: u32, first: bool, x: DevicePtr, rank: u32, b: u32) -> Result<()> {
        let (mut ps, mut pw, mut pp, mut v, mut f, mut px) = (state, w1.dptr(), prev, vocab, first as u32, x);
        let mut p = [param(&mut ps), param(&mut pw), param(&mut pp), param(&mut v), param(&mut f), param(&mut px)];
        Self::go(&self.markov_in_b, s, LaunchConfig { grid: (b, 1, 1), block: (rank, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// [`DsparkOps::markov_gate`] for `b` sequences; `ch` rows `ch_stride` floats apart; the
    /// latents go to rows 0..b of `qa`.
    #[allow(clippy::too_many_arguments)]
    pub fn markov_gate_b(&self, s: &Stream, raw: DevicePtr, ch: DevicePtr, ch_stride: u32, cf: DevicePtr, bias: &dyn Dev<f32>, state: DevicePtr, first: bool, qa: &mut QAct, b: u32) -> Result<()> {
        let rank = qa.k;
        let (mut pq, mut pd, _) = qa.planes_for(b)?;
        let (mut pr, mut pc, mut cs, mut pf, mut pb, mut pst, mut f) = (raw, ch, ch_stride, cf, bias.dptr(), state, first as u32);
        let mut p = [param(&mut pr), param(&mut pc), param(&mut cs), param(&mut pf), param(&mut pb), param(&mut pst), param(&mut f), param(&mut pq), param(&mut pd)];
        Self::go(&self.markov_gate_b, s, LaunchConfig { grid: (b, 1, 1), block: (rank, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `y[b][j] += base[b * base_stride + j] + fb[j]` for `j < n`, `rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rows(&self, s: &Stream, y: DevicePtr, y_stride: u32, base: DevicePtr, base_stride: u32, fb: &dyn Dev<f32>, n: u32, rows: u32) -> Result<()> {
        let (mut py, mut ys, mut pb, mut bs, mut pf, mut nn) = (y, y_stride, base, base_stride, fb.dptr(), n);
        let mut p = [param(&mut py), param(&mut ys), param(&mut pb), param(&mut bs), param(&mut pf), param(&mut nn)];
        Self::go(&self.add_rows, s, LaunchConfig { grid: (n.div_ceil(256), rows, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    fn go(f: &Function, stream: &Stream, cfg: LaunchConfig, params: &mut [*mut std::ffi::c_void]) -> Result<()> {
        // SAFETY: every caller builds `params` to match the kernel's C signature in dspark.cu
        // and checks buffer sizes first; device-side indices are clamped by the kernels.
        unsafe { f.launch(cfg, stream, params) }
    }

    /// For `rows` rows of `dim`: `r = x + add` (stored to `res` when given), `y = rms_norm(r) * w`
    /// (stored when given) and quantised into `qa` when given. `x` and `res` may alias.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rms_q8(
        &self,
        s: &Stream,
        x: DevicePtr,
        x_len: usize,
        add: Option<&dyn Dev<f32>>,
        res: Option<DevicePtr>,
        w: &dyn Dev<f32>,
        y: Option<&mut dyn DevMut<f32>>,
        qa: Option<&mut QAct>,
        rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        let n = (rows * dim) as usize;
        if rows == 0 || !dim.is_multiple_of(32) {
            return Err(invalid(format!("add_rms_q8: {rows} rows of {dim}")));
        }
        need("add_rms_q8 x", x_len, n)?;
        need("add_rms_q8 w", w.dlen(), dim as usize)?;
        if let Some(a) = add {
            need("add_rms_q8 add", a.dlen(), n)?;
        }
        let py = match y {
            Some(y) => {
                need("add_rms_q8 y", y.dlen(), n)?;
                y.dptr()
            }
            None => 0,
        };
        let (mut pq, mut pd, mut ph) = match qa {
            Some(q) => {
                if q.k != dim {
                    return Err(invalid(format!("add_rms_q8: activations k {} != dim {dim}", q.k)));
                }
                q.planes_for(rows)?
            }
            None => (0, 0, 0),
        };
        let (mut px, mut pa, mut pr, mut pw, mut py, mut d, mut e) = (x, add.map_or(0, |a| a.dptr()), res.unwrap_or(0), w.dptr(), py, dim, eps);
        let mut p = [param(&mut px), param(&mut pa), param(&mut pr), param(&mut pw), param(&mut py), param(&mut pq), param(&mut pd), param(&mut ph), param(&mut d), param(&mut e)];
        let block = (dim / 4).next_multiple_of(32).clamp(64, 1024);
        Self::go(&self.add_rms_q8, s, LaunchConfig { grid: (rows, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Quantise `rows` rows of `qa.k` values (row `t` at `x[t * x_stride]`) into `qa` (with its
    /// f16 copy when allocated).
    pub fn quant(&self, s: &Stream, x: &dyn Dev<f32>, x_stride: u32, rows: u32, qa: &mut QAct) -> Result<()> {
        let dim = qa.k;
        if rows == 0 {
            return Err(invalid("quant: no rows".into()));
        }
        need("quant x", x.dlen(), (rows as usize - 1) * x_stride as usize + dim as usize)?;
        let (mut pq, mut pd, mut ph) = qa.planes_for(rows)?;
        let (mut px, mut xs, mut d) = (x.dptr(), x_stride, dim);
        let mut p = [param(&mut px), param(&mut xs), param(&mut pq), param(&mut pd), param(&mut ph), param(&mut d)];
        let block = (dim / 4).next_multiple_of(32).clamp(32, 1024);
        Self::go(&self.quant, s, LaunchConfig { grid: (rows, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `out[t] = sum alpha[i] * feat[i][row0 + t]` for 4 taps of `tap_stride` floats, `rows` rows
    /// of `dim`; `row0` from the device when given (else 0).
    #[allow(clippy::too_many_arguments)]
    pub fn mix4(&self, s: &Stream, feat: &dyn Dev<f32>, tap_stride: usize, row0: Option<DevIndex>, alpha: [f32; 4], out: &mut dyn DevMut<f32>, rows: u32, dim: u32) -> Result<()> {
        need("mix4 feat", feat.dlen(), 3 * tap_stride + (rows * dim) as usize)?;
        need("mix4 out", out.dlen(), (rows * dim) as usize)?;
        let rp = row0.map_or(0, |r| r.ptr + 4 * r.idx as u64);
        let (mut pf, mut ts, mut pr, mut po, mut d) = (feat.dptr(), tap_stride as u64, rp, out.dptr(), dim);
        let [mut a0, mut a1, mut a2, mut a3] = alpha;
        let mut p = [param(&mut pf), param(&mut ts), param(&mut pr), param(&mut a0), param(&mut a1), param(&mut a2), param(&mut a3), param(&mut po), param(&mut d)];
        Self::go(&self.mix4, s, LaunchConfig { grid: (dim.div_ceil(256), rows, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Per (token, head) of 128 values: optional RMSNorm (`w`), optional neox RoPE at
    /// `pos + t`, written to `dst` row `t` or, `to_cache`, row = position (of `capacity`).
    #[allow(clippy::too_many_arguments)]
    pub fn head_prep(
        &self,
        s: &Stream,
        src: DevicePtr,
        src_stride: u32,
        dst: &mut dyn DevMut<f32>,
        dst_stride: u32,
        w: Option<&dyn Dev<f32>>,
        rope: bool,
        pos: DevIndex,
        to_cache: bool,
        capacity: u32,
        ntok: u32,
        heads: u32,
        theta: f32,
        eps: f32,
    ) -> Result<()> {
        if ntok == 0 || heads == 0 || capacity == 0 || heads * HEAD_DIM > dst_stride {
            return Err(invalid(format!("head_prep: {ntok} tokens x {heads} heads, dst stride {dst_stride}")));
        }
        need("head_prep dst", dst.dlen(), (if to_cache { capacity } else { ntok }) as usize * dst_stride as usize)?;
        if let Some(w) = w {
            need("head_prep w", w.dlen(), HEAD_DIM as usize)?;
        }
        let (mut ps, mut ss, mut pd, mut ds, mut pw) = (src, src_stride, dst.dptr(), dst_stride, w.map_or(0, |w| w.dptr()));
        let (mut r, mut pp, mut pi, mut tc, mut cap, mut th, mut e) = (rope as u32, pos.ptr, pos.idx, to_cache as u32, capacity, theta, eps);
        let mut p = [
            param(&mut ps),
            param(&mut ss),
            param(&mut pd),
            param(&mut ds),
            param(&mut pw),
            param(&mut r),
            param(&mut pp),
            param(&mut pi),
            param(&mut tc),
            param(&mut cap),
            param(&mut th),
            param(&mut e),
        ];
        Self::go(&self.head_prep, s, LaunchConfig { grid: (ntok, heads, 1), block: (64, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Draft block attention: `nblk` queries (row stride `q_stride`) over cached keys
    /// `[L - window, L)`, `L = pos + nblk` (window 0 = all). `out` is `[nblk][heads * 128]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn(
        &self,
        s: &Stream,
        q: DevicePtr,
        q_stride: u32,
        kc: &dyn Dev<f32>,
        vc: &dyn Dev<f32>,
        out: &mut dyn DevMut<f32>,
        pos: DevIndex,
        nblk: u32,
        heads: u32,
        kv_heads: u32,
        window: u32,
        capacity: u32,
    ) -> Result<()> {
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) || heads / kv_heads > MAX_GROUP || nblk == 0 {
            return Err(invalid(format!("dspark attn: {heads} heads over {kv_heads} KV heads")));
        }
        let kvw = (kv_heads * HEAD_DIM) as usize;
        need("dspark attn kc", kc.dlen(), capacity as usize * kvw)?;
        need("dspark attn vc", vc.dlen(), capacity as usize * kvw)?;
        need("dspark attn out", out.dlen(), (nblk * heads * HEAD_DIM) as usize)?;
        let (mut pq, mut qs, mut pk, mut pv, mut po, mut pp, mut pi) = (q, q_stride, kc.dptr(), vc.dptr(), out.dptr(), pos.ptr, pos.idx);
        let (mut nb, mut h, mut kh, mut win, mut sc, mut cap) = (nblk, heads, kv_heads, window, 1.0f32 / (HEAD_DIM as f32).sqrt(), capacity);
        let mut p = [
            param(&mut pq),
            param(&mut qs),
            param(&mut pk),
            param(&mut pv),
            param(&mut po),
            param(&mut pp),
            param(&mut pi),
            param(&mut nb),
            param(&mut h),
            param(&mut kh),
            param(&mut win),
            param(&mut sc),
            param(&mut cap),
        ];
        Self::go(&self.attn, s, LaunchConfig { grid: (nblk, kv_heads, 1), block: (32 * 16, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `silu(gate) * up` of `rows` rows of `[gate (n_ff) | up (n_ff)]`, quantised into `qa`.
    pub fn swiglu_q8(&self, s: &Stream, gu: &dyn Dev<f32>, rows: u32, qa: &mut QAct) -> Result<()> {
        let n_ff = qa.k;
        need("swiglu gu", gu.dlen(), (2 * n_ff * rows) as usize)?;
        let (mut pq, mut pd, mut ph) = qa.planes_for(rows)?;
        let (mut pg, mut n) = (gu.dptr(), n_ff);
        let mut p = [param(&mut pg), param(&mut n), param(&mut pq), param(&mut pd), param(&mut ph)];
        Self::go(&self.swiglu_q8, s, LaunchConfig { grid: (rows, 1, 1), block: (1024, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `x = [state | w1[prev]]` (`rank` each; state as 0 when `first`).
    #[allow(clippy::too_many_arguments)]
    pub fn markov_in(&self, s: &Stream, state: &dyn Dev<f32>, w1: &dyn Dev<f32>, prev: DevIndex, vocab: u32, first: bool, x: &mut dyn DevMut<f32>, rank: u32) -> Result<()> {
        need("markov_in state", state.dlen(), rank as usize)?;
        need("markov_in w1", w1.dlen(), (vocab * rank) as usize)?;
        need("markov_in x", x.dlen(), 2 * rank as usize)?;
        let (mut ps, mut pw, mut pp, mut v, mut f, mut px) = (state.dptr(), w1.dptr(), prev.ptr + 4 * prev.idx as u64, vocab, first as u32, x.dptr());
        let mut p = [param(&mut ps), param(&mut pw), param(&mut pp), param(&mut v), param(&mut f), param(&mut px)];
        Self::go(&self.markov_in, s, LaunchConfig { grid: (1, 1, 1), block: (rank, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// The recurrent head's gate: `raw + ch + cf + bias` → new `state`, `tanh(out)` into `qa`.
    #[allow(clippy::too_many_arguments)]
    pub fn markov_gate(&self, s: &Stream, raw: &dyn Dev<f32>, ch: &dyn Dev<f32>, cf: &dyn Dev<f32>, bias: &dyn Dev<f32>, state: &mut dyn DevMut<f32>, first: bool, qa: &mut QAct) -> Result<()> {
        let rank = qa.k;
        for (what, b) in [("raw", raw), ("ch", ch), ("cf", cf), ("bias", bias)] {
            need(what, b.dlen(), 3 * rank as usize)?;
        }
        need("markov_gate state", state.dlen(), rank as usize)?;
        let (mut pq, mut pd, _) = qa.planes_for(1)?;
        let (mut pr, mut pc, mut pf, mut pb, mut ps, mut fi) = (raw.dptr(), ch.dptr(), cf.dptr(), bias.dptr(), state.dptr(), first as u32);
        let mut p = [param(&mut pr), param(&mut pc), param(&mut pf), param(&mut pb), param(&mut ps), param(&mut fi), param(&mut pq), param(&mut pd)];
        Self::go(&self.markov_gate, s, LaunchConfig { grid: (1, 1, 1), block: (rank, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `out[0] = argmax(y + base + fb)` over the first `n` values.
    pub fn add_argmax(&self, s: &Stream, y: &dyn Dev<f32>, base: &dyn Dev<f32>, fb: &dyn Dev<f32>, n: u32, out: DevIndex) -> Result<()> {
        for (what, b) in [("y", y), ("base", base), ("fb", fb)] {
            need(what, b.dlen(), n as usize)?;
        }
        let (mut py, mut pb, mut pf, mut nn, mut po) = (y.dptr(), base.dptr(), fb.dptr(), n, out.ptr + 4 * out.idx as u64);
        let mut p = [param(&mut py), param(&mut pb), param(&mut pf), param(&mut nn), param(&mut po)];
        Self::go(&self.add_argmax, s, LaunchConfig { grid: (1, 1, 1), block: (1024, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Greedy acceptance of `k` drafts (`token[1..=k]`) against the target's picks `next_all`;
    /// see `astral_dsp_accept`.
    /// `forced` (diagnostics): accept against a fixed token sequence by position instead.
    #[allow(clippy::too_many_arguments)]
    pub fn accept(
        &self,
        s: &Stream,
        next_all: &dyn Dev<i32>,
        token: &mut dyn DevMut<i32>,
        k: u32,
        out: &mut dyn DevMut<i32>,
        info: &mut dyn DevMut<i32>,
        pos: &mut dyn DevMut<i32>,
        forced: Option<&dyn Dev<i32>>,
    ) -> Result<()> {
        need("accept next_all", next_all.dlen(), k as usize + 1)?;
        need("accept token", token.dlen(), k as usize + 1)?;
        need("accept out", out.dlen(), k as usize + 2)?;
        need("accept info", info.dlen(), 3)?;
        need("accept pos", pos.dlen(), 1)?;
        let (mut pn, mut pt, mut kk, mut po, mut pi, mut pp, mut pf) = (next_all.dptr(), token.dptr(), k, out.dptr(), info.dptr(), pos.dptr(), forced.map_or(0, |f| f.dptr()));
        let mut p = [param(&mut pn), param(&mut pt), param(&mut kk), param(&mut po), param(&mut pi), param(&mut pp), param(&mut pf)];
        Self::go(&self.accept, s, LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Short-conv rollback over a table of `(state, hist, channels)` u64 triples (`entries` of
    /// them, at most `max_channels` channels): keep the inputs of the `info[1] + 1` accepted
    /// tokens. The caller guarantees every pointer in the table stays valid.
    pub fn conv_commit(&self, s: &Stream, table: &dyn Dev<u64>, entries: u32, max_channels: u32, info: &dyn Dev<i32>) -> Result<()> {
        need("conv_commit table", table.dlen(), 3 * entries as usize)?;
        need("conv_commit info", info.dlen(), 2)?;
        if entries == 0 {
            return Ok(());
        }
        let (mut pt, mut pi) = (table.dptr(), info.dptr());
        let mut p = [param(&mut pt), param(&mut pi)];
        Self::go(&self.conv_commit, s, LaunchConfig { grid: (max_channels.div_ceil(256), entries, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `out[r] = argmax(x[r][..n])` for `rows` rows of stride `x_stride` (multi-block; `part_*`
    /// hold `rows * ARGMAX_PARTS` partials).
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_rows(&self, s: &Stream, x: &dyn Dev<f32>, x_stride: u32, n: u32, rows: u32, part_v: &mut dyn DevMut<f32>, part_i: &mut dyn DevMut<u32>, out: &mut dyn DevMut<i32>) -> Result<()> {
        need("argmax_rows x", x.dlen(), (rows as usize - 1) * x_stride as usize + n as usize)?;
        need("argmax_rows parts", part_v.dlen().min(part_i.dlen()), (rows * ARGMAX_PARTS) as usize)?;
        need("argmax_rows out", out.dlen(), rows as usize)?;
        let (mut px, mut xs, mut nn, mut pv, mut pi) = (x.dptr(), x_stride, n, part_v.dptr(), part_i.dptr());
        let mut p = [param(&mut px), param(&mut xs), param(&mut nn), param(&mut pv), param(&mut pi)];
        Self::go(&self.rowmax_part, s, LaunchConfig { grid: (ARGMAX_PARTS, rows, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        self.fin(s, part_v.dptr(), part_i.dptr(), rows, out.dptr(), 1)
    }

    fn fin(&self, s: &Stream, pv: DevicePtr, pi: DevicePtr, rows: u32, out: DevicePtr, out_stride: u32) -> Result<()> {
        let (mut a, mut b, mut np, mut o, mut os) = (pv, pi, ARGMAX_PARTS, out, out_stride);
        let mut p = [param(&mut a), param(&mut b), param(&mut np), param(&mut o), param(&mut os)];
        Self::go(&self.argmax_fin, s, LaunchConfig { grid: (rows, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// The Markov head's greedy pick: `out = argmax_i(fold[i] . latent + base[i] + fb[i])` over
    /// `i < n`, `fold` a Q8_0 matrix with k = 256 and `latent` its quantised input.
    #[allow(clippy::too_many_arguments)]
    pub fn fold_argmax(
        &self,
        s: &Stream,
        fold: &crate::ops::GemvMatrix,
        latent: &QAct,
        base: &dyn Dev<f32>,
        fb: &dyn Dev<f32>,
        n: u32,
        part_v: &mut dyn DevMut<f32>,
        part_i: &mut dyn DevMut<u32>,
        out: DevIndex,
    ) -> Result<()> {
        if fold.ty != crate::ops::GemvType::Q8_0 || fold.k != 256 || fold.n_mats != 1 || latent.k != 256 || n > fold.n {
            return Err(invalid(format!("fold_argmax: {:?} n {} k {} (latent k {}), n {n}", fold.ty, fold.n, fold.k, latent.k)));
        }
        need("fold_argmax base", base.dlen(), n as usize)?;
        need("fold_argmax fb", fb.dlen(), n as usize)?;
        need("fold_argmax parts", part_v.dlen().min(part_i.dlen()), ARGMAX_PARTS as usize)?;
        let (lq, ld) = latent.planes();
        let (mut pq, mut pd, mut a, mut b, mut pb, mut pf, mut nn, mut pv, mut pi) = (fold.qs.device_ptr(), fold.aux.device_ptr(), lq, ld, base.dptr(), fb.dptr(), n, part_v.dptr(), part_i.dptr());
        let mut p = [param(&mut pq), param(&mut pd), param(&mut a), param(&mut b), param(&mut pb), param(&mut pf), param(&mut nn), param(&mut pv), param(&mut pi)];
        Self::go(&self.fold_part, s, LaunchConfig { grid: (ARGMAX_PARTS, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        self.fin(s, part_v.dptr(), part_i.dptr(), 1, out.ptr + 4 * out.idx as u64, 1)
    }
}
