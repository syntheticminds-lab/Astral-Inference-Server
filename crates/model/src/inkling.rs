// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Inkling 2 Mini on the GPU (spec 06), P1 correctness-first decode: one token per step.
//!
//! The forward pass follows llama.cpp's `inkling2` / `inkling2_loop` graphs (MIT) step for step:
//! 24 stored layers run as 48 cache slots (layers 6..18 three times, each pass restarting from
//! the prelude output and steered by loop "messages"), with per-slot KV and short-conv state.

use crate::dspark::{hist_offsets, DraftWeights, Spec, SpecStats};
use astral_core::{AstralError, Result};
use astral_gpu::{Context, Dev, DevMut, DeviceBuffer, GraphExec, Stream};
use astral_kernels::dspark::{DevIndex, DsparkOps};
use astral_kernels::inkling::{AttnPrep, AttnShape, InklingOps, SeqRows};
use astral_kernels::ops::{Gemv, GemvMatrix, GemvType, MmaWork, MultiItem, QAct, QItem, RmsNormF32, QITEM_ROWS};
use astral_kernels::KernelRegistry;
use astral_loader::{Gguf, TensorInfo};
use astral_quant::GgmlType;
use std::sync::Arc;

fn bad(m: String) -> AstralError {
    AstralError::Invalid(format!("inkling: {m}"))
}

/// Model hyper-parameters, read from the GGUF.
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub vocab_padded: usize,
    pub vocab: usize,
    pub core_start: usize,
    pub core_end: usize,
    pub passes: usize,
    pub swa: Vec<bool>,
    pub n_swa: u32,
    pub n_rel: u32,
    pub rel_extent: u32,
    pub rms_eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub expert_scale: f32,
    pub softcap: f32,
    pub attn_res_block: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        if g.architecture() != Some("inkling2_loop") {
            return Err(bad(format!("architecture {:?}, expected inkling2_loop", g.architecture())));
        }
        let u = |k: &str| g.arch_key(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| bad(format!("missing key {k}")));
        let f = |k: &str| g.arch_key(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| bad(format!("missing key {k}")));
        let first = |k: &str| -> Result<usize> {
            let v = g.arch_key(k).ok_or_else(|| bad(format!("missing key {k}")))?;
            v.as_u64().or_else(|| v.as_array().and_then(|a| a.first()).and_then(|x| x.as_u64())).map(|x| x as usize).ok_or_else(|| bad(format!("bad key {k}")))
        };
        let swa: Vec<bool> = g
            .arch_key("attention.sliding_window_pattern")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad("missing sliding_window_pattern".into()))?
            .iter()
            .map(|v| v.as_bool().unwrap_or(false))
            .collect();
        let c = Self {
            n_layer: u("block_count")?,
            dim: u("embedding_length")?,
            heads: first("attention.head_count")?,
            kv_heads: first("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            vocab_padded: u("vocab_size")?,
            vocab: u("vocab_size_unpadded")?,
            core_start: u("loop.start")?,
            core_end: u("loop.end")?,
            passes: u("loop.count")?,
            swa,
            n_swa: u("attention.sliding_window")? as u32,
            n_rel: u("attention.relative_length")? as u32,
            rel_extent: u("attention.relative_extent")? as u32,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            qk_clip: f("attention.qk_clip")?,
            rel_clip: f("attention.relative_clip")?,
            log_floor: f("attention.log_scale_floor")?,
            log_alpha: f("attention.log_scale_alpha")?,
            situ_beta: f("activation.situ_beta")?,
            situ_lb: f("activation.situ_linear_beta")?,
            expert_scale: g.arch_key("expert_weights_scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            softcap: g.arch_key("final_logit_softcapping").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            attn_res_block: u("attn_res.block_size")?,
        };
        if c.swa.len() != c.n_layer || c.core_end > c.n_layer || c.core_start >= c.core_end || c.passes < 1 {
            return Err(bad("inconsistent layer / loop configuration".into()));
        }
        if c.n_swa != c.rel_extent {
            return Err(bad("P1 assumes the SWA window equals the relative extent".into()));
        }
        Ok(c)
    }

    /// Cache slots: n_layer + (passes - 1) * core length.
    pub fn n_slots(&self) -> usize {
        self.n_layer + (self.passes - 1) * (self.core_end - self.core_start)
    }

    fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Floats of short-conv state per cache slot and sequence (last 3 inputs per channel).
    fn conv_len(&self) -> usize {
        3 * self.conv_channels()
    }

    /// Short-conv channels: K, V, attention output, FFN output.
    fn conv_channels(&self) -> usize {
        2 * self.kv_width() + 2 * self.dim
    }
}

struct Layer {
    swa: bool,
    wq: GemvMatrix,
    wk: GemvMatrix,
    wv: GemvMatrix,
    wo: GemvMatrix,
    wgate: GemvMatrix,
    wrel: GemvMatrix,
    sink: GemvMatrix,
    k_conv_proj: Option<GemvMatrix>,
    v_conv_proj: Option<GemvMatrix>,
    /// Q8_0 copies of the three above, fused into the decode / small-batch projection launch.
    sink_q8: GemvMatrix,
    kp_q8: Option<GemvMatrix>,
    vp_q8: Option<GemvMatrix>,
    /// F32 copies of the same three, for the prefill GEMM.
    sink_f32: GemvMatrix,
    kp_f32: Option<GemvMatrix>,
    vp_f32: Option<GemvMatrix>,
    q_clip: DeviceBuffer<f32>,
    rel_bias: DeviceBuffer<f32>,
    rel_clip: DeviceBuffer<f32>,
    k_conv: DeviceBuffer<f32>,
    v_conv: DeviceBuffer<f32>,
    attn_post_conv: DeviceBuffer<f32>,
    ffn_post_conv: DeviceBuffer<f32>,
    attn_norm: Option<DeviceBuffer<f32>>,
    qkv_inp_norm: Option<[DeviceBuffer<f32>; 3]>,
    attn_res_query: Option<DeviceBuffer<f32>>,
    ffn_res_query: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    router: GemvMatrix,
    probs_b: Option<DeviceBuffer<f32>>,
    gate_up: GemvMatrix,
    down: GemvMatrix,
}

struct LoopLayer {
    msg_query: DeviceBuffer<f32>,
    down: [GemvMatrix; 4],
    /// Gain rows then forget rows (`[2 * dim, rank]`): one GEMV gives both.
    gain_forget: [GemvMatrix; 4],
}

/// All weights of the model on the GPU.
pub struct Weights {
    pub cfg: Config,
    pub(crate) tok_embd: GemvMatrix,
    pub(crate) tok_norm: DeviceBuffer<f32>,
    pub(crate) output: GemvMatrix,
    output_norm: DeviceBuffer<f32>,
    output_res_query: DeviceBuffer<f32>,
    layers: Vec<Layer>,
    loops: Vec<LoopLayer>,
    pub bytes: usize,
}

/// Reorders the rows of each `[n][..]` matrix of `raw` (gate rows then up rows, `n / 2` each)
/// to the interleaved order: row `16 b + i` is gate row `8 b + i`, row `16 b + 8 + i` up row
/// `8 b + i`. Rows are whole quantisation blocks, so this is type-agnostic.
/// The two-block-cluster dense prefill GEMM (`ASTRAL_DENSE_MC=1`; measured no faster).
fn dense_mc() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ASTRAL_DENSE_MC").is_ok_and(|v| v == "1"))
}

/// The persistent dense prefill GEMM (`ASTRAL_DENSE_P=0` turns it off).
fn dense_p() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ASTRAL_DENSE_P").map_or(true, |v| v != "0"))
}

/// The expert activation in the gate_up GEMM's epilogue (`ASTRAL_FUSE_SITU=0` turns it off).
fn fuse_situ() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ASTRAL_FUSE_SITU").map_or(true, |v| v != "0"))
}

fn interleave_gate_up(raw: &mut [u8], n: usize, mats: usize) -> std::result::Result<(), String> {
    let total = n * mats;
    if !n.is_multiple_of(16) || total == 0 || !raw.len().is_multiple_of(total) {
        return Err(format!("gate_up rows {n} x {mats} do not interleave"));
    }
    let row = raw.len() / total;
    let half = n / 2;
    let mut tmp = vec![0u8; n * row];
    for m in raw.chunks_exact_mut(n * row) {
        for r in 0..n {
            let (b, i) = (r / 16, r % 16);
            let src = if i < 8 { 8 * b + i } else { half + 8 * b + i - 8 };
            tmp[r * row..(r + 1) * row].copy_from_slice(&m[src * row..(src + 1) * row]);
        }
        m.copy_from_slice(&tmp);
    }
    Ok(())
}

struct Loader<'a> {
    g: &'a Gguf,
    ctx: &'a Arc<Context>,
    bytes: usize,
}

impl Loader<'_> {
    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.g.tensor(name).ok_or_else(|| bad(format!("missing tensor {name}")))
    }

    /// A weight matrix (`[k, n]` or `[k, n, mats]` in GGML order) in GEMV layout.
    fn mat(&mut self, name: &str) -> Result<GemvMatrix> {
        let t = self.info(name)?.clone();
        let (k, n, mats) = match t.dims.as_slice() {
            [k, n] => (*k, *n, 1),
            [k, n, m] => (*k, *n, *m),
            d => return Err(bad(format!("{name}: unexpected dims {d:?}"))),
        };
        let mut raw = self.g.read_tensor(&t)?;
        // Expert gate_up: rows interleaved by eight (row 16 b + i is gate row 8 b + i, row
        // 16 b + 8 + i its up row), so one 16-row MMA tile holds both halves of eight channels
        // and the activation can be applied in the GEMM epilogue.
        if name.contains("ffn_gate_up_exps") {
            interleave_gate_up(&mut raw, n as usize, mats as usize).map_err(|e| bad(format!("{name}: {e}")))?;
        }
        let (ty, qs, aux) = match t.ty {
            GgmlType::F32 => (GemvType::F32, raw, Vec::new()),
            GgmlType::Bf16 => (GemvType::Bf16, raw, Vec::new()),
            GgmlType::Q8_0 => {
                let (q, a) = astral_quant::repack_q8_0(&raw)?;
                (GemvType::Q8_0, q, a)
            }
            GgmlType::Nvfp4 => {
                let (q, a) = astral_quant::repack_nvfp4(&raw)?;
                (GemvType::Nvfp4, q, a)
            }
            other => return Err(bad(format!("{name}: type {} not supported by the P1 GEMV", other.name()))),
        };
        let scale_name = name.strip_suffix(".weight").map(|b| format!("{b}.scale"));
        let scale = match scale_name.as_deref().and_then(|s| self.g.tensor(s)) {
            Some(st) => Some(astral_quant::dequantize(st.ty, &self.g.read_tensor(st)?, st.n_elements as usize)?),
            None => None,
        };
        let mut m = GemvMatrix::upload(self.ctx, ty, n as u32, k as u32, mats as u32, &qs, &aux, scale.as_deref())?;
        // Experts: the prefill fragment layout for the warpgroup-MMA GEMM (a second copy of the
        // weights; ASTRAL_NO_FRAG=1 to skip it and use the mma.sync prefill kernels).
        if ty == GemvType::Nvfp4 && mats > 1 && n % 128 == 0 && k % 64 == 0 && std::env::var_os("ASTRAL_NO_FRAG").is_none() {
            m.add_frag(self.ctx, &qs, &aux)?;
        }
        // Q8_0 experts (MXFP8 checkpoints) on GPUs with room for an f16 copy (about 2 GB per
        // layer): the TMA prefill GEMM instead of the mma.sync one. ASTRAL_EXPERT_F16=0/1 forces.
        if ty == GemvType::Q8_0 && mats > 1 && n % 128 == 0 && k % 64 == 0 {
            let hopper = self.ctx.device().compute_capability().map(|(m, _)| m == 9).unwrap_or(false) && astral_kernels::ops::has_sm90a_build();
            let big = self.ctx.mem_info().map(|(_, total)| total >= 90usize << 30).unwrap_or(false);
            let want = match std::env::var("ASTRAL_EXPERT_F16").ok().as_deref() {
                Some("0") => false,
                Some(_) => true,
                None => big,
            };
            if hopper && want {
                m.add_w16(self.ctx, &qs, &aux)?;
            }
        }
        // Dense Q8_0 projections: an f16 copy for the prefill GEMMs -- row-major with a TMA tensor
        // map on Hopper (ASTRAL_NO_TMA=1: the fragment-order copy for the older kernel).
        if ty == GemvType::Q8_0 && mats == 1 && n % 128 == 0 && k % 64 == 0 && std::env::var_os("ASTRAL_NO_FRAG").is_none() {
            let hopper = self.ctx.device().compute_capability().map(|(m, _)| m == 9).unwrap_or(false) && astral_kernels::ops::has_sm90a_build();
            if hopper && std::env::var_os("ASTRAL_NO_TMA").is_none() {
                m.add_w16(self.ctx, &qs, &aux)?;
            } else {
                m.add_frag16(self.ctx, &qs, &aux)?;
            }
        }
        self.bytes += m.bytes();
        Ok(m)
    }

    /// A `[k, n]` matrix of any dequantisable type, requantised to Q8_0 (fused decode GEMV).
    fn mat_q8(&mut self, name: &str) -> Result<GemvMatrix> {
        let t = self.info(name)?.clone();
        let [k, n] = t.dims.as_slice() else { return Err(bad(format!("{name}: expected a matrix"))) };
        let v = astral_quant::dequantize(t.ty, &self.g.read_tensor(&t)?, t.n_elements as usize)?;
        let (qs, aux) = astral_quant::quantize_q8_0_rows(&v, *n as usize, *k as usize)?;
        let m = GemvMatrix::upload(self.ctx, GemvType::Q8_0, *n as u32, *k as u32, 1, &qs, &aux, None)?;
        self.bytes += m.bytes();
        Ok(m)
    }

    /// Matrices with the same `k`, stacked by rows into one F32 matrix.
    fn stacked_f32(&mut self, names: &[&str]) -> Result<GemvMatrix> {
        let (mut rows, mut k0, mut data) = (0usize, None, Vec::new());
        for name in names {
            let t = self.info(name)?.clone();
            let [k, n] = t.dims.as_slice() else { return Err(bad(format!("{name}: expected a matrix"))) };
            if k0.is_some_and(|k0| k0 != *k) {
                return Err(bad(format!("{name}: k {k} differs from the stack")));
            }
            k0 = Some(*k);
            rows += *n as usize;
            let v = astral_quant::dequantize(t.ty, &self.g.read_tensor(&t)?, t.n_elements as usize)?;
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        let k = k0.ok_or_else(|| bad("empty stack".into()))?;
        let m = GemvMatrix::upload(self.ctx, GemvType::F32, rows as u32, k as u32, 1, &data, &[], None)?;
        self.bytes += m.bytes();
        Ok(m)
    }

    /// A small tensor as f32 on the GPU.
    fn vec(&mut self, name: &str) -> Result<DeviceBuffer<f32>> {
        let t = self.info(name)?.clone();
        let v = astral_quant::dequantize(t.ty, &self.g.read_tensor(&t)?, t.n_elements as usize)?;
        self.bytes += v.len() * 4;
        DeviceBuffer::from_slice(self.ctx, &v)
    }

    fn opt_vec(&mut self, name: &str) -> Result<Option<DeviceBuffer<f32>>> {
        if self.g.tensor(name).is_some() {
            self.vec(name).map(Some)
        } else {
            Ok(None)
        }
    }
}

impl Weights {
    pub fn load(ctx: &Arc<Context>, g: &Gguf) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let mut l = Loader { g, ctx, bytes: 0 };
        let tok_embd = l.mat("token_embd.weight")?;
        let tok_norm = l.vec("token_embd_norm.weight")?;
        let output = l.mat("output.weight")?;
        let output_norm = l.vec("output_norm.weight")?;
        let output_res_query = l.vec("output_res_query.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{i}.{s}");
            let swa = cfg.swa[i];
            layers.push(Layer {
                swa,
                wq: l.mat(&b("attn_q.weight"))?,
                wk: l.mat(&b("attn_k.weight"))?,
                wv: l.mat(&b("attn_v.weight"))?,
                wo: l.mat(&b("attn_output.weight"))?,
                wgate: l.mat(&b("attn_gate.weight"))?,
                wrel: l.mat(&b("attn_rel.weight"))?,
                sink: l.mat(&b("attn_sink_proj.weight"))?,
                k_conv_proj: if swa { None } else { Some(l.mat(&b("attn_k_conv_proj.weight"))?) },
                v_conv_proj: if swa { None } else { Some(l.mat(&b("attn_v_conv_proj.weight"))?) },
                sink_q8: l.mat_q8(&b("attn_sink_proj.weight"))?,
                kp_q8: if swa { None } else { Some(l.mat_q8(&b("attn_k_conv_proj.weight"))?) },
                vp_q8: if swa { None } else { Some(l.mat_q8(&b("attn_v_conv_proj.weight"))?) },
                sink_f32: l.stacked_f32(&[&b("attn_sink_proj.weight")])?,
                kp_f32: if swa { None } else { Some(l.stacked_f32(&[&b("attn_k_conv_proj.weight")])?) },
                vp_f32: if swa { None } else { Some(l.stacked_f32(&[&b("attn_v_conv_proj.weight")])?) },
                q_clip: l.vec(&b("attn_q_clip.weight"))?,
                rel_bias: l.vec(&b("attn_rel_bias.weight"))?,
                rel_clip: l.vec(&b("attn_rel_clip.weight"))?,
                k_conv: l.vec(&b("attn_k_conv.weight"))?,
                v_conv: l.vec(&b("attn_v_conv.weight"))?,
                attn_post_conv: l.vec(&b("attn_post_conv.weight"))?,
                ffn_post_conv: l.vec(&b("ffn_post_conv.weight"))?,
                attn_norm: if swa { Some(l.vec(&b("attn_norm.weight"))?) } else { None },
                qkv_inp_norm: if swa { None } else { Some([l.vec(&b("attn_q_inp_norm.weight"))?, l.vec(&b("attn_k_inp_norm.weight"))?, l.vec(&b("attn_v_inp_norm.weight"))?]) },
                attn_res_query: l.opt_vec(&b("attn_res_query.weight"))?,
                ffn_res_query: l.vec(&b("ffn_res_query.weight"))?,
                ffn_norm: l.vec(&b("ffn_norm.weight"))?,
                router: l.mat(&b("ffn_gate_inp.weight"))?,
                probs_b: l.opt_vec(&b("exp_probs_b.bias"))?,
                gate_up: l.mat(&b("ffn_gate_up_exps.weight"))?,
                down: l.mat(&b("ffn_down_exps.weight"))?,
            });
        }
        let mut loops = Vec::new();
        for i in cfg.core_start..cfg.core_end {
            let b = |s: &str| format!("blk.{i}.{s}");
            let mut m = |kind: &str| -> Result<[GemvMatrix; 4]> {
                Ok([
                    l.mat(&b(&format!("loop_q_ctrl_{kind}.weight")))?,
                    l.mat(&b(&format!("loop_k_ctrl_{kind}.weight")))?,
                    l.mat(&b(&format!("loop_v_ctrl_{kind}.weight")))?,
                    l.mat(&b(&format!("loop_ffn_ctrl_{kind}.weight")))?,
                ])
            };
            let down = m("down")?;
            let mut gf = Vec::with_capacity(4);
            for c in ["q", "k", "v", "ffn"] {
                gf.push(l.stacked_f32(&[&b(&format!("loop_{c}_ctrl_gain.weight")), &b(&format!("loop_{c}_ctrl_forget.weight"))])?);
            }
            let gain_forget: [GemvMatrix; 4] = gf.try_into().map_err(|_| bad("loop controllers".into()))?;
            loops.push(LoopLayer { msg_query: l.vec(&b("loop_msg_query.weight"))?, down, gain_forget });
        }
        Ok(Self { cfg, tok_embd, tok_norm, output, output_norm, output_res_query, layers, loops, bytes: l.bytes })
    }
}

/// Most prompt tokens one forward pass takes (prefill chunk).
pub const PREFILL_TMAX: usize = 2048;
/// Upper bound on rows per forward pass (`ASTRAL_TMAX`).
pub const MAX_PASS_ROWS: usize = 8192;

/// Scratch buffers for up to `tmax` tokens per forward pass, all token-major (decode uses row
/// 0). The attention-residual bank holds `BANK_ROWS` rows of `[tmax][dim]`.
struct Scratch {
    tmax: usize,
    cur: DeviceBuffer<f32>,
    inp_sa: DeviceBuffer<f32>,
    mixed: DeviceBuffer<f32>,
    inputs: [DeviceBuffer<f32>; 3],
    normed: [DeviceBuffer<f32>; 3],
    msgs: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    kdyn: DeviceBuffer<f32>,
    vdyn: DeviceBuffer<f32>,
    r: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    attn_work: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    m: DeviceBuffer<f32>,
    ctmp: DeviceBuffer<f32>,
    mn: DeviceBuffer<f32>,
    rlogits: DeviceBuffer<f32>,
    ids: DeviceBuffer<i32>,
    zero_ids: DeviceBuffer<i32>,
    /// Per-token completion counters of the fused router (always zero between launches).
    router_cnt: DeviceBuffer<u32>,
    w: DeviceBuffer<f32>,
    gu: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    cd: DeviceBuffer<f32>,
    crms: DeviceBuffer<f32>,
    bank: DeviceBuffer<f32>,
    prelude: DeviceBuffer<f32>,
    bank_out: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    token: DeviceBuffer<i32>,
    pos: DeviceBuffer<i32>,
    next: DeviceBuffer<i32>,
    qa: [QAct; 3],
    qa_attn: QAct,
    qa_ff: QAct,
    qa_last: QAct,
    /// Prefill work items: dense rows, MoE gate_up and MoE down.
    items_dense: DeviceBuffer<u32>,
    items_gu: DeviceBuffer<u32>,
    items_dn: DeviceBuffer<u32>,
    /// Tensor-core prefill work lists: dense rows, MoE gate_up and MoE down.
    work: [DevWork; 3],
    /// The dense work list in items of up to 128 rows (warpgroup-MMA prefill GEMMs).
    work128: DevWork,
    /// The dense work list in items of up to 256 rows (TMA prefill GEMMs).
    work256: DevWork,
    /// Layer intermediates (name, values), collected on decode steps when set (verification).
    dbg: Option<Vec<(&'static str, Vec<f32>)>>,
}

/// A [`MmaWork`] on the device (fixed-size buffers, rewritten per use).
pub(crate) struct DevWork {
    pub(crate) items: DeviceBuffer<u32>,
    pub(crate) xidx: DeviceBuffer<u32>,
    pub(crate) yidx: DeviceBuffer<u32>,
    pub(crate) n_items: u32,
    pub(crate) idx_len: u32,
}

impl DevWork {
    pub(crate) fn new(ctx: &Arc<Context>, max_items: usize, max_idx: usize) -> Result<Self> {
        Ok(Self { items: DeviceBuffer::zeroed(ctx, 3 * max_items)?, xidx: DeviceBuffer::zeroed(ctx, max_idx)?, yidx: DeviceBuffer::zeroed(ctx, max_idx)?, n_items: 0, idx_len: 0 })
    }

    /// Synchronous upload; the caller guarantees no kernel is reading the buffers.
    pub(crate) fn upload(&mut self, w: &MmaWork) -> Result<()> {
        let put = |buf: &mut DeviceBuffer<u32>, v: &[u32]| -> Result<()> {
            if v.len() > buf.len() {
                return Err(bad("prefill: work list too long".into()));
            }
            let mut p = v.to_vec();
            p.resize(buf.len(), 0);
            buf.copy_from_slice(&p)
        };
        put(&mut self.items, &w.items)?;
        put(&mut self.xidx, &w.xidx)?;
        put(&mut self.yidx, &w.yidx)?;
        self.n_items = w.n_items();
        self.idx_len = w.xidx.len() as u32;
        Ok(())
    }
}

/// Rows of the attention-residual bank: prelude + 2 per pass + postlude, with headroom.
const BANK_ROWS: usize = 15;

/// Kernels and the stream, borrowed together with the scratch buffers.
struct Exec<'a> {
    st: &'a Stream,
    gemv: &'a Gemv,
    ops: &'a InklingOps,
    rms: &'a RmsNormF32,
}

impl Exec<'_> {
    /// `y = W x` for `ntok` quantised rows in `qa`: the int8 GEMV for one token, the grouped
    /// GEMM over the dense items for more.
    fn mm(&self, m: &GemvMatrix, qa: &QAct, y: &mut DeviceBuffer<f32>, ntok: usize, dense: &DeviceBuffer<u32>, work: &DevWork) -> Result<()> {
        if ntok == 1 {
            self.gemv.launch_q(self.st, m, qa, y, None)
        } else if small_batch(ntok) {
            // Every weight row read once for all rows (verify passes): the fused T-token GEMV,
            // else one grouped item.
            let mut it = [MultiItem { m, x: qa, y }];
            if Gemv::multi_t_ok(&it, ntok as u32) {
                return self.gemv.launch_q_multi_t(self.st, &mut it, ntok as u32);
            }
            let [MultiItem { y, .. }] = it;
            self.gemv.launch_items(self.st, m, qa, ntok as u32, y, ntok as u32, dense, 1)
        } else if use_mma() && m.n.is_multiple_of(16) {
            let n = ntok as u32;
            self.gemv.launch_mma(self.st, m, qa, n, y, n, &work.items, work.n_items, &work.xidx, &work.yidx, work.idx_len)
        } else {
            let items = ntok.div_ceil(QITEM_ROWS) as u32;
            self.gemv.launch_items(self.st, m, qa, ntok as u32, y, ntok as u32, dense, items)
        }
    }

    /// Decode-time dense Q8_0 product of `ntok` rows (single token or batched decode): the dp4a
    /// GEMV for one token, and for several the rows kernels that compute every row exactly as
    /// that GEMV does (so batched results do not depend on the batch).
    fn mm_decode(&self, m: &GemvMatrix, qa: &QAct, y: &mut dyn DevMut<f32>, ntok: usize) -> Result<()> {
        if ntok == 1 {
            return self.gemv.launch_q(self.st, m, qa, y, None);
        }
        self.gemv.launch_q_rows(self.st, m, qa, y, ntok as u32)
    }

    /// [`Exec::mm`], or for prefill passes (`prefill`) always the prefill GEMMs, whatever the
    /// token count.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn mm_any(&self, m: &GemvMatrix, qa: &QAct, y: &mut DeviceBuffer<f32>, ntok: usize, dense: &DeviceBuffer<u32>, work: &DevWork, work128: &DevWork, prefill: bool) -> Result<()> {
        if !prefill {
            return self.mm(m, qa, y, ntok, dense, work);
        }
        if m.frag16.is_some() && Gemv::has_wgmma(self.st) {
            let n = ntok as u32;
            return self.gemv.launch_wgmma_f16(self.st, &mut [MultiItem { m, x: qa, y }], n, n, &work128.items, work128.n_items, &work128.xidx, &work128.yidx, work128.idx_len);
        }
        if use_mma() && m.n.is_multiple_of(16) {
            let n = ntok as u32;
            self.gemv.launch_mma(self.st, m, qa, n, y, n, &work.items, work.n_items, &work.xidx, &work.yidx, work.idx_len)
        } else {
            let items = ntok.div_ceil(QITEM_ROWS) as u32;
            self.gemv.launch_items(self.st, m, qa, ntok as u32, y, ntok as u32, dense, items)
        }
    }

    /// `y[t] = W x[t]` with f32 activations (F32 / BF16 weights), `x` token-major of width k.
    fn mm_f32(&self, m: &GemvMatrix, x: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>, ntok: usize, zero_ids: &DeviceBuffer<i32>) -> Result<()> {
        if ntok == 1 {
            self.gemv.launch(self.st, m, x, 0, y)
        } else {
            self.gemv.launch_ids(self.st, m, x, m.k, y, &zero_ids.view(0, ntok)?)
        }
    }

    fn norm(&self, x: &DeviceBuffer<f32>, w: &DeviceBuffer<f32>, out: &mut DeviceBuffer<f32>, rows: usize, d: usize, eps: f32) -> Result<()> {
        self.rms.launch(self.st, x, Some(w as &dyn Dev<f32>), out, rows as u32, d as u32, eps)
    }
}

/// One sequence on the GPU: batched prefill, then one token per decode step.
pub struct Session {
    pub w: Weights,
    stream: Stream,
    /// Stream for eager work (prefill, verification); the decode graph replays on `stream`.
    eager_stream: Stream,
    gemv: Gemv,
    ops: InklingOps,
    rms: RmsNormF32,
    kc: Vec<DeviceBuffer<u16>>,
    vc: Vec<DeviceBuffer<u16>>,
    conv: Vec<DeviceBuffer<f32>>,
    s: Scratch,
    capacity: u32,
    pub pos: u32,
    /// When set, `l_out` of every slot is copied to the host each forward (verification).
    pub capture: bool,
    pub captured: Vec<(usize, Vec<f32>)>,
    graph: Option<GraphExec>,
    /// `reset()` as a graph (the conv-state memsets), for the same reason as graph prefill.
    reset_graph: Option<GraphExec>,
    /// DSpark speculative decoding, when enabled ([`Session::enable_spec`]).
    spec: Option<Box<Spec>>,
    /// Sequences with their own KV cache and conv state (sequence 0 is the single-sequence API's).
    n_seqs: usize,
    /// Batched decode: rows per pass, per-row inputs and outputs, and one graph per row count.
    max_batch: usize,
    b: BatchBufs,
    batch_graphs: Vec<Option<GraphExec>>,
    /// Per-sequence `reset_seq` graphs (zeroing that sequence's conv state).
    seq_reset: Vec<Option<GraphExec>>,
    /// Row count the dense work lists (`s.work[0]`, `s.items_dense`) currently describe.
    dense_rows: usize,
    /// `ASTRAL_STEP_TIMING`: decode_batch calls and seconds in (prepare, launch, wait, read).
    phase: (u64, [f64; 4], u64),
    /// Every row's pick of the last verification pass.
    last_rows: Vec<u32>,
    /// Verification-pass graphs by (parts, tokens per part).
    verify_graphs: std::collections::HashMap<(usize, usize), GraphExec>,
    /// Chained verification on the decode kernels: per-slot row-state buffers, the device
    /// pointer tables of conv states and row states, and graphs by row count.
    rowstates: Vec<DeviceBuffer<f32>>,
    state_ptrs: Option<(DeviceBuffer<u64>, DeviceBuffer<u64>)>,
    chain_graphs: std::collections::HashMap<usize, GraphExec>,
    /// The next prefill stores batched drafting's context (set by prefill_seq_spec).
    batch_prefill: bool,
    /// The next step_mixed captures the target features (set by step_mixed_spec).
    mixed_feats: bool,
}

/// Device buffers of the batched decode pass.
struct BatchBufs {
    seq: DeviceBuffer<i32>,
    pos: DeviceBuffer<i32>,
    logits: DeviceBuffer<f32>,
    next: DeviceBuffer<i32>,
    part_v: DeviceBuffer<f32>,
    part_i: DeviceBuffer<u32>,
    dops: DsparkOps,
    /// First position of each segment of a packed prefill pass.
    seg_pos: DeviceBuffer<i32>,
    /// Per-row sequence and position of a packed pass (tmax rows), for verification passes.
    vseq: DeviceBuffer<i32>,
    vpos: DeviceBuffer<i32>,
    /// Chained verification: per-row chain lengths, and the (sequence, row) commit lists.
    chain: DeviceBuffer<i32>,
    commit_seq: DeviceBuffer<i32>,
    commit_row: DeviceBuffer<i32>,
}

/// One row of a batched decode step: feed `token` to sequence `seq` at position `pos`.
#[derive(Debug, Clone, Copy)]
pub struct DecodeRow {
    pub seq: usize,
    pub pos: u32,
    pub token: u32,
}

/// Rows of the batched output buffers: decode rows, or verification passes of up to 4 tokens
/// per sequence.
fn vrows_max(max_batch: usize, tmax: usize) -> usize {
    (4 * max_batch).min(tmax).max(max_batch)
}

/// Floats of attention workspace: decode at full capacity, or `tmax` tokens over 512 positions.
fn attn_work_len(c: &Config, capacity: u32, tmax: usize, max_batch: usize) -> usize {
    use astral_kernels::inkling::{attn_workspace, ATT_CHUNK};
    let (h, hd) = (c.heads as u32, c.head_dim as u32);
    // Decode (up to `max_batch` sequences), and passes of up to QITEM_ROWS tokens (speculative
    // verification), at full capacity.
    let decode = attn_workspace(QITEM_ROWS.max(4 * max_batch).min(tmax) as u32, h, hd, capacity.div_ceil(ATT_CHUNK));
    if split_prefill() {
        decode.max(attn_workspace(tmax.min(512) as u32, h, hd, 512u32.div_ceil(ATT_CHUNK)))
    } else {
        decode
    }
}

/// Bytes of KV cache and conv state one sequence of `capacity` positions takes.
pub fn seq_bytes(c: &Config, capacity: u32) -> usize {
    c.n_slots() * (2 * 2 * capacity as usize * c.kv_width() + 4 * c.conv_len())
}

impl Session {
    pub fn new(ctx: &Arc<Context>, w: Weights, capacity: u32) -> Result<Self> {
        Self::with_seqs(ctx, w, capacity, 1, 1)
    }

    /// A session with `n_seqs` independent sequences of `capacity` positions each, decoded
    /// together up to `max_batch` at a time ([`Session::decode_batch`]).
    pub fn with_seqs(ctx: &Arc<Context>, w: Weights, capacity: u32, n_seqs: usize, max_batch: usize) -> Result<Self> {
        if capacity == 0 || capacity > astral_kernels::inkling::MAX_CTX {
            return Err(bad(format!("capacity {capacity} outside 1..={}", astral_kernels::inkling::MAX_CTX)));
        }
        let n_seqs = n_seqs.max(1);
        let reg = KernelRegistry::load(ctx)?;
        let c = &w.cfg;
        // Rows per pass. Packed passes hold several sequences, so this may exceed the context.
        let tmax = std::env::var("ASTRAL_TMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(PREFILL_TMAX).clamp(1, MAX_PASS_ROWS);
        let max_batch = max_batch.clamp(1, n_seqs).min(PREFILL_TMAX);
        // Batch rows share the prefill scratch: it must hold max_batch tokens.
        let tmax = tmax.max(max_batch);
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(ctx, n);
        let (d, kvw, slots, t, ke) = (c.dim, c.kv_width(), c.n_slots(), tmax, c.n_expert_used);
        let mut kc = Vec::with_capacity(slots);
        let mut vc = Vec::with_capacity(slots);
        let mut conv = Vec::with_capacity(slots);
        for _ in 0..slots {
            kc.push(DeviceBuffer::<u16>::zeroed(ctx, n_seqs * capacity as usize * kvw)?);
            vc.push(DeviceBuffer::<u16>::zeroed(ctx, n_seqs * capacity as usize * kvw)?);
            conv.push(z(n_seqs * c.conv_len())?);
        }
        let rank = w.loops.first().map_or(1, |l| l.down[0].n as usize);
        let dynw = w.layers.iter().filter_map(|l| l.k_conv_proj.as_ref()).map(|m| m.n as usize).max().unwrap_or(4);
        let max_items = t + c.n_expert + 1;
        let s = Scratch {
            tmax,
            cur: z(t * d)?,
            inp_sa: z(t * d)?,
            mixed: z(3 * t * d)?,
            inputs: [z(t * d)?, z(t * d)?, z(t * d)?],
            normed: [z(t * d)?, z(t * d)?, z(t * d)?],
            msgs: z(4 * t * d)?,
            q: z(t * c.heads * c.head_dim)?,
            k: z(t * kvw)?,
            v: z(t * kvw)?,
            kdyn: z(t * dynw)?,
            vdyn: z(t * dynw)?,
            r: z(t * c.heads * c.n_rel as usize)?,
            sinks: z(t * c.heads)?,
            attn: z(t * c.heads * c.head_dim)?,
            attn_work: z(attn_work_len(c, capacity, t, max_batch))?,
            gate: z(t * c.heads * c.head_dim)?,
            o: z(t * d)?,
            m: z(t * d)?,
            ctmp: z(t * d)?,
            mn: z(t * d)?,
            rlogits: z(t * c.n_expert)?,
            ids: DeviceBuffer::<i32>::zeroed(ctx, t * ke)?,
            zero_ids: DeviceBuffer::<i32>::zeroed(ctx, t)?,
            router_cnt: DeviceBuffer::<u32>::zeroed(ctx, t)?,
            w: z(t * ke)?,
            gu: z(t * ke * 2 * c.n_ff_exp)?,
            y: z(t * ke * d)?,
            cd: z(3 * t * rank)?,
            crms: z(2 * t)?,
            bank: z(BANK_ROWS * t * d)?,
            prelude: z(t * d)?,
            bank_out: z(t * d)?,
            logits: z(c.vocab_padded)?,
            token: DeviceBuffer::<i32>::zeroed(ctx, t)?,
            pos: DeviceBuffer::<i32>::zeroed(ctx, 1)?,
            next: DeviceBuffer::<i32>::zeroed(ctx, 1)?,
            qa: [QAct::with_f16(ctx, d as u32, t as u32)?, QAct::with_f16(ctx, d as u32, t as u32)?, QAct::with_f16(ctx, d as u32, t as u32)?],
            qa_attn: QAct::with_f16(ctx, (c.heads * c.head_dim) as u32, t as u32)?,
            qa_ff: QAct::with_f16(ctx, c.n_ff_exp as u32, (t * ke) as u32)?,
            qa_last: QAct::new(ctx, d as u32, 1)?,
            items_dense: DeviceBuffer::<u32>::zeroed(ctx, max_items * QItem::WORDS)?,
            items_gu: DeviceBuffer::<u32>::zeroed(ctx, max_items * QItem::WORDS)?,
            items_dn: DeviceBuffer::<u32>::zeroed(ctx, max_items * QItem::WORDS)?,
            work: {
                let (mi, mx) = (t + c.n_expert + 1, t * ke);
                [DevWork::new(ctx, mi, mx)?, DevWork::new(ctx, mi, mx)?, DevWork::new(ctx, mi, mx)?]
            },
            work128: DevWork::new(ctx, t.div_ceil(128) + 1, t)?,
            work256: DevWork::new(ctx, t.div_ceil(256) + 1, t)?,
            dbg: None,
        };
        let b = BatchBufs {
            seq: DeviceBuffer::<i32>::zeroed(ctx, vrows_max(max_batch, t))?,
            pos: DeviceBuffer::<i32>::zeroed(ctx, vrows_max(max_batch, t))?,
            chain: DeviceBuffer::<i32>::zeroed(ctx, vrows_max(max_batch, t))?,
            commit_seq: DeviceBuffer::<i32>::zeroed(ctx, max_batch)?,
            commit_row: DeviceBuffer::<i32>::zeroed(ctx, max_batch)?,
            // Rows of a batched pass: decode (max_batch) or verification (up to 4 per sequence).
            logits: z(vrows_max(max_batch, t) * w.output.n as usize)?,
            next: DeviceBuffer::<i32>::zeroed(ctx, vrows_max(max_batch, t))?,
            part_v: z(vrows_max(max_batch, t) * astral_kernels::dspark::ARGMAX_PARTS as usize)?,
            part_i: DeviceBuffer::<u32>::zeroed(ctx, vrows_max(max_batch, t) * astral_kernels::dspark::ARGMAX_PARTS as usize)?,
            dops: DsparkOps::new(&reg)?,
            // Start positions, then (packed prefill) the 4-word segment table for attention.
            seg_pos: DeviceBuffer::<i32>::zeroed(ctx, 9 * max_batch)?,
            vseq: DeviceBuffer::<i32>::zeroed(ctx, t)?,
            vpos: DeviceBuffer::<i32>::zeroed(ctx, t)?,
        };
        let mut me = Self {
            stream: Stream::new(ctx)?,
            eager_stream: Stream::new(ctx)?,
            gemv: Gemv::new(&reg)?,
            ops: InklingOps::new(&reg)?,
            rms: RmsNormF32::new(&reg)?,
            kc,
            vc,
            conv,
            s,
            capacity,
            pos: 0,
            capture: false,
            captured: Vec::new(),
            graph: None,
            reset_graph: None,
            spec: None,
            n_seqs,
            max_batch,
            b,
            batch_graphs: (0..=max_batch).map(|_| None).collect(),
            seq_reset: (0..n_seqs).map(|_| None).collect(),
            dense_rows: 0,
            phase: (0, [0.0; 4], 0),
            last_rows: Vec::new(),
            verify_graphs: std::collections::HashMap::new(),
            rowstates: Vec::new(),
            state_ptrs: None,
            chain_graphs: std::collections::HashMap::new(),
            batch_prefill: false,
            mixed_feats: false,
            w,
        };
        me.capture_graph()?;
        {
            let cl = me.w.cfg.conv_len();
            let Session { stream, conv, reset_graph, .. } = &mut me;
            *reset_graph = Some(stream.capture(|st| {
                for c in conv.iter_mut() {
                    c.zero_range_async(0, cl, st)?;
                }
                Ok(())
            })?);
        }
        Ok(me)
    }

    /// Sequences this session holds state for.
    pub fn n_seqs(&self) -> usize {
        self.n_seqs
    }

    /// Most rows one [`Session::decode_batch`] call takes.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Forget sequence `seq`: zero its short-conv state (its KV cache is overwritten as it
    /// fills). Runs as a graph, like [`Session::reset`].
    pub fn reset_seq(&mut self, seq: usize) -> Result<()> {
        if seq >= self.n_seqs {
            return Err(bad(format!("reset_seq: sequence {seq} of {}", self.n_seqs)));
        }
        if seq == 0 {
            return self.reset();
        }
        let cl = self.w.cfg.conv_len();
        let Session { stream, conv, seq_reset, .. } = self;
        if seq_reset[seq].is_none() {
            seq_reset[seq] = Some(stream.capture(|st| {
                for c in conv.iter_mut() {
                    c.zero_range_async(seq * cl, cl, st)?;
                }
                Ok(())
            })?);
        }
        seq_reset[seq].as_ref().expect("captured above").launch(stream)?;
        stream.synchronize()
    }

    /// Points the dense work lists at `rows` consecutive rows (prefill chunks, verify passes and
    /// batched decode share them).
    fn set_dense(&mut self, rows: usize) -> Result<()> {
        if self.dense_rows == rows {
            return Ok(());
        }
        let s = &mut self.s;
        let mut words = QItem::words(&QItem::dense(rows as u32));
        words.resize(s.items_dense.len(), 0);
        s.items_dense.copy_from_slice(&words)?;
        let pairs: Vec<(u32, u32)> = (0..rows as u32).map(|r| (r, r)).collect();
        let mut dense = MmaWork::default();
        dense.push(0, &pairs);
        s.work[0].upload(&dense)?;
        let mut dense128 = MmaWork::default();
        dense128.push_chunked(0, &pairs, 128);
        s.work128.upload(&dense128)?;
        let mut dense256 = MmaWork::default();
        dense256.push_chunked(0, &pairs, 256);
        s.work256.upload(&dense256)?;
        self.dense_rows = rows;
        Ok(())
    }

    /// One decode step for several sequences at once: row `i` feeds `rows[i].token` to
    /// `rows[i].seq` at `rows[i].pos` and gets back that sequence's greedy next token. Each row
    /// count runs as its own CUDA graph (captured on first use, see [`Session::capture_batch`]).
    /// The sequences must be distinct and each must have been prefilled up to `pos`.
    pub fn decode_batch(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        let n = rows.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.max_batch {
            return Err(bad(format!("decode_batch: {n} rows, at most {}", self.max_batch)));
        }
        for r in rows {
            if r.seq >= self.n_seqs || r.pos >= self.capacity {
                return Err(bad(format!("decode_batch: sequence {} at position {} (sequences {}, context {})", r.seq, r.pos, self.n_seqs, self.capacity)));
            }
        }
        let t0 = std::time::Instant::now();
        self.set_dense(n)?;
        if !self.capture {
            self.capture_batch(n)?;
        }
        let tmax = self.s.tmax;
        let mut tok = vec![0i32; tmax];
        let (mut sq, mut ps) = (vec![0i32; self.b.seq.len()], vec![0i32; self.b.pos.len()]);
        for (i, r) in rows.iter().enumerate() {
            tok[i] = r.token as i32;
            sq[i] = r.seq as i32;
            ps[i] = r.pos as i32;
        }
        self.s.token.copy_from_slice(&tok)?;
        self.b.seq.copy_from_slice(&sq)?;
        self.b.pos.copy_from_slice(&ps)?;
        if self.capture {
            // Verification: eager, with every slot's output copied to the host.
            let conv_stride = self.w.cfg.conv_len() as u32;
            let n_seqs = self.n_seqs as u32;
            let Session { w, stream, gemv, ops, rms, kc, vc, conv, s, capacity, b, captured, .. } = self;
            let BatchBufs { seq, pos, logits, next, part_v, part_i, dops, .. } = b;
            captured.clear();
            let mut io = BatchIo { rows: SeqRows { seq: &*seq, n_seqs, conv_stride, chain: 0, rowstate: 0, conv_base: 0 }, pos: &*pos, verify: None, rowstates: None, logits, next, part_v, part_i, dops };
            forward(&Exec { st: stream, gemv, ops, rms }, w, kc, vc, conv, s, *capacity, 0, n, Some(captured), &mut SpecIo::none(), 0, Some(&mut io), None)?;
        } else {
            let t1 = std::time::Instant::now();
            // ASTRAL_PROFILE_STEP=k: profile the k-th batched step of at least 48 rows.
            let prof = n >= 48 && std::env::var("ASTRAL_PROFILE_STEP").ok().and_then(|v| v.parse::<u64>().ok()).is_some_and(|k| {
                self.phase.2 += 1;
                self.phase.2 == k
            });
            if prof {
                self.stream.synchronize()?;
                astral_gpu::profiler(true)?;
            }
            self.batch_graphs[n].as_ref().expect("captured above").launch(&self.stream)?;
            let t2 = std::time::Instant::now();
            self.stream.synchronize()?;
            if prof {
                astral_gpu::profiler(false)?;
            }
            let t3 = std::time::Instant::now();
            let next = self.b.next.to_vec()?;
            if std::env::var_os("ASTRAL_STEP_TIMING").is_some() {
                let p = &mut self.phase;
                p.0 += 1;
                for (k, d) in [(t1 - t0), (t2 - t1), (t3 - t2), t3.elapsed()].iter().enumerate() {
                    p.1[k] += d.as_secs_f64();
                }
                if p.0 % 200 == 0 {
                    let a = |k: usize| p.1[k] * 1e3 / p.0 as f64;
                    eprintln!("decode_batch: {} steps: prepare {:.2} ms, launch {:.2} ms, wait {:.2} ms, read {:.2} ms", p.0, a(0), a(1), a(2), a(3));
                }
            }
            return Ok(next[..n].iter().map(|&t| t as u32).collect());
        }
        self.stream.synchronize()?;
        let next = self.b.next.to_vec()?;
        Ok(next[..n].iter().map(|&t| t as u32).collect())
    }

    /// Verification on the batched decode kernels: `rows` hold, per sequence, consecutive
    /// tokens at consecutive positions (a chain; chains back to back). Every row's greedy pick
    /// is returned. The short convs run each chain's rows in order and keep every row's state;
    /// [`Session::commit_verify`] must follow, keeping each sequence's state at its last
    /// accepted row.
    pub fn decode_verify(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        let n = rows.len();
        let cap_rows = self.b.chain.len().min(self.s.tmax);
        if n == 0 || n > cap_rows {
            return Err(bad(format!("decode_verify: {n} rows, at most {cap_rows}")));
        }
        let mut chain = vec![0i32; cap_rows];
        let mut i = 0usize;
        while i < n {
            let r = rows[i];
            if r.seq >= self.n_seqs || r.pos >= self.capacity {
                return Err(bad(format!("decode_verify: sequence {} at position {}", r.seq, r.pos)));
            }
            let mut j = i + 1;
            while j < n && rows[j].seq == r.seq {
                if rows[j].pos != rows[j - 1].pos + 1 {
                    return Err(bad("decode_verify: a sequence's rows must be at consecutive positions".into()));
                }
                j += 1;
            }
            chain[i] = (j - i) as i32;
            i = j;
        }
        if self.rowstates.is_empty() {
            let ctx = self.stream.context().clone();
            let len = cap_rows * self.w.cfg.conv_len();
            self.rowstates = (0..self.conv.len()).map(|_| DeviceBuffer::<f32>::zeroed(&ctx, len)).collect::<Result<_>>()?;
            let sp: Vec<u64> = self.conv.iter().map(|c| c.device_ptr()).collect();
            let rp: Vec<u64> = self.rowstates.iter().map(|c| c.device_ptr()).collect();
            self.state_ptrs = Some((DeviceBuffer::from_slice(&ctx, &sp)?, DeviceBuffer::from_slice(&ctx, &rp)?));
        }
        self.set_dense(n)?;
        let tmax = self.s.tmax;
        let mut tok = vec![0i32; tmax];
        let (mut sq, mut ps) = (vec![0i32; self.b.seq.len()], vec![0i32; self.b.pos.len()]);
        for (i, r) in rows.iter().enumerate() {
            tok[i] = r.token as i32;
            sq[i] = r.seq as i32;
            ps[i] = r.pos as i32;
        }
        self.s.token.copy_from_slice(&tok)?;
        self.b.seq.copy_from_slice(&sq)?;
        self.b.pos.copy_from_slice(&ps)?;
        self.b.chain.copy_from_slice(&chain)?;
        if !self.chain_graphs.contains_key(&n) {
            let conv_stride = self.w.cfg.conv_len() as u32;
            let n_seqs = self.n_seqs as u32;
            let Session { w, stream, gemv, ops, rms, kc, vc, conv, s, capacity, b, chain_graphs, rowstates, spec, .. } = self;
            let BatchBufs { seq, pos, logits, next, part_v, part_i, dops, chain, .. } = b;
            let cp = chain.device_ptr();
            // Batched drafting: every row's target features for the draft.
            let feats = spec.as_deref_mut().filter(|sp| sp.batch.is_some()).map(|sp| ([sp.w.cfg.taps[0], sp.w.cfg.taps[1], sp.w.cfg.taps[2]], &mut sp.feats));
            let g = stream.capture(|st| {
                let mut sio = match feats {
                    Some((taps, f)) => SpecIo { feats: Some(f), taps, hist: None, hist_rows: 0, all: None, argmax: None },
                    None => SpecIo::none(),
                };
                let mut io = BatchIo { rows: SeqRows { seq: &*seq, n_seqs, conv_stride, chain: cp, rowstate: 0, conv_base: 0 }, pos: &*pos, verify: None, rowstates: Some(&rowstates[..]), logits, next, part_v, part_i, dops };
                forward(&Exec { st, gemv, ops, rms }, w, kc, vc, conv, s, *capacity, 0, n, None, &mut sio, 0, Some(&mut io), None)
            })?;
            chain_graphs.insert(n, g);
        }
        self.chain_graphs[&n].launch(&self.stream)?;
        self.stream.synchronize()?;
        let next = self.b.next.to_vec()?;
        Ok(next[..n].iter().map(|&t| t as u32).collect())
    }

    /// Batched drafting for up to `bmax` sequences (after [`Session::enable_spec`]).
    pub fn enable_batch_spec(&mut self, bmax: usize) -> Result<()> {
        let ctx = self.stream.context().clone();
        let (n, cap, vp) = (self.n_seqs, self.capacity, self.w.output.n as usize);
        let sp = self.spec.as_deref_mut().ok_or_else(|| bad("enable_batch_spec: speculative decoding is not enabled".into()))?;
        sp.enable_batch(&ctx, bmax.min(self.max_batch), n, cap, vp)?;
        self.chain_graphs.clear();
        Ok(())
    }

    /// Drafts per round of batched speculative decoding (0 when not enabled).
    pub fn batch_spec_k(&self) -> usize {
        self.spec.as_ref().filter(|sp| sp.batch.is_some()).map_or(0, |sp| sp.k)
    }

    /// Runs one batched draft: block b is sequence `blocks[b].0`, anchor token `.2` at position
    /// `.1`, Markov context feature row `.3` (of the last pass); returns each block's drafts.
    fn batch_draft_run(&mut self, blocks: &[(usize, u32, u32, usize)]) -> Result<Vec<Vec<u32>>> {
        let nb = blocks.len();
        let Session { w, stream, gemv, ops, spec, .. } = self;
        let sp = spec.as_deref_mut().ok_or_else(|| bad("spec_round: speculative decoding is not enabled".into()))?;
        let k = sp.k;
        let b = sp.batch.as_deref_mut().ok_or_else(|| bad("spec_round: batched drafting is not enabled".into()))?;
        if nb == 0 || nb > b.bmax {
            return Err(bad(format!("spec_round: {nb} sequences, at most {}", b.bmax)));
        }
        let mut sq = vec![0i32; b.bmax];
        let mut st = vec![0i32; b.bmax];
        let mut mk = vec![0i32; b.bmax];
        let mut tk = vec![0i32; b.bmax * (k + 1)];
        for (i, &(q, p, t, m)) in blocks.iter().enumerate() {
            sq[i] = q as i32;
            st[i] = p as i32;
            mk[i] = m as i32;
            tk[i] = t as i32;
        }
        b.seq_b.copy_from_slice(&sq)?;
        b.start_b.copy_from_slice(&st)?;
        b.mark_b.copy_from_slice(&mk)?;
        b.tok.copy_from_slice(&tk)?;
        if !b.draft_graphs.contains_key(&nb) {
            let g = stream.capture(|s| sp.batch_draft(s, gemv, ops, w, nb))?;
            sp.batch.as_deref_mut().expect("checked").draft_graphs.insert(nb, g);
        }
        let b = sp.batch.as_deref_mut().expect("checked");
        b.draft_graphs[&nb].launch(stream)?;
        stream.synchronize()?;
        let tok = b.tok.to_vec()?;
        Ok((0..nb).map(|i| (1..=k).map(|r| tok[r * nb + i] as u32).collect()).collect())
    }

    /// After a prompt (prefilled with the draft context through [`Session::prefill_seq_spec`]):
    /// the first drafts of sequence `seq`, whose next token `next` sits at position `pos`.
    pub fn spec_first_draft(&mut self, seq: usize, pos: u32, next: u32) -> Result<Vec<u32>> {
        let last = self.spec.as_ref().map_or(1, |sp| sp.last_rows).max(1) - 1;
        Ok(self.batch_draft_run(&[(seq, pos, next, last)])?.remove(0))
    }

    /// [`Session::prefill_seq`] that also stores the draft's context (batched drafting).
    pub fn prefill_seq_spec(&mut self, seq: usize, pos: u32, tokens: &[u32]) -> Result<u32> {
        if seq >= self.n_seqs {
            return Err(bad(format!("prefill_seq_spec: sequence {seq} of {}", self.n_seqs)));
        }
        self.batch_prefill = true;
        self.prefill_inner(seq, pos, tokens, true)
    }

    /// [`Session::step_mixed`] that also stores batched drafting's context for the parts marked
    /// in `spec` (prompt chunks of drafting requests), and drafts for the marked parts in `first`
    /// (prompts this pass finishes: their pick is the anchor at the part's end). Returns the
    /// picks and, per `first` part, its drafts.
    pub fn step_mixed_spec(&mut self, parts: &[(usize, u32, &[u32])], spec: &[bool], first: &[bool]) -> Result<(Vec<u32>, Vec<Option<Vec<u32>>>)> {
        let any = spec.iter().any(|&x| x) && self.batch_spec_k() > 0;
        self.mixed_feats = any;
        let picks = self.step_mixed(parts)?;
        let mut drafts = vec![None; parts.len()];
        if !any {
            return Ok((picks, drafts));
        }
        let mut row0 = 0usize;
        let (mut f, mut sq, mut ps) = (Vec::new(), Vec::new(), Vec::new());
        let mut blocks = Vec::new();
        let mut which = Vec::new();
        for (i, &(q, p, t)) in parts.iter().enumerate() {
            if spec[i] {
                for r in 0..t.len() {
                    f.push((row0 + r) as i32);
                    sq.push(q as i32);
                    ps.push((p as usize + r) as i32);
                }
                if first[i] {
                    blocks.push((q, p + t.len() as u32, picks[i], row0 + t.len() - 1));
                    which.push(i);
                }
            }
            row0 += t.len();
        }
        {
            let Session { stream, gemv, spec: sp, .. } = self;
            let sp = sp.as_deref_mut().expect("checked by batch_spec_k");
            let b = sp.batch.as_deref_mut().expect("checked");
            let n = b.ctx_rows;
            let nr = f.len();
            f.resize(n, 0);
            sq.resize(n, 0);
            ps.resize(n, 0);
            b.ctx_feat.copy_from_slice(&f)?;
            b.ctx_seq.copy_from_slice(&sq)?;
            b.ctx_pos.copy_from_slice(&ps)?;
            let prev = stream.context().set_pdl(false);
            let g = stream.capture(|s| sp.batch_store(s, gemv, nr));
            stream.context().set_pdl(prev);
            g?.launch(stream)?;
            stream.synchronize()?;
        }
        if !blocks.is_empty() {
            for (i, d) in which.into_iter().zip(self.batch_draft_run(&blocks)?) {
                drafts[i] = Some(d);
            }
        }
        Ok((picks, drafts))
    }

    /// One round of batched speculative decoding. Entry `(seq, pos, anchor, drafts)`: the
    /// sequence's next input `anchor` at `pos` and its `k` drafts. Verifies all, keeps each
    /// sequence's matching prefix plus the target's next pick, rolls the conv states back,
    /// stores the draft context and drafts again. Returns per entry the accepted tokens (the
    /// matching drafts, then the target's pick: the new anchor, at `pos + len`) and its new drafts.
    pub fn spec_round(&mut self, entries: &[(usize, u32, u32, Vec<u32>)]) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
        let k = self.batch_spec_k();
        if k == 0 || entries.iter().any(|e| e.3.len() != k) {
            return Err(bad("spec_round: batched drafting is not enabled or drafts are not k long".into()));
        }
        // Drafts verified this round (ASTRAL_SPEC_KV, default 2 of the k drafted): fewer rows per
        // sequence keep the verify pass (a MoE pass touching more experts per row) near the cost
        // of a plain decode step; 2 measured best serving 8 requests.
        // Default: a budget of ASTRAL_SPEC_ROWS (24) verify rows per round, so fewer sequences
        // verify more drafts each.
        let budget = std::env::var("ASTRAL_SPEC_ROWS").ok().and_then(|x| x.parse::<usize>().ok()).unwrap_or(24);
        let kv = std::env::var("ASTRAL_SPEC_KV").ok().and_then(|x| x.parse::<usize>().ok()).unwrap_or((budget / entries.len().max(1)).saturating_sub(1)).clamp(1, k);
        let v = kv + 1;
        let mut rows = Vec::with_capacity(entries.len() * v);
        for &(q, p, a, ref d) in entries {
            rows.push(DecodeRow { seq: q, pos: p, token: a });
            for (i, &t) in d[..kv].iter().enumerate() {
                rows.push(DecodeRow { seq: q, pos: p + 1 + i as u32, token: t });
            }
        }
        let tq0 = std::time::Instant::now();
        let picks = self.decode_verify(&rows)?;
        let tq1 = std::time::Instant::now();
        let mut keep = Vec::with_capacity(entries.len());
        let mut acc = Vec::with_capacity(entries.len());
        let mut blocks = Vec::with_capacity(entries.len());
        for (e, &(q, p, _, ref d)) in entries.iter().enumerate() {
            let pk = &picks[e * v..(e + 1) * v];
            let j = d[..kv].iter().zip(pk).take_while(|(x, y)| x == y).count();
            let mut out = d[..j].to_vec();
            out.push(pk[j]);
            keep.push((q, e * v + j));
            blocks.push((q, p + j as u32 + 1, pk[j], e * v + j));
            acc.push(out);
        }
        self.commit_verify(&keep)?;
        let tq2 = std::time::Instant::now();
        // The draft context of every verified row (rows past the accepted ones are rewritten
        // by later rounds before they are read).
        {
            let Session { stream, gemv, spec, .. } = self;
            let sp = spec.as_deref_mut().expect("checked by batch_spec_k");
            let b = sp.batch.as_deref_mut().expect("checked");
            let n = b.ctx_rows;
            let (mut f, mut sq, mut ps) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
            for (r, row) in rows.iter().enumerate() {
                f[r] = r as i32;
                sq[r] = row.seq as i32;
                ps[r] = row.pos as i32;
            }
            b.ctx_feat.copy_from_slice(&f)?;
            b.ctx_seq.copy_from_slice(&sq)?;
            b.ctx_pos.copy_from_slice(&ps)?;
            let nr = rows.len();
            if !b.store_graphs.contains_key(&nr) {
                let g = stream.capture(|s| sp.batch_store(s, gemv, nr))?;
                sp.batch.as_deref_mut().expect("checked").store_graphs.insert(nr, g);
            }
            sp.batch.as_deref_mut().expect("checked").store_graphs[&nr].launch(stream)?;
        }
        let tq3 = std::time::Instant::now();
        let drafts = self.batch_draft_run(&blocks)?;
        if std::env::var_os("ASTRAL_SPEC_TIMING").is_some() {
            static T: std::sync::Mutex<(u64, [f64; 4])> = std::sync::Mutex::new((0, [0.0; 4]));
            let mut t = T.lock().unwrap_or_else(|e| e.into_inner());
            t.0 += 1;
            for (i, d) in [tq1 - tq0, tq2 - tq1, tq3 - tq2, tq3.elapsed()].iter().enumerate() {
                t.1[i] += d.as_secs_f64() * 1e3;
            }
            if t.0 % 20 == 0 {
                let n = t.0 as f64;
                eprintln!("spec_round {} x {}: verify {:.2} commit {:.2} store {:.2} draft {:.2} ms", t.0, entries.len(), t.1[0] / n, t.1[1] / n, t.1[2] / n, t.1[3] / n);
            }
        }
        if let Some(sp) = self.spec.as_deref_mut() {
            sp.stats.rounds += entries.len() as u64;
            sp.stats.drafted += (entries.len() * kv) as u64;
            sp.stats.accepted += acc.iter().map(|a| a.len() as u64 - 1).sum::<u64>();
        }
        Ok(acc.into_iter().zip(drafts).collect())
    }

    /// After [`Session::decode_verify`]: sequence `seq` keeps the conv state after verify row
    /// `row` (its last accepted token), for every `(seq, row)`.
    pub fn commit_verify(&mut self, keep: &[(usize, usize)]) -> Result<()> {
        let Some((sp, rp)) = self.state_ptrs.as_ref() else { return Err(bad("commit_verify: no verification pass".into())) };
        if keep.len() > self.b.commit_seq.len() {
            return Err(bad("commit_verify: too many sequences".into()));
        }
        let mut a = vec![0i32; self.b.commit_seq.len()];
        let mut c = vec![0i32; self.b.commit_row.len()];
        for (i, &(q, r)) in keep.iter().enumerate() {
            a[i] = q as i32;
            c[i] = r as i32;
        }
        self.b.commit_seq.copy_from_slice(&a)?;
        self.b.commit_row.copy_from_slice(&c)?;
        let cl = self.w.cfg.conv_len() as u32;
        self.ops.conv_commit_rows(&self.stream, sp, rp, self.conv.len() as u32, &self.b.commit_seq, &self.b.commit_row, keep.len() as u32, cl, cl)?;
        self.stream.synchronize()
    }

    /// Logits of the `n` rows of the last [`Session::decode_batch`] (unpadded vocabulary, row-major).
    pub fn batch_logits(&self, n: usize) -> Result<Vec<f32>> {
        let (vp, v) = (self.w.output.n as usize, self.w.cfg.vocab);
        let all = self.b.logits.to_vec()?;
        Ok((0..n.min(self.max_batch)).flat_map(|r| all[r * vp..r * vp + v].to_vec()).collect())
    }

    /// Records the batched decode pass for `n` rows as a CUDA graph (nothing runs).
    pub fn capture_batch(&mut self, n: usize) -> Result<()> {
        if n == 0 || n > self.max_batch {
            return Err(bad(format!("capture_batch: {n} rows, at most {}", self.max_batch)));
        }
        if self.batch_graphs[n].is_some() {
            return Ok(());
        }
        self.set_dense(n)?;
        let conv_stride = self.w.cfg.conv_len() as u32;
        let n_seqs = self.n_seqs as u32;
        let Session { w, stream, gemv, ops, rms, kc, vc, conv, s, capacity, b, batch_graphs, .. } = self;
        let BatchBufs { seq, pos, logits, next, part_v, part_i, dops, .. } = b;
        let g = stream.capture(|st| {
            let mut io = BatchIo { rows: SeqRows { seq: &*seq, n_seqs, conv_stride, chain: 0, rowstate: 0, conv_base: 0 }, pos: &*pos, verify: None, rowstates: None, logits, next, part_v, part_i, dops };
            forward(&Exec { st, gemv, ops, rms }, w, kc, vc, conv, s, *capacity, 0, n, None, &mut SpecIo::none(), 0, Some(&mut io), None)
        })?;
        batch_graphs[n] = Some(g);
        Ok(())
    }

    /// Records the one-token forward pass as a CUDA graph (nothing runs), so the first decode
    /// step does not pay for capture and instantiation. No-op with `ASTRAL_NO_GRAPH=1`.
    fn capture_graph(&mut self) -> Result<()> {
        if self.graph.is_some() || std::env::var_os("ASTRAL_NO_GRAPH").is_some() {
            return Ok(());
        }
        let Session { w, stream, gemv, ops, rms, kc, vc, conv, s, capacity, graph, .. } = self;
        let g = stream.capture(|st| forward(&Exec { st, gemv, ops, rms }, w, kc, vc, conv, s, *capacity, 0, 1, None, &mut SpecIo::none(), 0, None, None))?;
        *graph = Some(g);
        Ok(())
    }

    /// Runs one token at position `self.pos`; returns the greedy next token and advances.
    ///
    /// The first plain step captures the whole forward pass into a CUDA graph; later steps
    /// replay it (token and position live in device buffers, so the graph never changes).
    /// Verification (`capture`, debug taps) and `ASTRAL_NO_GRAPH=1` run eagerly instead.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        if self.pos >= self.capacity {
            return Err(bad(format!("context full ({} positions)", self.capacity)));
        }
        let Session { w, stream, gemv, ops, rms, kc, vc, conv, s, capacity, pos, capture, captured, graph, .. } = self;
        // The previous call ended with a stream sync, so these synchronous copies cannot race.
        let t0 = std::time::Instant::now();
        let mut tok = vec![0i32; s.tmax];
        tok[0] = token as i32;
        s.token.copy_from_slice(&tok)?;
        s.pos.copy_from_slice(&[*pos as i32])?;
        let t1 = std::time::Instant::now();
        let eager = *capture || s.dbg.is_some() || std::env::var_os("ASTRAL_NO_GRAPH").is_some();
        if eager {
            captured.clear();
            let ex = Exec { st: stream, gemv, ops, rms };
            forward(&ex, w, kc, vc, conv, s, *capacity, *pos, 1, capture.then_some(captured), &mut SpecIo::none(), 0, None, None)?;
        } else {
            if graph.is_none() {
                let g = stream.capture(|st| forward(&Exec { st, gemv, ops, rms }, w, kc, vc, conv, s, *capacity, 0, 1, None, &mut SpecIo::none(), 0, None, None))?;
                *graph = Some(g);
            }
            graph.as_ref().expect("graph captured above").launch(stream)?;
        }
        let t2 = std::time::Instant::now();
        stream.synchronize()?;
        let t3 = std::time::Instant::now();
        *pos += 1;
        let r = s.next.to_vec()?[0] as u32;
        if std::env::var_os("ASTRAL_STEP_TIMING").is_some() {
            eprintln!(
                "step copy {:.0} us launch {:.0} us sync {:.0} us read {:.0} us",
                (t1 - t0).as_secs_f64() * 1e6,
                (t2 - t1).as_secs_f64() * 1e6,
                (t3 - t2).as_secs_f64() * 1e6,
                t3.elapsed().as_secs_f64() * 1e6
            );
        }
        Ok(r)
    }

    /// Processes `tokens` (the prompt) in batched chunks; returns the greedy next token after
    /// the last one. Eager (no graph): every chunk shape differs and MoE grouping reads the
    /// router's picks on the host once per layer.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        let pos = self.pos;
        let draft = self.spec.is_some();
        let next = self.prefill_inner(0, pos, tokens, draft)?;
        self.pos = pos + tokens.len() as u32;
        if std::env::var_os("ASTRAL_RECAPTURE").is_some() {
            self.graph = None;
            self.capture_graph()?;
        }
        Ok(next)
    }

    /// Prefills several new sequences together (each after [`Session::reset_seq`]): `jobs[i] =
    /// (sequence, prompt)`. Prompts are packed into passes of up to the prefill chunk size (a
    /// long prompt continues in the next pass), so the weights are read once per pass rather than
    /// once per prompt. Returns each sequence's greedy next token. No draft context is stored.
    pub fn prefill_many(&mut self, jobs: &[(usize, &[u32])]) -> Result<Vec<u32>> {
        for &(q, t) in jobs {
            if q >= self.n_seqs || t.is_empty() || t.len() > self.capacity as usize {
                return Err(bad(format!("prefill_many: sequence {q} with {} tokens (sequences {}, context {})", t.len(), self.n_seqs, self.capacity)));
            }
        }
        let mut out = vec![0u32; jobs.len()];
        let mut done = vec![0usize; jobs.len()];
        loop {
            let tmax = self.s.tmax;
            let (mut parts, mut owners, mut rows) = (Vec::new(), Vec::new(), 0usize);
            for (i, &(q, t)) in jobs.iter().enumerate() {
                if done[i] < t.len() && rows < tmax && parts.len() < self.max_batch {
                    let n = (t.len() - done[i]).min(tmax - rows);
                    parts.push((q, done[i] as u32, &t[done[i]..done[i] + n]));
                    owners.push(i);
                    rows += n;
                }
            }
            if parts.is_empty() {
                return Ok(out);
            }
            let next = self.step_mixed(&parts)?;
            for ((&(_, _, t), &i), n) in parts.iter().zip(&owners).zip(next) {
                done[i] += t.len();
                if done[i] == jobs[i].1.len() {
                    out[i] = n;
                }
            }
        }
    }

    /// Rows one [`Session::step_mixed`] pass can hold.
    pub fn token_budget(&self) -> usize {
        self.s.tmax
    }

    /// One forward pass over several sequences at once: `parts[i] = (sequence, position,
    /// tokens)` feeds `tokens` to that sequence from `position` (a prompt chunk, or one token of
    /// a sequence that is decoding), and returns the greedy next token after each part's last
    /// token. The parts must name distinct sequences, hold at most [`Session::token_budget`]
    /// tokens in all and be at most [`Session::max_batch`]; each sequence must already hold its
    /// first `position` tokens. No draft context is stored.
    pub fn step_mixed(&mut self, parts: &[(usize, u32, &[u32])]) -> Result<Vec<u32>> {
        let tmax = self.s.tmax;
        let rows: usize = parts.iter().map(|p| p.2.len()).sum();
        if parts.is_empty() || parts.len() > self.max_batch || rows > tmax {
            return Err(bad(format!("step_mixed: {} parts of {rows} tokens (at most {} parts, {tmax} tokens)", parts.len(), self.max_batch)));
        }
        let mut segs = Vec::with_capacity(parts.len());
        let mut tok = vec![0i32; tmax];
        let mut row0 = 0usize;
        for &(q, pos, t) in parts {
            if q >= self.n_seqs || t.is_empty() || pos as usize + t.len() > self.capacity as usize {
                return Err(bad(format!("step_mixed: sequence {q}, {} tokens at {pos} (sequences {}, context {})", t.len(), self.n_seqs, self.capacity)));
            }
            for (r, &x) in t.iter().enumerate() {
                tok[row0 + r] = x as i32;
            }
            segs.push(Seg { seq: q, row0, len: t.len(), pos0: pos });
            row0 += t.len();
        }
        let mut sp = vec![0i32; 9 * self.max_batch];
        let kv_span = self.capacity as usize * self.w.cfg.kv_width();
        let conv_span = self.w.cfg.conv_len();
        for (k, sg) in segs.iter().enumerate() {
            sp[k] = sg.pos0 as i32;
            let t = self.max_batch + 8 * k;
            sp[t..t + 5].copy_from_slice(&[sg.row0 as i32, sg.len as i32, (sg.seq * kv_span) as i32, k as i32, (sg.seq * conv_span) as i32]);
        }
        // Leading one-token parts (decoding sequences) run on the batched decode kernels, which
        // read each row's sequence and position from the batch buffers.
        let (mut sq, mut ps) = (vec![0i32; self.b.seq.len()], vec![0i32; self.b.pos.len()]);
        for (i, sg) in segs.iter().enumerate().take_while(|(i, sg)| sg.len == 1 && sg.row0 == *i) {
            sq[i] = sg.seq as i32;
            ps[i] = sg.pos0 as i32;
        }
        self.b.seq.copy_from_slice(&sq)?;
        self.b.pos.copy_from_slice(&ps)?;
        // Short segments only (verification): every row's sequence and position.
        let verify = segs.iter().all(|sg| sg.len <= 8) && segs.iter().any(|sg| sg.len > 1);
        if verify {
            let (mut vs, mut vp) = (vec![0i32; tmax], vec![0i32; tmax]);
            for sg in &segs {
                for r in 0..sg.len {
                    vs[sg.row0 + r] = sg.seq as i32;
                    vp[sg.row0 + r] = (sg.pos0 as usize + r) as i32;
                }
            }
            self.b.vseq.copy_from_slice(&vs)?;
            self.b.vpos.copy_from_slice(&vp)?;
        }
        self.set_dense(rows)?;
        self.s.token.copy_from_slice(&tok)?;
        self.b.seg_pos.copy_from_slice(&sp)?;
        self.captured.clear();
        let conv_stride = self.w.cfg.conv_len() as u32;
        let n_seqs = self.n_seqs as u32;
        {
            let want_feats = std::mem::take(&mut self.mixed_feats);
            let Session { w, eager_stream: stream, gemv, ops, rms, kc, vc, conv, s, capacity, b, verify_graphs, spec, .. } = self;
            // Batched drafting: the pass's target features (for the draft context of prompts).
            let mut feats = if want_feats { spec.as_deref_mut().filter(|sp| sp.batch.is_some()).map(|sp| ([sp.w.cfg.taps[0], sp.w.cfg.taps[1], sp.w.cfg.taps[2]], &mut sp.feats)) } else { None };
            let BatchBufs { seq, pos, logits, next, part_v, part_i, dops, seg_pos, vseq, vpos, .. } = b;
            let cap = *capacity;
            // Verification passes of equal-length parts (at least two: their segments are all
            // table-driven, so the graph replays for any sequences and positions) reuse a graph
            // per (parts, tokens); anything else runs a one-off graph (as in prefill).
            let key = (feats.is_none() && verify && segs.len() >= 2 && segs.iter().all(|sg| sg.len == segs[0].len) && std::env::var_os("ASTRAL_VERIFY_NO_CACHE").is_none()).then(|| (segs.len(), segs[0].len));
            if let Some(g) = key.and_then(|k| verify_graphs.get(&k)) {
                g.launch(stream)?;
                stream.synchronize()?;
            } else {
            // PDL off for the large prefill kernels.
            let prev = stream.context().set_pdl(std::env::var_os("ASTRAL_PREFILL_PDL").is_some());
            let g = stream.capture(|st| {
                let vr = verify.then_some(BatchRows { rows: SeqRows { seq: &*vseq, n_seqs, conv_stride, chain: 0, rowstate: 0, conv_base: 0 }, pos: &*vpos });
                let mut io = BatchIo { rows: SeqRows { seq: &*seq, n_seqs, conv_stride, chain: 0, rowstate: 0, conv_base: 0 }, pos: &*pos, verify: vr, rowstates: None, logits, next, part_v, part_i, dops };
                let mut sio = match feats.take() {
                    Some((taps, f)) => SpecIo { feats: Some(f), taps, hist: None, hist_rows: 0, all: None, argmax: None },
                    None => SpecIo::none(),
                };
                forward(&Exec { st, gemv, ops, rms }, w, kc, vc, conv, s, cap, 0, rows, None, &mut sio, 0, Some(&mut io), Some((&segs, &*seg_pos)))
            });
            stream.context().set_pdl(prev);
            let g = g?;
            g.launch(stream)?;
            stream.synchronize()?;
            if let Some(k) = key {
                verify_graphs.insert(k, g);
            }
            }
        }
        let next = self.b.next.to_vec()?;
        if verify && self.w.output.ty == GemvType::Q8_0 && rows <= self.b.next.len() {
            // Every row's pick: the segment's last row is its next token.
            self.last_rows = next[..rows].iter().map(|&t| t as u32).collect();
            return Ok(segs.iter().map(|sg| next[sg.row0 + sg.len - 1] as u32).collect());
        }
        self.last_rows.clear();
        Ok(next[..segs.len()].iter().map(|&t| t as u32).collect())
    }

    /// Verification pass: every part's tokens (at most 8 each) fed to its sequence, returning
    /// the greedy pick after each of them (`picks[p][i]` follows token i of part p).
    pub fn step_verify(&mut self, parts: &[(usize, u32, &[u32])]) -> Result<Vec<Vec<u32>>> {
        if parts.iter().any(|p| p.2.len() > 8) || parts.iter().all(|p| p.2.len() == 1) || self.w.output.ty != GemvType::Q8_0 {
            return Err(bad("step_verify: parts of 1..=8 tokens (not all one), Q8_0 LM head".into()));
        }
        self.step_mixed(parts)?;
        let mut out = Vec::with_capacity(parts.len());
        let mut r = 0usize;
        for p in parts {
            out.push(self.last_rows[r..r + p.2.len()].to_vec());
            r += p.2.len();
        }
        Ok(out)
    }

    /// Prefills `tokens` into sequence `seq` from position `pos` (0 for a new sequence, after
    /// [`Session::reset_seq`]); returns the greedy next token. No draft context is stored: the
    /// sequence decodes through [`Session::decode_batch`].
    pub fn prefill_seq(&mut self, seq: usize, pos: u32, tokens: &[u32]) -> Result<u32> {
        if seq >= self.n_seqs {
            return Err(bad(format!("prefill_seq: sequence {seq} of {}", self.n_seqs)));
        }
        self.prefill_inner(seq, pos, tokens, false)
    }

    fn prefill_inner(&mut self, seq: usize, pos_start: u32, tokens: &[u32], draft: bool) -> Result<u32> {
        // Batched drafting's context (prefill_seq_spec), or the lone draft's.
        let batch_ctx = std::mem::take(&mut self.batch_prefill);
        if tokens.is_empty() {
            return Err(bad("prefill: no tokens".into()));
        }
        if pos_start as usize + tokens.len() > self.capacity as usize {
            return Err(bad(format!("prefill: {} tokens at position {} exceed the context of {}", tokens.len(), pos_start, self.capacity)));
        }
        let mut cur_pos = pos_start;
        use astral_kernels::inkling::{attn_workspace, ATT_CHUNK};
        let mut next = 0;
        let mut done = 0;
        self.captured.clear();
        while done < tokens.len() {
            // Largest chunk whose attention workspace fits.
            let (h, hd) = (self.w.cfg.heads as u32, self.w.cfg.head_dim as u32);
            let mut t = (tokens.len() - done).min(self.s.tmax);
            // Only the split-K fallback needs workspace; flash-attention prefill does not.
            while split_prefill() && t > 1 && attn_workspace(t as u32, h, hd, (cur_pos + t as u32).div_ceil(ATT_CHUNK)) > self.s.attn_work.len() {
                t /= 2;
            }
            if t > 1 {
                self.set_dense(t)?;
            }
            let chunk = &tokens[done..done + t];
            let pos = &mut cur_pos;
            {
                let mut sp = vec![0i32; 9 * self.max_batch];
                sp[0] = *pos as i32;
                self.b.seg_pos.copy_from_slice(&sp)?;
            }
            let seg1 = [Seg { seq, row0: 0, len: t, pos0: *pos }];
            let Session { w, eager_stream: stream, gemv, ops, rms, kc, vc, conv, s, capacity, capture, captured, spec, b, .. } = self;
            let segs = Some((&seg1[..], &b.seg_pos));
            let mut no_spec: Option<Box<Spec>> = None;
            let spec = if draft { spec } else { &mut no_spec };
            let mut tok = vec![0i32; s.tmax];
            for (i, &x) in chunk.iter().enumerate() {
                tok[i] = x as i32;
            }
            s.token.copy_from_slice(&tok)?;
            s.pos.copy_from_slice(&[*pos as i32])?;
            let ex = Exec { st: stream, gemv, ops, rms };
            if let Some(k) = std::env::var("ASTRAL_EAGER_DIAG").ok().and_then(|v| v.parse().ok()) {
                DIAG_LIMIT.store(k, std::sync::atomic::Ordering::Relaxed);
            }
            // Prefill runs as a one-off CUDA graph: bursts of eager launches (or memsets) leave the
            // process's later graph replays ~6% slower on this driver (measured: decode 135 vs 144
            // tok/s), so a serving process makes none. ASTRAL_PREFILL_EAGER=1 launches eagerly.
            let r = if std::env::var_os("ASTRAL_PREFILL_EAGER").is_none() && !*capture {
                let p0 = *pos;
                let cap = *capacity;
                // PDL helps decode's small kernels but slows prefill's large ones: off while capturing.
                let prev = stream.context().set_pdl(std::env::var_os("ASTRAL_PREFILL_PDL").is_some());
                let captured_graph = stream.capture(|st| forward(&Exec { st, gemv: ex.gemv, ops: ex.ops, rms: ex.rms }, w, kc, vc, conv, s, cap, p0, t, None, &mut spec_io(spec), seq, None, segs));
                stream.context().set_pdl(prev);
                match captured_graph {
                    Ok(g) => g.launch(stream).and_then(|_| stream.synchronize()),
                    Err(e) => Err(e),
                }
            } else {
                forward(&ex, w, kc, vc, conv, s, *capacity, *pos, t, capture.then_some(captured), &mut spec_io(spec), seq, None, segs)
            };
            DIAG_LIMIT.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
            r?;
            stream.synchronize()?;
            // Speculative decoding: the draft's context K/V for this chunk (one-off graph).
            if let Some(sp) = spec.as_deref_mut().filter(|sp| sp.batch.is_some() && batch_ctx) {
                // Batched drafting: this sequence's draft context for the chunk's rows.
                let b = sp.batch.as_deref_mut().expect("checked");
                let n = b.ctx_rows;
                let rows: Vec<i32> = (0..n as i32).collect();
                let seqs = vec![seq as i32; n];
                let poss: Vec<i32> = (0..n).map(|i| *pos as i32 + i.min(t - 1) as i32).collect();
                b.ctx_feat.copy_from_slice(&rows)?;
                b.ctx_seq.copy_from_slice(&seqs)?;
                b.ctx_pos.copy_from_slice(&poss)?;
                let prev = stream.context().set_pdl(false);
                let g = stream.capture(|st| sp.batch_store(st, gemv, t));
                stream.context().set_pdl(prev);
                g?.launch(stream)?;
                stream.synchronize()?;
                sp.last_rows = t;
            } else if let Some(sp) = spec.as_deref_mut() {
                let mut wk = MmaWork::default();
                wk.push(0, &(0..t as u32).map(|r| (r, r)).collect::<Vec<_>>());
                sp.work_pf.upload(&wk)?;
                let at = DevIndex::of(&s.pos, 0)?;
                let prev = stream.context().set_pdl(false);
                let g = stream.capture(|st| sp.store_ctx(st, gemv, t, at, false));
                stream.context().set_pdl(prev);
                g?.launch(stream)?;
                stream.synchronize()?;
                sp.last_rows = t;
            }
            *pos += t as u32;
            next = s.next.to_vec()?[0] as u32;
            done += t;
        }
        Ok(next)
    }

    /// Enables DSpark speculative decoding with the draft of the native checkpoint in `dir`
    /// (`config.json` with `mtp_config`, `mtp.safetensors`), verifying `k` drafts per round.
    pub fn enable_spec(&mut self, dir: &std::path::Path, k: usize) -> Result<()> {
        let ctx = self.stream.context().clone();
        let reg = KernelRegistry::load(&ctx)?;
        let dops = DsparkOps::new(&reg)?;
        let dw = DraftWeights::load(&ctx, dir, &self.w, &self.gemv, &dops)?;
        let c = &self.w.cfg;
        if dw.cfg.taps[3] != c.n_slots() || dw.cfg.taps[..3].iter().any(|&v| v >= c.n_slots()) {
            return Err(bad(format!("draft taps {:?} do not fit {} layer visits", dw.cfg.taps, c.n_slots())));
        }
        let mut sp = Spec::new(&ctx, dw, dops, k, self.s.tmax, self.capacity, c.n_slots(), c.kv_width(), c.dim, self.w.output.n as usize)?;
        // Conv rollback table: (state, hist, channels) per slot and site.
        let (kvw, d) = (c.kv_width(), c.dim);
        let ho = hist_offsets(k + 1, kvw, d);
        let so = [0, 3 * kvw, 6 * kvw, (2 * kvw + d) * 3];
        let ch = [kvw, kvw, d, d];
        let mut t = Vec::new();
        for (cv, h) in self.conv.iter().zip(&sp.hist) {
            for i in 0..4 {
                t.extend([cv.device_ptr() + 4 * so[i] as u64, h.device_ptr() + 4 * ho[i] as u64, ch[i] as u64]);
            }
        }
        sp.table.copy_from_slice(&t)?;
        sp.entries = (t.len() / 3) as u32;
        sp.max_ch = kvw.max(d) as u32;
        eprintln!("dspark draft: {:.2} GB, k = {k}", sp.w.bytes as f64 / 1e9);
        self.spec = Some(Box::new(sp));
        Ok(())
    }

    /// Diagnostics: make speculative rounds accept against `seq` (prompt + continuation, token
    /// by position) instead of the target's greedy picks. Call before the first `spec_generate`.
    pub fn spec_force(&mut self, seq: &[u32]) -> Result<()> {
        let sp = self.spec.as_deref_mut().ok_or_else(|| bad("spec_force: speculative decoding is not enabled".into()))?;
        let mut v: Vec<i32> = seq.iter().map(|&t| t as i32).collect();
        v.resize(self.capacity as usize + 16, 0);
        sp.forced = Some(DeviceBuffer::from_slice(self.stream.context(), &v)?);
        sp.forced_host = seq.to_vec();
        sp.rounds.iter_mut().for_each(|r| *r = None);
        Ok(())
    }

    pub fn spec_stats(&self) -> Option<SpecStats> {
        self.spec.as_ref().map(|s| s.stats)
    }

    /// Greedy generation of `n` tokens after `prompt` with speculative decoding (needs
    /// [`Session::enable_spec`]). Each round is one CUDA graph: verify the anchor and `k` drafts,
    /// accept the matching prefix plus the target's next pick, roll the short-conv states back,
    /// extend the draft's context and draft the next block.
    pub fn spec_generate(&mut self, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
        let mut out = vec![self.spec_begin(prompt)?];
        while out.len() < n {
            let r = self.spec_next()?;
            if r.is_empty() {
                break;
            }
            out.extend(r);
        }
        // Diagnostics: time the draft-only graph (ASTRAL_SPEC_PROFILE=1).
        if std::env::var_os("ASTRAL_SPEC_PROFILE").is_some() {
            let Session { stream, spec, .. } = self;
            let sp = spec.as_deref_mut().expect("spec_begin checked");
            let reps = 50;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                sp.first.as_ref().expect("captured in spec_begin").launch(stream)?;
            }
            stream.synchronize()?;
            eprintln!("profile: draft step {:.3} ms (graph replay)", t.elapsed().as_secs_f64() / reps as f64 * 1e3);
        }
        out.truncate(n);
        Ok(out)
    }

    /// Starts a speculative generation (needs [`Session::enable_spec`]): prefills `prompt`
    /// from the current position (call [`Session::reset`] first for a new sequence) and drafts
    /// the first block. Returns the first generated token; [`Session::spec_next`] yields the rest.
    pub fn spec_begin(&mut self, prompt: &[u32]) -> Result<u32> {
        if self.spec.is_none() {
            return Err(bad("spec_begin: speculative decoding is not enabled".into()));
        }
        let mut next = self.prefill(prompt)?;
        if let Some(&t) = self.spec.as_ref().and_then(|sp| sp.forced_host.get(prompt.len())) {
            next = t;
        }
        let Session { w, stream, gemv, ops, rms, kc, vc, conv, s, capacity, pos, spec, dense_rows, .. } = self;
        let sp = spec.as_deref_mut().expect("checked above");
        let vmax = sp.k + 1;
        // The verify pass's dense work list (prefill left its own in work[0]).
        let pairs: Vec<(u32, u32)> = (0..vmax as u32).map(|r| (r, r)).collect();
        let mut dense = MmaWork::default();
        dense.push(0, &pairs);
        s.work[0].upload(&dense)?;
        let mut dense128 = MmaWork::default();
        dense128.push_chunked(0, &pairs, 128);
        s.work128.upload(&dense128)?;
        let mut dense256 = MmaWork::default();
        dense256.push_chunked(0, &pairs, 256);
        s.work256.upload(&dense256)?;
        let mut words = QItem::words(&QItem::dense(vmax as u32));
        words.resize(s.items_dense.len(), 0);
        s.items_dense.copy_from_slice(&words)?;
        *dense_rows = vmax;
        let mut tok = vec![0i32; s.tmax];
        tok[0] = next as i32;
        s.token.copy_from_slice(&tok)?;
        s.pos.copy_from_slice(&[*pos as i32])?;
        sp.info.copy_from_slice(&[*pos as i32, sp.last_rows as i32 - 1, *pos as i32, 0])?;
        if sp.first.is_none() {
            let kmax = sp.k;
            let g = stream.capture(|st| sp.draft(st, gemv, ops, w, &s.token, kmax))?;
            sp.first = Some(g);
        }
        // One round graph per verified length (captured once; capture runs nothing).
        let (lo, hi) = if sp.policy.adaptive { (sp.policy.kmin, sp.k) } else { (sp.k, sp.k) };
        for kv in lo..=hi {
            if sp.rounds[kv].is_some() {
                continue;
            }
            let taps = [sp.w.cfg.taps[0], sp.w.cfg.taps[1], sp.w.cfg.taps[2]];
            let nv = kv + 1;
            let g = stream.capture(|st| {
                let ex = Exec { st, gemv, ops, rms };
                {
                    let mut io = SpecIo {
                        feats: Some(&mut sp.feats),
                        taps,
                        hist: Some(&mut sp.hist),
                        hist_rows: vmax,
                        all: Some((&mut sp.logits_all, &mut sp.next_all)),
                        argmax: Some((&sp.ops, &mut sp.part_v, &mut sp.part_i)),
                    };
                    forward(&ex, w, kc, vc, conv, s, *capacity, 0, nv, None, &mut io, 0, None, None)?;
                }
                sp.ops.accept(st, &sp.next_all, &mut s.token, kv as u32, &mut sp.out, &mut sp.info, &mut s.pos, sp.forced.as_ref().map(|f| f as &dyn Dev<i32>))?;
                sp.ops.conv_commit(st, &sp.table, sp.entries, sp.max_ch, &sp.info)?;
                let at = DevIndex::of(&sp.info, 0)?;
                sp.store_ctx(st, gemv, nv, at, true)?;
                let nd = sp.policy.drafts_after(kv);
                sp.draft(st, gemv, ops, w, &s.token, nd)
            })?;
            sp.rounds[kv] = Some(g);
        }
        sp.first.as_ref().expect("captured above").launch(stream)?;
        stream.synchronize()?;
        Ok(next)
    }

    /// One speculative round after [`Session::spec_begin`]: verifies the drafts and returns the
    /// accepted tokens plus the target's next one (1..=k+1 tokens), or none when the context
    /// is full.
    pub fn spec_next(&mut self) -> Result<Vec<u32>> {
        let Session { stream, capacity, pos, spec, .. } = self;
        let sp = spec.as_deref_mut().ok_or_else(|| bad("spec_next: speculative decoding is not enabled".into()))?;
        let vmax = sp.k + 1;
        if *pos as usize + vmax + 1 > *capacity as usize {
            return Ok(Vec::new());
        }
        let kv = sp.policy.cur;
        let round = sp.rounds[kv].as_ref().ok_or_else(|| bad("spec_next: call spec_begin first".into()))?;
        let t0 = std::time::Instant::now();
        round.launch(stream)?;
        stream.synchronize()?;
        let o = sp.out.to_vec()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let c = (o[0].max(1) as usize).min(kv + 1);
        *pos += c as u32;
        sp.stats.rounds += 1;
        sp.stats.accepted += (c - 1) as u64;
        sp.stats.drafted += kv as u64;
        sp.policy.update(kv, c - 1, ms);
        Ok(o[1..=c].iter().map(|&t| t as u32).collect())
    }

    /// Whether [`Session::enable_spec`] was called.
    pub fn spec_enabled(&self) -> bool {
        self.spec.is_some()
    }

    /// Context capacity in positions.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Collect layer intermediates on every following decode step (see [`Session::take_debug`]).
    pub fn set_debug(&mut self, on: bool) {
        self.s.dbg = on.then(Vec::new);
    }

    pub fn take_debug(&mut self) -> Vec<(&'static str, Vec<f32>)> {
        self.s.dbg.as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// Logits of the last token of the last call (unpadded vocabulary), copied to the host.
    pub fn logits(&self) -> Result<Vec<f32>> {
        let mut v = self.s.logits.to_vec()?;
        v.truncate(self.w.cfg.vocab);
        Ok(v)
    }

    /// Diagnostic: one tiny eager kernel launch (argmax over the logits) on `stream` ("same") or
    /// the eager stream ("eager"), to test whether any non-graph launch slows later graph replays.
    pub fn touch(&mut self, which: &str) -> Result<()> {
        if let Some(rest) = which.strip_prefix("warmgraph_") {
            self.step(0)?;
            self.reset()?;
            return self.touch(rest);
        }
        let st = &self.eager_stream;
        match which {
            "memcpy" => {
                let d = self.w.cfg.dim;
                self.s.prelude.copy_range_from_async(0, &self.s.cur, 0, d, st)?;
            }
            "gemv" => {
                let l = &self.w.layers[0];
                self.gemv.launch_q(st, &l.wo, &self.s.qa_attn, &mut self.s.o, None)?;
            }
            "embed" => self.ops.embed(st, &self.w.tok_embd, &self.s.token, &mut self.s.inp_sa, 1)?,
            "rmsnorm" => {
                let d = self.w.cfg.dim;
                self.rms.launch(st, &self.s.inp_sa, Some(&self.w.tok_norm as &dyn Dev<f32>), &mut self.s.cur, 1, d as u32, 1e-6)?;
            }
            "mix" => {
                let d = self.w.cfg.dim as u32;
                let rs = self.s.tmax * self.w.cfg.dim;
                self.ops.mix(st, &self.s.bank, &[], rs, &self.s.cur, &self.w.output_res_query, 1, &mut self.s.m, d, 1)?;
            }
            "rms_q8" => {
                let d = self.w.cfg.dim;
                self.ops.rms_q8(st, &self.s.m, d, Some(&self.w.output_norm), Some(&mut self.s.mn), &mut self.s.qa_last, 1, 1e-6)?;
            }
            "softcap" => self.ops.softcap(st, &mut self.s.logits, 16, 30.0)?,
            "many" => {
                for _ in 0..100 {
                    self.ops.argmax(st, &self.s.logits, 16, &mut self.s.next)?;
                }
            }
            "many_same" => {
                for _ in 0..100 {
                    self.ops.argmax(&self.stream, &self.s.logits, 16, &mut self.s.next)?;
                }
                self.stream.synchronize()?;
            }
            _ => self.ops.argmax(st, &self.s.logits, 16, &mut self.s.next)?,
        }
        st.synchronize()
    }

    /// Forget the sequence: position 0, zeroed short-conv state (the KV cache is overwritten).
    pub fn reset(&mut self) -> Result<()> {
        match &self.reset_graph {
            Some(g) => g.launch(&self.stream)?,
            None => {
                for c in &mut self.conv {
                    c.zero_async(&self.stream)?;
                }
            }
        }
        self.stream.synchronize()?;
        self.pos = 0;
        Ok(())
    }
}

/// Speculative-decoding extras of one forward pass ([`SpecIo::none`] for plain passes).
pub(crate) struct SpecIo<'a> {
    /// Target features `[4][tmax][dim]`: the stream entering visits `taps`, then the final-norm
    /// output, for every token of the pass.
    pub(crate) feats: Option<&'a mut DeviceBuffer<f32>>,
    pub(crate) taps: [usize; 3],
    /// Verify: per slot, the short-conv inputs are saved here (`rows` rows per site) instead of
    /// advancing the conv state; `astral_dsp_conv_commit` applies the accepted prefix later.
    pub(crate) hist: Option<&'a mut [DeviceBuffer<f32>]>,
    pub(crate) hist_rows: usize,
    /// Verify: logits (`[ntok][vocab_padded]`) and greedy pick of every token.
    pub(crate) all: Option<(&'a mut DeviceBuffer<f32>, &'a mut DeviceBuffer<i32>)>,
    /// Verify: the multi-block argmax (ops and partials) for those picks.
    pub(crate) argmax: Option<(&'a DsparkOps, &'a mut DeviceBuffer<f32>, &'a mut DeviceBuffer<u32>)>,
}

impl SpecIo<'_> {
    pub(crate) fn none() -> Self {
        Self { feats: None, taps: [usize::MAX; 3], hist: None, hist_rows: 0, all: None, argmax: None }
    }
}

/// Rows `row0..row0 + len` of a prefill pass: tokens of sequence `seq` from position `pos0`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Seg {
    pub(crate) seq: usize,
    pub(crate) row0: usize,
    pub(crate) len: usize,
    pub(crate) pos0: u32,
}

/// The rows of a batched decode pass, as the layers see them.
#[derive(Clone, Copy)]
struct BatchRows<'a> {
    rows: SeqRows<'a>,
    pos: &'a DeviceBuffer<i32>,
}

/// A batched decode pass: row `t` of `s.token` is the next input of sequence `rows.seq[t]` at
/// position `pos[t]`; the greedy pick of every row lands in `next`.
struct BatchIo<'a> {
    rows: SeqRows<'a>,
    pos: &'a DeviceBuffer<i32>,
    /// Packed passes of short segments (verification): every row's sequence and position, for
    /// the decode attention over each row's own cache.
    verify: Option<BatchRows<'a>>,
    /// Chained verification on the decode kernels: every slot's per-row conv-state buffer.
    rowstates: Option<&'a [DeviceBuffer<f32>]>,
    logits: &'a mut DeviceBuffer<f32>,
    next: &'a mut DeviceBuffer<i32>,
    part_v: &'a mut DeviceBuffer<f32>,
    part_i: &'a mut DeviceBuffer<u32>,
    dops: &'a DsparkOps,
}

/// A slot's conv-input history for a verify pass.
struct Hist<'a> {
    buf: &'a mut DeviceBuffer<f32>,
    rows: usize,
}

/// The whole forward pass for `ntok` tokens (`s.token[..ntok]`, positions from `s.pos`), ending
/// with the greedy pick after the last token in `s.next`. With one token it enqueues work only
/// (graph-capturable); with more it syncs once per layer to group the MoE experts. `pos0` is the
/// host copy of the first position (prefill only; sizes the attention splits).
#[allow(clippy::too_many_arguments)]
fn forward(
    ex: &Exec,
    w: &Weights,
    kc: &mut [DeviceBuffer<u16>],
    vc: &mut [DeviceBuffer<u16>],
    conv: &mut [DeviceBuffer<f32>],
    s: &mut Scratch,
    capacity: u32,
    pos0: u32,
    ntok: usize,
    mut captured: Option<&mut Vec<(usize, Vec<f32>)>>,
    io: &mut SpecIo,
    seq: usize,
    mut batch: Option<&mut BatchIo>,
    segs: Option<(&[Seg], &DeviceBuffer<i32>)>,
) -> Result<()> {
    let cfg = &w.cfg;
    // Packed prefill (segs) reuses the batch buffers only for its outputs.
    let brows = if segs.is_some() { None } else { batch.as_deref().map(|b| BatchRows { rows: b.rows, pos: b.pos }) };
    let rowstates = batch.as_deref().and_then(|b| b.rowstates);
    // Packed passes: leading one-token segments (sequences that are decoding) take the batched
    // decode kernels, one launch for all of them; their sequences and positions are in the batch
    // buffers.
    let dec = match (segs, batch.as_deref()) {
        (Some((list, _)), Some(b)) => {
            let n = list.iter().enumerate().take_while(|(i, sg)| sg.len == 1 && sg.row0 == *i).count();
            (n > 0).then_some((BatchRows { rows: b.rows, pos: b.pos }, n))
        }
        _ => None,
    };
    let vrows = match (segs, batch.as_deref()) {
        (Some(_), Some(b)) if dec.is_none() => b.verify,
        _ => None,
    };
    let d = cfg.dim;
    let td = ntok * d;
    let rs = s.tmax * d;
    // Embedding and its norm.
    ex.ops.embed(ex.st, &w.tok_embd, &s.token, &mut s.inp_sa, ntok as u32)?;
    ex.norm(&s.inp_sa, &w.tok_norm, &mut s.cur, ntok, d, cfg.rms_eps)?;

    let mut blocks: Vec<u32> = Vec::new();
    let mut next_row = 0usize;
    let mut visit = 0usize;
    let taps = io.taps;
    let hist_rows = io.hist_rows;
    let mut run = |slot: usize, ip: usize, loop_i: Option<usize>, blocks: &mut Vec<u32>, next_row: &mut usize, s: &mut Scratch| -> Result<()> {
        // Diagnostic (ASTRAL_EAGER_DIAG=k): an eager pass stops after k layer slots.
        if pos0 != u32::MAX && ntok >= 1 && slot >= DIAG_LIMIT.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }
        if let Some(f) = io.feats.as_deref_mut()
            && let Some(t) = taps.iter().position(|&v| v == visit)
        {
            f.copy_range_from_async(t * s.tmax * d, &s.cur, 0, td, ex.st)?;
        }
        visit += 1;
        let mut hist = io.hist.as_deref_mut().map(|h| Hist { buf: &mut h[slot], rows: hist_rows });
        // Chained verification: this slot's row-state buffer and the conv state it mirrors.
        let brows = match (brows, rowstates) {
            (Some(mut b), Some(rs)) => {
                b.rows.rowstate = rs[slot].device_ptr();
                b.rows.conv_base = conv[slot].device_ptr();
                Some(b)
            }
            (b, _) => b,
        };
        layer(ex, w, s, &mut kc[slot], &mut vc[slot], &mut conv[slot], ip, loop_i, blocks, next_row, capacity, pos0, ntok, hist.as_mut(), seq, brows, segs, dec, vrows)?;
        if let Some(c) = captured.as_deref_mut() {
            ex.st.synchronize()?;
            let mut v = s.cur.to_vec()?;
            v.truncate(td);
            c.push((slot, v));
        }
        Ok(())
    };

    let (start, end) = (cfg.core_start, cfg.core_end);
    let core = end - start;
    for ip in 0..start {
        run(ip, ip, None, &mut blocks, &mut next_row, s)?;
    }
    // Every pass starts from the prelude output.
    s.prelude.copy_range_from_async(0, &s.cur, 0, td, ex.st)?;
    let prelude_blocks = blocks.clone();
    for pass in 0..cfg.passes {
        let mut bank: Vec<u32> = Vec::new();
        if pass > 0 {
            // The previous pass's blocks after its first core block, and its output.
            bank = blocks.get(prelude_blocks.len() + 1..).map(|b| b.to_vec()).unwrap_or_default();
            s.bank_out.copy_range_from_async(0, &s.cur, 0, td, ex.st)?;
            s.cur.copy_range_from_async(0, &s.prelude, 0, td, ex.st)?;
            blocks = prelude_blocks.clone();
        }
        for ip in start..end {
            if pass == 0 {
                run(ip, ip, None, &mut blocks, &mut next_row, s)?;
                continue;
            }
            let li = ip - start;
            let mq = &w.loops[li].msg_query;
            ex.ops.mix(ex.st, &s.bank, &bank, rs, &s.bank_out, mq, (mq.len() / d) as u32, &mut s.msgs, d as u32, ntok as u32)?;
            run(cfg.n_layer + (pass - 1) * core + li, ip, Some(li), &mut blocks, &mut next_row, s)?;
        }
    }
    for ip in end..cfg.n_layer {
        run(ip, ip, None, &mut blocks, &mut next_row, s)?;
    }

    // Speculative decoding: the final-norm output of every token (a feature), and in verify
    // passes the logits and greedy pick of every token.
    if io.feats.is_some() || io.all.is_some() {
        ex.ops.mix(ex.st, &s.bank, &blocks, rs, &s.cur, &w.output_res_query, 1, &mut s.m, d as u32, ntok as u32)?;
        ex.ops.rms_q8(ex.st, &s.m, d, Some(&w.output_norm), Some(&mut s.mn), &mut s.qa[0], ntok as u32, cfg.rms_eps)?;
        if let Some(f) = io.feats.as_deref_mut() {
            f.copy_range_from_async(3 * s.tmax * d, &s.mn, 0, td, ex.st)?;
        }
        if let Some((lg, na)) = io.all.as_mut() {
            let vp = w.output.n as usize;
            ex.mm(&w.output, &s.qa[0], lg, ntok, &s.items_dense, &s.work[0])?;
            if cfg.softcap != 0.0 {
                ex.ops.softcap(ex.st, &mut **lg, (vp * ntok) as u32, cfg.softcap)?;
            }
            match io.argmax.as_mut() {
                Some((dops, pv, pi)) => dops.argmax_rows(ex.st, &**lg, vp as u32, cfg.vocab as u32, ntok as u32, &mut **pv, &mut **pi, &mut **na)?,
                None => {
                    for t in 0..ntok {
                        ex.ops.argmax(ex.st, &lg.view(t * vp, vp)?, cfg.vocab as u32, &mut na.view_mut(t, 1)?)?;
                    }
                }
            }
            return Ok(());
        }
    }

    // Verification passes: logits and greedy pick of every row (row r into output row r).
    if let (Some(_), Some(b), Some(_)) = (segs, batch.as_deref_mut(), vrows.as_ref()) {
        if w.output.ty == GemvType::Q8_0 && ntok <= b.next.len() {
            let vp = w.output.n as usize;
            ex.ops.mix(ex.st, &s.bank, &blocks, rs, &s.cur, &w.output_res_query, 1, &mut s.m, d as u32, ntok as u32)?;
            ex.ops.rms_q8(ex.st, &s.m, d, Some(&w.output_norm), Some(&mut s.mn), &mut s.qa[0], ntok as u32, cfg.rms_eps)?;
            ex.gemv.launch_q_dense(ex.st, &mut [MultiItem { m: &w.output, x: &s.qa[0], y: &mut *b.logits }], ntok as u32)?;
            if cfg.softcap != 0.0 {
                ex.ops.softcap(ex.st, &mut *b.logits, (vp * ntok) as u32, cfg.softcap)?;
            }
            return b.dops.argmax_rows(ex.st, &*b.logits, vp as u32, cfg.vocab as u32, ntok as u32, &mut *b.part_v, &mut *b.part_i, &mut *b.next);
        }
    }
    // Packed prefill: logits and greedy pick after the last row of every segment (into the
    // batch output buffers, row i for segment i).
    if let (Some((list, _)), Some(b)) = (segs, batch.as_deref_mut()) {
        let vp = w.output.n as usize;
        // Leading one-token segments: their rows are the first rows, so one batched LM head.
        let nd = dec.as_ref().map_or(0, |d| d.1);
        if nd > 0 && w.output.ty == GemvType::Q8_0 {
            ex.ops.mix(ex.st, &s.bank, &blocks, rs, &s.cur, &w.output_res_query, 1, &mut s.m, d as u32, nd as u32)?;
            ex.ops.rms_q8(ex.st, &s.m, d, Some(&w.output_norm), Some(&mut s.mn), &mut s.qa[0], nd as u32, cfg.rms_eps)?;
            ex.gemv.launch_q_dense(ex.st, &mut [MultiItem { m: &w.output, x: &s.qa[0], y: &mut *b.logits }], nd as u32)?;
            if cfg.softcap != 0.0 {
                ex.ops.softcap(ex.st, &mut *b.logits, (vp * nd) as u32, cfg.softcap)?;
            }
            b.dops.argmax_rows(ex.st, &*b.logits, vp as u32, cfg.vocab as u32, nd as u32, &mut *b.part_v, &mut *b.part_i, &mut *b.next)?;
        }
        let skip = if w.output.ty == GemvType::Q8_0 { nd } else { 0 };
        for (i, sg) in list.iter().enumerate().skip(skip) {
            let last = (sg.row0 + sg.len - 1) * d;
            {
                let bank = s.bank.view(last, s.bank.len() - last)?;
                let cur = s.cur.view(last, d)?;
                ex.ops.mix(ex.st, &bank, &blocks, rs, &cur, &w.output_res_query, 1, &mut s.m, d as u32, 1)?;
            }
            ex.ops.rms_q8(ex.st, &s.m, d, Some(&w.output_norm), Some(&mut s.mn), &mut s.qa_last, 1, cfg.rms_eps)?;
            let mut lg = b.logits.view_mut(i * vp, vp)?;
            match w.output.ty {
                GemvType::Q8_0 => ex.mm_decode(&w.output, &s.qa_last, &mut lg, 1)?,
                GemvType::Nvfp4 => ex.gemv.launch_q(ex.st, &w.output, &s.qa_last, &mut lg, None)?,
                _ => ex.gemv.launch(ex.st, &w.output, &s.mn, 0, &mut lg)?,
            }
            if cfg.softcap != 0.0 {
                ex.ops.softcap(ex.st, &mut lg, w.output.n, cfg.softcap)?;
            }
            ex.ops.argmax(ex.st, &lg, cfg.vocab as u32, &mut b.next.view_mut(i, 1)?)?;
        }
        return Ok(());
    }

    // Batched decode: logits and greedy pick of every row.
    if let Some(b) = batch {
        let vp = w.output.n as usize;
        ex.ops.mix(ex.st, &s.bank, &blocks, rs, &s.cur, &w.output_res_query, 1, &mut s.m, d as u32, ntok as u32)?;
        ex.ops.rms_q8(ex.st, &s.m, d, Some(&w.output_norm), Some(&mut s.mn), &mut s.qa[0], ntok as u32, cfg.rms_eps)?;
        match w.output.ty {
            GemvType::Q8_0 if fast_batch(ntok) && q_tiled() && ntok > tiled_min() && ntok <= 64 && w.output.k.is_multiple_of(128) => ex.gemv.launch_q_tiled(ex.st, &mut [MultiItem { m: &w.output, x: &s.qa[0], y: &mut *b.logits }], ntok as u32)?,
            GemvType::Q8_0 if fast_batch(ntok) => ex.gemv.launch_q_dense(ex.st, &mut [MultiItem { m: &w.output, x: &s.qa[0], y: &mut *b.logits }], ntok as u32)?,
            GemvType::Q8_0 => ex.mm_decode(&w.output, &s.qa[0], &mut *b.logits, ntok)?,
            _ => ex.mm(&w.output, &s.qa[0], b.logits, ntok, &s.items_dense, &s.work[0])?,
        }
        if cfg.softcap != 0.0 {
            ex.ops.softcap(ex.st, &mut *b.logits, (vp * ntok) as u32, cfg.softcap)?;
        }
        return b.dops.argmax_rows(ex.st, &*b.logits, vp as u32, cfg.vocab as u32, ntok as u32, &mut *b.part_v, &mut *b.part_i, &mut *b.next);
    }

    // Output for the last token: depth mix, norm, LM head, softcap, greedy pick.
    let last = (ntok - 1) * d;
    {
        let bank = s.bank.view(last, s.bank.len() - last)?;
        let cur = s.cur.view(last, d)?;
        ex.ops.mix(ex.st, &bank, &blocks, rs, &cur, &w.output_res_query, 1, &mut s.m, d as u32, 1)?;
    }
    ex.ops.rms_q8(ex.st, &s.m, d, Some(&w.output_norm), Some(&mut s.mn), &mut s.qa_last, 1, cfg.rms_eps)?;
    match w.output.ty {
        GemvType::Q8_0 => ex.mm_decode(&w.output, &s.qa_last, &mut s.logits, 1)?,
        GemvType::Nvfp4 => ex.gemv.launch_q(ex.st, &w.output, &s.qa_last, &mut s.logits, None)?,
        _ => ex.gemv.launch(ex.st, &w.output, &s.mn, 0, &mut s.logits)?,
    }
    if cfg.softcap != 0.0 {
        ex.ops.softcap(ex.st, &mut s.logits, w.output.n, cfg.softcap)?;
    }
    ex.ops.argmax(ex.st, &s.logits, cfg.vocab as u32, &mut s.next)
}

/// The prefill extras when speculative decoding is on: every token's features.
fn spec_io(spec: &mut Option<Box<Spec>>) -> SpecIo<'_> {
    match spec.as_deref_mut() {
        Some(sp) => {
            let t = &sp.w.cfg.taps;
            let taps = [t[0], t[1], t[2]];
            SpecIo { feats: Some(&mut sp.feats), taps, hist: None, hist_rows: 0, all: None, argmax: None }
        }
        None => SpecIo::none(),
    }
}

/// Most attention splits of a graph-captured pass (decode, verify); `ASTRAL_ATT_SPLITS` to tune.
fn graph_splits() -> u32 {
    std::env::var("ASTRAL_ATT_SPLITS").ok().and_then(|v| v.parse().ok()).filter(|&n: &u32| n > 0).unwrap_or(32)
}

/// Prefill attention through the split-K decode kernel instead of flash attention.
fn split_prefill() -> bool {
    std::env::var_os("ASTRAL_ATTN_SPLIT_PREFILL").is_some()
}

/// Passes of 2..=8 tokens (speculative verification) use the grouped dp4a GEMV instead of the
/// tensor-core prefill GEMMs (`ASTRAL_SMALL_MMA=1` to keep the latter).
/// Batched decode of more than [`QITEM_ROWS`] rows on the prefill tensor-core GEMMs (the
/// default). `ASTRAL_EXACT_BATCH=1` keeps the batch-exact kernels, which give every row the
/// same result as decoding it alone.
fn fast_batch(ntok: usize) -> bool {
    // Rows above which batched decode takes the tensor-core kernels rather than the
    // batch-exact ones (ASTRAL_FAST_MIN, default 4: 5 to 8 rows measured 7-15% faster on tensor
    // cores, 2 to 4 rows no faster).
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let min = *MIN.get_or_init(|| std::env::var("ASTRAL_FAST_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4));
    ntok > min && std::env::var_os("ASTRAL_EXACT_BATCH").is_none()
}

/// A dense prefill product on the TMA GEMM: 256-token items when they still give two waves of
/// blocks, else 128.
fn mm_tma(ex: &Exec, m: &GemvMatrix, qa: &QAct, y: &mut DeviceBuffer<f32>, ntok: usize, w128: &DevWork, w256: &DevWork) -> Result<()> {
    let (_, tm) = m.w16.as_ref().ok_or_else(|| bad("mm_tma: no f16 copy".into()))?;
    let wide = (m.n as usize / 128) * ntok.div_ceil(256) >= 2 * 132;
    // The persistent dense GEMM (contiguous tokens; ASTRAL_DENSE_P=0: the per-tile kernel, and
    // ASTRAL_DENSE_MC=1 the two-block-cluster one).
    if m.n_mats == 1 && dense_p() {
        return ex.gemv.launch_gemm_f16_dense(ex.st, tm, m.n, m.k, qa.f16_ptr()?, ntok as u32, y);
    }
    if wide && m.n.is_multiple_of(256) && m.n_mats == 1 && dense_mc() {
        return ex.gemv.launch_gemm_f16_mc(ex.st, tm, m.n, m.k, qa.f16_ptr()?, ntok as u32, y);
    }
    let (w, tile) = if wide { (w256, 256) } else { (w128, 128) };
    let n = ntok as u32;
    ex.gemv.launch_gemm_f16_tma(ex.st, tile, tm, m.n, m.k, 1, None, qa.f16_ptr()?, n, y, n, &w.items, w.n_items, &w.xidx, &w.yidx, w.idx_len)
}

/// Largest KV-cache offset (in halves) a segment table would hold.
fn sg_max_kv(list: &[Seg], kv_span: usize) -> usize {
    list.iter().map(|sg| (sg.seq + 1) * kv_span).max().unwrap_or(0)
}

/// Experiment: prefill Q, gate, R, K, V on [`Gemv::launch_q_dense`] (`ASTRAL_PREFILL_DENSE_Q8=1`).
fn prefill_dense_q8() -> bool {
    std::env::var_os("ASTRAL_PREFILL_DENSE_Q8").is_some()
}

/// Batched decode's dense projections on the TMA f16 GEMM (`ASTRAL_DECODE_TMA=1`; measured
/// slower than the int8 kernel at 32 and 64 rows on a GH200, 21.1 vs 17.8 ms per 64-row step).
fn decode_tma() -> bool {
    std::env::var("ASTRAL_DECODE_TMA").is_ok_and(|v| v == "1")
}

/// Batched decode's dense projections on the shared-memory tiled kernel (`ASTRAL_Q_TILED=0`
/// for the register kernel).
/// Rows above which batched decode uses the smem-tiled int8 GEMM (`ASTRAL_TILED_MIN`).
fn tiled_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ASTRAL_TILED_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(32))
}

fn q_tiled() -> bool {
    std::env::var("ASTRAL_Q_TILED").map_or(true, |v| v != "0")
}

fn small_batch(ntok: usize) -> bool {
    ntok > 1 && ntok <= QITEM_ROWS && std::env::var_os("ASTRAL_SMALL_MMA").is_none()
}

/// Depth mix fused with the following RMSNorm + quantisation (`ASTRAL_NO_FUSE_MIX=1` to split).
fn fuse_mix() -> bool {
    std::env::var_os("ASTRAL_NO_FUSE_MIX").is_none()
}

/// Sink and conv-tap projections as Q8_0 inside the fused projection launch (decode and small
/// batches); `ASTRAL_SIDE_BF16=1` keeps the separate BF16 GEMVs.
fn side_q8() -> bool {
    std::env::var_os("ASTRAL_SIDE_BF16").is_none()
}

/// Decode GEMVs (single token and batched) on int8 tensor cores; `ASTRAL_DECODE_DP4A=1` for
/// the dp4a kernels (single token only: batched decode needs the tensor-core path for
/// batch-independent results).
fn decode_mma() -> bool {
    Gemv::small_mma() && std::env::var_os("ASTRAL_DECODE_DP4A").is_none()
}

/// Group MoE picks on the host (one sync per layer) instead of on the GPU.
fn host_group() -> bool {
    std::env::var_os("ASTRAL_HOST_MOE_GROUP").is_some()
}

/// Prefill GEMMs on tensor cores (default) or the dp4a grouped kernel (`ASTRAL_PREFILL_DP4A=1`).
fn use_mma() -> bool {
    std::env::var_os("ASTRAL_PREFILL_DP4A").is_none()
}

static DIAG_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Physical layer whose intermediates the debug taps collect (`ASTRAL_TAP_LAYER`, default 0).
pub fn tap_layer() -> usize {
    std::env::var("ASTRAL_TAP_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// Groups the router's (token, expert) picks by expert into gate_up and down work items.
fn moe_items(ids: &[i32], k: usize, n_expert: usize) -> (Vec<QItem>, Vec<QItem>) {
    let mut by_e: Vec<Vec<u32>> = vec![Vec::new(); n_expert];
    for (p, &e) in ids.iter().enumerate() {
        by_e[(e.max(0) as usize).min(n_expert - 1)].push(p as u32);
    }
    let (mut gu, mut dn) = (Vec::new(), Vec::new());
    for (e, pairs) in by_e.iter().enumerate() {
        for chunk in pairs.chunks(QITEM_ROWS) {
            let mut a = QItem { e: e as u32, n: chunk.len() as u32, ..Default::default() };
            let mut b = a;
            for (i, &p) in chunk.iter().enumerate() {
                a.xi[i] = p / k as u32; // gate_up reads the token's normalised input
                a.yi[i] = p;
                b.xi[i] = p; // down reads the pair's activation
                b.yi[i] = p;
            }
            gu.push(a);
            dn.push(b);
        }
    }
    (gu, dn)
}

/// One `build_layer` of llama.cpp's inkling2 graph for `ntok` tokens; `s.cur` is updated in
/// place. `ip` is the physical layer; `loop_i` selects the loop controller (later passes only).
#[allow(clippy::too_many_arguments)]
fn layer(
    ex: &Exec,
    w: &Weights,
    s: &mut Scratch,
    kc: &mut DeviceBuffer<u16>,
    vc: &mut DeviceBuffer<u16>,
    conv: &mut DeviceBuffer<f32>,
    ip: usize,
    loop_i: Option<usize>,
    blocks: &mut Vec<u32>,
    next_row: &mut usize,
    capacity: u32,
    pos0: u32,
    ntok: usize,
    mut hist: Option<&mut Hist>,
    seq: usize,
    batch: Option<BatchRows>,
    segs: Option<(&[Seg], &DeviceBuffer<i32>)>,
    dec: Option<(BatchRows, usize)>,
    verify: Option<BatchRows>,
) -> Result<()> {
    let cfg = &w.cfg;
    let l = &w.layers[ip];
    let n_dec = dec.as_ref().map_or(0, |d| d.1);
    // One sequence's cache and conv state, or (batched decode) every sequence's, rows indexed.
    let (kv_span, conv_span) = (capacity as usize * cfg.kv_width(), cfg.conv_len());
    let (kv_off, conv_off) = if batch.is_some() { (0, 0) } else { (seq * kv_span, seq * conv_span) };
    let kv_len = if batch.is_some() { kc.len() } else { kv_span };
    let conv_len = if batch.is_some() { conv.len() } else { conv_span };
    let (kc_all, vc_all, conv_all) = (kc, vc, conv);
    // The sequence segments of a prefill / verify pass (one unless prompts are packed).
    let one_seg = [Seg { seq, row0: 0, len: ntok, pos0 }];
    let seg_list: &[Seg] = segs.map_or(&one_seg[..], |(l, _)| l);
    let rows = batch.map(|b| b.rows);
    // Prefill passes (segments) pick their kernels independently of the token count, so a
    // prompt gets the same result alone or packed with others.
    let prefill = segs.is_some();
    // Per-token decode kernels: a single token, or one token of each batched sequence.
    let per_tok = batch.is_some() || (ntok == 1 && !prefill);
    // Large batched decode: grouped tensor-core products (not batch-exact).
    let fast = batch.is_some() && fast_batch(ntok);
    let hoff = hist.as_ref().map(|h| crate::dspark::hist_offsets(h.rows, cfg.kv_width(), cfg.dim));
    let (st, d, du, nt) = (ex.st, cfg.dim, cfg.dim as u32, ntok as u32);
    let (heads, kvh, hd, n_rel) = (cfg.heads as u32, cfg.kv_heads as u32, cfg.head_dim as u32, cfg.n_rel);
    let kvw = cfg.kv_width();
    let rs = s.tmax * d;
    let tap = (ntok == 1 || batch.is_some()) && s.dbg.is_some() && loop_i.is_none() && ip == tap_layer();
    // Batched decode: the taps record row ASTRAL_TAP_ROW.
    let tap_row = if batch.is_some() { std::env::var("ASTRAL_TAP_ROW").ok().and_then(|v| v.parse().ok()).unwrap_or(0usize).min(ntok - 1) } else { 0 };
    macro_rules! tap {
        ($name:literal, $buf:ident) => {
            if tap {
                st.synchronize()?;
                let v = s.$buf.to_vec()?;
                let w = v.len() / s.tmax;
                s.dbg.as_mut().unwrap().push(($name, v[tap_row * w..(tap_row + 1) * w].to_vec()));
            }
        };
    }

    // Attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V; the
    // loop controllers then make all three differ. `n_in` distinct inputs are normalised.
    let nq = l.attn_res_query.as_ref().map_or(0, |q| q.len() / d);
    let row = |j: usize| if l.swa || nq <= 1 { 0 } else { j };
    let x_stride = if nq > 0 { nq * d } else { d };
    let n_in = if loop_i.is_some() || !l.swa { 3 } else { 1 };
    // Outside the loop the mix feeds the input norms directly: one fused kernel.
    let fuse_in = fuse_mix() && loop_i.is_none() && nq == n_in && s.dbg.is_none();
    if let Some(qry) = &l.attn_res_query {
        if fuse_in {
            let norms: Vec<&DeviceBuffer<f32>> = (0..n_in)
                .map(|j| match (&l.attn_norm, &l.qkv_inp_norm) {
                    (Some(an), _) => Ok(an),
                    (None, Some(qkv)) => Ok(&qkv[j]),
                    _ => Err(bad(format!("layer {ip}: no attention norm"))),
                })
                .collect::<Result<_>>()?;
            let Scratch { bank, cur, mixed, normed, qa, .. } = &mut *s;
            let mut outs: Vec<(&dyn Dev<f32>, Option<&mut dyn DevMut<f32>>, &mut QAct)> = Vec::with_capacity(n_in);
            for ((w, y), a) in norms.iter().zip(normed.iter_mut()).zip(qa.iter_mut()).take(n_in) {
                outs.push((*w as &dyn Dev<f32>, Some(y as &mut dyn DevMut<f32>), a));
            }
            ex.ops.mix_rms_q8(st, &*bank, blocks, rs, &*cur, qry, Some(mixed as &mut dyn DevMut<f32>), du, nt, &mut outs, cfg.rms_eps)?;
        } else {
            ex.ops.mix(st, &s.bank, blocks, rs, &s.cur, qry, nq as u32, &mut s.mixed, du, nt)?;
        }
    }
    let batched_ctrl = (per_tok || (!prefill && small_batch(ntok))) && std::env::var_os("ASTRAL_NO_BATCH_CTRL").is_none();
    if let (Some(li), true) = (loop_i, batched_ctrl) {
        // Q, K and V controllers in two launches, then their three input norms in one.
        let lp = &w.loops[li];
        let rank = lp.down[0].n as usize;
        let Scratch { mixed, cur, msgs, cd, inputs, normed, qa, crms, .. } = &mut *s;
        let xs: Vec<_> = (0..3).map(|j| if nq > 0 { mixed.view(row(j) * d, mixed.len() - row(j) * d) } else { cur.view(0, cur.len()) }).collect::<Result<_>>()?;
        let ms: Vec<_> = (0..3).map(|j| msgs.view(j * d, msgs.len() - j * d)).collect::<Result<_>>()?;
        let (cd0, rest) = cd.split_at_mut_views(ntok * rank)?;
        let (cd1, cd2) = rest.split_at_mut_views(ntok * rank)?;
        let [i0, i1, i2] = inputs;
        {
            let mut cds = [cd0, cd1, cd2];
            let [c0, c1, c2] = &mut cds;
            let mut jobs: [(&GemvMatrix, &GemvMatrix, &dyn Dev<f32>, &dyn Dev<f32>, &mut dyn DevMut<f32>, &mut dyn DevMut<f32>); 3] = [
                (&lp.down[0], &lp.gain_forget[0], &xs[0], &ms[0], c0, &mut *i0),
                (&lp.down[1], &lp.gain_forget[1], &xs[1], &ms[1], c1, &mut *i1),
                (&lp.down[2], &lp.gain_forget[2], &xs[2], &ms[2], c2, &mut *i2),
            ];
            ex.ops.ctrl_multi(st, &mut jobs, x_stride, 4 * d, du, nt, !fast, &mut *crms)?;
        }
        let norm = |j: usize| -> Result<&DeviceBuffer<f32>> {
            match (&l.attn_norm, &l.qkv_inp_norm) {
                (Some(an), _) => Ok(an),
                (None, Some(qkv)) => Ok(&qkv[j]),
                _ => Err(bad(format!("layer {ip}: no attention norm"))),
            }
        };
        let [n0, n1, n2] = normed;
        let [q0, q1, q2] = qa;
        let mut rj: [(&dyn Dev<f32>, &dyn Dev<f32>, Option<&mut dyn DevMut<f32>>, &mut QAct); 3] = [(&*i0, norm(0)?, Some(n0), q0), (&*i1, norm(1)?, Some(n1), q1), (&*i2, norm(2)?, Some(n2), q2)];
        ex.ops.rms_q8_multi(st, &mut rj, d, nt, cfg.rms_eps)?;
    } else if let Some(li) = loop_i {
        let lp = &w.loops[li];
        for j in 0..3 {
            let x = if nq > 0 { s.mixed.view(row(j) * d, s.mixed.len() - row(j) * d)? } else { s.cur.view(0, s.cur.len())? };
            let msg = s.msgs.view(j * d, s.msgs.len() - j * d)?;
            if per_tok || (!prefill && small_batch(ntok)) {
                ex.ops.ctrl_down(st, &lp.down[j], &x, x_stride, &msg, 4 * d, &mut s.cd, du, nt)?;
                ex.ops.ctrl_combine(st, &lp.gain_forget[j], &s.cd, &x, x_stride, &msg, 4 * d, &mut s.inputs[j], du, nt)?;
            } else {
                ex.ops.ctrl_t(st, &lp.down[j], &lp.gain_forget[j], &x, x_stride, &msg, 4 * d, &mut s.crms, &mut s.cd, &mut s.gu, &mut s.inputs[j], du, nt)?;
            }
        }
    }
    for j in 0..n_in {
        if (fuse_in && nq > 0) || (loop_i.is_some() && batched_ctrl) {
            break;
        }
        let nw: &DeviceBuffer<f32> = match (&l.attn_norm, &l.qkv_inp_norm) {
            (Some(an), _) => an,
            (None, Some(qkv)) => &qkv[j],
            _ => return Err(bad(format!("layer {ip}: no attention norm"))),
        };
        let (x, xs) = if loop_i.is_some() {
            (s.inputs[j].view(0, s.inputs[j].len())?, d)
        } else if nq > 0 {
            (s.mixed.view(row(j) * d, s.mixed.len() - row(j) * d)?, x_stride)
        } else {
            (s.cur.view(0, s.cur.len())?, d)
        };
        ex.ops.rms_q8(st, &x, xs, Some(nw), Some(&mut s.normed[j]), &mut s.qa[j], nt, cfg.rms_eps)?;
    }
    let (ik, iv) = if n_in == 3 { (1, 2) } else { (0, 0) };
    if tap {
        st.synchronize()?;
        let v = s.normed[0].to_vec()?;
        s.dbg.as_mut().unwrap().push(("attn_in_normed", v[tap_row * d..(tap_row + 1) * d].to_vec()));
    }

    // Projections: Q, gate, R, K and V (one launch when decoding); the BF16 sink and tap
    // projections.
    // Decode and small batches: one launch for Q, gate, R, K, V and (as Q8_0 copies) the sink
    // and conv-tap projections.
    let mut side_fused = false;
    let multi_path = per_tok || (!prefill && small_batch(ntok) && [&l.wq, &l.wgate, &l.wrel, &l.wk, &l.wv].iter().all(|m| m.ty == GemvType::Q8_0 && m.n_mats == 1 && m.mat_scale.is_none()));
    if multi_path {
        let side = side_q8() && s.dbg.is_none();
        let mut items: Vec<MultiItem> = vec![
            MultiItem { m: &l.wq, x: &s.qa[0], y: &mut s.q },
            MultiItem { m: &l.wgate, x: &s.qa[0], y: &mut s.gate },
            MultiItem { m: &l.wrel, x: &s.qa[0], y: &mut s.r },
            MultiItem { m: &l.wk, x: &s.qa[ik], y: &mut s.k },
            MultiItem { m: &l.wv, x: &s.qa[iv], y: &mut s.v },
        ];
        if side {
            items.push(MultiItem { m: &l.sink_q8, x: &s.qa[0], y: &mut s.sinks });
            if let (Some(kq), Some(vq)) = (l.kp_q8.as_ref(), l.vp_q8.as_ref()) {
                items.push(MultiItem { m: kq, x: &s.qa[ik], y: &mut s.kdyn });
                items.push(MultiItem { m: vq, x: &s.qa[iv], y: &mut s.vdyn });
            }
        }
        // Decode: the tensor-core fused GEMV (measured faster than dp4a here: 174.5 vs 171.4 tok/s);
        // ASTRAL_DECODE_DP4A=1 for the dp4a kernel.
        let decode_mma = decode_mma() && l.wq.k.is_multiple_of(128);
        if ntok == 1 && decode_mma {
            ex.gemv.launch_q_multi_mma(st, &mut items, 1)?;
        } else if ntok == 1 {
            ex.gemv.launch_q_multi(st, &mut items)?;
        } else if fast && decode_tma() && items[..5].iter().all(|it| it.m.w16.is_some()) && Gemv::has_wgmma(st) {
            // Large batched decode on Hopper: the f16 TMA GEMM for Q, gate, R, K, V (the side
            // projections stay on the int8 kernel).
            let rest = items.split_off(5);
            for it in items.iter_mut() {
                let (_, tm) = it.m.w16.as_ref().expect("checked");
                let w = &s.work128;
                ex.gemv.launch_gemm_f16_tma(st, 128, tm, it.m.n, it.m.k, 1, None, it.x.f16_ptr()?, nt, &mut *it.y, nt, &w.items, w.n_items, &w.xidx, &w.yidx, w.idx_len)?;
            }
            let mut rest = rest;
            if !rest.is_empty() {
                ex.gemv.launch_q_dense(st, &mut rest, nt)?;
            }
        } else if fast && q_tiled() && nt as usize > tiled_min() && nt <= 64 && l.wq.k.is_multiple_of(128) {
            // Large batched decode: weights and activations staged through shared memory.
            ex.gemv.launch_q_tiled(st, &mut items, nt)?;
        } else if fast {
            // Large batched decode: one launch, each weight row read once for the batch.
            ex.gemv.launch_q_dense(st, &mut items, nt)?;
        } else if batch.is_some() {
            // Batched decode: every row exactly as the one-token launch computes it.
            if decode_mma {
                ex.gemv.launch_q_multi_mma_rows(st, &mut items, nt)?;
            } else {
                for it in items.iter_mut() {
                    ex.gemv.launch_q_rows(st, it.m, it.x, &mut *it.y, nt)?;
                }
            }
        } else {
            ex.gemv.launch_q_multi_t(st, &mut items, nt)?;
        }
        side_fused = side;
    } else if prefill && prefill_dense_q8() && [&l.wq, &l.wgate, &l.wrel, &l.wk, &l.wv].iter().all(|m| m.ty == GemvType::Q8_0 && m.n_mats == 1 && m.mat_scale.is_none() && m.k.is_multiple_of(128)) {
        // Experiment: the batched dense kernel for prefill.
        let mut items: Vec<MultiItem> = vec![
            MultiItem { m: &l.wq, x: &s.qa[0], y: &mut s.q },
            MultiItem { m: &l.wgate, x: &s.qa[0], y: &mut s.gate },
            MultiItem { m: &l.wrel, x: &s.qa[0], y: &mut s.r },
            MultiItem { m: &l.wk, x: &s.qa[ik], y: &mut s.k },
            MultiItem { m: &l.wv, x: &s.qa[iv], y: &mut s.v },
        ];
        if side_q8() && s.dbg.is_none() {
            items.push(MultiItem { m: &l.sink_q8, x: &s.qa[0], y: &mut s.sinks });
            if let (Some(kq), Some(vq)) = (l.kp_q8.as_ref(), l.vp_q8.as_ref()) {
                items.push(MultiItem { m: kq, x: &s.qa[ik], y: &mut s.kdyn });
                items.push(MultiItem { m: vq, x: &s.qa[iv], y: &mut s.vdyn });
            }
            side_fused = true;
        }
        ex.gemv.launch_q_dense(st, &mut items, nt)?;
    } else if prefill && Gemv::has_wgmma(st) && [&l.wq, &l.wgate, &l.wrel, &l.wk, &l.wv].iter().all(|m| m.w16.is_some()) {
        // Prefill on Hopper: the TMA GEMM per projection.
        mm_tma(ex, &l.wq, &s.qa[0], &mut s.q, ntok, &s.work128, &s.work256)?;
        mm_tma(ex, &l.wgate, &s.qa[0], &mut s.gate, ntok, &s.work128, &s.work256)?;
        mm_tma(ex, &l.wrel, &s.qa[0], &mut s.r, ntok, &s.work128, &s.work256)?;
        mm_tma(ex, &l.wk, &s.qa[ik], &mut s.k, ntok, &s.work128, &s.work256)?;
        mm_tma(ex, &l.wv, &s.qa[iv], &mut s.v, ntok, &s.work128, &s.work256)?;
    } else if prefill && Gemv::has_wgmma(st) && [&l.wq, &l.wgate, &l.wrel, &l.wk, &l.wv].iter().all(|m| m.frag16.is_some()) {
        // Prefill on Hopper: Q, gate, R, K and V in one warpgroup-MMA launch.
        let mut prods = [
            MultiItem { m: &l.wq, x: &s.qa[0], y: &mut s.q },
            MultiItem { m: &l.wgate, x: &s.qa[0], y: &mut s.gate },
            MultiItem { m: &l.wrel, x: &s.qa[0], y: &mut s.r },
            MultiItem { m: &l.wk, x: &s.qa[ik], y: &mut s.k },
            MultiItem { m: &l.wv, x: &s.qa[iv], y: &mut s.v },
        ];
        let w = &s.work128;
        ex.gemv.launch_wgmma_f16(st, &mut prods, nt, nt, &w.items, w.n_items, &w.xidx, &w.yidx, w.idx_len)?;
    } else {
        ex.mm_any(&l.wq, &s.qa[0], &mut s.q, ntok, &s.items_dense, &s.work[0], &s.work128, prefill)?;
        ex.mm_any(&l.wgate, &s.qa[0], &mut s.gate, ntok, &s.items_dense, &s.work[0], &s.work128, prefill)?;
        ex.mm_any(&l.wrel, &s.qa[0], &mut s.r, ntok, &s.items_dense, &s.work[0], &s.work128, prefill)?;
        ex.mm_any(&l.wk, &s.qa[ik], &mut s.k, ntok, &s.items_dense, &s.work[0], &s.work128, prefill)?;
        ex.mm_any(&l.wv, &s.qa[iv], &mut s.v, ntok, &s.items_dense, &s.work[0], &s.work128, prefill)?;
    }
    let (kp, vp) = (l.k_conv_proj.as_ref(), l.v_conv_proj.as_ref());
    if !side_fused && prefill && side_q8() && s.dbg.is_none() {
        // Prefill: the Q8_0 copies of the sink and conv-tap projections, one launch.
        let mut items: Vec<MultiItem> = vec![MultiItem { m: &l.sink_q8, x: &s.qa[0], y: &mut s.sinks }];
        if let (Some(kq), Some(vq)) = (l.kp_q8.as_ref(), l.vp_q8.as_ref()) {
            items.push(MultiItem { m: kq, x: &s.qa[ik], y: &mut s.kdyn });
            items.push(MultiItem { m: vq, x: &s.qa[iv], y: &mut s.vdyn });
        }
        ex.gemv.launch_q_dense(st, &mut items, nt)?;
    } else if !side_fused && prefill {
        // Prefill: tiled f32 GEMMs (the same kernels for any token count).
        ex.ops.gemm_f32_ws(st, &s.normed[0], d, &l.sink_f32, &mut s.sinks, nt, &mut s.gu)?;
        if let (Some(kq), Some(vq)) = (l.kp_f32.as_ref(), l.vp_f32.as_ref()) {
            ex.ops.gemm_f32_ws(st, &s.normed[ik], d, kq, &mut s.kdyn, nt, &mut s.gu)?;
            ex.ops.gemm_f32_ws(st, &s.normed[iv], d, vq, &mut s.vdyn, nt, &mut s.gu)?;
        }
    } else if !side_fused {
        ex.mm_f32(&l.sink, &s.normed[0], &mut s.sinks, ntok, &s.zero_ids)?;
        if let (Some(kp), Some(vp)) = (kp, vp) {
            ex.mm_f32(kp, &s.normed[ik], &mut s.kdyn, ntok, &s.zero_ids)?;
            ex.mm_f32(vp, &s.normed[iv], &mut s.vdyn, ntok, &s.zero_ids)?;
        }
    }

    // Short convs (per token in attn_prep when decoding; sequentially over the chunk when
    // prefilling), clips, value norm, log-length scale and the KV append.
    let dyn_heads = kp.map_or(0, |m| m.n / 4);
    // Prefill: convolved K and V go to s.attn and s.o (free until the attention and the output
    // projection), which attn_prep then reads.
    let shape = AttnShape { heads, kv_heads: kvh, head_dim: hd, n_rel, extent: cfg.rel_extent, window: if l.swa { cfg.n_swa } else { 0 }, capacity };
    use astral_kernels::inkling::ATT_CHUNK;
    let mut gated = false;
    if !per_tok {
        // Prefill / verify: per sequence segment (one, or several prompts packed in the pass),
        // the convs, attention prep and attention on that segment's rows and its own state.
        let one = [Seg { seq, row0: 0, len: ntok, pos0 }];
        let (list, seg_pos): (&[Seg], Option<&DeviceBuffer<i32>>) = match segs {
            Some((l, p)) => (l, Some(p)),
            None => (&one, None),
        };
        let packed = segs.is_some();
        let (q_w, r_w, dyn_w) = ((heads * hd) as usize, (heads * n_rel) as usize, kp.map_or(0, |m| m.n as usize));
        if let Some((dr, nd)) = dec.as_ref() {
            // Decoding sequences: conv, clips, value norm and KV append per row, one launch.
            let nd = *nd;
            let (q_w, r_w, dyn_w) = ((heads * hd) as usize, (heads * n_rel) as usize, kp.map_or(0, |m| m.n as usize));
            let kl = kc_all.len();
            let cl = conv_all.len();
            let mut kcv = kc_all.view_mut(0, kl)?;
            let mut vcv = vc_all.view_mut(0, kl)?;
            let mut cv = conv_all.view_mut(0, cl)?;
            let kd = s.kdyn.view(0, nd * dyn_w.max(1))?;
            let vd = s.vdyn.view(0, nd * dyn_w.max(1))?;
            ex.ops.attn_prep(
                st,
                AttnPrep {
                    q: &mut s.q.view_mut(0, nd * q_w)?,
                    r: &mut s.r.view_mut(0, nd * r_w)?,
                    k: &mut s.k.view_mut(0, nd * kvw)?,
                    v: &mut s.v.view_mut(0, nd * kvw)?,
                    q_scale: &l.q_clip,
                    r_scale: &l.rel_clip,
                    conv_kv: &mut cv,
                    k_conv: &l.k_conv,
                    v_conv: &l.v_conv,
                    kdyn: kp.map(|_| &kd as &dyn Dev<f32>),
                    vdyn: vp.map(|_| &vd as &dyn Dev<f32>),
                    dyn_heads,
                    ntok: nd as u32,
                    conv: true,
                    kc: &mut kcv,
                    vc: &mut vcv,
                    pos: dr.pos,
                    heads,
                    kv_heads: kvh,
                    head_dim: hd,
                    n_rel,
                    thr_qk: cfg.qk_clip,
                    thr_rel: cfg.rel_clip,
                    eps: cfg.rms_eps,
                    global: !l.swa,
                    log_floor: cfg.log_floor,
                    log_alpha: cfg.log_alpha,
                    capacity,
                    rows: Some(dr.rows),
                },
            )?;
        }
        // Several prompt segments without verify history: one launch per step for all of them.
        let rest = list.len() - n_dec;
        let seg_launch = packed && rest >= 2 && hist.is_none() && std::env::var_os("ASTRAL_PER_SEG").is_none();
        if seg_launch {
            let sp = seg_pos.expect("packed segments carry positions");
            let nb = sp.len() / 9;
            let table = sp.view(nb + 8 * n_dec, 8 * rest)?;
            let max_rows = list[n_dec..].iter().map(|sg| sg.len).max().unwrap_or(1) as u32;
            let dyn_w = kp.map_or(0, |m| m.n);
            ex.ops.attn_inputs_segs(
                st,
                astral_kernels::inkling::SegAttnInputs {
                    q: s.q.device_ptr(),
                    r: s.r.device_ptr(),
                    k_raw: s.k.device_ptr(),
                    v_raw: s.v.device_ptr(),
                    k_out: s.attn.device_ptr(),
                    v_out: s.o.device_ptr(),
                    kdyn: kp.map(|_| (s.kdyn.device_ptr(), dyn_w)),
                    vdyn: vp.map(|_| (s.vdyn.device_ptr(), dyn_w)),
                    dyn_heads,
                    conv: conv_all.device_ptr(),
                    k_conv: l.k_conv.device_ptr(),
                    v_conv: l.v_conv.device_ptr(),
                    q_scale: l.q_clip.device_ptr(),
                    r_scale: l.rel_clip.device_ptr(),
                    kc: kc_all.device_ptr(),
                    vc: vc_all.device_ptr(),
                    pos: sp.device_ptr(),
                    seg: &table,
                    n_seg: rest as u32,
                    max_rows,
                    heads,
                    kv_heads: kvh,
                    head_dim: hd,
                    n_rel,
                    thr_qk: cfg.qk_clip,
                    thr_rel: cfg.rel_clip,
                    eps: cfg.rms_eps,
                    global: !l.swa,
                    log_floor: cfg.log_floor,
                    log_alpha: cfg.log_alpha,
                    capacity,
                },
            )?;
        }
        for (si, sg) in list.iter().enumerate() {
            if si < n_dec || seg_launch {
                continue;
            }
            let (r0, n) = (sg.row0, sg.len);
            let nn = n as u32;
            let (ko, co) = (sg.seq * kv_span, sg.seq * conv_span);
            let mut cv = conv_all.view_mut(co, conv_span)?;
            let taps_k = kp.map(|m| -> Result<_> { Ok((s.kdyn.view(r0 * dyn_w, n * dyn_w)?, m.n as usize, dyn_heads)) }).transpose()?;
            let taps_v = vp.map(|m| -> Result<_> { Ok((s.vdyn.view(r0 * dyn_w, n * dyn_w)?, m.n as usize, dyn_heads)) }).transpose()?;
            // Verify passes: the conv kernels also save their inputs for the rollback (no state update).
            {
                let mut hk = match (hist.as_mut(), hoff) {
                    (Some(h), Some(o)) => Some(h.buf.view_mut(o[0], ntok * kvw)?),
                    _ => None,
                };
                let src = s.k.view(r0 * kvw, n * kvw)?;
                let mut dst = s.attn.view_mut(r0 * kvw, n * kvw)?;
                ex.ops.conv_par(st, &src, kvw, &mut dst, kvw, nn, &cv.view(0, 3 * kvw)?, &l.k_conv, taps_k.as_ref().map(|(v, a, b)| (v as &dyn Dev<f32>, *a, *b)), kvw as u32, hk.as_mut().map(|v| v as &mut dyn DevMut<f32>))?;
                if hk.is_none() {
                    ex.ops.conv_state(st, &src, kvw, nn, &mut cv.view_mut(0, 3 * kvw)?, kvw as u32)?;
                }
            }
            {
                let mut hv = match (hist.as_mut(), hoff) {
                    (Some(h), Some(o)) => Some(h.buf.view_mut(o[1], ntok * kvw)?),
                    _ => None,
                };
                let src = s.v.view(r0 * kvw, n * kvw)?;
                let mut dst = s.o.view_mut(r0 * kvw, n * kvw)?;
                ex.ops.conv_par(st, &src, kvw, &mut dst, kvw, nn, &cv.view(3 * kvw, 3 * kvw)?, &l.v_conv, taps_v.as_ref().map(|(v, a, b)| (v as &dyn Dev<f32>, *a, *b)), kvw as u32, hv.as_mut().map(|v| v as &mut dyn DevMut<f32>))?;
                if hv.is_none() {
                    ex.ops.conv_state(st, &src, kvw, nn, &mut cv.view_mut(3 * kvw, 3 * kvw)?, kvw as u32)?;
                }
            }
            let pos_v = match seg_pos {
                Some(p) => p.view(si, 1)?,
                None => s.pos.view(0, 1)?,
            };
            let mut kcv = kc_all.view_mut(ko, kv_span)?;
            let mut vcv = vc_all.view_mut(ko, kv_span)?;
            ex.ops.attn_prep(
                st,
                AttnPrep {
                    q: &mut s.q.view_mut(r0 * q_w, n * q_w)?,
                    r: &mut s.r.view_mut(r0 * r_w, n * r_w)?,
                    k: &mut s.attn.view_mut(r0 * kvw, n * kvw)?,
                    v: &mut s.o.view_mut(r0 * kvw, n * kvw)?,
                    q_scale: &l.q_clip,
                    r_scale: &l.rel_clip,
                    conv_kv: &mut cv,
                    k_conv: &l.k_conv,
                    v_conv: &l.v_conv,
                    kdyn: None,
                    vdyn: None,
                    dyn_heads,
                    ntok: nn,
                    conv: false,
                    kc: &mut kcv,
                    vc: &mut vcv,
                    pos: &pos_v,
                    heads,
                    kv_heads: kvh,
                    head_dim: hd,
                    n_rel,
                    thr_qk: cfg.qk_clip,
                    thr_rel: cfg.rel_clip,
                    eps: cfg.rms_eps,
                    global: !l.swa,
                    log_floor: cfg.log_floor,
                    log_alpha: cfg.log_alpha,
                    capacity,
                    rows: None,
                },
            )?;
        }
        tap!("Qcur", q);
        tap!("Kcur", k);
        tap!("Vcur", v);
        tap!("Rcur", r);
        tap!("sinks", sinks);
        // Packed prefill: every segment's attention in one launch.
        let all_flash = list.iter().all(|sg| prefill && hd == 128 && n_rel <= 16 && n_rel.is_multiple_of(4) && !split_prefill() && sg.len > 0);
        let one_launch = packed && (list.len() > 1 || n_dec > 0) && all_flash && (sg_max_kv(list, kv_span) < i32::MAX as usize) && std::env::var_os("ASTRAL_ATTN_PER_SEG").is_none();
        // Verification passes (short segments, rows' sequences and positions given): the decode
        // attention per row over its sequence's cache, which already holds the pass's K / V.
        let per_row = verify.is_some() && n_dec == 0 && list.iter().all(|sg| sg.len <= 8) && std::env::var_os("ASTRAL_VERIFY_SEGS").is_none();
        if per_row {
            let vr = verify.as_ref().expect("checked");
            let splits = capacity.div_ceil(ATT_CHUNK);
            ex.ops.attn_decode_split(st, &s.q, &s.r, &s.sinks, &*kc_all, &*vc_all, &l.rel_bias, &mut s.attn_work, &mut s.attn, vr.pos, shape, nt, splits, false, None, Some(vr.rows))?;
        }
        if one_launch && !per_row {
            if let Some((dr, nd)) = dec.as_ref() {
                // Decoding sequences: the batched decode attention over each row's own cache.
                let splits = capacity.div_ceil(ATT_CHUNK);
                ex.ops.attn_decode_split(st, &s.q, &s.r, &s.sinks, &*kc_all, &*vc_all, &l.rel_bias, &mut s.attn_work, &mut s.attn, dr.pos, shape, *nd as u32, splits, false, None, Some(dr.rows))?;
            }
            if list.len() > n_dec {
                let sp = seg_pos.expect("packed segments carry positions");
                let nb = sp.len() / 9;
                let table = sp.view(nb + 8 * n_dec, 8 * (list.len() - n_dec))?;
                let max_rows = list[n_dec..].iter().map(|sg| sg.len).max().unwrap_or(1) as u32;
                let (q, r, sk) = (&s.q, &s.r, &s.sinks);
                ex.ops.attn_prefill_segs(st, q, r, sk, &*kc_all, &*vc_all, &l.rel_bias, &mut s.attn, &sp.view(0, nb)?, &table, (list.len() - n_dec) as u32, max_rows, nt, shape)?;
            }
        }
        for (si, sg) in list.iter().enumerate() {
            if one_launch || per_row {
                break;
            }
            let (r0, n) = (sg.row0, sg.len);
            let nn = n as u32;
            let ko = sg.seq * kv_span;
            let pos_v = match seg_pos {
                Some(p) => p.view(si, 1)?,
                None => s.pos.view(0, 1)?,
            };
            let kcv = kc_all.view(ko, kv_span)?;
            let vcv = vc_all.view(ko, kv_span)?;
            let q = s.q.view(r0 * q_w, n * q_w)?;
            let r = s.r.view(r0 * r_w, n * r_w)?;
            let sk = s.sinks.view(r0 * heads as usize, n * heads as usize)?;
            if (prefill || !small_batch(n)) && hd == 128 && n_rel <= 16 && n_rel.is_multiple_of(4) && !split_prefill() {
                let mut out = s.attn.view_mut(r0 * q_w, n * q_w)?;
                ex.ops.attn_prefill(st, &q, &r, &sk, &kcv, &vcv, &l.rel_bias, &mut out, &pos_v, shape, nn)?;
            } else {
                // Verify passes (graph-captured at any position): a capped split count, each split
                // looping over chunks; prefill fallback: exact splits for the segment's last position.
                let chunk_loop = !prefill && small_batch(n);
                let splits = if chunk_loop { capacity.div_ceil(ATT_CHUNK).min(graph_splits()) } else { (sg.pos0 + nn).div_ceil(ATT_CHUNK) };
                // The merge also applies the output gate and quantises for the O projection (one
                // unpacked segment only: the quantised rows start at row 0).
                let fuse_gate = s.dbg.is_none() && !packed;
                let gate = if fuse_gate { Some((&s.gate as &dyn Dev<f32>, &mut s.qa_attn)) } else { None };
                let mut out = s.attn.view_mut(r0 * q_w, n * q_w)?;
                ex.ops.attn_decode_split(st, &q, &r, &sk, &kcv, &vcv, &l.rel_bias, &mut s.attn_work, &mut out, &pos_v, shape, nn, splits, chunk_loop, gate, None)?;
                gated = fuse_gate;
            }
        }
    } else {
        let mut kc = kc_all.view_mut(kv_off, kv_len)?;
        let mut vc = vc_all.view_mut(kv_off, kv_len)?;
        let mut conv = conv_all.view_mut(conv_off, conv_len)?;
        ex.ops.attn_prep(
            st,
            AttnPrep {
                q: &mut s.q,
                r: &mut s.r,
                k: &mut s.k,
                v: &mut s.v,
                q_scale: &l.q_clip,
                r_scale: &l.rel_clip,
                conv_kv: &mut conv.view_mut(0, conv_len)?,
                k_conv: &l.k_conv,
                v_conv: &l.v_conv,
                kdyn: kp.map(|_| &s.kdyn as &dyn Dev<f32>),
                vdyn: vp.map(|_| &s.vdyn as &dyn Dev<f32>),
                dyn_heads,
                ntok: nt,
                conv: true,
                kc: &mut kc,
                vc: &mut vc,
                pos: match batch {
                    Some(b) => b.pos,
                    None => &s.pos,
                },
                heads,
                kv_heads: kvh,
                head_dim: hd,
                n_rel,
                thr_qk: cfg.qk_clip,
                thr_rel: cfg.rel_clip,
                eps: cfg.rms_eps,
                global: !l.swa,
                log_floor: cfg.log_floor,
                log_alpha: cfg.log_alpha,
                capacity,
                rows,
            },
        )?;
        tap!("Qcur", q);
        tap!("Kcur", k);
        tap!("Vcur", v);
        tap!("Rcur", r);
        tap!("sinks", sinks);

        // Attention, output gate, output projection, post conv into the residual stream. The
        // first layer of each block starts a new stream; the old one joins the blocks.
        if batch.is_none() && ntok == 1 && (hd != 128 || n_rel > 32 || std::env::var_os("ASTRAL_ATTN_SIMPLE").is_some()) {
            ex.ops.attn_decode(st, &s.q, &s.r, &s.sinks, &kc, &vc, &l.rel_bias, &mut s.attn, &s.pos, shape)?;
        } else {
            // Decode: one chunk per split over the whole capacity (graph-capturable, measured
            // fastest). The merge also applies the output gate and quantises for the O
            // projection (except when the debug taps want the ungated attention output).
            let splits = capacity.div_ceil(ATT_CHUNK);
            let fuse_gate = s.dbg.is_none();
            let gate = if fuse_gate { Some((&s.gate as &dyn Dev<f32>, &mut s.qa_attn)) } else { None };
            let pos = match batch {
                Some(b) => b.pos,
                None => &s.pos,
            };
            ex.ops.attn_decode_split(st, &s.q, &s.r, &s.sinks, &kc, &vc, &l.rel_bias, &mut s.attn_work, &mut s.attn, pos, shape, nt, splits, false, gate, rows)?;
            gated = fuse_gate;
        }
    }
    tap!("kqv_out", attn);
    if !gated {
        ex.ops.gate_q8(st, &mut s.attn, &s.gate, &mut s.qa_attn, nt)?;
    }
    if fast && decode_tma() && l.wo.w16.is_some() && Gemv::has_wgmma(st) {
        let (_, tm) = l.wo.w16.as_ref().expect("checked");
        let w = &s.work128;
        ex.gemv.launch_gemm_f16_tma(st, 128, tm, l.wo.n, l.wo.k, 1, None, s.qa_attn.f16_ptr()?, nt, &mut s.o, nt, &w.items, w.n_items, &w.xidx, &w.yidx, w.idx_len)?;
    } else if fast && q_tiled() && nt as usize > tiled_min() && nt <= 64 && l.wo.k.is_multiple_of(128) {
        ex.gemv.launch_q_tiled(st, &mut [MultiItem { m: &l.wo, x: &s.qa_attn, y: &mut s.o }], nt)?;
    } else if fast {
        ex.gemv.launch_q_dense(st, &mut [MultiItem { m: &l.wo, x: &s.qa_attn, y: &mut s.o }], nt)?;
    } else if per_tok {
        ex.mm_decode(&l.wo, &s.qa_attn, &mut s.o, ntok)?;
    } else {
        if prefill && l.wo.w16.is_some() && Gemv::has_wgmma(st) {
            mm_tma(ex, &l.wo, &s.qa_attn, &mut s.o, ntok, &s.work128, &s.work256)?;
        } else {
            ex.mm_any(&l.wo, &s.qa_attn, &mut s.o, ntok, &s.items_dense, &s.work[0], &s.work128, prefill)?;
        }
    }
    let starts_block = ip % cfg.attn_res_block == 0;
    if starts_block && *next_row >= BANK_ROWS {
        return Err(bad("attention-residual bank full".into()));
    }
    {
        let mut bank_row = if starts_block { Some(s.bank.view_mut(*next_row * rs, rs)?) } else { None };
        if per_tok {
            let br = bank_row.as_mut().map(|b| b as &mut dyn DevMut<f32>);
            let mut conv = conv_all.view_mut(conv_off, conv_len)?;
            ex.ops.conv_residual(st, &s.o, None, nt, &mut conv.view_mut(2 * kvw * 3, conv_len - 2 * kvw * 3)?, &l.attn_post_conv, &mut s.cur, br, du, rows)?;
        } else {
            if let Some((dr, nd)) = dec.as_ref() {
                // Decoding sequences: the post conv into the residual stream, one launch.
                let nd = *nd;
                let cl = conv_all.len();
                let o = s.o.view(0, nd * d)?;
                let mut cur = s.cur.view_mut(0, nd * d)?;
                let mut bv = match bank_row.as_mut() {
                    Some(b) => Some(b.view_mut(0, nd * d)?),
                    None => None,
                };
                ex.ops.conv_residual(st, &o, None, nd as u32, &mut conv_all.view_mut(2 * kvw * 3, cl - 2 * kvw * 3)?, &l.attn_post_conv, &mut cur, bv.as_mut().map(|v| v as &mut dyn DevMut<f32>), du, Some(dr.rows))?;
            }
            let rest = seg_list.len() - n_dec;
            let seg_launch = segs.is_some() && rest >= 2 && hist.is_none() && std::env::var_os("ASTRAL_PER_SEG").is_none();
            if seg_launch {
                let sp = segs.expect("checked").1;
                let nb = sp.len() / 9;
                let table = sp.view(nb + 8 * n_dec, 8 * rest)?;
                let max_rows = seg_list[n_dec..].iter().map(|sg| sg.len).max().unwrap_or(1) as u32;
                let bank = bank_row.as_ref().map(|b| b.dptr());
                ex.ops.conv_residual_segs(st, s.o.device_ptr(), conv_all.device_ptr() + 4 * (2 * kvw * 3) as u64, l.attn_post_conv.device_ptr(), s.cur.device_ptr(), bank, du, &table, rest as u32, max_rows)?;
            }
            for sg in seg_list.iter().skip(if seg_launch { seg_list.len() } else { n_dec }) {
                let mut cv = conv_all.view_mut(sg.seq * conv_span, conv_span)?;
                let (r0, n) = (sg.row0 * d, sg.len * d);
                let mut ha = match (hist.as_mut(), hoff) {
                    (Some(h), Some(o)) => Some(h.buf.view_mut(o[2], ntok * d)?),
                    _ => None,
                };
                let src = s.o.view(r0, n)?;
                let mut cur = s.cur.view_mut(r0, n)?;
                let mut bv = match bank_row.as_mut() {
                    Some(b) => Some(b.view_mut(r0, n)?),
                    None => None,
                };
                ex.ops.conv_res_par(st, &src, sg.len as u32, &cv.view(2 * kvw * 3, 3 * d)?, &l.attn_post_conv, &mut cur, bv.as_mut().map(|v| v as &mut dyn DevMut<f32>), du, ha.as_mut().map(|v| v as &mut dyn DevMut<f32>))?;
                if ha.is_none() {
                    ex.ops.conv_state(st, &src, d, sg.len as u32, &mut cv.view_mut(2 * kvw * 3, 3 * d)?, du)?;
                }
            }
        }
    }
    if starts_block {
        blocks.push(*next_row as u32);
        *next_row += 1;
    }
    tap!("ffn_inp", cur);

    // MoE.
    let fuse_ffn = fuse_mix() && loop_i.is_none() && s.dbg.is_none();
    if fuse_ffn {
        let Scratch { bank, cur, m, mn, qa, .. } = &mut *s;
        let mut outs: [(&dyn Dev<f32>, Option<&mut dyn DevMut<f32>>, &mut QAct); 1] = [(&l.ffn_norm, Some(mn as &mut dyn DevMut<f32>), &mut qa[0])];
        ex.ops.mix_rms_q8(st, &*bank, blocks, rs, &*cur, &l.ffn_res_query, Some(m as &mut dyn DevMut<f32>), du, nt, &mut outs, cfg.rms_eps)?;
    } else {
        ex.ops.mix(st, &s.bank, blocks, rs, &s.cur, &l.ffn_res_query, 1, &mut s.m, du, nt)?;
    }
    if let Some(li) = loop_i {
        let lp = &w.loops[li];
        if per_tok || (!prefill && small_batch(ntok)) {
            // The same launch as the attention-input controllers (one job), so one token or many
            // share the kernels and their per-token arithmetic.
            let Scratch { msgs, m, cd, ctmp, crms, .. } = &mut *s;
            let msg = msgs.view(3 * d, msgs.len() - 3 * d)?;
            let rank = lp.down[3].n as usize;
            let mut cdv = cd.view_mut(0, ntok * rank)?;
            #[allow(clippy::type_complexity)]
            let mut jobs: [(&GemvMatrix, &GemvMatrix, &dyn Dev<f32>, &dyn Dev<f32>, &mut dyn DevMut<f32>, &mut dyn DevMut<f32>); 1] = [(&lp.down[3], &lp.gain_forget[3], &*m, &msg, &mut cdv, &mut *ctmp)];
            ex.ops.ctrl_multi(st, &mut jobs, d, 4 * d, du, nt, !fast, &mut *crms)?;
        } else {
            let msg = s.msgs.view(3 * d, s.msgs.len() - 3 * d)?;
            ex.ops.ctrl_t(st, &lp.down[3], &lp.gain_forget[3], &s.m, d, &msg, 4 * d, &mut s.crms, &mut s.cd, &mut s.gu, &mut s.ctmp, du, nt)?;
        }
    }
    if !fuse_ffn {
        let ffn_x: &DeviceBuffer<f32> = if loop_i.is_some() { &s.ctmp } else { &s.m };
        ex.ops.rms_q8(st, ffn_x, d, Some(&l.ffn_norm), Some(&mut s.mn), &mut s.qa[0], nt, cfg.rms_eps)?;
    }
    tap!("ffn_normed", mn);
    let (ne, k) = (cfg.n_expert as u32, cfg.n_expert_used as u32);
    // Set when the router launch also wrote the small-batch expert work items.
    let mut grouped = false;
    if fast && l.router.ty == GemvType::F32 && s.dbg.is_none() {
        // Large batches: tiled logits, then the top-k. ASTRAL_ROUTER_FULL=1: logits, top-k and
        // the expert work items in one launch (its serial last-block tail measured slower).
        if std::env::var_os("ASTRAL_ROUTER_FULL").is_some() {
            let bias = l.probs_b.as_ref().map(|b| b as &dyn Dev<f32>);
            let mi = astral_kernels::inkling::moe_items_bound(nt * k, ne);
            ex.ops.router_b_full(st, &l.router, &s.mn, &mut s.rlogits, bias, &mut s.ids, &mut s.w, k, cfg.expert_scale, nt, &mut s.router_cnt, &mut s.items_gu, &mut s.items_dn, mi)?;
            grouped = true;
        } else {
            ex.ops.router_logits_b(st, &l.router, &s.mn, &mut s.rlogits, nt)?;
            ex.ops.router_topk(st, &s.rlogits, l.probs_b.as_ref(), &mut s.ids, &mut s.w, ne, k, cfg.expert_scale, nt)?;
        }
    } else if l.router.ty == GemvType::F32 && !prefill && (ntok <= QITEM_ROWS || batch.is_some()) && s.dbg.is_none() && std::env::var_os("ASTRAL_NO_FUSE_ROUTER").is_none() {
        // Router GEMV + top-k in one kernel (the last block per token selects).
        let bias = l.probs_b.as_ref().map(|b| b as &dyn Dev<f32>);
        ex.ops.router_fused(st, &l.router, &s.mn, &mut s.rlogits, bias, &mut s.ids, &mut s.w, k, cfg.expert_scale, nt, &mut s.router_cnt)?;
    } else {
        if prefill {
            ex.ops.gemm_f32_ws(st, &s.mn, d, &l.router, &mut s.rlogits, nt, &mut s.gu)?;
        } else {
            ex.mm_f32(&l.router, &s.mn, &mut s.rlogits, ntok, &s.zero_ids)?;
        }
        tap!("ffn_moe_logits", rlogits);
        ex.ops.router_topk(st, &s.rlogits, l.probs_b.as_ref(), &mut s.ids, &mut s.w, ne, k, cfg.expert_scale, nt)?;
    }
    let pairs = ntok * k as usize;
    // Set when the expert GEMM already wrote the weighted combine into s.o.
    let mut combined = false;
    // Set when the expert GEMM wrote f16 per-pair outputs.
    let mut y16 = false;
    let fuse_combine = std::env::var("ASTRAL_FUSE_COMBINE").map_or(true, |v| v != "0");
    if ntok == 1 && !prefill {
        let ids = s.ids.view(0, k as usize)?;
        ex.gemv.launch_q(st, &l.gate_up, &s.qa[0], &mut s.gu, Some(&ids as &dyn Dev<i32>))?;
        ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, k, cfg.situ_beta, cfg.situ_lb)?;
        ex.gemv.launch_q(st, &l.down, &s.qa_ff, &mut s.y, Some(&ids as &dyn Dev<i32>))?;
    } else {
        let p32 = pairs as u32;
        // Mid-sized packed passes (a decode batch plus a prompt chunk, ASTRAL_MOE_ITEMS_MAX
        // pairs or fewer) can take the decode path's 8-token items instead of the prefill GEMM.
        let items_max: usize = std::env::var("ASTRAL_MOE_ITEMS_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let items_ok = s.qa[0].f16_ptr().is_ok() && l.gate_up.frag.is_some() && pairs <= items_max && pairs <= s.items_gu.len() / QItem::WORDS * 8;
        if fast || (prefill && items_ok) {
            // Large batched decode: tensor-core grouped products, each expert's weights read
            // once per 8 of its tokens.
            let mi = astral_kernels::inkling::moe_items_bound(p32, cfg.n_expert as u32);
            if !grouped {
                ex.ops.moe_group_q(st, &s.ids, p32, k, cfg.n_expert as u32, &mut s.items_gu, &mut s.items_dn, mi)?;
            }
            // NVFP4 experts: SITU in the gate_up epilogue (the down kernel reads the f16 copy).
            if fuse_situ() && l.gate_up.ty == GemvType::Nvfp4 && l.down.ty == GemvType::Nvfp4 {
                ex.gemv.launch_items_mma_act(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &s.items_gu, mi, Some((&mut s.qa_ff, cfg.situ_beta, cfg.situ_lb)))?;
            } else {
                ex.gemv.launch_items_mma(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &s.items_gu, mi)?;
                ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
            }
            ex.gemv.launch_items_mma(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &s.items_dn, mi)?;
        } else if batch.is_some() {
            // Batched decode, every (token, expert) pair as the one-token dp4a GEMV computes it.
            let mi = astral_kernels::inkling::moe_items_bound(p32, cfg.n_expert as u32);
            ex.ops.moe_group_q(st, &s.ids, p32, k, cfg.n_expert as u32, &mut s.items_gu, &mut s.items_dn, mi)?;
            ex.gemv.launch_items_exact(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &s.items_gu, mi)?;
            ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
            ex.gemv.launch_items_exact(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &s.items_dn, mi)?;
        } else if !prefill && small_batch(ntok) {
            // Few tokens (verify passes): dp4a grouped GEMV, each used expert read once, grouped
            // on the GPU (graph-capturable).
            let mi = astral_kernels::inkling::moe_items_bound(p32, cfg.n_expert as u32);
            ex.ops.moe_group_q(st, &s.ids, p32, k, cfg.n_expert as u32, &mut s.items_gu, &mut s.items_dn, mi)?;
            ex.gemv.launch_items(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &s.items_gu, mi)?;
            ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
            ex.gemv.launch_items(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &s.items_dn, mi)?;
        } else if use_mma() && !host_group() {
            // Group the (token, expert) pairs by expert on the GPU: no host sync.
            // Few tokens (verify passes): launch only as many items as can be used.
            // Hopper with fragment-layout experts: the warp-specialised GEMM, with 64-token items
            // when experts average few tokens (mixed decode + prompt passes: one item per expert
            // without padding a 128-token tile) and 128-token items otherwise.
            let wg = |m: &GemvMatrix| m.frag.is_some() && Gemv::has_wgmma(st);
            let tma = |m: &GemvMatrix| m.w16.is_some() && Gemv::has_wgmma(st);
            let ws = wg(&l.gate_up) && wg(&l.down) && l.gate_up.n.is_multiple_of(256) && l.down.n.is_multiple_of(256) && std::env::var_os("ASTRAL_NO_WS256").is_none();
            let avg = pairs / cfg.n_expert.max(1);
            let tile: u32 = match std::env::var("ASTRAL_MOE_TILE").ok().and_then(|v| v.parse().ok()) {
                Some(t @ (32 | 64 | 128 | 256)) => t,
                _ if avg <= 96 => 64,
                // 256 (128 weight rows x 256 tokens, ASTRAL_MOE_TILE=256) halves the dequantisation
                // per MMA but measured slower on real routing: uneven experts pad more.
                _ => 128,
            };
            let chunk = if ws { tile } else if (wg(&l.gate_up) && wg(&l.down)) || (tma(&l.gate_up) && tma(&l.down)) { 128 } else { 64 };
            let max_items = ((s.work[1].items.len() / 3) as u32).min((p32.div_ceil(chunk.min(64)) + cfg.n_expert as u32).min(p32));
            let [_, w1, w2] = &mut s.work;
            ex.ops.moe_group(st, &s.ids, p32, k, cfg.n_expert as u32, (&mut w1.items, &mut w1.xidx, &mut w1.yidx), (&mut w2.items, &mut w2.xidx, &mut w2.yidx), max_items, chunk)?;
            let (w1, w2) = (&s.work[1], &s.work[2]);
            // Experts with a fragment layout: the warpgroup-MMA GEMM (Hopper).
            if tma(&l.gate_up) && chunk == 128 {
                let m = &l.gate_up;
                let (_, tm) = m.w16.as_ref().expect("checked");
                ex.gemv.launch_gemm_f16_tma(st, 128, tm, m.n, m.k, m.n_mats, m.mat_scale.as_ref(), s.qa[0].f16_ptr()?, nt, &mut s.gu, p32, &w1.items, max_items, &w1.xidx, &w1.yidx, p32)?;
            } else if ws {
                // SITU in the GEMM epilogue: the down GEMM reads only the f16 activations
                // (ASTRAL_FUSE_SITU=0: the separate kernel).
                let situ = fuse_situ().then_some((&mut s.qa_ff, cfg.situ_beta, cfg.situ_lb));
                ex.gemv.launch_ws256(st, chunk, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &w1.items, max_items, &w1.xidx, &w1.yidx, p32, situ, None, false)?;
            } else if wg(&l.gate_up) {
                ex.gemv.launch_mma5(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &w1.items, max_items, &w1.xidx, &w1.yidx, p32)?;
            } else {
                ex.gemv.launch_mma(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &w1.items, max_items, &w1.xidx, &w1.yidx, p32)?;
            }
            if !ws || !fuse_situ() || (tma(&l.gate_up) && chunk == 128) {
                ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
            }
            if tma(&l.down) && chunk == 128 {
                let m = &l.down;
                let (_, tm) = m.w16.as_ref().expect("checked");
                ex.gemv.launch_gemm_f16_tma(st, 128, tm, m.n, m.k, m.n_mats, m.mat_scale.as_ref(), s.qa_ff.f16_ptr()?, p32, &mut s.y, p32, &w2.items, max_items, &w2.xidx, &w2.yidx, p32)?;
            } else if ws {
                // The weighted combine in the GEMM epilogue (atomic adds into the zeroed s.o) when
                // nothing needs the per-pair outputs (ASTRAL_FUSE_COMBINE=0: separate kernel).
                let comb = if fuse_combine && !per_tok && std::env::var("ASTRAL_FUSE_COMBINE").is_ok_and(|v| v == "1") {
                    s.o.zero_range_async(0, ntok * d, st)?;
                    combined = true;
                    Some((&mut s.o as &mut dyn DevMut<f32>, &s.w as &dyn Dev<f32>, k))
                } else {
                    None
                };
                // Otherwise f16 per-pair outputs for the combine kernel (ASTRAL_Y16=0: f32).
                y16 = comb.is_none() && !per_tok && std::env::var("ASTRAL_Y16").map_or(true, |v| v != "0");
                ex.gemv.launch_ws256(st, chunk, &l.down, &s.qa_ff, p32, &mut s.y, p32, &w2.items, max_items, &w2.xidx, &w2.yidx, p32, None, comb, y16)?;
            } else if wg(&l.down) {
                ex.gemv.launch_mma5(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &w2.items, max_items, &w2.xidx, &w2.yidx, p32)?;
            } else {
                ex.gemv.launch_mma(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &w2.items, max_items, &w2.xidx, &w2.yidx, p32)?;
            }
        } else {
            // Group the (token, expert) pairs by expert on the host, so each expert's weights are
            // read once per 8 (dp4a) or 64 (tensor cores) of its tokens.
            st.synchronize()?;
            let mut ids = s.ids.to_vec()?;
            ids.truncate(pairs);
            if use_mma() {
                let (mut gu, mut dn) = (MmaWork::default(), MmaWork::default());
                let mut by_e: Vec<Vec<u32>> = vec![Vec::new(); cfg.n_expert];
                for (p, &e) in ids.iter().enumerate() {
                    by_e[(e.max(0) as usize).min(cfg.n_expert - 1)].push(p as u32);
                }
                for (e, ps) in by_e.iter().enumerate() {
                    if ps.is_empty() {
                        continue;
                    }
                    let g: Vec<(u32, u32)> = ps.iter().map(|&p| (p / k, p)).collect();
                    let dd: Vec<(u32, u32)> = ps.iter().map(|&p| (p, p)).collect();
                    gu.push(e as u32, &g);
                    dn.push(e as u32, &dd);
                }
                s.work[1].upload(&gu)?;
                s.work[2].upload(&dn)?;
                let (w1, w2) = (&s.work[1], &s.work[2]);
                ex.gemv.launch_mma(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &w1.items, w1.n_items, &w1.xidx, &w1.yidx, w1.idx_len)?;
                ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
                ex.gemv.launch_mma(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &w2.items, w2.n_items, &w2.xidx, &w2.yidx, w2.idx_len)?;
            } else {
                let (gu_items, dn_items) = moe_items(&ids, k as usize, cfg.n_expert);
                for (items, buf) in [(&gu_items, &mut s.items_gu), (&dn_items, &mut s.items_dn)] {
                    let mut words = QItem::words(items);
                    if words.len() > buf.len() {
                        return Err(bad("moe: too many work items".into()));
                    }
                    words.resize(buf.len(), 0);
                    buf.copy_from_slice(&words)?;
                }
                ex.gemv.launch_items(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &s.items_gu, gu_items.len() as u32)?;
                ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
                ex.gemv.launch_items(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &s.items_dn, dn_items.len() as u32)?;
            }
        }
    }
    // Combine, post conv and the residual add.
    let wv = s.w.view(0, pairs)?;
    let st_off = (2 * kvw + d) * 3;
    if per_tok {
        let mut conv = conv_all.view_mut(conv_off, conv_len)?;
        ex.ops.conv_residual(st, &s.y, Some((&wv as &dyn Dev<f32>, k)), nt, &mut conv.view_mut(st_off, conv_len - st_off)?, &l.ffn_post_conv, &mut s.cur, None, du, rows)
    } else {
        if y16 {
            ex.ops.moe_combine_t16(st, s.y.dptr(), &wv, &mut s.o, du, k, nt)?;
        } else if !combined {
            ex.ops.moe_combine_t(st, &s.y, &wv, &mut s.o, du, k, nt)?;
        }
        if let Some((dr, nd)) = dec.as_ref() {
            let nd = *nd;
            let cl = conv_all.len();
            let o = s.o.view(0, nd * d)?;
            let mut cur = s.cur.view_mut(0, nd * d)?;
            ex.ops.conv_residual(st, &o, None, nd as u32, &mut conv_all.view_mut(st_off, cl - st_off)?, &l.ffn_post_conv, &mut cur, None, du, Some(dr.rows))?;
        }
        let rest = seg_list.len() - n_dec;
        let seg_launch = segs.is_some() && rest >= 2 && hist.is_none() && std::env::var_os("ASTRAL_PER_SEG").is_none();
        if seg_launch {
            let sp = segs.expect("checked").1;
            let nb = sp.len() / 9;
            let table = sp.view(nb + 8 * n_dec, 8 * rest)?;
            let max_rows = seg_list[n_dec..].iter().map(|sg| sg.len).max().unwrap_or(1) as u32;
            ex.ops.conv_residual_segs(st, s.o.device_ptr(), conv_all.device_ptr() + 4 * st_off as u64, l.ffn_post_conv.device_ptr(), s.cur.device_ptr(), None, du, &table, rest as u32, max_rows)?;
        }
        for sg in seg_list.iter().skip(if seg_launch { seg_list.len() } else { n_dec }) {
            let mut cv = conv_all.view_mut(sg.seq * conv_span, conv_span)?;
            let (r0, n) = (sg.row0 * d, sg.len * d);
            let mut hf = match (hist.as_mut(), hoff) {
                (Some(h), Some(o)) => Some(h.buf.view_mut(o[3], ntok * d)?),
                _ => None,
            };
            let src = s.o.view(r0, n)?;
            let mut cur = s.cur.view_mut(r0, n)?;
            ex.ops.conv_res_par(st, &src, sg.len as u32, &cv.view(st_off, 3 * d)?, &l.ffn_post_conv, &mut cur, None, du, hf.as_mut().map(|v| v as &mut dyn DevMut<f32>))?;
            if hf.is_none() {
                ex.ops.conv_state(st, &src, d, sg.len as u32, &mut cv.view_mut(st_off, 3 * d)?, du)?;
            }
        }
        Ok(())
    }
}
