// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Inkling 2 DSpark speculative decoding with the checkpoint's own draft (`mtp.safetensors`,
//! `mtp_config` in `config.json`).
//!
//! The draft is a 3-layer dense transformer. Each target token leaves four hidden states (the
//! residual stream entering layer visits 5, 23 and 41, and the final-norm output); every draft
//! layer mixes them (softmax of `fuse_weight`), normalises and projects them to its own K/V
//! cache ("context"). A draft step then runs one block of 8 tokens — the anchor (the last
//! accepted token, by its normalised target embedding) and 7 `mask_embed` rows at the next
//! positions — bidirectionally over that context, and a recurrent Markov head turns the 7 mask
//! outputs into drafted tokens left to right (greedy). The target verifies the anchor plus `k`
//! drafts in one batched pass; the short-conv states are rolled back to the accepted prefix.
//!
//! Matches vLLM's `Inkling2DSparkModel` (greedy): weights are Q8_0 here (BF16 there), and the
//! Markov head's `hidden_out` is folded into the target LM head (`fold = lm_head @ hidden_out`).

use crate::inkling::{DevWork, Weights as TargetWeights};
use astral_core::{AstralError, Result};
use astral_gpu::{Context, Dev, DeviceBuffer, GraphExec, Stream};
use astral_kernels::dspark::{DevIndex, DsparkOps, HEAD_DIM};
use astral_kernels::inkling::InklingOps;
use astral_kernels::ops::{Gemv, GemvMatrix, GemvType, MmaWork, MultiItem, QAct, QItem};
use astral_loader::safetensors::Safetensors;
use std::path::Path;
use std::sync::Arc;

fn bad(m: String) -> AstralError {
    AstralError::Invalid(m)
}

/// `mtp_config` of the checkpoint.
#[derive(Debug, Clone)]
pub struct DraftConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub inter: usize,
    pub rank: usize,
    pub eps: f32,
    pub theta: f32,
    pub window: u32,
    /// Tokens per draft block (anchor + masks).
    pub block: usize,
    pub local: Vec<bool>,
    /// Target layer visits whose input is a feature; the last entry is the final-norm output.
    pub taps: Vec<usize>,
}

impl DraftConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| bad(format!("config.json: {e}")))?;
        let m = v.get("mtp_config").ok_or_else(|| bad("config.json has no mtp_config".into()))?;
        let u = |k: &str| m.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| bad(format!("mtp_config.{k} missing")));
        let f = |k: &str| m.get(k).and_then(|x| x.as_f64()).map(|x| x as f32).ok_or_else(|| bad(format!("mtp_config.{k} missing")));
        let list = |k: &str| -> Result<Vec<usize>> {
            m.get(k).and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).ok_or_else(|| bad(format!("mtp_config.{k} missing")))
        };
        let layers = u("num_hidden_layers")?;
        let local_ids = list("local_layer_ids")?;
        let c = Self {
            hidden: u("hidden_size")?,
            layers,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            inter: u("intermediate_size")?,
            rank: u("markov_rank")?,
            eps: f("rms_norm_eps")?,
            theta: f("rope_theta")?,
            window: u("sliding_window_size")? as u32,
            block: u("block_size")?,
            local: (0..layers).map(|i| local_ids.contains(&i)).collect(),
            taps: list("target_layer_ids")?,
        };
        if u("head_dim")? != HEAD_DIM as usize || c.taps.len() != 4 || c.block < 2 || m.get("markov_head_type").and_then(|x| x.as_str()) != Some("rnn_hidden") {
            return Err(bad(format!("unsupported mtp_config {m}")));
        }
        Ok(c)
    }
}

struct DraftLayer {
    wqkv: GemvMatrix,
    wo: GemvMatrix,
    wgu: GemvMatrix,
    wdown: GemvMatrix,
    wkv_ctx: GemvMatrix,
    in_ln: DeviceBuffer<f32>,
    post_ln: DeviceBuffer<f32>,
    dflare_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    k_norm_ctx: DeviceBuffer<f32>,
}

/// The draft's weights on the GPU.
pub struct DraftWeights {
    pub cfg: DraftConfig,
    layers: Vec<DraftLayer>,
    /// `softmax(fuse_weight)` rows: one per layer, then the Markov head's.
    alpha: Vec<[f32; 4]>,
    mask_embed: Vec<f32>,
    norm: DeviceBuffer<f32>,
    mk_norm: DeviceBuffer<f32>,
    /// `rnn_proj` columns: [state | w1[prev]] (F32 3r x 2r), hidden (3r x d), f_rnn (3r x d).
    w_s: GemvMatrix,
    w_h: GemvMatrix,
    w_f: GemvMatrix,
    rnn_bias: DeviceBuffer<f32>,
    w1: DeviceBuffer<f32>,
    w1_rows: u32,
    /// `lm_head @ hidden_out.weight` (Q8_0, vocab x r) and `lm_head @ hidden_out.bias`.
    fold: GemvMatrix,
    fold_bias: DeviceBuffer<f32>,
    /// Opt-in (`ASTRAL_DRAFT_HEAD_FP4=1`): an NVFP4 copy of the target LM head for the draft's
    /// base logits. Measured no faster than the Q8_0 head on tensor cores, with slightly lower
    /// acceptance, so off by default. The target always verifies with its own head.
    head_fp4: Option<GemvMatrix>,
    pub bytes: usize,
}

fn q8(ctx: &Arc<Context>, x: &[f32], n: usize, k: usize) -> Result<GemvMatrix> {
    let (qs, aux) = astral_quant::quantize_q8_0_rows(x, n, k)?;
    GemvMatrix::upload(ctx, GemvType::Q8_0, n as u32, k as u32, 1, &qs, &aux, None)
}

fn f32m(ctx: &Arc<Context>, x: &[f32], n: usize, k: usize) -> Result<GemvMatrix> {
    let b: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    GemvMatrix::upload(ctx, GemvType::F32, n as u32, k as u32, 1, &b, &[], None)
}

/// The target's Q8_0 LM head re-encoded as NVFP4 (dequantised and quantised on the host, rows
/// split over the CPU cores).
fn head_to_nvfp4(ctx: &Arc<Context>, lm: &GemvMatrix) -> Result<GemvMatrix> {
    let (n, k) = (lm.n as usize, lm.k as usize);
    let qs = lm.qs.to_vec()?;
    let aux = lm.aux.to_vec()?;
    let threads = std::thread::available_parallelism().map_or(8, |p| p.get()).min(32);
    let per = n.div_ceil(threads);
    let parts: Vec<Result<Vec<u8>>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let (qs, aux) = (&qs, &aux);
                sc.spawn(move || {
                    let (r0, r1) = (t * per, ((t + 1) * per).min(n));
                    let mut row = vec![0f32; k];
                    let mut blocks = Vec::with_capacity((r1.saturating_sub(r0)) * k / 64 * 36);
                    for r in r0..r1 {
                        for (c, v) in row.iter_mut().enumerate() {
                            let b = r * (k / 32) + c / 32;
                            let d = astral_quant::f16_to_f32(u16::from_le_bytes([aux[2 * b], aux[2 * b + 1]]));
                            *v = (qs[r * k + c] as i8) as f32 * d;
                        }
                        blocks.extend(astral_quant::quantize_nvfp4(&row)?);
                    }
                    Ok(blocks)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(bad("nvfp4 head: worker panicked".into())))).collect()
    });
    let mut blocks = Vec::with_capacity(n * k / 64 * 36);
    for p in parts {
        blocks.extend(p?);
    }
    let (q, a) = astral_quant::repack_nvfp4(&blocks)?;
    GemvMatrix::upload(ctx, GemvType::Nvfp4, n as u32, k as u32, 1, &q, &a, None)
}

/// One grouped dp4a work item (rows <= 8): activation rows `x0 + r` to output rows `r`.
fn item(ctx: &Arc<Context>, rows: usize, x0: u32) -> Result<DeviceBuffer<u32>> {
    let mut it = QItem { e: 0, n: rows as u32, ..Default::default() };
    for r in 0..rows {
        it.xi[r] = x0 + r as u32;
        it.yi[r] = r as u32;
    }
    DeviceBuffer::from_slice(ctx, &QItem::words(&[it]))
}

/// `y[t] = m x[t]` for the `rows` (<= 8) activation rows of `x`: the fused T-token GEMV when it
/// applies (Q8_0), else one grouped dp4a item (`items`).
fn mm_small(gemv: &Gemv, st: &Stream, m: &GemvMatrix, x: &QAct, rows: u32, y: &mut DeviceBuffer<f32>, items: &DeviceBuffer<u32>) -> Result<()> {
    let mut it = [MultiItem { m, x, y }];
    if Gemv::small_mma() && m.ty == GemvType::Q8_0 && m.k.is_multiple_of(128) && x.slots >= rows {
        return gemv.launch_q_multi_mma(st, &mut it, rows);
    }
    if rows >= 2 && Gemv::multi_t_ok(&it, rows) {
        return gemv.launch_q_multi_t(st, &mut it, rows);
    }
    let [MultiItem { y, .. }] = it;
    gemv.launch_items(st, m, x, rows, y, rows, items, 1)
}

/// Dense rows `(x0 + r, r)` for `r < rows`.
fn dense_work(ctx: &Arc<Context>, rows: usize, x0: u32) -> Result<DevWork> {
    let mut w = MmaWork::default();
    let pairs: Vec<(u32, u32)> = (0..rows as u32).map(|r| (x0 + r, r)).collect();
    w.push(0, &pairs);
    let mut d = DevWork::new(ctx, w.n_items() as usize, rows)?;
    d.upload(&w)?;
    Ok(d)
}

impl DraftWeights {
    /// Loads `mtp.safetensors` from the native checkpoint directory `dir`, and folds the Markov
    /// head into the target's LM head on the GPU.
    pub fn load(ctx: &Arc<Context>, dir: &Path, target: &TargetWeights, gemv: &Gemv, ops: &DsparkOps) -> Result<Self> {
        let cfg = DraftConfig::from_dir(dir)?;
        let st = Safetensors::open(dir.join("mtp.safetensors"))?;
        let get = |name: &str| -> Result<(Vec<f32>, Vec<u64>)> {
            let t = st.tensor(name).ok_or_else(|| bad(format!("mtp.safetensors: missing {name}")))?;
            Ok((st.read_f32(t)?, t.shape.clone()))
        };
        let vecd = |name: &str| -> Result<DeviceBuffer<f32>> { DeviceBuffer::from_slice(ctx, &get(name)?.0) };
        let (d, r) = (cfg.hidden, cfg.rank);
        let (qw, kvw) = (cfg.heads * HEAD_DIM as usize, cfg.kv_heads * HEAD_DIM as usize);
        let mut bytes = 0usize;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            let mut qkv = get(&p("self_attn.q_proj.weight"))?.0;
            qkv.extend(get(&p("self_attn.k_proj.weight"))?.0);
            qkv.extend(get(&p("self_attn.v_proj.weight"))?.0);
            let mut gu = get(&p("mlp.gate_proj.weight"))?.0;
            gu.extend(get(&p("mlp.up_proj.weight"))?.0);
            let mut kv = get(&p("self_attn.k_proj_ctx.weight"))?.0;
            kv.extend(get(&p("self_attn.v_proj_ctx.weight"))?.0);
            let l = DraftLayer {
                wqkv: q8(ctx, &qkv, qw + 2 * kvw, d)?,
                wo: q8(ctx, &get(&p("self_attn.o_proj.weight"))?.0, d, qw)?,
                wgu: q8(ctx, &gu, 2 * cfg.inter, d)?,
                wdown: q8(ctx, &get(&p("mlp.down_proj.weight"))?.0, d, cfg.inter)?,
                wkv_ctx: q8(ctx, &kv, 2 * kvw, d)?,
                in_ln: vecd(&p("input_layernorm.weight"))?,
                post_ln: vecd(&p("post_attention_layernorm.weight"))?,
                dflare_norm: vecd(&p("dflare_norm.weight"))?,
                q_norm: vecd(&p("self_attn.q_norm.weight"))?,
                k_norm: vecd(&p("self_attn.k_norm.weight"))?,
                k_norm_ctx: vecd(&p("self_attn.k_norm_ctx.weight"))?,
            };
            bytes += [&l.wqkv, &l.wo, &l.wgu, &l.wdown, &l.wkv_ctx].iter().map(|m| m.bytes()).sum::<usize>();
            layers.push(l);
        }
        let (fw, fshape) = get("fuse_weight")?;
        let taps = cfg.taps.len();
        if fshape != [cfg.layers as u64 + 1, taps as u64] {
            return Err(bad(format!("fuse_weight shape {fshape:?}")));
        }
        let alpha: Vec<[f32; 4]> = fw
            .chunks_exact(taps)
            .map(|row| {
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
                let s: f32 = e.iter().sum();
                [e[0] / s, e[1] / s, e[2] / s, e[3] / s]
            })
            .collect();
        // rnn_proj [3r, 2r + 2d]: columns state | w1 | hidden | f_rnn.
        let (rp, rs) = get("markov_head.rnn_proj.weight")?;
        let cols = 2 * r + 2 * d;
        if rs != [3 * r as u64, cols as u64] {
            return Err(bad(format!("rnn_proj shape {rs:?}")));
        }
        let slice = |c0: usize, w: usize| -> Vec<f32> { rp.chunks_exact(cols).flat_map(|row| row[c0..c0 + w].to_vec()).collect() };
        let w_s = f32m(ctx, &slice(0, 2 * r), 3 * r, 2 * r)?;
        let w_h = f32m(ctx, &slice(2 * r, d), 3 * r, d)?;
        let w_f = f32m(ctx, &slice(2 * r + d, d), 3 * r, d)?;
        let (w1h, w1s) = get("markov_head.markov_w1.weight")?;
        let w1_rows = w1s[0] as u32;
        let w1 = DeviceBuffer::from_slice(ctx, &w1h)?;
        bytes += w1.bytes() + w_s.bytes() + w_h.bytes() + w_f.bytes();

        // Fold hidden_out into the LM head: y[j] = lm_head @ x_j for the r columns of
        // hidden_out.weight [d, r] and its bias, as one tensor-core GEMM over r + 1 rows.
        let (how, hos) = get("markov_head.hidden_out.weight")?;
        let hob = get("markov_head.hidden_out.bias")?.0;
        if hos != [d as u64, r as u64] {
            return Err(bad(format!("hidden_out shape {hos:?}")));
        }
        let rows = r + 1;
        let mut xs = vec![0f32; rows * d];
        for c in 0..d {
            for j in 0..r {
                xs[j * d + c] = how[c * r + j];
            }
            xs[r * d + c] = hob[c];
        }
        let lm = &target.output;
        let vp = lm.n as usize;
        let fold_t = {
            let st = Stream::new(ctx)?;
            let x = DeviceBuffer::from_slice(ctx, &xs)?;
            let mut qa = QAct::with_f16(ctx, d as u32, rows as u32)?;
            ops.quant(&st, &x, d as u32, rows as u32, &mut qa)?;
            let work = dense_work(ctx, rows, 0)?;
            let mut y = DeviceBuffer::<f32>::zeroed(ctx, rows * vp)?;
            gemv.launch_mma(&st, lm, &qa, rows as u32, &mut y, rows as u32, &work.items, work.n_items, &work.xidx, &work.yidx, work.idx_len)?;
            st.synchronize()?;
            y.to_vec()?
        };
        let mut fold = vec![0f32; vp * r];
        for j in 0..r {
            let row = &fold_t[j * vp..(j + 1) * vp];
            for (v, &x) in row.iter().enumerate() {
                fold[v * r + j] = x;
            }
        }
        let fold = q8(ctx, &fold, vp, r)?;
        let head_fp4 = if std::env::var_os("ASTRAL_DRAFT_HEAD_FP4").is_some() && lm.ty == GemvType::Q8_0 && lm.n_mats == 1 && lm.k.is_multiple_of(64) { Some(head_to_nvfp4(ctx, lm)?) } else { None };
        bytes += head_fp4.as_ref().map_or(0, |m| m.bytes());
        let fold_bias = DeviceBuffer::from_slice(ctx, &fold_t[r * vp..])?;
        bytes += fold.bytes() + fold_bias.bytes();
        Ok(Self {
            layers,
            alpha,
            mask_embed: get("mask_embed")?.0,
            norm: vecd("norm.weight")?,
            mk_norm: vecd("markov_head.dflare_norm.weight")?,
            w_s,
            w_h,
            w_f,
            rnn_bias: vecd("markov_head.rnn_proj.bias")?,
            w1,
            w1_rows,
            fold,
            fold_bias,
            head_fp4,
            bytes,
            cfg,
        })
    }
}

/// Adaptive draft length. The draft always proposes `kmax` tokens; each round verifies the
/// first `cur` of them. The policy keeps a decayed acceptance rate per draft position and the
/// measured round time per verified length, and picks the length with the most expected
/// tokens per millisecond. Acceptance is far from geometric (the first draft is accepted
/// much more often than later ones, and predictable stretches come in bursts), hence per
/// position.
#[derive(Debug, Clone)]
pub struct Policy {
    pub adaptive: bool,
    pub kmin: usize,
    pub kmax: usize,
    /// Drafts verified next round.
    pub cur: usize,
    att: Vec<f64>,
    acc: Vec<f64>,
    time_ms: Vec<f64>,
    decay: f64,
    /// "burst" (default): grow after a fully accepted round, shrink after an early reject;
    /// "model": expected tokens per millisecond from per-position rates.
    burst: bool,
    /// Burst rule: drafts added after a fully accepted round, and the shrink rule after an
    /// early reject ("dec": one fewer, "jump": accepted + 1, "half": halfway between).
    grow: usize,
    shrink: u8,
}

impl Policy {
    /// Drafts the round that verifies `k` must leave for the next one (the longest the policy
    /// can pick after it).
    pub fn drafts_after(&self, k: usize) -> usize {
        if self.adaptive && self.burst {
            (k + self.grow).min(self.kmax)
        } else {
            self.kmax
        }
    }

    pub fn new(kmax: usize, adaptive: bool) -> Self {
        let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let kmin = (env("ASTRAL_SPEC_KMIN", 2.0) as usize).clamp(1, kmax);
        Self {
            adaptive,
            kmin,
            kmax,
            cur: if adaptive { 3.clamp(kmin, kmax) } else { kmax },
            att: vec![0.0; kmax + 1],
            acc: vec![0.0; kmax + 1],
            time_ms: vec![0.0; kmax + 1],
            decay: env("ASTRAL_SPEC_DECAY", 0.9),
            burst: std::env::var("ASTRAL_SPEC_POLICY").map_or(true, |v| v != "model"),
            grow: (env("ASTRAL_SPEC_GROW", 1.0) as usize).max(1),
            shrink: match std::env::var("ASTRAL_SPEC_SHRINK").as_deref() {
                Ok("jump") => 1,
                Ok("half") => 2,
                _ => 0,
            },
        }
    }

    /// Acceptance rate of draft position `i` given all earlier drafts were accepted (a weak
    /// prior keeps unexplored positions worth trying).
    fn rate(&self, i: usize) -> f64 {
        const PRIOR: f64 = 0.6;
        const W: f64 = 1.0;
        (self.acc[i] + PRIOR * W) / (self.att[i] + W)
    }

    /// Round time of verifying `k` drafts: measured, else a linear model fitted to what is
    /// measured (about 7.4 + 0.9 k ms on an H100 PCIe), scaled to this machine.
    fn cost(&self, k: usize) -> f64 {
        if self.time_ms[k] > 0.0 {
            return self.time_ms[k];
        }
        let model = |k: usize| 7.4 + 0.9 * k as f64;
        let known: Vec<(f64, f64)> = (1..=self.kmax).filter(|&i| self.time_ms[i] > 0.0).map(|i| (self.time_ms[i], model(i))).collect();
        let scale = if known.is_empty() { 1.0 } else { known.iter().map(|(t, m)| t / m).sum::<f64>() / known.len() as f64 };
        model(k) * scale
    }

    /// Records a round that verified `k` drafts, accepted `j` of them and took `ms`.
    pub fn update(&mut self, k: usize, j: usize, ms: f64) {
        for i in 1..=self.kmax {
            self.att[i] *= self.decay;
            self.acc[i] *= self.decay;
        }
        for i in 1..=(j + 1).min(k) {
            self.att[i] += 1.0;
            if i <= j {
                self.acc[i] += 1.0;
            }
        }
        let t = &mut self.time_ms[k];
        *t = if *t > 0.0 { 0.8 * *t + 0.2 * ms } else { ms };
        if self.adaptive {
            self.cur = if self.burst {
                // Predictable text comes in runs: follow them up, back off on early rejects.
                if j == k {
                    (k + self.grow).min(self.kmax)
                } else if j + 1 < k {
                    let to = match self.shrink {
                        1 => j + 1,
                        2 => (k + j + 1) / 2,
                        _ => k - 1,
                    };
                    to.clamp(self.kmin, self.kmax)
                } else {
                    k
                }
            } else {
                self.choose()
            };
        }
    }

    /// The length with the most expected tokens per millisecond.
    fn choose(&self) -> usize {
        let (mut best, mut best_rate) = (self.cur, 0.0);
        let mut p = 1.0;
        let mut expected = 1.0;
        for k in 1..=self.kmax {
            p *= self.rate(k);
            expected += p;
            if k < self.kmin {
                continue;
            }
            let r = expected / self.cost(k);
            if r > best_rate {
                best_rate = r;
                best = k;
            }
        }
        best
    }
}

/// Counters of a speculative run.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    pub rounds: u64,
    /// Drafted tokens the target accepted.
    pub accepted: u64,
    pub drafted: u64,
}

/// Speculative-decoding state of a [`crate::inkling::Session`]: the draft, its KV cache and
/// the verify-side buffers the target's forward pass fills.
pub struct Spec {
    /// Drafted tokens verified per round (1..block).
    pub k: usize,
    pub w: DraftWeights,
    pub ops: DsparkOps,
    pub(crate) tmax: usize,
    capacity: u32,
    /// Target features `[4][tmax][dim]`.
    pub(crate) feats: DeviceBuffer<f32>,
    /// Per target slot: the short-conv inputs of the verified tokens (`[4 sites][k + 1][channels]`).
    pub(crate) hist: Vec<DeviceBuffer<f32>>,
    pub(crate) table: DeviceBuffer<u64>,
    pub(crate) entries: u32,
    pub(crate) max_ch: u32,
    pub(crate) logits_all: DeviceBuffer<f32>,
    pub(crate) next_all: DeviceBuffer<i32>,
    pub(crate) out: DeviceBuffer<i32>,
    pub(crate) info: DeviceBuffer<i32>,
    kc: Vec<DeviceBuffer<f32>>,
    vc: Vec<DeviceBuffer<f32>>,
    xin: DeviceBuffer<f32>,
    res: DeviceBuffer<f32>,
    hid: DeviceBuffer<f32>,
    tmp: DeviceBuffer<f32>,
    mlp: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    gu: DeviceBuffer<f32>,
    qa_d: QAct,
    qa_o: QAct,
    qa_f: QAct,
    qa_lat: QAct,
    qa_h: QAct,
    ctxmix: DeviceBuffer<f32>,
    kvctx: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    fmix: DeviceBuffer<f32>,
    fnorm: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    cf: DeviceBuffer<f32>,
    ch: DeviceBuffer<f32>,
    x2r: DeviceBuffer<f32>,
    raw: DeviceBuffer<f32>,
    state: DeviceBuffer<f32>,
    zero_ids: DeviceBuffer<i32>,
    /// Multi-block argmax partials (`[8][ARGMAX_PARTS]`).
    pub(crate) part_v: DeviceBuffer<f32>,
    pub(crate) part_i: DeviceBuffer<u32>,
    items_blk: DeviceBuffer<u32>,
    items_head: DeviceBuffer<u32>,
    /// Grouped items of 1..=k rows (index n - 1), for the NVFP4 draft head.
    items_head_n: Vec<DeviceBuffer<u32>>,
    items_ctxv: DeviceBuffer<u32>,
    pub(crate) work_pf: DevWork,
    /// Round graphs by the number of drafts verified (index k, 1..=self.k).
    pub(crate) rounds: Vec<Option<GraphExec>>,
    pub(crate) first: Option<GraphExec>,
    /// Chooses how many of the drafted tokens each round verifies.
    pub policy: Policy,
    /// Diagnostics: accept against this fixed sequence (token by position) instead of the
    /// target's picks (set before the round graph is captured).
    pub forced: Option<DeviceBuffer<i32>>,
    pub(crate) forced_host: Vec<u32>,
    /// Rows of the last prefill chunk (its last row is the newest committed token's features).
    pub(crate) last_rows: usize,
    pub stats: SpecStats,
    /// Batched drafting for many sequences ([`Spec::enable_batch`]).
    pub(crate) batch: Option<Box<BatchDraft>>,
}

impl Spec {
    /// Buffers for a target with `tmax`-token passes, `capacity` positions and `slots` layer
    /// slots of conv `channels` (`kvw` + `kvw` + `dim` + `dim`), verifying `k` drafts per round.
    #[allow(clippy::too_many_arguments)]
    pub fn new(ctx: &Arc<Context>, w: DraftWeights, ops: DsparkOps, k: usize, tmax: usize, capacity: u32, slots: usize, target_kvw: usize, target_dim: usize, vocab_padded: usize) -> Result<Self> {
        let c = w.cfg.clone();
        if k == 0 || k >= c.block || c.block > astral_kernels::ops::QITEM_ROWS {
            return Err(bad(format!("speculative tokens {k} outside 1..{}", c.block)));
        }
        if c.hidden != target_dim {
            return Err(bad(format!("draft hidden {} != target {}", c.hidden, target_dim)));
        }
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(ctx, n);
        let (d, nb, r) = (c.hidden, c.block, c.rank);
        let (qw, kvw) = (c.heads * HEAD_DIM as usize, c.kv_heads * HEAD_DIM as usize);
        let vmax = k + 1;
        let dcap = capacity + nb as u32 + 8;
        let mut xin = vec![0f32; nb * d];
        for row in 1..nb {
            xin[row * d..(row + 1) * d].copy_from_slice(&w.mask_embed);
        }
        let hist_len = vmax * (2 * target_kvw + 2 * target_dim);
        let pf_rows = tmax;
        Ok(Self {
            k,
            tmax,
            capacity: dcap,
            feats: z(4 * tmax * d)?,
            hist: (0..slots).map(|_| z(hist_len)).collect::<Result<_>>()?,
            table: DeviceBuffer::zeroed(ctx, 3 * 4 * slots)?,
            entries: 0,
            max_ch: 0,
            logits_all: z(vmax * vocab_padded)?,
            next_all: DeviceBuffer::zeroed(ctx, vmax.max(8))?,
            out: DeviceBuffer::zeroed(ctx, vmax + 2)?,
            info: DeviceBuffer::zeroed(ctx, 4)?,
            kc: (0..c.layers).map(|_| z(dcap as usize * kvw)).collect::<Result<_>>()?,
            vc: (0..c.layers).map(|_| z(dcap as usize * kvw)).collect::<Result<_>>()?,
            xin: DeviceBuffer::from_slice(ctx, &xin)?,
            res: z(nb * d)?,
            hid: z(nb * d)?,
            tmp: z(nb * d)?,
            mlp: z(nb * d)?,
            qkv: z(nb * (qw + 2 * kvw))?,
            att: z(nb * qw)?,
            gu: z(nb * 2 * c.inter)?,
            qa_d: QAct::with_f16(ctx, d as u32, pf_rows.max(nb) as u32)?,
            qa_o: QAct::with_f16(ctx, qw as u32, nb as u32)?,
            qa_f: QAct::with_f16(ctx, c.inter as u32, nb as u32)?,
            qa_lat: QAct::new(ctx, r as u32, 1)?,
            qa_h: QAct::new(ctx, d as u32, nb as u32)?,
            ctxmix: z(pf_rows * d)?,
            kvctx: z(pf_rows * 2 * kvw)?,
            base: z((nb - 1) * vocab_padded)?,
            fmix: z(d)?,
            fnorm: z(d)?,
            emb: z(d)?,
            cf: z(3 * r)?,
            ch: z((nb - 1) * 3 * r)?,
            x2r: z(2 * r)?,
            raw: z(3 * r)?,
            state: z(r)?,
            zero_ids: DeviceBuffer::zeroed(ctx, nb)?,
            part_v: z(8 * astral_kernels::dspark::ARGMAX_PARTS as usize)?,
            part_i: DeviceBuffer::zeroed(ctx, 8 * astral_kernels::dspark::ARGMAX_PARTS as usize)?,
            items_blk: item(ctx, nb, 0)?,
            items_head: item(ctx, k, 0)?,
            items_head_n: (1..=k).map(|n| item(ctx, n, 0)).collect::<Result<_>>()?,
            items_ctxv: item(ctx, vmax, 0)?,
            work_pf: DevWork::new(ctx, pf_rows.div_ceil(astral_kernels::ops::MMA_TOKENS), pf_rows)?,
            rounds: (0..=k).map(|_| None).collect(),
            first: None,
            policy: Policy::new(k, std::env::var_os("ASTRAL_SPEC_FIXED").is_none()),
            last_rows: 1,
            forced: None,
            forced_host: Vec::new(),
            stats: SpecStats::default(),
            batch: None,
            ops,
            w,
        })
    }

    /// Writes the draft's context K/V for `rows` target tokens (feature rows `0..rows`) at
    /// positions `pos..`; `verify` selects the fixed (k + 1)-row work list, else `work_pf`
    /// (uploaded by the caller for `rows`).
    pub(crate) fn store_ctx(&mut self, st: &Stream, gemv: &Gemv, rows: usize, pos: DevIndex, verify: bool) -> Result<()> {
        let c = &self.w.cfg;
        let (d, kvh) = (c.hidden, c.kv_heads as u32);
        let kvw = kvh * HEAD_DIM;
        let tap = self.tmax * d;
        for l in 0..c.layers {
            let lw = &self.w.layers[l];
            self.ops.mix4(st, &self.feats, tap, None, self.w.alpha[l], &mut self.ctxmix, rows as u32, d as u32)?;
            self.ops.add_rms_q8(st, self.ctxmix.dptr(), self.ctxmix.len(), None, None, &lw.dflare_norm, None, Some(&mut self.qa_d), rows as u32, d as u32, c.eps)?;
            if verify {
                mm_small(gemv, st, &lw.wkv_ctx, &self.qa_d, rows as u32, &mut self.kvctx, &self.items_ctxv)?;
            } else {
                let wk = &self.work_pf;
                gemv.launch_mma(st, &lw.wkv_ctx, &self.qa_d, rows as u32, &mut self.kvctx, rows as u32, &wk.items, wk.n_items, &wk.xidx, &wk.yidx, wk.idx_len)?;
            }
            let kp = self.kvctx.dptr();
            self.ops.head_prep(st, kp, 2 * kvw, &mut self.kc[l], kvw, Some(&lw.k_norm_ctx), true, pos, true, self.capacity, rows as u32, kvh, c.theta, c.eps)?;
            self.ops.head_prep(st, kp + 4 * kvw as u64, 2 * kvw, &mut self.vc[l], kvw, None, false, pos, true, self.capacity, rows as u32, kvh, c.theta, c.eps)?;
        }
        Ok(())
    }

    /// One draft step: the block for anchor `token[0]` at position `info[2]`, with the Markov
    /// context from feature row `info[1]`; writes the `k` drafts to `token[1..=k]`.
    /// `n` (1..=k) drafts are made; the rest of `token[1..=k]` keeps stale values the next
    /// round does not verify.
    pub(crate) fn draft(&mut self, st: &Stream, gemv: &Gemv, iops: &InklingOps, tw: &TargetWeights, token: &DeviceBuffer<i32>, n: usize) -> Result<()> {
        let n = n.clamp(1, self.k);
        let c = &self.w.cfg;
        let (d, nb, r) = (c.hidden, c.block, c.rank);
        let (du, nbu) = (d as u32, nb as u32);
        let (heads, kvh) = (c.heads as u32, c.kv_heads as u32);
        let (qw, kvw) = (heads * HEAD_DIM, kvh * HEAD_DIM);
        let qkvw = qw + 2 * kvw;
        let start = DevIndex::of(&self.info, 2)?;
        let tap = self.tmax * d;
        // Markov context: the last committed token's features.
        self.ops.mix4(st, &self.feats, tap, Some(DevIndex::of(&self.info, 1)?), self.w.alpha[c.layers], &mut self.fmix, 1, du)?;
        self.ops.add_rms_q8(st, self.fmix.dptr(), d, None, None, &self.w.mk_norm, Some(&mut self.fnorm), None, 1, du, c.eps)?;
        gemv.launch(st, &self.w.w_f, &self.fnorm, 0, &mut self.cf)?;
        // Block input: the anchor's normalised target embedding, then the masks (fixed rows).
        iops.embed(st, &tw.tok_embd, token, &mut self.emb, 1)?;
        self.ops.add_rms_q8(st, self.emb.dptr(), d, None, None, &tw.tok_norm, Some(&mut self.xin.view_mut(0, d)?), None, 1, du, tw.cfg.rms_eps)?;
        let blk = &self.items_blk;
        for l in 0..c.layers {
            let lw = &self.w.layers[l];
            let (x, add) = if l == 0 { (self.xin.dptr(), None) } else { (self.res.dptr(), Some(&self.mlp as &dyn Dev<f32>)) };
            self.ops.add_rms_q8(st, x, nb * d, add, Some(self.res.dptr()), &lw.in_ln, None, Some(&mut self.qa_d), nbu, du, c.eps)?;
            mm_small(gemv, st, &lw.wqkv, &self.qa_d, nbu, &mut self.qkv, blk)?;
            let qp = self.qkv.dptr();
            self.ops.head_prep(st, qp, qkvw, &mut self.qkv, qkvw, Some(&lw.q_norm), true, start, false, self.capacity, nbu, heads, c.theta, c.eps)?;
            self.ops.head_prep(st, qp + 4 * qw as u64, qkvw, &mut self.kc[l], kvw, Some(&lw.k_norm), true, start, true, self.capacity, nbu, kvh, c.theta, c.eps)?;
            self.ops.head_prep(st, qp + 4 * (qw + kvw) as u64, qkvw, &mut self.vc[l], kvw, None, false, start, true, self.capacity, nbu, kvh, c.theta, c.eps)?;
            let window = if c.local[l] { c.window + nbu } else { 0 };
            self.ops.attn(st, qp, qkvw, &self.kc[l], &self.vc[l], &mut self.att, start, nbu, heads, kvh, window, self.capacity)?;
            self.ops.quant(st, &self.att, qw, nbu, &mut self.qa_o)?;
            mm_small(gemv, st, &lw.wo, &self.qa_o, nbu, &mut self.tmp, blk)?;
            self.ops.add_rms_q8(st, self.res.dptr(), nb * d, Some(&self.tmp), Some(self.res.dptr()), &lw.post_ln, None, Some(&mut self.qa_d), nbu, du, c.eps)?;
            mm_small(gemv, st, &lw.wgu, &self.qa_d, nbu, &mut self.gu, blk)?;
            self.ops.swiglu_q8(st, &self.gu, nbu, &mut self.qa_f)?;
            mm_small(gemv, st, &lw.wdown, &self.qa_f, nbu, &mut self.mlp, blk)?;
        }
        self.ops.add_rms_q8(st, self.res.dptr(), nb * d, Some(&self.mlp), Some(self.res.dptr()), &self.w.norm, Some(&mut self.hid), Some(&mut self.qa_d), nbu, du, c.eps)?;
        // Base logits of the mask rows (block rows 1..nb), and their rnn_proj hidden terms.
        // Only the k mask rows that are drafted from (block rows 1..=k).
        let kr = n as u32;
        self.ops.quant(st, &self.hid.view(d, n * d)?, du, kr, &mut self.qa_h)?;
        match &self.w.head_fp4 {
            Some(h) => gemv.launch_items(st, h, &self.qa_h, kr, &mut self.base, kr, &self.items_head_n[n - 1], 1)?,
            None => mm_small(gemv, st, &tw.output, &self.qa_h, kr, &mut self.base, &self.items_head)?,
        }
        gemv.launch_ids(st, &self.w.w_h, &self.hid.view(d, n * d)?, du, &mut self.ch, &self.zero_ids.view(0, n)?)?;
        // Sequential Markov head (greedy).
        let vp = tw.output.n as usize;
        let vocab = tw.cfg.vocab as u32;
        for i in 0..n {
            self.ops.markov_in(st, &self.state, &self.w.w1, DevIndex::of(token, i as u32)?, self.w.w1_rows, i == 0, &mut self.x2r, r as u32)?;
            gemv.launch(st, &self.w.w_s, &self.x2r, 0, &mut self.raw)?;
            self.ops.markov_gate(st, &self.raw, &self.ch.view(i * 3 * r, 3 * r)?, &self.cf, &self.w.rnn_bias, &mut self.state, i == 0, &mut self.qa_lat)?;
            self.ops.fold_argmax(st, &self.w.fold, &self.qa_lat, &self.base.view(i * vp, vp)?, &self.w.fold_bias, vocab, &mut self.part_v, &mut self.part_i, DevIndex::of(token, i as u32 + 1)?)?;
        }
        Ok(())
    }
}

/// Drafting for up to `bmax` sequences at once: every sequence keeps its own draft K/V cache
/// (`dcap` positions); a draft call runs B blocks of `block` rows through the draft and the
/// Markov head for all of them; a context store writes the draft K/V of many target tokens
/// (rows of the target features) at their sequences and positions.
pub struct BatchDraft {
    pub bmax: usize,
    pub(crate) dcap: u32,
    pub(crate) ctx_rows: usize,
    kc: Vec<DeviceBuffer<f32>>,
    vc: Vec<DeviceBuffer<f32>>,
    xin: DeviceBuffer<f32>,
    res: DeviceBuffer<f32>,
    hid: DeviceBuffer<f32>,
    hidk: DeviceBuffer<f32>,
    tmp: DeviceBuffer<f32>,
    mlp: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    gu: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    anc: DeviceBuffer<f32>,
    qa_d: QAct,
    qa_o: QAct,
    qa_f: QAct,
    qa_h: QAct,
    qa_lat: QAct,
    ctxmix: DeviceBuffer<f32>,
    kvctx: DeviceBuffer<f32>,
    fmix: DeviceBuffer<f32>,
    fnorm: DeviceBuffer<f32>,
    cf: DeviceBuffer<f32>,
    ch: DeviceBuffer<f32>,
    x2r: DeviceBuffer<f32>,
    raw: DeviceBuffer<f32>,
    state: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_i: DeviceBuffer<u32>,
    zero_ids: DeviceBuffer<i32>,
    /// Per block: sequence, block start position, feature row of the Markov context.
    pub(crate) seq_b: DeviceBuffer<i32>,
    pub(crate) start_b: DeviceBuffer<i32>,
    pub(crate) mark_b: DeviceBuffer<i32>,
    /// Tokens `[block row i][b]`: anchors in row 0, then the drafts.
    pub(crate) tok: DeviceBuffer<i32>,
    /// Context store rows: feature row, sequence and position of each target token.
    pub(crate) ctx_feat: DeviceBuffer<i32>,
    pub(crate) ctx_seq: DeviceBuffer<i32>,
    pub(crate) ctx_pos: DeviceBuffer<i32>,
    /// Draft graphs by block count, context-store graphs by row count.
    pub(crate) draft_graphs: std::collections::HashMap<usize, GraphExec>,
    pub(crate) store_graphs: std::collections::HashMap<usize, GraphExec>,
}

impl Spec {
    /// Batched drafting for up to `bmax` of `n_seqs` sequences of `capacity` positions.
    pub fn enable_batch(&mut self, ctx: &Arc<Context>, bmax: usize, n_seqs: usize, capacity: u32, vocab_padded: usize) -> Result<()> {
        let c = self.w.cfg.clone();
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(ctx, n);
        let (d, nb, r, k) = (c.hidden, c.block, c.rank, self.k);
        let (qw, kvw) = (c.heads * HEAD_DIM as usize, c.kv_heads * HEAD_DIM as usize);
        let rows = bmax * nb;
        let ctx_rows = self.tmax;
        let dcap = capacity + nb as u32 + 8;
        let mut xin = vec![0f32; rows * d];
        for row in 0..rows {
            if row % nb != 0 {
                xin[row * d..(row + 1) * d].copy_from_slice(&self.w.mask_embed);
            }
        }
        let b = BatchDraft {
            bmax,
            dcap,
            ctx_rows,
            kc: (0..c.layers).map(|_| z(n_seqs * dcap as usize * kvw)).collect::<Result<_>>()?,
            vc: (0..c.layers).map(|_| z(n_seqs * dcap as usize * kvw)).collect::<Result<_>>()?,
            xin: DeviceBuffer::from_slice(ctx, &xin)?,
            res: z(rows * d)?,
            hid: z(rows * d)?,
            hidk: z(bmax * k * d)?,
            tmp: z(rows * d)?,
            mlp: z(rows * d)?,
            qkv: z(rows * (qw + 2 * kvw))?,
            att: z(rows * qw)?,
            gu: z(rows * 2 * c.inter)?,
            emb: z(bmax * d)?,
            anc: z(bmax * d)?,
            qa_d: QAct::with_f16(ctx, d as u32, rows.max(ctx_rows) as u32)?,
            qa_o: QAct::with_f16(ctx, qw as u32, rows as u32)?,
            qa_f: QAct::with_f16(ctx, c.inter as u32, rows as u32)?,
            qa_h: QAct::with_f16(ctx, d as u32, (bmax * k) as u32)?,
            qa_lat: QAct::with_f16(ctx, r as u32, bmax as u32)?,
            ctxmix: z(ctx_rows * d)?,
            kvctx: z(ctx_rows * 2 * kvw)?,
            fmix: z(bmax * d)?,
            fnorm: z(bmax * d)?,
            cf: z(bmax * 3 * r)?,
            ch: z(bmax * k * 3 * r)?,
            x2r: z(bmax * 2 * r)?,
            raw: z(bmax * 3 * r)?,
            state: z(bmax * r)?,
            base: z(bmax * k * vocab_padded)?,
            y: z(bmax * vocab_padded)?,
            part_v: z(bmax * astral_kernels::dspark::ARGMAX_PARTS as usize)?,
            part_i: DeviceBuffer::zeroed(ctx, bmax * astral_kernels::dspark::ARGMAX_PARTS as usize)?,
            zero_ids: DeviceBuffer::zeroed(ctx, (bmax * k).max(bmax))?,
            seq_b: DeviceBuffer::zeroed(ctx, bmax)?,
            start_b: DeviceBuffer::zeroed(ctx, bmax)?,
            mark_b: DeviceBuffer::zeroed(ctx, bmax)?,
            tok: DeviceBuffer::zeroed(ctx, bmax * (k + 1))?,
            ctx_feat: DeviceBuffer::zeroed(ctx, ctx_rows)?,
            ctx_seq: DeviceBuffer::zeroed(ctx, ctx_rows)?,
            ctx_pos: DeviceBuffer::zeroed(ctx, ctx_rows)?,
            draft_graphs: std::collections::HashMap::new(),
            store_graphs: std::collections::HashMap::new(),
        };
        self.batch = Some(Box::new(b));
        Ok(())
    }

    /// Draft K/V of `rows` target tokens: feature rows `ctx_feat`, at sequences `ctx_seq` and
    /// positions `ctx_pos` (uploaded by the caller).
    pub(crate) fn batch_store(&mut self, st: &Stream, gemv: &Gemv, rows: usize) -> Result<()> {
        let Spec { w, ops, feats, tmax, batch, .. } = self;
        let b = batch.as_deref_mut().ok_or_else(|| bad("batch_store: not enabled".into()))?;
        let c = &w.cfg;
        let (d, kvh) = (c.hidden, c.kv_heads as u32);
        let kvw = kvh * HEAD_DIM;
        let tap = *tmax * d;
        let nr = rows as u32;
        for l in 0..c.layers {
            let lw = &w.layers[l];
            ops.mix4_b(st, &*feats, tap, &b.ctx_feat, w.alpha[l], &mut b.ctxmix, nr, d as u32)?;
            ops.add_rms_q8(st, b.ctxmix.dptr(), rows * d, None, None, &lw.dflare_norm, None, Some(&mut b.qa_d), nr, d as u32, c.eps)?;
            gemv.launch_q_dense(st, &mut [MultiItem { m: &lw.wkv_ctx, x: &b.qa_d, y: &mut b.kvctx }], nr)?;
            let kp = b.kvctx.dptr();
            ops.head_prep_b(st, kp, 2 * kvw, b.kc[l].dptr(), kvw, Some(&lw.k_norm_ctx), true, &b.ctx_pos, &b.ctx_seq, 1, true, b.dcap, nr, kvh, c.theta, c.eps)?;
            ops.head_prep_b(st, kp + 4 * kvw as u64, 2 * kvw, b.vc[l].dptr(), kvw, None, false, &b.ctx_pos, &b.ctx_seq, 1, true, b.dcap, nr, kvh, c.theta, c.eps)?;
        }
        Ok(())
    }

    /// One draft for `nb` blocks: block b is sequence `seq_b[b]`, anchor `tok[0][b]` at position
    /// `start_b[b]`, Markov context feature row `mark_b[b]`; writes the `k` drafts to
    /// `tok[1..=k][b]`.
    pub(crate) fn batch_draft(&mut self, st: &Stream, gemv: &Gemv, iops: &InklingOps, tw: &TargetWeights, nblocks: usize) -> Result<()> {
        let Spec { w, ops, feats, tmax, batch, k, .. } = self;
        let b = batch.as_deref_mut().ok_or_else(|| bad("batch_draft: not enabled".into()))?;
        let c = &w.cfg;
        let (d, nb, r, n) = (c.hidden, c.block, c.rank, *k);
        let (du, nbu, bu) = (d as u32, nb as u32, nblocks as u32);
        let (heads, kvh) = (c.heads as u32, c.kv_heads as u32);
        let (qw, kvw) = (heads * HEAD_DIM, kvh * HEAD_DIM);
        let qkvw = qw + 2 * kvw;
        let rows = nblocks * nb;
        let ru = rows as u32;
        let tap = *tmax * d;
        // Markov context: each sequence's last committed token's features.
        ops.mix4_b(st, &*feats, tap, &b.mark_b, w.alpha[c.layers], &mut b.fmix, bu, du)?;
        ops.add_rms_q8(st, b.fmix.dptr(), nblocks * d, None, None, &w.mk_norm, Some(&mut b.fnorm), None, bu, du, c.eps)?;
        gemv.launch_ids(st, &w.w_f, &b.fnorm.view(0, nblocks * d)?, du, &mut b.cf, &b.zero_ids.view(0, nblocks)?)?;
        // Anchors: normalised target embeddings into row 0 of every block.
        iops.embed(st, &tw.tok_embd, &b.tok.view(0, nblocks)?, &mut b.emb, bu)?;
        ops.add_rms_q8(st, b.emb.dptr(), nblocks * d, None, None, &tw.tok_norm, Some(&mut b.anc), None, bu, du, tw.cfg.rms_eps)?;
        for i in 0..nblocks {
            b.xin.copy_range_from_async(i * nb * d, &b.anc, i * d, d, st)?;
        }
        for l in 0..c.layers {
            let lw = &w.layers[l];
            let (x, add) = if l == 0 { (b.xin.dptr(), None) } else { (b.res.dptr(), Some(&b.mlp as &dyn Dev<f32>)) };
            ops.add_rms_q8(st, x, rows * d, add, Some(b.res.dptr()), &lw.in_ln, None, Some(&mut b.qa_d), ru, du, c.eps)?;
            gemv.launch_q_dense(st, &mut [MultiItem { m: &lw.wqkv, x: &b.qa_d, y: &mut b.qkv }], ru)?;
            let qp = b.qkv.dptr();
            ops.head_prep_b(st, qp, qkvw, qp, qkvw, Some(&lw.q_norm), true, &b.start_b, &b.seq_b, nbu, false, b.dcap, ru, heads, c.theta, c.eps)?;
            ops.head_prep_b(st, qp + 4 * qw as u64, qkvw, b.kc[l].dptr(), kvw, Some(&lw.k_norm), true, &b.start_b, &b.seq_b, nbu, true, b.dcap, ru, kvh, c.theta, c.eps)?;
            ops.head_prep_b(st, qp + 4 * (qw + kvw) as u64, qkvw, b.vc[l].dptr(), kvw, None, false, &b.start_b, &b.seq_b, nbu, true, b.dcap, ru, kvh, c.theta, c.eps)?;
            let window = if c.local[l] { c.window + nbu } else { 0 };
            ops.attn_b(st, qp, qkvw, b.kc[l].dptr(), b.vc[l].dptr(), b.att.dptr(), &b.start_b, &b.seq_b, bu, nbu, heads, kvh, window, b.dcap)?;
            ops.quant(st, &b.att, qw, ru, &mut b.qa_o)?;
            gemv.launch_q_dense(st, &mut [MultiItem { m: &lw.wo, x: &b.qa_o, y: &mut b.tmp }], ru)?;
            ops.add_rms_q8(st, b.res.dptr(), rows * d, Some(&b.tmp), Some(b.res.dptr()), &lw.post_ln, None, Some(&mut b.qa_d), ru, du, c.eps)?;
            gemv.launch_q_dense(st, &mut [MultiItem { m: &lw.wgu, x: &b.qa_d, y: &mut b.gu }], ru)?;
            ops.swiglu_q8(st, &b.gu, ru, &mut b.qa_f)?;
            gemv.launch_q_dense(st, &mut [MultiItem { m: &lw.wdown, x: &b.qa_f, y: &mut b.mlp }], ru)?;
        }
        ops.add_rms_q8(st, b.res.dptr(), rows * d, Some(&b.mlp), Some(b.res.dptr()), &w.norm, Some(&mut b.hid), None, ru, du, c.eps)?;
        // Mask rows 1..=n of every block, compacted: base logits and the rnn_proj hidden terms.
        for i in 0..nblocks {
            b.hidk.copy_range_from_async(i * n * d, &b.hid, (i * nb + 1) * d, n * d, st)?;
        }
        let kr = (nblocks * n) as u32;
        ops.quant(st, &b.hidk, du, kr, &mut b.qa_h)?;
        gemv.launch_q_dense(st, &mut [MultiItem { m: &tw.output, x: &b.qa_h, y: &mut b.base }], kr)?;
        gemv.launch_ids(st, &w.w_h, &b.hidk.view(0, nblocks * n * d)?, du, &mut b.ch, &b.zero_ids.view(0, nblocks * n)?)?;
        // The Markov head, step by step for all blocks.
        let vp = tw.output.n;
        let vocab = tw.cfg.vocab as u32;
        let ru32 = r as u32;
        for i in 0..n {
            let ti = b.tok.dptr() + 4 * (i * nblocks) as u64;
            ops.markov_in_b(st, b.state.dptr(), &w.w1, ti, w.w1_rows, i == 0, b.x2r.dptr(), ru32, bu)?;
            gemv.launch_ids(st, &w.w_s, &b.x2r.view(0, nblocks * 2 * r)?, 2 * ru32, &mut b.raw, &b.zero_ids.view(0, nblocks)?)?;
            ops.markov_gate_b(st, b.raw.dptr(), b.ch.dptr() + 4 * (i * 3 * r) as u64, (n * 3 * r) as u32, b.cf.dptr(), &w.rnn_bias, b.state.dptr(), i == 0, &mut b.qa_lat, bu)?;
            gemv.launch_q_dense(st, &mut [MultiItem { m: &w.fold, x: &b.qa_lat, y: &mut b.y }], bu)?;
            ops.add_rows(st, b.y.dptr(), vp, b.base.dptr() + 4 * (i as u64 * vp as u64), n as u32 * vp, &w.fold_bias, vocab, bu)?;
            let mut out = b.tok.view_mut((i + 1) * nblocks, nblocks)?;
            ops.argmax_rows(st, &b.y, vp, vocab, bu, &mut b.part_v, &mut b.part_i, &mut out)?;
        }
        Ok(())
    }
}

/// Offsets (floats) of the four short-conv sites in a slot's `hist` buffer for `rows` rows:
/// K, V (`kvw` channels each), attention output and FFN output (`dim` each).
pub(crate) fn hist_offsets(rows: usize, kvw: usize, dim: usize) -> [usize; 4] {
    [0, rows * kvw, 2 * rows * kvw, 2 * rows * kvw + rows * dim]
}
