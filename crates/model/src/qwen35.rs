// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Qwen3.8-27B (`qwen3_5`, text model) on the GPU from the compressed-tensors MXFP4
//! checkpoint (`Qwen3.8-27B-MXFP4`).
//!
//! 64 layers: every 4th is gated full attention (24 query / 4 KV heads of 256, q/k RMS norms,
//! NeoX RoPE on the first 64 dims, sigmoid output gate), the rest Gated DeltaNet linear
//! attention (16 key / 48 value heads of 128, causal conv + SiLU, L2-normalised q/k, gated
//! delta rule, gated RMSNorm). Dense SwiGLU MLP. Qwen3.5 RMSNorm weights are zero-centred
//! (`1 + w`), except the gated norm's.
//!
//! MXFP4 weights are converted at load time, losslessly (see
//! [`astral_quant::mxfp4_ct_to_nvfp4`]), to the NVFP4 sub-block layout with a per-matrix
//! power-of-two scale, so the tuned NVFP4 kernels serve them: int8 dp4a GEMV for decode, f16
//! tensor-core GEMM for prefill. The tied-off embedding and LM head stay BF16.

use astral_core::{AstralError, Result};
use astral_gpu::{Context, Dev, DeviceBuffer, GraphExec, Stream};
use astral_kernels::inkling::InklingOps;
use astral_kernels::ops::{Gemv, GemvMatrix, GemvType, MmaWork, QAct};
use astral_kernels::qwen::{attn_workspace, GdnShape, QwenOps, ATTN_CHUNK};
use astral_kernels::KernelRegistry;
use astral_loader::safetensors::Safetensors;
use std::path::Path;
use std::sync::Arc;

fn bad(m: String) -> AstralError {
    AstralError::Invalid(format!("qwen35: {m}"))
}

/// Text-model hyper-parameters (`config.json` → `text_config`).
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub dim: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rot: usize,
    pub rope_theta: f32,
    pub gdn: GdnShape,
    pub conv_k: usize,
    pub eps: f32,
    pub full_attn: Vec<bool>,
    pub eos: Vec<u32>,
}

impl Config {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let c: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?).map_err(|e| bad(format!("config.json: {e}")))?;
        let t = c.get("text_config").unwrap_or(&c);
        let u = |k: &str| t[k].as_u64().map(|v| v as usize).ok_or_else(|| bad(format!("missing {k}")));
        let n_layer = u("num_hidden_layers")?;
        let full_attn: Vec<bool> = match t["layer_types"].as_array() {
            Some(a) => a.iter().map(|v| v.as_str() == Some("full_attention")).collect(),
            None => (0..n_layer).map(|i| (i + 1) % u("full_attention_interval").unwrap_or(4) == 0).collect(),
        };
        let head_dim = u("head_dim")?;
        let prf = t["rope_parameters"]["partial_rotary_factor"].as_f64().or(t["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
        let theta = t["rope_parameters"]["rope_theta"].as_f64().or(t["rope_theta"].as_f64()).unwrap_or(10000.0);
        let eos = match &t["eos_token_id"] {
            serde_json::Value::Array(a) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
            v => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
        };
        let cfg = Self {
            n_layer,
            dim: u("hidden_size")?,
            n_ff: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim,
            rot: (head_dim as f64 * prf) as usize,
            rope_theta: theta as f32,
            gdn: GdnShape { k_heads: u("linear_num_key_heads")? as u32, v_heads: u("linear_num_value_heads")? as u32, head_dim: u("linear_key_head_dim")? as u32 },
            conv_k: u("linear_conv_kernel_dim")?,
            eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            full_attn,
            eos,
        };
        if cfg.full_attn.len() != n_layer || cfg.head_dim != 256 || cfg.conv_k != 4 || u("linear_value_head_dim")? != cfg.gdn.head_dim as usize {
            return Err(bad(format!("unsupported configuration {cfg:?}")));
        }
        Ok(cfg)
    }
}

enum Mixer {
    Linear {
        /// [in_proj_qkv; in_proj_z] stacked by rows.
        wqkvz: GemvMatrix,
        /// [in_proj_a; in_proj_b] (BF16), and an F32 copy for the prefill GEMM.
        wab: GemvMatrix,
        wab32: GemvMatrix,
        conv: DeviceBuffer<f32>,
        neg_a: DeviceBuffer<f32>,
        dt_bias: DeviceBuffer<f32>,
        norm: DeviceBuffer<f32>,
        out: GemvMatrix,
    },
    Full {
        /// [q_proj (query, gate per head); k_proj; v_proj] stacked by rows.
        wqkv: GemvMatrix,
        wo: GemvMatrix,
        q_norm: DeviceBuffer<f32>,
        k_norm: DeviceBuffer<f32>,
    },
}

struct Layer {
    in_norm: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
    mixer: Mixer,
    /// [gate_proj; up_proj] stacked by rows.
    gate_up: GemvMatrix,
    down: GemvMatrix,
}

pub struct Weights {
    pub cfg: Config,
    embed: DeviceBuffer<u16>,
    norm: DeviceBuffer<f32>,
    lm_head: GemvMatrix,
    layers: Vec<Layer>,
    pub bytes: usize,
    /// MXFP4 blocks that fell outside a matrix's 18-binade window and were re-encoded.
    pub reencoded_blocks: usize,
}

struct Loader<'a> {
    st: &'a Safetensors,
    ctx: &'a Arc<Context>,
    bytes: usize,
    reencoded: usize,
}

const P: &str = "model.language_model.";

impl Loader<'_> {
    fn t(&self, name: &str) -> Result<&astral_loader::safetensors::StTensor> {
        self.st.tensor(name).ok_or_else(|| bad(format!("missing tensor {name}")))
    }

    fn vec(&mut self, name: &str, plus_one: bool) -> Result<DeviceBuffer<f32>> {
        let mut v = self.st.read_f32(self.t(name)?)?;
        if plus_one {
            v.iter_mut().for_each(|x| *x += 1.0);
        }
        self.bytes += v.len() * 4;
        DeviceBuffer::from_slice(self.ctx, &v)
    }

    /// One linear layer, or several with the same input stacked by rows into one matrix (one
    /// launch). MXFP4 (`.weight_packed` + `.weight_scale`) becomes the NVFP4 layout with one
    /// shared exponent window; BF16 stays BF16.
    fn mat(&mut self, bases: &[&str]) -> Result<GemvMatrix> {
        let first = bases[0];
        if self.st.tensor(&format!("{first}.weight_packed")).is_some() {
            let (mut packed, mut scales, mut n, mut k) = (Vec::new(), Vec::new(), 0usize, 0usize);
            for base in bases {
                let pk = self.t(&format!("{base}.weight_packed"))?.clone();
                let sc = self.t(&format!("{base}.weight_scale"))?.clone();
                let [rows, k2] = pk.shape.as_slice() else { return Err(bad(format!("{base}: packed shape {:?}", pk.shape))) };
                if k != 0 && k != 2 * *k2 as usize {
                    return Err(bad(format!("{base}: k differs from the stack")));
                }
                k = 2 * *k2 as usize;
                n += *rows as usize;
                packed.extend_from_slice(&self.st.read(&pk)?);
                scales.extend_from_slice(&self.st.read(&sc)?);
            }
            let c = convert_parallel(&packed, &scales, n, k)?;
            self.reencoded += c.reencoded_blocks;
            // Decode reads the checkpoint's own MXFP4 (reordered values + E8M0 scales: exact);
            // the NVFP4 conversion feeds the prefill fragment layout.
            let qs_mx = if c.reencoded_blocks == 0 { c.qs.clone() } else { astral_quant::mxfp4_ct_reorder(&packed, n, k)? };
            let mut m = GemvMatrix::upload(self.ctx, GemvType::Nvfp4, n as u32, k as u32, 1, &qs_mx, &c.aux, Some(&[c.mat_scale]))?;
            m.mx_scale = Some(DeviceBuffer::from_slice(self.ctx, &scales)?);
            // Prefill fragment layout (doubles the weight memory of this matrix).
            if n.is_multiple_of(128) && std::env::var_os("ASTRAL_NO_FRAG").is_none() {
                m.add_frag(self.ctx, &c.qs, &c.aux)?;
            }
            self.bytes += m.bytes();
            return Ok(m);
        }
        // Dense BF16 checkpoints: the weights are quantised to Q8_0 (the int8-activation kernels
        // of the prefill and decode paths read them); ASTRAL_BF16_WEIGHTS=1 keeps BF16 (only
        // valid where the f32-activation path is used).
        if self.dense_checkpoint() && std::env::var_os("ASTRAL_BF16_WEIGHTS").is_none() {
            let (mut v, mut n, mut k) = (Vec::new(), 0usize, 0usize);
            for base in bases {
                let t = self.t(&format!("{base}.weight"))?.clone();
                let [rows, cols] = t.shape.as_slice() else { return Err(bad(format!("{base}: shape {:?}", t.shape))) };
                if (k != 0 && k != *cols as usize) || cols % 32 != 0 {
                    return Err(bad(format!("{base}: k {cols} (a multiple of 32, the same for stacked matrices)")));
                }
                k = *cols as usize;
                n += *rows as usize;
                v.extend(self.st.read_f32(&t)?);
            }
            let (q, d8) = astral_quant::quantize_q8_0_rows(&v, n, k)?;
            let m = GemvMatrix::upload(self.ctx, GemvType::Q8_0, n as u32, k as u32, 1, &q, &d8, None)?;
            self.bytes += m.bytes();
            return Ok(m);
        }
        self.mat_bf16(bases)
    }

    /// A dense BF16 checkpoint (no MXFP4 weights).
    fn dense_checkpoint(&self) -> bool {
        self.st.tensor(&format!("{P}layers.0.mlp.down_proj.weight_packed")).is_none()
    }

    /// BF16 matrices stacked by rows, kept BF16.
    fn mat_bf16(&mut self, bases: &[&str]) -> Result<GemvMatrix> {
        let (mut raw, mut n, mut k) = (Vec::new(), 0u64, 0u64);
        for base in bases {
            let t = self.t(&format!("{base}.weight"))?.clone();
            let [rows, cols] = t.shape.as_slice() else { return Err(bad(format!("{base}: shape {:?}", t.shape))) };
            if t.dtype != "BF16" || (k != 0 && k != *cols) {
                return Err(bad(format!("{base}: dtype {} / k {cols} (expected BF16 or MXFP4, same k)", t.dtype)));
            }
            k = *cols;
            n += rows;
            raw.extend_from_slice(&self.st.read(&t)?);
        }
        let m = GemvMatrix::upload(self.ctx, GemvType::Bf16, n as u32, k as u32, 1, &raw, &[], None)?;
        self.bytes += m.bytes();
        Ok(m)
    }
}

impl Loader<'_> {
    /// BF16 matrices stacked by rows, as one F32 matrix.
    fn mat_f32(&mut self, bases: &[&str]) -> Result<GemvMatrix> {
        let (mut data, mut n, mut k) = (Vec::new(), 0u64, 0u64);
        for base in bases {
            let t = self.t(&format!("{base}.weight"))?.clone();
            let [rows, cols] = t.shape.as_slice() else { return Err(bad(format!("{base}: shape {:?}", t.shape))) };
            if k != 0 && k != *cols {
                return Err(bad(format!("{base}: k differs from the stack")));
            }
            k = *cols;
            n += rows;
            data.extend(self.st.read_f32(&t)?.iter().flat_map(|x| x.to_le_bytes()));
        }
        let m = GemvMatrix::upload(self.ctx, GemvType::F32, n as u32, k as u32, 1, &data, &[], None)?;
        self.bytes += m.bytes();
        Ok(m)
    }
}

/// [`astral_quant::mxfp4_ct_to_nvfp4`] over row ranges on all cores (load-time only).
fn convert_parallel(packed: &[u8], scales: &[u8], n: usize, k: usize) -> Result<astral_quant::Mxfp4AsNvfp4> {
    // The per-matrix offset needs the global maximum first; converting row slices with a shared
    // maximum gives the same bytes as one call.
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get()).min(32);
    let rows = n.div_ceil(threads).max(1);
    let emax = *scales.iter().max().unwrap_or(&127);
    let parts: Vec<Result<astral_quant::Mxfp4AsNvfp4>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..n.div_ceil(rows))
            .map(|i| {
                let (r0, r1) = (i * rows, ((i + 1) * rows).min(n));
                sc.spawn(move || {
                    // Prepend one row whose scale is emax so the slice sees the same maximum.
                    let mut s = scales[r0 * k / 32..r1 * k / 32].to_vec();
                    let mut pk = packed[r0 * k / 2..r1 * k / 2].to_vec();
                    s.splice(0..0, std::iter::repeat_n(emax, k / 32));
                    pk.splice(0..0, std::iter::repeat_n(0u8, k / 2));
                    let mut c = astral_quant::mxfp4_ct_to_nvfp4(&pk, &s, r1 - r0 + 1, k)?;
                    c.qs.drain(..k / 2);
                    c.aux.drain(..k / 16);
                    Ok(c)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(bad("conversion thread panicked".into())))).collect()
    });
    let mut out = astral_quant::Mxfp4AsNvfp4 { qs: Vec::with_capacity(n * k / 2), aux: Vec::with_capacity(n * k / 16), mat_scale: 0.0, reencoded_blocks: 0 };
    for p in parts {
        let p = p?;
        out.qs.extend_from_slice(&p.qs);
        out.aux.extend_from_slice(&p.aux);
        out.mat_scale = p.mat_scale;
        out.reencoded_blocks += p.reencoded_blocks;
    }
    Ok(out)
}

impl Weights {
    pub fn load(ctx: &Arc<Context>, dir: &Path) -> Result<Self> {
        let cfg = Config::from_dir(dir)?;
        let st = Safetensors::open_dir(dir)?;
        let mut l = Loader { st: &st, ctx, bytes: 0, reencoded: 0 };
        let et = l.t(&format!("{P}embed_tokens.weight"))?.clone();
        if et.dtype != "BF16" || et.shape != [cfg.vocab as u64, cfg.dim as u64] {
            return Err(bad(format!("embed_tokens: {} {:?}", et.dtype, et.shape)));
        }
        let raw = st.read(&et)?;
        let e16: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let embed = DeviceBuffer::from_slice(ctx, &e16)?;
        l.bytes += raw.len();
        drop((raw, e16));
        let norm = l.vec(&format!("{P}norm.weight"), true)?;
        // LM head: BF16 as in the checkpoint; ASTRAL_LM_HEAD_Q8=1 quantises it to Q8_0 (half the
        // bytes per decode step; not bit-identical to the reference).
        // Tied embeddings (no lm_head.weight): the head is the embedding matrix.
        let head_base = if st.tensor("lm_head.weight").is_some() { "lm_head".to_string() } else { format!("{P}embed_tokens") };
        let head = &format!("{head_base}.weight");
        let lm_head = if std::env::var_os("ASTRAL_LM_HEAD_Q8").is_some() {
            let t = l.t(head)?.clone();
            let (n, k) = (t.shape[0] as usize, t.shape[1] as usize);
            let v = st.read_f32(&t)?;
            let (q, d8) = astral_quant::quantize_q8_0_rows(&v, n, k)?;
            let m = GemvMatrix::upload(ctx, GemvType::Q8_0, n as u32, k as u32, 1, &q, &d8, None)?;
            l.bytes += m.bytes();
            m
        } else {
            l.mat_bf16(&[&head_base])?
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let b = |s: &str| format!("{P}layers.{i}.{s}");
            let mixer = if cfg.full_attn[i] {
                Mixer::Full {
                    wqkv: l.mat(&[&b("self_attn.q_proj"), &b("self_attn.k_proj"), &b("self_attn.v_proj")])?,
                    wo: l.mat(&[&b("self_attn.o_proj")])?,
                    q_norm: l.vec(&b("self_attn.q_norm.weight"), true)?,
                    k_norm: l.vec(&b("self_attn.k_norm.weight"), true)?,
                }
            } else {
                let a_log = st.read_f32(l.t(&b("linear_attn.A_log"))?)?;
                let neg_a: Vec<f32> = a_log.iter().map(|x| -x.exp()).collect();
                Mixer::Linear {
                    wqkvz: l.mat(&[&b("linear_attn.in_proj_qkv"), &b("linear_attn.in_proj_z")])?,
                    wab: l.mat_bf16(&[&b("linear_attn.in_proj_a"), &b("linear_attn.in_proj_b")])?,
                    wab32: l.mat_f32(&[&b("linear_attn.in_proj_a"), &b("linear_attn.in_proj_b")])?,
                    conv: l.vec(&b("linear_attn.conv1d.weight"), false)?,
                    neg_a: DeviceBuffer::from_slice(ctx, &neg_a)?,
                    dt_bias: l.vec(&b("linear_attn.dt_bias"), false)?,
                    norm: l.vec(&b("linear_attn.norm.weight"), false)?,
                    out: l.mat(&[&b("linear_attn.out_proj")])?,
                }
            };
            layers.push(Layer {
                in_norm: l.vec(&b("input_layernorm.weight"), true)?,
                post_norm: l.vec(&b("post_attention_layernorm.weight"), true)?,
                mixer,
                gate_up: l.mat(&[&b("mlp.gate_proj"), &b("mlp.up_proj")])?,
                down: l.mat(&[&b("mlp.down_proj")])?,
            });
        }
        Ok(Self { cfg, embed, norm, lm_head, layers, bytes: l.bytes, reencoded_blocks: l.reencoded })
    }
}

/// Tensor-core prefill work list on the device (dense rows), in items of `chunk` tokens.
struct DevWork {
    items: DeviceBuffer<u32>,
    xidx: DeviceBuffer<u32>,
    yidx: DeviceBuffer<u32>,
    n_items: u32,
    idx_len: u32,
    chunk: usize,
}

impl DevWork {
    fn new(ctx: &Arc<Context>, tmax: usize, chunk: usize) -> Result<Self> {
        Ok(Self { items: DeviceBuffer::zeroed(ctx, 3 * (tmax + 1))?, xidx: DeviceBuffer::zeroed(ctx, tmax)?, yidx: DeviceBuffer::zeroed(ctx, tmax)?, n_items: 0, idx_len: 0, chunk })
    }
    fn dense(&mut self, t: usize) -> Result<()> {
        let mut w = MmaWork::default();
        let rows: Vec<(u32, u32)> = (0..t as u32).map(|r| (r, r)).collect();
        w.push_chunked(0, &rows, self.chunk);
        let put = |b: &mut DeviceBuffer<u32>, v: &[u32]| -> Result<()> {
            let mut p = v.to_vec();
            p.resize(b.len(), 0);
            b.copy_from_slice(&p)
        };
        put(&mut self.items, &w.items)?;
        put(&mut self.xidx, &w.xidx)?;
        put(&mut self.yidx, &w.yidx)?;
        self.n_items = w.n_items();
        self.idx_len = t as u32;
        Ok(())
    }
}

pub const PREFILL_TMAX: usize = 2048;

struct Scratch {
    tmax: usize,
    cur: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    qa_x: QAct,
    qkvz: DeviceBuffer<f32>,
    qkvc: DeviceBuffer<f32>,
    ab: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    core: DeviceBuffer<f32>,
    gdn_scratch: DeviceBuffer<f32>,
    qa_core: QAct,
    qkv: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    work: DeviceBuffer<f32>,
    fgu: DeviceBuffer<f32>,
    qa_ff: QAct,
    o: DeviceBuffer<f32>,
    last: QAct,
    logits: DeviceBuffer<f32>,
    token: DeviceBuffer<i32>,
    pos: DeviceBuffer<i32>,
    next: DeviceBuffer<i32>,
    zero_ids: DeviceBuffer<i32>,
    dense: DevWork,
    dense128: DevWork,
}

struct Exec<'a> {
    st: &'a Stream,
    gemv: &'a Gemv,
    qw: &'a QwenOps,
    ink: &'a InklingOps,
}

impl Exec<'_> {
    /// `y = W x` for `ntok` quantised rows (MXFP4-as-NVFP4 weights).
    fn mm(&self, m: &GemvMatrix, qa: &QAct, y: &mut DeviceBuffer<f32>, ntok: usize, dense: &DevWork, dense128: &DevWork) -> Result<()> {
        if ntok == 1 {
            self.gemv.launch_q(self.st, m, qa, y, None)
        } else if m.frag.is_some() && std::env::var_os("ASTRAL_MMA_V3").is_none() {
            let (n, d) = (ntok as u32, dense128);
            self.gemv.launch_mma5(self.st, m, qa, n, y, n, &d.items, d.n_items, &d.xidx, &d.yidx, d.idx_len)
        } else if m.ty == GemvType::Nvfp4 && m.n.is_multiple_of(128) && std::env::var_os("ASTRAL_MMA4").is_some() {
            let (n, d) = (ntok as u32, dense128);
            self.gemv.launch_mma4(self.st, m, qa, n, y, n, &d.items, d.n_items, &d.xidx, &d.yidx, d.idx_len)
        } else {
            let n = ntok as u32;
            self.gemv.launch_mma(self.st, m, qa, n, y, n, &dense.items, dense.n_items, &dense.xidx, &dense.yidx, dense.idx_len)
        }
    }
    /// `y = W x` with f32 activations (BF16 weights).
    fn mm_f32(&self, m: &GemvMatrix, x: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>, ntok: usize, zero_ids: &DeviceBuffer<i32>) -> Result<()> {
        if ntok == 1 {
            self.gemv.launch(self.st, m, x, 0, y)
        } else {
            self.gemv.launch_ids(self.st, m, x, m.k, y, &zero_ids.view(0, ntok)?)
        }
    }
}

/// Recurrent state of one layer.
enum LayerState {
    Linear { conv: DeviceBuffer<f32>, ssm: DeviceBuffer<f32> },
    Full { kc: DeviceBuffer<u16>, vc: DeviceBuffer<u16> },
}

pub struct Session {
    pub w: Weights,
    stream: Stream,
    eager: Stream,
    gemv: Gemv,
    qw: QwenOps,
    ink: InklingOps,
    states: Vec<LayerState>,
    s: Scratch,
    capacity: u32,
    pub pos: u32,
    graph: Option<GraphExec>,
    reset_graph: Option<GraphExec>,
}

impl Session {
    pub fn new(ctx: &Arc<Context>, w: Weights, capacity: u32) -> Result<Self> {
        if capacity == 0 || capacity > astral_kernels::inkling::MAX_CTX {
            return Err(bad(format!("capacity {capacity} outside 1..={}", astral_kernels::inkling::MAX_CTX)));
        }
        let reg = KernelRegistry::load(ctx)?;
        let c = w.cfg.clone();
        let t = PREFILL_TMAX.min(capacity as usize);
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(ctx, n);
        let (d, f, vh, gd) = (c.dim, c.n_ff, c.gdn.v_heads as usize, c.gdn.head_dim as usize);
        let qkvw = c.gdn.qkv_width() as usize;
        let attn_w = c.heads * c.head_dim;
        let kvw = c.kv_heads * c.head_dim;
        let mut states = Vec::with_capacity(c.n_layer);
        for &full in &c.full_attn {
            states.push(if full {
                LayerState::Full { kc: DeviceBuffer::zeroed(ctx, capacity as usize * kvw)?, vc: DeviceBuffer::zeroed(ctx, capacity as usize * kvw)? }
            } else {
                LayerState::Linear { conv: z(3 * qkvw)?, ssm: z(vh * gd * gd)? }
            });
        }
        let decode_work = attn_workspace(1, c.heads as u32, capacity.div_ceil(ATTN_CHUNK));
        let prefill_work = attn_workspace(t as u32, c.heads as u32, 512u32.min(capacity).div_ceil(ATTN_CHUNK));
        let s = Scratch {
            tmax: t,
            cur: z(t * d)?,
            normed: z(t * d)?,
            qa_x: QAct::with_f16(ctx, d as u32, t as u32)?,
            qkvz: z(t * (qkvw + vh * gd))?,
            qkvc: z(t * qkvw)?,
            ab: z(t * 2 * vh)?,
            beta: z(t * vh)?,
            g: z(t * vh)?,
            core: z(t * vh * gd)?,
            gdn_scratch: z(astral_kernels::qwen::gdn_scratch(t as u32, vh as u32))?,
            qa_core: QAct::with_f16(ctx, (vh * gd) as u32, t as u32)?,
            qkv: z(t * (2 * attn_w + 2 * kvw))?,
            q: z(t * attn_w)?,
            gate: z(t * attn_w)?,
            attn: z(t * attn_w)?,
            work: z(decode_work.max(prefill_work))?,
            fgu: z(t * 2 * f)?,
            qa_ff: QAct::with_f16(ctx, f as u32, t as u32)?,
            o: z(t * d)?,
            last: QAct::new(ctx, d as u32, 1)?,
            logits: z(c.vocab)?,
            token: DeviceBuffer::zeroed(ctx, t)?,
            pos: DeviceBuffer::zeroed(ctx, 1)?,
            next: DeviceBuffer::zeroed(ctx, 1)?,
            zero_ids: DeviceBuffer::zeroed(ctx, t)?,
            dense: DevWork::new(ctx, t, 64)?,
            dense128: DevWork::new(ctx, t, 128)?,
        };
        let mut me = Self {
            stream: Stream::new(ctx)?,
            eager: Stream::new(ctx)?,
            gemv: Gemv::new(&reg)?,
            qw: QwenOps::new(&reg)?,
            ink: InklingOps::new(&reg)?,
            states,
            s,
            capacity,
            pos: 0,
            graph: None,
            reset_graph: None,
            w,
        };
        if std::env::var_os("ASTRAL_NO_GRAPH").is_none() {
            let Session { w, stream, gemv, qw, ink, states, s, capacity, graph, .. } = &mut me;
            *graph = Some(stream.capture(|st| forward(&Exec { st, gemv, qw, ink }, w, states, s, *capacity, 0, 1))?);
        }
        {
            let Session { stream, states, reset_graph, .. } = &mut me;
            *reset_graph = Some(stream.capture(|st| {
                for state in states.iter_mut() {
                    if let LayerState::Linear { conv, ssm } = state {
                        conv.zero_async(st)?;
                        ssm.zero_async(st)?;
                    }
                }
                Ok(())
            })?);
        }
        Ok(me)
    }

    fn set_tokens(&mut self, toks: &[u32]) -> Result<()> {
        let mut v = vec![0i32; self.s.tmax];
        for (i, &t) in toks.iter().enumerate() {
            v[i] = t as i32;
        }
        self.s.token.copy_from_slice(&v)?;
        self.s.pos.copy_from_slice(&[self.pos as i32])
    }

    /// One decode step (graph replay); returns the greedy next token.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        if self.pos >= self.capacity {
            return Err(bad(format!("context full ({} positions)", self.capacity)));
        }
        self.set_tokens(&[token])?;
        let Session { w, stream, gemv, qw, ink, states, s, capacity, graph, .. } = self;
        match graph {
            Some(g) => g.launch(stream)?,
            None => forward(&Exec { st: stream, gemv, qw, ink }, w, states, s, *capacity, 0, 1)?,
        }
        stream.synchronize()?;
        self.pos += 1;
        Ok(self.s.next.to_vec()?[0] as u32)
    }

    /// The prompt in batched chunks; returns the greedy next token after it.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        if tokens.is_empty() || self.pos as usize + tokens.len() > self.capacity as usize {
            return Err(bad(format!("prefill: {} tokens at {} (context {})", tokens.len(), self.pos, self.capacity)));
        }
        let mut done = 0;
        let mut next = 0;
        while done < tokens.len() {
            let heads = self.w.cfg.heads as u32;
            let mut t = (tokens.len() - done).min(self.s.tmax);
            // Only the split-K fallback needs workspace (flash-attention prefill does not).
            while std::env::var_os("ASTRAL_ATTN_SPLIT_PREFILL").is_some() && t > 1 && attn_workspace(t as u32, heads, (self.pos + t as u32).div_ceil(ATTN_CHUNK)) > self.s.work.len() {
                t /= 2;
            }
            self.set_tokens(&tokens[done..done + t])?;
            if t > 1 {
                self.s.dense.dense(t)?;
                self.s.dense128.dense(t)?;
            }
            let pos0 = self.pos;
            let Session { w, eager, stream, gemv, qw, ink, states, s, capacity, .. } = self;
            // A one-off CUDA graph, not eager launches: bursts of eager launches leave later graph
            // replays slower on this driver (see the Inkling session). ASTRAL_PREFILL_EAGER=1.
            if std::env::var_os("ASTRAL_PREFILL_EAGER").is_none() {
                let cap = *capacity;
                let prev = stream.context().set_pdl(std::env::var_os("ASTRAL_PREFILL_PDL").is_some());
                let g = stream.capture(|st| forward(&Exec { st, gemv, qw, ink }, w, states, s, cap, pos0, t));
                stream.context().set_pdl(prev);
                let g = g?;
                g.launch(stream)?;
                stream.synchronize()?;
            } else {
                forward(&Exec { st: eager, gemv, qw, ink }, w, states, s, *capacity, pos0, t)?;
                eager.synchronize()?;
            }
            self.pos += t as u32;
            next = self.s.next.to_vec()?[0] as u32;
            done += t;
        }
        Ok(next)
    }

    /// Logits after the last processed token.
    /// Context capacity (positions).
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn logits(&self) -> Result<Vec<f32>> {
        self.s.logits.to_vec()
    }

    /// Forget the sequence (zero the recurrent and conv states; the KV cache is overwritten).
    pub fn reset(&mut self) -> Result<()> {
        if let Some(g) = &self.reset_graph {
            g.launch(&self.stream)?;
        } else {
            for st in &mut self.states {
                if let LayerState::Linear { conv, ssm } = st {
                    conv.zero_async(&self.stream)?;
                    ssm.zero_async(&self.stream)?;
                }
            }
        }
        self.stream.synchronize()?;
        self.pos = 0;
        Ok(())
    }
}

/// The forward pass for `ntok` tokens in `s.token` at positions `s.pos[0]..`, ending with the
/// greedy pick after the last one in `s.next`. Enqueue-only (graph-capturable for one token).
fn forward(ex: &Exec, w: &Weights, states: &mut [LayerState], s: &mut Scratch, capacity: u32, pos0: u32, ntok: usize) -> Result<()> {
    let c = &w.cfg;
    let (st, d, nt) = (ex.st, c.dim, ntok as u32);
    let td = ntok * d;
    ex.qw.embed(st, &w.embed, c.vocab as u32, d as u32, &s.token, &mut s.cur, nt)?;
    let fuse = std::env::var_os("ASTRAL_NO_ADD_RMS").is_none();
    for (li, (l, state)) in w.layers.iter().zip(states.iter_mut()).enumerate() {
        // Input norm; after layer 0 the previous MLP's residual add is fused into it.
        if li == 0 || !fuse {
            ex.ink.rms_q8(st, &s.cur, d, Some(&l.in_norm), Some(&mut s.normed), &mut s.qa_x, nt, c.eps)?;
        } else {
            ex.qw.add_rms_q8(st, &mut s.cur, &s.o, &l.in_norm, Some(&mut s.normed), &mut s.qa_x, nt, c.eps)?;
        }
        match (&l.mixer, state) {
            (Mixer::Linear { wqkvz, wab, wab32, conv, neg_a, dt_bias, norm, out }, LayerState::Linear { conv: cst, ssm }) => {
                let (qkvw, zw, vh) = (c.gdn.qkv_width() as usize, (c.gdn.v_heads * c.gdn.head_dim) as usize, c.gdn.v_heads as usize);
                let row = qkvw + zw;
                ex.mm(wqkvz, &s.qa_x, &mut s.qkvz, ntok, &s.dense, &s.dense128)?;
                if ntok == 1 {
                    ex.mm_f32(wab, &s.normed, &mut s.ab, ntok, &s.zero_ids)?;
                } else {
                    ex.ink.gemm_f32(st, &s.normed, d, wab32, &mut s.ab, nt)?;
                }
                ex.qw.conv_silu(st, &s.qkvz, row, &mut s.qkvc, nt, cst, conv, qkvw as u32)?;
                let (a, b) = (s.ab.view(0, s.ab.len())?, s.ab.view(vh, s.ab.len() - vh)?);
                ex.qw.gdn(st, c.gdn, &mut s.qkvc, &b, &a, 2 * vh, neg_a, dt_bias, &mut s.beta, &mut s.g, ssm, &mut s.core, nt, Some(&mut s.gdn_scratch))?;
                let z = s.qkvz.view(qkvw, s.qkvz.len() - qkvw)?;
                ex.qw.gated_norm_q8(st, &s.core, &z, row, norm, &mut s.qa_core, c.gdn.v_heads, c.gdn.head_dim, c.eps, nt)?;
                ex.mm(out, &s.qa_core, &mut s.o, ntok, &s.dense, &s.dense128)?;
            }
            (Mixer::Full { wqkv, wo, q_norm, k_norm }, LayerState::Full { kc, vc }) => {
                let (heads, kvh) = (c.heads as u32, c.kv_heads as u32);
                let row = (2 * c.heads + 2 * c.kv_heads) * c.head_dim;
                ex.mm(wqkv, &s.qa_x, &mut s.qkv, ntok, &s.dense, &s.dense128)?;
                ex.qw.attn_prep(st, &s.qkv, row, q_norm, k_norm, &mut s.q, &mut s.gate, kc, vc, &s.pos, heads, kvh, c.rot as u32, c.rope_theta, c.eps, capacity, nt)?;
                if ntok > 1 && std::env::var_os("ASTRAL_ATTN_SPLIT_PREFILL").is_none() {
                    ex.qw.attention_prefill(st, &s.q, &*kc, &*vc, &mut s.attn, &s.pos, heads, kvh, capacity, nt)?;
                } else {
                    let splits = if ntok == 1 { capacity.div_ceil(ATTN_CHUNK) } else { (pos0 + nt).div_ceil(ATTN_CHUNK) };
                    ex.qw.attention(st, &s.q, &*kc, &*vc, &mut s.work, &mut s.attn, &s.pos, heads, kvh, capacity, nt, splits)?;
                }
                ex.ink.gate_q8(st, &mut s.attn, &s.gate, &mut s.qa_core, nt)?;
                ex.mm(wo, &s.qa_core, &mut s.o, ntok, &s.dense, &s.dense128)?;
            }
            _ => return Err(bad("layer / state kind mismatch".into())),
        }
        if fuse {
            ex.qw.add_rms_q8(st, &mut s.cur, &s.o, &l.post_norm, None, &mut s.qa_x, nt, c.eps)?;
        } else {
            ex.qw.add(st, &mut s.cur, &s.o, td)?;
            ex.ink.rms_q8(st, &s.cur, d, Some(&l.post_norm), None, &mut s.qa_x, nt, c.eps)?;
        }
        ex.mm(&l.gate_up, &s.qa_x, &mut s.fgu, ntok, &s.dense, &s.dense128)?;
        ex.qw.swiglu_q8(st, &s.fgu, 2 * c.n_ff, &mut s.qa_ff, nt)?;
        ex.mm(&l.down, &s.qa_ff, &mut s.o, ntok, &s.dense, &s.dense128)?;
        if !fuse || li + 1 == w.layers.len() {
            ex.qw.add(st, &mut s.cur, &s.o, td)?;
        }
    }
    // Final norm and LM head for the last token.
    let last = s.cur.view((ntok - 1) * d, d)?;
    ex.ink.rms_q8(st, &last, d, Some(&w.norm), Some(&mut s.normed), &mut s.last, 1, c.eps)?;
    if w.lm_head.ty == GemvType::Q8_0 {
        ex.gemv.launch_q(st, &w.lm_head, &s.last, &mut s.logits, None)?;
    } else {
        ex.gemv.launch(st, &w.lm_head, &s.normed, 0, &mut s.logits)?;
    }
    ex.ink.argmax(st, &s.logits, c.vocab as u32, &mut s.next)
}

/// Loads the checkpoint directory onto device 0.
pub fn load(ctx: &Arc<Context>, dir: &Path) -> Result<Weights> {
    Weights::load(ctx, dir)
}

#[allow(dead_code)]
fn _assert_dev(_: &dyn Dev<f32>) {}
