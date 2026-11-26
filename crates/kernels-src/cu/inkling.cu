// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//
// Inkling 2 Mini decode operations (P1, one token per step, correctness first). Each kernel
// reproduces one step of llama.cpp's inkling2 graph (src/models/inkling2.cpp, MIT) so results
// can be checked against llama.cpp's per-op dumps on the same GGUF. All activations are f32;
// the KV cache is f16, as in llama.cpp's default.
// Parameter lists are mirrored by the launchers in crates/kernels/src/inkling.rs.

#include <cuda_fp16.h>

// Programmatic dependent launch: on Hopper every kernel may be launched before its predecessor
// finishes (overlapping launch latency); griddepcontrol.wait blocks until the predecessor's
// results are visible, so it is the first thing every kernel does. A no-op otherwise.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if !defined(ASTRAL_PDL_LATE)  // small, mostly single-wave kernels: let the successor start scheduling now
#define PDL_WAIT() asm volatile("griddepcontrol.wait;\n griddepcontrol.launch_dependents;" ::: "memory")
#else
#define PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#endif
#else
#define PDL_WAIT() do {} while (0)
#endif

typedef unsigned long long u64;
typedef unsigned char u8;
typedef unsigned short u16;

#define MAX_DEPTH 8      // attention-residual states mixed at once (Inkling uses at most 4 blocks + cur);
                         // kept small so the per-state partial sums of astral_ink_mix stay in registers

// Rows of a pass. Without `seq_p`, row t is token t of one sequence at position pos_p[0] + t.
// With it (batched decode), row t belongs to sequence seq_p[t] (clamped to n_seqs - 1) at
// position pos_p[t]; each sequence owns `capacity` cache positions and one conv state.
__device__ __forceinline__ int row_pos(const int* pos_p, const int* seq_p, unsigned t) {
    return seq_p ? pos_p[t] : pos_p[0] + (int)t;
}
__device__ __forceinline__ u64 row_seq(const int* seq_p, unsigned n_seqs, unsigned t) {
    return seq_p ? (u64)min((unsigned)max(seq_p[t], 0), n_seqs - 1u) : 0ull;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Sum over the whole block (blockDim.x a multiple of 32, <= 1024). All threads get the result.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    v = warp_sum(v);
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = lane < (blockDim.x >> 5) ? part[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------------------------------
// Token embedding: dequantise row `token` of a repacked Q8_0 matrix (K = dim).
extern "C" __global__ void astral_ink_embed_q8_0(const u8* __restrict__ qs, const u8* __restrict__ d,
                                                 const int* __restrict__ token, float* __restrict__ out, unsigned dim,
                                                 unsigned n_rows) { PDL_WAIT();
    // One block per token (blockIdx.x); the row index is clamped to the table.
    const unsigned t = blockIdx.x;
    const u64 row = (u64)min((unsigned)token[t], n_rows - 1u);
    const u16* dr = reinterpret_cast<const u16*>(d) + row * (dim >> 5);
    float* o = out + (u64)t * dim;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        const float s = __half2float(__ushort_as_half(dr[i >> 5]));
        o[i] = (float)(signed char)qs[row * dim + i] * s;
    }
}

// ---------------------------------------------------------------------------------------------
// clip(): per head of `hd` values, optional per-channel scale, then cap the L2 norm at thr:
//   x *= scale; n = sqrt(sum x^2) + 1e-6; x *= clamp(thr / n, 0, 1).   One warp per head.
extern "C" __global__ void astral_ink_clip_heads(float* __restrict__ x, const float* __restrict__ scale,
                                                 unsigned heads, unsigned hd, float thr) { PDL_WAIT();
    const unsigned h = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (h >= heads) return;
    float* xh = x + (u64)h * hd;
    float ss = 0.0f;
    for (unsigned i = lane; i < hd; i += 32u) {
        const float v = scale ? xh[i] * scale[(u64)h * hd + i] : xh[i];
        ss += v * v;
    }
    ss = warp_sum(ss);
    const float n = sqrtf(ss) + 1e-6f;
    const float f = fminf(fmaxf(thr / n, 0.0f), 1.0f);
    for (unsigned i = lane; i < hd; i += 32u) {
        const float v = scale ? xh[i] * scale[(u64)h * hd + i] : xh[i];
        xh[i] = v * f;
    }
}

// x *= tau, tau = 1 + alpha * log(max(1, (pos + 1) / floor)): the log-length scale applied to
// q and r on global layers, computed from the position on the device.
extern "C" __global__ void astral_ink_scale_tau(float* __restrict__ x, unsigned n, const int* __restrict__ pos,
                                                float floor_len, float alpha) { PDL_WAIT();
    const float tau = 1.0f + alpha * logf(fmaxf(((float)pos[0] + 1.0f) / floor_len, 1.0f));
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] *= tau;
}

// ---------------------------------------------------------------------------------------------
// mix(): attention over depth. States = blocks[idx[0..depth)] (rows of `bank`) followed by `cur`.
// Each state is RMS-normalised (no weight, eps 1e-6); for every query q_j (rows of `query`,
// nq of them): p = softmax_depth(states . q_j / sqrt(dim)); out_j = sum p * normalised state.
// One block per query; 256 threads.
struct DepthIdx {
    unsigned v[MAX_DEPTH];
};
extern "C" __global__ void __launch_bounds__(1024) astral_ink_mix(const float* __restrict__ bank, DepthIdx idx, unsigned depth,
                                                                  unsigned row_stride, const float* __restrict__ cur,
                                                                  const float* __restrict__ query, float* __restrict__ out,
                                                                  unsigned dim) { PDL_WAIT();
    // blockIdx.x = query j, blockIdx.y = token t. Bank row r of token t starts at
    // bank + r * row_stride + t * dim; cur and out are token-major ([t][dim], [t][nq][dim]).
    constexpr unsigned S = MAX_DEPTH + 1;
    const unsigned j = blockIdx.x, t = blockIdx.y, nq = gridDim.x;
    const float* q = query + (u64)j * dim;
    const float* ct = cur + (u64)t * dim;
    const float* bt = bank + (u64)t * dim;
    const unsigned n_states = depth + 1;
    float ss[S], dot[S];
#pragma unroll
    for (unsigned s = 0; s < S; ++s) { ss[s] = 0.0f; dot[s] = 0.0f; }
    const unsigned d4 = dim >> 2;
    for (unsigned i = threadIdx.x; i < d4; i += blockDim.x) {
        const float4 qi = reinterpret_cast<const float4*>(q)[i];
#pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            if (s < n_states) {
                const float4 v = reinterpret_cast<const float4*>(s < depth ? bt + (u64)idx.v[s] * row_stride : ct)[i];
                ss[s] += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                dot[s] += v.x * qi.x + v.y * qi.y + v.z * qi.z + v.w * qi.w;
            }
        }
    }
    __shared__ float part[32][2 * S];
    __shared__ float p[S];
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
#pragma unroll
    for (unsigned s = 0; s < S; ++s) {
        if (s < n_states) {
            const float a = warp_sum(ss[s]), b = warp_sum(dot[s]);
            if (lane == 0) { part[warp][2 * s] = a; part[warp][2 * s + 1] = b; }
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const unsigned nw = blockDim.x >> 5;
        float sc = -INFINITY, inv = 0.0f;
        if (lane < n_states) {
            float a = 0.0f, b = 0.0f;
            for (unsigned w = 0; w < nw; ++w) { a += part[w][2 * lane]; b += part[w][2 * lane + 1]; }
            inv = rsqrtf(a / (float)dim + 1e-6f);
            sc = b * inv * rsqrtf((float)dim);
        }
        const float m = warp_max(sc);
        const float e = lane < n_states ? expf(sc - m) : 0.0f;
        const float sum = warp_sum(e);
        if (lane < n_states) p[lane] = e / sum * inv;   // fold the RMS scale in
    }
    __syncthreads();
    float* o = out + ((u64)t * nq + j) * dim;
    for (unsigned i = threadIdx.x; i < d4; i += blockDim.x) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (unsigned s = 0; s < n_states; ++s) {
            const float4 v = reinterpret_cast<const float4*>(s < depth ? bt + (u64)idx.v[s] * row_stride : ct)[i];
            acc.x = fmaf(p[s], v.x, acc.x); acc.y = fmaf(p[s], v.y, acc.y);
            acc.z = fmaf(p[s], v.z, acc.z); acc.w = fmaf(p[s], v.w, acc.w);
        }
        reinterpret_cast<float4*>(o)[i] = acc;
    }
}

// astral_ink_mix fused with the RMSNorm + int8 quantisation that follows it (rms_q8): for query
// j the mixed row (written to `out` when given) is normalised with weight w[j], stored to y[j]
// (optional, [t][dim]) and quantised into (xq[j], xd[j], xh[j]) (token-major). Needs
// blockDim >= dim / 4 (each thread holds one float4 of the row) and blockDim % 32 == 0.
#define MIX_OUTS 3
struct MixRmsOut {
    const float* w[MIX_OUTS];
    float* y[MIX_OUTS];
    signed char* xq[MIX_OUTS];
    float* xd[MIX_OUTS];
    __half* xh[MIX_OUTS];
};
extern "C" __global__ void __launch_bounds__(1024) astral_ink_mix_rms_q8(const float* __restrict__ bank, DepthIdx idx, unsigned depth,
                                                                          unsigned row_stride, const float* __restrict__ cur,
                                                                          const float* __restrict__ query, float* __restrict__ out,
                                                                          unsigned dim, MixRmsOut mo, float eps) { PDL_WAIT();
    constexpr unsigned S = MAX_DEPTH + 1;
    const unsigned j = blockIdx.x, t = blockIdx.y, nq = gridDim.x;
    const float* q = query + (u64)j * dim;
    const float* ct = cur + (u64)t * dim;
    const float* bt = bank + (u64)t * dim;
    const unsigned n_states = depth + 1;
    const unsigned d4 = dim >> 2, i = threadIdx.x;
    float ss[S], dot[S];
#pragma unroll
    for (unsigned s = 0; s < S; ++s) { ss[s] = 0.0f; dot[s] = 0.0f; }
    if (i < d4) {
        const float4 qi = reinterpret_cast<const float4*>(q)[i];
#pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            if (s < n_states) {
                const float4 v = reinterpret_cast<const float4*>(s < depth ? bt + (u64)idx.v[s] * row_stride : ct)[i];
                ss[s] += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                dot[s] += v.x * qi.x + v.y * qi.y + v.z * qi.z + v.w * qi.w;
            }
        }
    }
    __shared__ float part[32][2 * S];
    __shared__ float p[S];
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
#pragma unroll
    for (unsigned s = 0; s < S; ++s) {
        if (s < n_states) {
            const float a = warp_sum(ss[s]), b = warp_sum(dot[s]);
            if (lane == 0) { part[warp][2 * s] = a; part[warp][2 * s + 1] = b; }
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const unsigned nw = blockDim.x >> 5;
        float sc = -INFINITY, inv = 0.0f;
        if (lane < n_states) {
            float a = 0.0f, b = 0.0f;
            for (unsigned w = 0; w < nw; ++w) { a += part[w][2 * lane]; b += part[w][2 * lane + 1]; }
            inv = rsqrtf(a / (float)dim + 1e-6f);
            sc = b * inv * rsqrtf((float)dim);
        }
        const float m = warp_max(sc);
        const float e = lane < n_states ? expf(sc - m) : 0.0f;
        const float sum = warp_sum(e);
        if (lane < n_states) p[lane] = e / sum * inv;
    }
    __syncthreads();
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < d4) {
        for (unsigned s = 0; s < n_states; ++s) {
            const float4 v = reinterpret_cast<const float4*>(s < depth ? bt + (u64)idx.v[s] * row_stride : ct)[i];
            acc.x = fmaf(p[s], v.x, acc.x); acc.y = fmaf(p[s], v.y, acc.y);
            acc.z = fmaf(p[s], v.z, acc.z); acc.w = fmaf(p[s], v.w, acc.w);
        }
        if (out) reinterpret_cast<float4*>(out + ((u64)t * nq + j) * dim)[i] = acc;
    }
    // RMSNorm of the mixed row (as astral_ink_rms_q8: rsqrt(mean(x^2) + eps) * w).
    const float inv = rsqrtf(block_sum(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w) / (float)dim + eps);
    if (i < d4) {
        const float4 w = reinterpret_cast<const float4*>(mo.w[j])[i];
        const float4 v = make_float4(acc.x * inv * w.x, acc.y * inv * w.y, acc.z * inv * w.z, acc.w * inv * w.w);
        const u64 o = (u64)t * dim;
        if (mo.y[j]) reinterpret_cast<float4*>(mo.y[j] + o)[i] = v;
        // 32-value quantisation groups = 8 neighbouring threads.
        float a = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4));
        const float d = a * (1.0f / 127.0f);
        const auto qz = [&](float x) { return (signed char)(a == 0.0f ? 0 : __float2int_rn(x / d)); };
        char4 c;
        c.x = qz(v.x); c.y = qz(v.y); c.z = qz(v.z); c.w = qz(v.w);
        reinterpret_cast<char4*>(mo.xq[j] + o)[i] = c;
        if ((i & 7u) == 0) mo.xd[j][(o >> 5) + (i >> 3)] = d;
        if (mo.xh[j]) {
            __half2* h = reinterpret_cast<__half2*>(mo.xh[j] + o) + 2 * i;
            h[0] = __floats2half2_rn(v.x, v.y);
            h[1] = __floats2half2_rn(v.z, v.w);
        }
    } else {
        // Keep the group shuffles convergent for a partial last warp (unused lanes).
        float a = 0.0f;
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4));
    }
}

// ---------------------------------------------------------------------------------------------
// Loop controller pieces. gate = concat(rms(cur), rms(msg)) (each weight-less, eps 1e-6).
extern "C" __global__ void __launch_bounds__(256) astral_ink_concat_rms2(const float* __restrict__ a, const float* __restrict__ b,
                                                                         float* __restrict__ out, unsigned dim) { PDL_WAIT();
    float sa = 0.0f, sb = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) { sa += a[i] * a[i]; sb += b[i] * b[i]; }
    sa = block_sum(sa);
    sb = block_sum(sb);
    const float ra = rsqrtf(sa / (float)dim + 1e-6f), rb = rsqrtf(sb / (float)dim + 1e-6f);
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) { out[i] = a[i] * ra; out[dim + i] = b[i] * rb; }
}

// out = cur - tanh(f) * cur + tanh(g) * msg    (f, g: the forget and gain projections)
extern "C" __global__ void astral_ink_control_combine(const float* __restrict__ cur, const float* __restrict__ msg,
                                                      const float* __restrict__ gain, const float* __restrict__ forget,
                                                      float* __restrict__ out, unsigned dim) { PDL_WAIT();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += gridDim.x * blockDim.x) {
        const float c = cur[i];
        out[i] = (c - tanhf(forget[i]) * c) + tanhf(gain[i]) * msg[i];
    }
}

// ---------------------------------------------------------------------------------------------
// convolve(): causal 4-tap depthwise conv with residual, one token. `state` holds the previous
// n_conv - 1 = 3 inputs per channel (state[c*3 + 0] oldest). weight[c*4 + j], j = 0 oldest .. 3
// current. Optional dynamic taps dyn[h*4 + lag] for head h = c / (channels / heads); lag 0 =
// current input. out = (x + conv + taps) / sqrt((dyn ? 2 : 1) * 4 + 1). The state is advanced.
extern "C" __global__ void astral_ink_conv_step(float* __restrict__ x, float* __restrict__ state,
                                                const float* __restrict__ weight, const float* __restrict__ dyn,
                                                unsigned channels, unsigned heads) { PDL_WAIT();
    const float norm = rsqrtf(dyn ? 9.0f : 5.0f);
    const unsigned per_head = heads ? channels / heads : 1u;
    for (unsigned c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        const float in0 = state[c * 3 + 0], in1 = state[c * 3 + 1], in2 = state[c * 3 + 2], cur = x[c];
        const float* w = weight + (u64)c * 4;
        float out = in0 * w[0];
        out += in1 * w[1];
        out += in2 * w[2];
        out += cur * w[3];
        if (dyn) {
            const float* t = dyn + (u64)(c / per_head) * 4;
            out += cur * t[0];
            out += in2 * t[1];
            out += in1 * t[2];
            out += in0 * t[3];
        }
        state[c * 3 + 0] = in1;
        state[c * 3 + 1] = in2;
        state[c * 3 + 2] = cur;
        x[c] = (cur + out) * norm;
    }
}

// ---------------------------------------------------------------------------------------------
// Append this token's K and V (f32, [kv_heads * hd]) to an f16 cache at position `pos`.
extern "C" __global__ void astral_ink_kv_append(const float* __restrict__ k, const float* __restrict__ v,
                                                __half* __restrict__ kc, __half* __restrict__ vc,
                                                const int* __restrict__ pos, unsigned width, unsigned capacity) { PDL_WAIT();
    const u64 base = (u64)min((unsigned)pos[0], capacity - 1u) * width;   // clamped to the cache
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        kc[base + i] = __float2half_rn(k[i]);
        vc[base + i] = __float2half_rn(v[i]);
    }
}

// ---------------------------------------------------------------------------------------------
// Attention for one query token at position pos over cache positions [lo, pos]:
//   bias(h, d) = sum_k rel_bias[min(d, extent) * n_rel + k] * R[h, k]   (row `extent` is zero)
//   logit_j    = scale * (q_h . k_j) + bias(h, pos - j) - sink_h   (bias and sink enter as the
//                mask, unscaled, as in llama.cpp's build_attn_rel)
//   out_h      = sum_j softmax_j(logit, plus a sink logit of 0) * v_j
// SWA layers pass window = n_swa (positions with pos - j >= window are masked); global layers
// pass window = 0 (no window). GQA: kv head = h / (heads / kv_heads). One block per head; the
// logits of up to MAX_CTX positions live in shared memory.
#define MAX_CTX 8192
extern "C" __global__ void __launch_bounds__(128) astral_ink_attn_decode(
    const float* __restrict__ q, const float* __restrict__ r, const float* __restrict__ sink,
    const __half* __restrict__ kc, const __half* __restrict__ vc, const float* __restrict__ rel_bias,
    float* __restrict__ out, const int* __restrict__ pos_p, unsigned heads, unsigned kv_heads, unsigned hd,
    unsigned n_rel, unsigned extent, unsigned window, float scale, unsigned capacity) { PDL_WAIT();
    __shared__ float logit[MAX_CTX];
    __shared__ float rs[64];
    const unsigned h = blockIdx.x;
    const unsigned kvh = h / (heads / kv_heads);
    const unsigned pos = min((unsigned)pos_p[0], min(capacity, (unsigned)MAX_CTX) - 1u);   // clamped
    const unsigned lo = (window && pos + 1 > window) ? pos + 1 - window : 0u;
    const unsigned n = pos + 1 - lo;
    const float* qh = q + (u64)h * hd;
    const unsigned width = kv_heads * hd;
    for (unsigned i = threadIdx.x; i < n_rel; i += blockDim.x) rs[i] = r[(u64)h * n_rel + i];
    __syncthreads();
    const float sk = sink[h];
    float m = 0.0f;   // the sink logit is 0
    for (unsigned t = threadIdx.x; t < n; t += blockDim.x) {
        const unsigned j = lo + t;
        const __half* kj = kc + (u64)j * width + (u64)kvh * hd;
        float dot = 0.0f;
        for (unsigned d = 0; d < hd; ++d) dot = fmaf(qh[d], __half2float(kj[d]), dot);
        const unsigned dist = pos - j;
        float bias = 0.0f;
        if (dist < extent) {
            const float* row = rel_bias + (u64)dist * n_rel;
            for (unsigned k = 0; k < n_rel; ++k) bias = fmaf(row[k], rs[k], bias);
        }
        const float l = fmaf(scale, dot, bias) - sk;
        logit[t] = l;
        m = fmaxf(m, l);
    }
    // block max
    __shared__ float red[4];
    m = warp_max(m);
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    float sum = 0.0f;
    for (unsigned t = threadIdx.x; t < n; t += blockDim.x) {
        const float e = expf(logit[t] - m);
        logit[t] = e;
        sum += e;
    }
    sum = block_sum(sum) + expf(0.0f - m);
    __syncthreads();
    const float inv = 1.0f / sum;
    for (unsigned d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned t = 0; t < n; ++t) acc = fmaf(logit[t], __half2float(vc[(u64)(lo + t) * width + (u64)kvh * hd + d]), acc);
        out[(u64)h * hd + d] = acc * inv;
    }
}

// ---------------------------------------------------------------------------------------------
// MoE router: probs = sigmoid(logits); pick top-k of probs + bias; weights = probs[picked],
// normalised by their sum (+1e-20), then times w_scale. One warp; n_expert <= 256, k <= 32.
extern "C" __global__ void __launch_bounds__(256) astral_ink_router_topk(const float* __restrict__ logits, const float* __restrict__ bias,
                                                                         int* __restrict__ ids, float* __restrict__ weights,
                                                                         unsigned n_expert, unsigned k, float w_scale) { PDL_WAIT();
    // One block per token (blockIdx.x), one thread per expert. An expert's rank is the number of
    // experts ahead of it (higher score, or equal score and lower index: a stable descending
    // sort); ranks < k are picked.
    __shared__ float prob[256], sel[256], wsel[32];
    const unsigned t = threadIdx.x, tok = blockIdx.x;
    logits += (u64)tok * n_expert;
    ids += (u64)tok * k;
    weights += (u64)tok * k;
    if (t < n_expert) {
        const float pr = 1.0f / (1.0f + expf(-logits[t]));
        prob[t] = pr;
        sel[t] = bias ? pr + bias[t] : pr;
    }
    __syncthreads();
    if (t < n_expert) {
        const float v = sel[t];
        unsigned rank = 0;
        for (unsigned j = 0; j < n_expert; ++j) {
            const float o = sel[j];
            rank += (o > v || (o == v && j < t)) ? 1u : 0u;
        }
        if (rank < k) { ids[rank] = (int)t; wsel[rank] = prob[t]; }
    }
    __syncthreads();
    if (t < k) {
        float sum = 0.0f;
        for (unsigned j = 0; j < k; ++j) sum += wsel[j];   // in rank order
        weights[t] = wsel[t] / (sum + 1e-20f) * w_scale;
    }
}

// Router GEMV (F32 weights [n_expert][K], x rows of K) fused with astral_ink_router_topk: block
// (e, t) writes logit e of token t; the last block of token t to finish (counter[t], reset
// after use so graph replays start from zero) then runs the top-k selection for that token.
// Grid (n_expert, ntok), 256 threads (n_expert <= 256).
extern "C" __global__ void __launch_bounds__(256) astral_ink_router_fused(const float* __restrict__ w, const float* __restrict__ x, unsigned K,
                                                                          float* logits, const float* __restrict__ bias,
                                                                          int* __restrict__ ids, float* __restrict__ weights,
                                                                          unsigned n_expert, unsigned k, float w_scale,
                                                                          unsigned* __restrict__ counter) { PDL_WAIT();
    const unsigned e = blockIdx.x, tok = blockIdx.y, t = threadIdx.x;
    const float4* wv = reinterpret_cast<const float4*>(w + (u64)e * K);
    const float4* xv = reinterpret_cast<const float4*>(x + (u64)tok * K);
    float acc = 0.0f;
    for (unsigned i = t; i < (K >> 2); i += blockDim.x) {
        const float4 a = __ldg(wv + i), b = xv[i];
        acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc); acc = fmaf(a.z, b.z, acc); acc = fmaf(a.w, b.w, acc);
    }
    acc = block_sum(acc);
    __shared__ unsigned last;
    if (t == 0) {
        logits[(u64)tok * n_expert + e] = acc;
        __threadfence();
        last = atomicAdd(&counter[tok], 1u) == n_expert - 1u;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    __shared__ float prob[256], sel[256], wsel[32];
    const float* lg = logits + (u64)tok * n_expert;
    ids += (u64)tok * k;
    weights += (u64)tok * k;
    if (t < n_expert) {
        const float pr = 1.0f / (1.0f + expf(-__ldcg(lg + t)));
        prob[t] = pr;
        sel[t] = bias ? pr + bias[t] : pr;
    }
    __syncthreads();
    if (t < n_expert) {
        const float v = sel[t];
        unsigned rank = 0;
        for (unsigned j = 0; j < n_expert; ++j) {
            const float o = sel[j];
            rank += (o > v || (o == v && j < t)) ? 1u : 0u;
        }
        if (rank < k) { ids[rank] = (int)t; wsel[rank] = prob[t]; }
    }
    __syncthreads();
    if (t < k) {
        float sum = 0.0f;
        for (unsigned j = 0; j < k; ++j) sum += wsel[j];
        weights[t] = wsel[t] / (sum + 1e-20f) * w_scale;
    }
    if (t == 0) counter[tok] = 0u;
}

// SITU on the fused gate_up output (rows of 2*n_ff: gate first, up second):
//   act = beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)
extern "C" __global__ void astral_ink_situ(const float* __restrict__ gu, float* __restrict__ act, unsigned n_ff,
                                           unsigned slots, float beta, float lb) { PDL_WAIT();
    const unsigned total = n_ff * slots;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const unsigned s = i / n_ff, c = i - s * n_ff;
        const float g = gu[(u64)s * 2 * n_ff + c];
        const float u = gu[(u64)s * 2 * n_ff + n_ff + c];
        const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
        act[i] = a * (lb > 0.0f ? lb * tanhf(u / lb) : u);
    }
}

// out[c] = sum_s w[s] * y[s, c]
extern "C" __global__ void astral_ink_moe_combine(const float* __restrict__ y, const float* __restrict__ w,
                                                  float* __restrict__ out, unsigned dim, unsigned slots) { PDL_WAIT();
    for (unsigned c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (unsigned s = 0; s < slots; ++s) acc += w[s] * y[(u64)s * dim + c];
        out[c] = acc;
    }
}

// ---------------------------------------------------------------------------------------------
// Elementwise helpers.
extern "C" __global__ void astral_ink_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, unsigned n) { PDL_WAIT();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = a[i] + b[i];
}
// x *= sigmoid(g)
extern "C" __global__ void astral_ink_mul_sigmoid(float* __restrict__ x, const float* __restrict__ g, unsigned n) { PDL_WAIT();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] *= 1.0f / (1.0f + expf(-g[i]));
}
// x = cap * tanh(x / cap)
extern "C" __global__ void astral_ink_softcap(float* __restrict__ x, unsigned n, float cap) { PDL_WAIT();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] = cap * tanhf(x[i] / cap);
}
// Greedy sampling: index of the largest of the first n logits (lowest index on ties). One block.
extern "C" __global__ void __launch_bounds__(1024) astral_ink_argmax(const float* __restrict__ x, unsigned n, int* __restrict__ out) { PDL_WAIT();
    __shared__ float bv[32];
    __shared__ unsigned bi[32];
    float best = -INFINITY;
    unsigned idx = 0xffffffffu;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > best) { best = x[i]; idx = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const unsigned oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
    }
    if ((threadIdx.x & 31u) == 0) { bv[threadIdx.x >> 5] = best; bi[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if (threadIdx.x < 32) {
        best = threadIdx.x < (blockDim.x >> 5) ? bv[threadIdx.x] : -INFINITY;
        idx = threadIdx.x < (blockDim.x >> 5) ? bi[threadIdx.x] : 0xffffffffu;
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
        }
        if (threadIdx.x == 0) out[0] = (int)idx;
    }
}

// =============================================================================================
// Fused decode kernels (P1 optimisation). Each replaces a chain of the kernels above with one
// launch; the arithmetic per element is unchanged, so results match the unfused path up to
// summation order. Mirrored by the `fused` launchers in crates/kernels/src/inkling.rs.

// int8 quantisation of one 32-value block held one value per lane (llama.cpp q8_1 scheme,
// as astral_quantize_x_q8 in gemv.cu).
__device__ __forceinline__ void q8_store(float v, signed char* __restrict__ xq, float* __restrict__ xd, unsigned blk, unsigned lane,
                                         __half* __restrict__ xh) {
    if (xh) xh[blk * 32u + lane] = __float2half_rn(v);   // f16 copy for the prefill tensor-core GEMM
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d = a * (1.0f / 127.0f);
    xq[blk * 32u + lane] = (signed char)(a == 0.0f ? 0 : __float2int_rn(v / d));
    if (lane == 0) xd[blk] = d;
}

// y = rms_norm(x) * w (y optional), and y quantised to int8 per 32 values. One block.
extern "C" __global__ void __launch_bounds__(1024) astral_ink_rms_q8(const float* __restrict__ x, unsigned x_stride, const float* __restrict__ w,
                                                                     float* __restrict__ y, signed char* __restrict__ xq,
                                                                     float* __restrict__ xd, __half* __restrict__ xh, unsigned dim, float eps) { PDL_WAIT();
    // One block per row t: x row at x + t * x_stride; y, xq token-major.
    const unsigned t = blockIdx.x;
    x += (u64)t * x_stride;
    if (y) y += (u64)t * dim;
    xq += (u64)t * dim;
    xd += (u64)t * (dim >> 5);
    if (xh) xh += (u64)t * dim;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) ss = fmaf(x[i], x[i], ss);
    const float inv = rsqrtf(block_sum(ss) / (float)dim + eps);
    const unsigned lane = threadIdx.x & 31u, nw = blockDim.x >> 5;
    for (unsigned b = threadIdx.x >> 5; b < (dim >> 5); b += nw) {
        const unsigned i = b * 32u + lane;
        const float v = x[i] * inv * (w ? w[i] : 1.0f);
        if (y) y[i] = v;
        q8_store(v, xq, xd, b, lane, xh);
    }
}

// Attention prep for one token, one block (128 threads) per job:
//   jobs [0, H)        Q head: x *= q_scale; clip(thr_qk); global: *= tau
//   jobs [H, 2H)       R head: x *= r_scale; clip(thr_rel); global: *= tau
//   jobs [2H, 2H+KH)   K head: conv (+ dyn taps) ; clip(thr_qk) ; store f16 at pos (and f32)
//   jobs [2H+KH, +KH)  V head: conv (+ dyn taps) ; rms_norm(eps) ; store f16 at pos (and f32)
// conv_kv = this slot's conv state at channel 0 (K) / width (V); dyn_heads taps per projection.
extern "C" __global__ void __launch_bounds__(128) astral_ink_attn_prep(
    float* __restrict__ q, float* __restrict__ r, float* __restrict__ k, float* __restrict__ v,
    const float* __restrict__ q_scale, const float* __restrict__ r_scale,
    float* __restrict__ conv_kv, const float* __restrict__ k_conv_w, const float* __restrict__ v_conv_w,
    const float* __restrict__ kdyn, const float* __restrict__ vdyn, unsigned dyn_heads,
    __half* __restrict__ kc, __half* __restrict__ vc, const int* __restrict__ pos_p,
    unsigned heads, unsigned kv_heads, unsigned hd, unsigned n_rel, float thr_qk, float thr_rel, float eps,
    unsigned global, float floor_len, float alpha, unsigned capacity, unsigned do_conv,
    const int* __restrict__ seq_p, unsigned n_seqs, unsigned conv_stride) { PDL_WAIT();
    // blockIdx.y = row t (see row_pos); buffers are token-major. With do_conv = 0 the K/V convs
    // were already applied (astral_ink_conv_seq, prefill) and are skipped.
    const unsigned job = blockIdx.x, t = threadIdx.x, tok = blockIdx.y;
    const unsigned width = kv_heads * hd;
    q += (u64)tok * heads * hd;
    r += (u64)tok * heads * n_rel;
    k += (u64)tok * width;
    v += (u64)tok * width;
    const int posi = row_pos(pos_p, seq_p, tok);
    {
        const u64 sq = row_seq(seq_p, n_seqs, tok);
        conv_kv += sq * conv_stride;
        kc += sq * capacity * width;
        vc += sq * capacity * width;
        if (seq_p && kdyn) { kdyn += (u64)tok * dyn_heads * 4u; vdyn += (u64)tok * dyn_heads * 4u; }
    }
    const float tau = 1.0f + alpha * logf(fmaxf(((float)posi + 1.0f) / floor_len, 1.0f));
    if (job < 2u * heads) {
        const bool is_r = job >= heads;
        const unsigned h = is_r ? job - heads : job, n = is_r ? n_rel : hd;
        float* xh = (is_r ? r : q) + (u64)h * n;
        const float* sc = (is_r ? r_scale : q_scale) + (u64)h * n;
        float val[2] = {0.0f, 0.0f};   // hd <= 256
        float ss = 0.0f;
        for (unsigned i = t, j = 0; i < n && j < 2; i += blockDim.x, ++j) { val[j] = xh[i] * sc[i]; ss = fmaf(val[j], val[j], ss); }
        ss = block_sum(ss);
        const float f = fminf(fmaxf((is_r ? thr_rel : thr_qk) / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f) * (global ? tau : 1.0f);
        for (unsigned i = t, j = 0; i < n && j < 2; i += blockDim.x, ++j) xh[i] = val[j] * f;
        return;
    }
    const bool is_v = job >= 2u * heads + kv_heads;
    const unsigned h = job - 2u * heads - (is_v ? kv_heads : 0u);
    float* x = is_v ? v : k;
    float* st = conv_kv + (is_v ? (u64)width * 3u : 0u);
    const float* wt = is_v ? v_conv_w : k_conv_w;
    const float* dyn = is_v ? vdyn : kdyn;
    const unsigned per_head = dyn_heads ? width / dyn_heads : 1u;
    const float norm = rsqrtf(dyn ? 9.0f : 5.0f);
    float val[2] = {0.0f, 0.0f};
    float ss = 0.0f;
    for (unsigned i = t, j = 0; i < hd && j < 2; i += blockDim.x, ++j) {
        const unsigned c = h * hd + i;
        if (do_conv) {
            const float in0 = st[c * 3 + 0], in1 = st[c * 3 + 1], in2 = st[c * 3 + 2], cur = x[c];
            const float* w = wt + (u64)c * 4;
            float out = in0 * w[0];
            out += in1 * w[1];
            out += in2 * w[2];
            out += cur * w[3];
            if (dyn) {
                const float* tp = dyn + (u64)(c / per_head) * 4;
                out += cur * tp[0];
                out += in2 * tp[1];
                out += in1 * tp[2];
                out += in0 * tp[3];
            }
            st[c * 3 + 0] = in1;
            st[c * 3 + 1] = in2;
            st[c * 3 + 2] = cur;
            val[j] = (cur + out) * norm;
        } else {
            val[j] = x[c];
        }
        ss = fmaf(val[j], val[j], ss);
    }
    ss = block_sum(ss);
    const float f = is_v ? rsqrtf(ss / (float)hd + eps) : fminf(fmaxf(thr_qk / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
    const u64 base = (u64)min((unsigned)posi, capacity - 1u) * width;
    __half* cache = is_v ? vc : kc;
    for (unsigned i = t, j = 0; i < hd && j < 2; i += blockDim.x, ++j) {
        const unsigned c = h * hd + i;
        const float o = val[j] * f;
        x[c] = o;
        cache[base + c] = __float2half_rn(o);
    }
}

// Short conv over `ntok` consecutive tokens (prefill), one thread per channel, in place:
// x[t * x_stride + c]. Optional per-token dynamic taps dyn[t * dyn_stride + head * 4 + lag].
// Same arithmetic per token as astral_ink_conv_step; the state ends at the last token.
extern "C" __global__ void astral_ink_conv_seq(float* __restrict__ x, unsigned x_stride, unsigned ntok, float* __restrict__ state,
                                               const float* __restrict__ weight, const float* __restrict__ dyn, unsigned dyn_stride,
                                               unsigned dyn_heads, unsigned channels) { PDL_WAIT();
    const float norm = rsqrtf(dyn ? 9.0f : 5.0f);
    const unsigned per_head = dyn_heads ? channels / dyn_heads : 1u;
    for (unsigned c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        float in0 = state[c * 3 + 0], in1 = state[c * 3 + 1], in2 = state[c * 3 + 2];
        const float w0 = weight[c * 4 + 0], w1 = weight[c * 4 + 1], w2 = weight[c * 4 + 2], w3 = weight[c * 4 + 3];
        for (unsigned t = 0; t < ntok; ++t) {
            float* xp = x + (u64)t * x_stride + c;
            const float cur = *xp;
            float out = in0 * w0;
            out += in1 * w1;
            out += in2 * w2;
            out += cur * w3;
            if (dyn) {
                const float* tp = dyn + (u64)t * dyn_stride + (u64)(c / per_head) * 4;
                out += cur * tp[0];
                out += in2 * tp[1];
                out += in1 * tp[2];
                out += in0 * tp[3];
            }
            in0 = in1;
            in1 = in2;
            in2 = cur;
            *xp = (cur + out) * norm;
        }
        state[c * 3 + 0] = in0;
        state[c * 3 + 1] = in1;
        state[c * 3 + 2] = in2;
    }
}

// Loop controller, part 1: cd[r] = W_down[r] . concat(rms(x), rms(msg)) (W_down F32 [rank, 2*dim]).
// One block per rank row; every block recomputes the two RMS scales (cheap, from L2).
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_down(const float* __restrict__ wd, const float* __restrict__ x,
                                                                       unsigned x_stride, const float* __restrict__ msg,
                                                                       unsigned msg_stride, float* __restrict__ cd, unsigned dim) { PDL_WAIT();
    // blockIdx.x = rank row, blockIdx.y = token.
    const unsigned t = blockIdx.y;
    x += (u64)t * x_stride;
    msg += (u64)t * msg_stride;
    const float* w = wd + (u64)blockIdx.x * 2u * dim;
    float sx = 0.0f, sm = 0.0f, dx = 0.0f, dm = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        const float a = x[i], b = msg[i];
        sx = fmaf(a, a, sx);
        sm = fmaf(b, b, sm);
        dx = fmaf(w[i], a, dx);
        dm = fmaf(w[dim + i], b, dm);
    }
    sx = block_sum(sx);
    sm = block_sum(sm);
    dx = block_sum(dx);
    dm = block_sum(dm);
    if (threadIdx.x == 0) cd[(u64)t * gridDim.x + blockIdx.x] = dx * rsqrtf(sx / (float)dim + 1e-6f) + dm * rsqrtf(sm / (float)dim + 1e-6f);
}

// Loop controller, part 2: gain = Wg cd, forget = Wf cd (W_gf F32 [2*dim, rank], gain rows
// first); out = x - tanh(forget) x + tanh(gain) msg. One warp per output element.
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_combine(const float* __restrict__ wgf, const float* __restrict__ cd,
                                                                          const float* __restrict__ x, unsigned x_stride,
                                                                          const float* __restrict__ msg, unsigned msg_stride,
                                                                          float* __restrict__ out, unsigned dim, unsigned rank) { PDL_WAIT();
    // blockIdx.y = token; out is token-major [t][dim].
    const unsigned tok = blockIdx.y;
    const unsigned n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (n >= dim) return;
    cd += (u64)tok * rank;
    const float* wg = wgf + (u64)n * rank;
    const float* wf = wgf + (u64)(dim + n) * rank;
    float g = 0.0f, f = 0.0f;
    for (unsigned i = lane; i < rank; i += 32u) {
        const float c = cd[i];
        g = fmaf(wg[i], c, g);
        f = fmaf(wf[i], c, f);
    }
    g = warp_sum(g);
    f = warp_sum(f);
    if (lane == 0) {
        const float c = x[(u64)tok * x_stride + n];
        out[(u64)tok * dim + n] = (c - tanhf(f) * c) + tanhf(g) * msg[(u64)tok * msg_stride + n];
    }
}

// Up to three independent loop-controller jobs in one launch (the Q, K and V inputs of a
// looped layer): job j = blockIdx.z with its own weights, input, message and outputs.
struct Ctrl3 {
    const float* wd[3];
    const float* wgf[3];
    const float* x[3];
    const float* msg[3];
    float* cd[3];
    float* out[3];
};
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_down3(Ctrl3 c, unsigned x_stride, unsigned msg_stride, unsigned dim) { PDL_WAIT();
    const unsigned j = blockIdx.z, t = blockIdx.y;
    const float* x = c.x[j] + (u64)t * x_stride;
    const float* msg = c.msg[j] + (u64)t * msg_stride;
    const float* w = c.wd[j] + (u64)blockIdx.x * 2u * dim;
    float sx = 0.0f, sm = 0.0f, dx = 0.0f, dm = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        const float a = x[i], b = msg[i];
        sx = fmaf(a, a, sx);
        sm = fmaf(b, b, sm);
        dx = fmaf(w[i], a, dx);
        dm = fmaf(w[dim + i], b, dm);
    }
    sx = block_sum(sx);
    sm = block_sum(sm);
    dx = block_sum(dx);
    dm = block_sum(dm);
    if (threadIdx.x == 0) c.cd[j][(u64)t * gridDim.x + blockIdx.x] = dx * rsqrtf(sx / (float)dim + 1e-6f) + dm * rsqrtf(sm / (float)dim + 1e-6f);
}
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_combine3(Ctrl3 c, unsigned x_stride, unsigned msg_stride, unsigned dim,
                                                                           unsigned rank) { PDL_WAIT();
    const unsigned j = blockIdx.z, tok = blockIdx.y;
    const unsigned n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (n >= dim) return;
    const float* cd = c.cd[j] + (u64)tok * rank;
    const float* wg = c.wgf[j] + (u64)n * rank;
    const float* wf = c.wgf[j] + (u64)(dim + n) * rank;
    float g = 0.0f, f = 0.0f;
    for (unsigned i = lane; i < rank; i += 32u) {
        const float v = cd[i];
        g = fmaf(wg[i], v, g);
        f = fmaf(wf[i], v, f);
    }
    g = warp_sum(g);
    f = warp_sum(f);
    if (lane == 0) {
        const float xv = c.x[j][(u64)tok * x_stride + n];
        c.out[j][(u64)tok * dim + n] = (xv - tanhf(f) * xv) + tanhf(g) * c.msg[j][(u64)tok * msg_stride + n];
    }
}

// ctrl_down3 and ctrl_combine3 for CT tokens per block (tokens CT * blockIdx.y ..): each
// weight load serves CT tokens; every token's arithmetic is that of the one-token kernels.
#define CT 8u
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_down3_t(Ctrl3 c, unsigned x_stride, unsigned msg_stride, unsigned dim,
                                                                          unsigned ntok) { PDL_WAIT();
    const unsigned j = blockIdx.z, t0 = blockIdx.y * CT;
    const unsigned nt = min(CT, ntok - t0);
    const float* w = c.wd[j] + (u64)blockIdx.x * 2u * dim;
    float sx[CT], sm[CT], dx[CT], dm[CT];
#pragma unroll
    for (unsigned t = 0; t < CT; ++t) sx[t] = sm[t] = dx[t] = dm[t] = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        const float wa = w[i], wb = w[dim + i];
#pragma unroll
        for (unsigned t = 0; t < CT; ++t) {
            if (t < nt) {
                const float a = c.x[j][(u64)(t0 + t) * x_stride + i], b = c.msg[j][(u64)(t0 + t) * msg_stride + i];
                sx[t] = fmaf(a, a, sx[t]);
                sm[t] = fmaf(b, b, sm[t]);
                dx[t] = fmaf(wa, a, dx[t]);
                dm[t] = fmaf(wb, b, dm[t]);
            }
        }
    }
    for (unsigned t = 0; t < nt; ++t) {
        const float vsx = block_sum(sx[t]);
        const float vsm = block_sum(sm[t]);
        const float vdx = block_sum(dx[t]);
        const float vdm = block_sum(dm[t]);
        if (threadIdx.x == 0) c.cd[j][(u64)(t0 + t) * gridDim.x + blockIdx.x] = vdx * rsqrtf(vsx / (float)dim + 1e-6f) + vdm * rsqrtf(vsm / (float)dim + 1e-6f);
    }
}
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_combine3_t(Ctrl3 c, unsigned x_stride, unsigned msg_stride, unsigned dim,
                                                                             unsigned rank, unsigned ntok) { PDL_WAIT();
    const unsigned j = blockIdx.z, t0 = blockIdx.y * CT;
    const unsigned nt = min(CT, ntok - t0);
    const unsigned n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (n >= dim) return;
    const float* wg = c.wgf[j] + (u64)n * rank;
    const float* wf = c.wgf[j] + (u64)(dim + n) * rank;
    // rank <= 64: each lane holds its (at most 2) weights of both rows.
    float g0 = 0.0f, g1 = 0.0f, f0 = 0.0f, f1 = 0.0f;
    if (lane < rank) { g0 = wg[lane]; f0 = wf[lane]; }
    if (lane + 32u < rank) { g1 = wg[lane + 32u]; f1 = wf[lane + 32u]; }
    for (unsigned t = 0; t < nt; ++t) {
        const unsigned tok = t0 + t;
        const float* cd = c.cd[j] + (u64)tok * rank;
        float g = 0.0f, f = 0.0f;
        if (lane < rank) { const float v = cd[lane]; g = fmaf(g0, v, g); f = fmaf(f0, v, f); }
        if (lane + 32u < rank) { const float v = cd[lane + 32u]; g = fmaf(g1, v, g); f = fmaf(f1, v, f); }
        g = warp_sum(g);
        f = warp_sum(f);
        if (lane == 0) {
            const float xv = c.x[j][(u64)tok * x_stride + n];
            c.out[j][(u64)tok * dim + n] = (xv - tanhf(f) * xv) + tanhf(g) * c.msg[j][(u64)tok * msg_stride + n];
        }
    }
}

// Batched (not batch-exact) loop controllers for many tokens. down: block = 8 ranks x 8 tokens
// of one job, warp w on rank 8 * blockIdx.x + w; the 8 tokens' rows are read by all 8 warps
// (shared through L1), each weight row by one warp. grid (rank / 8, ceil(ntok / 8), jobs).
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_down3_b(Ctrl3 c, unsigned x_stride, unsigned msg_stride, unsigned dim,
                                                                          unsigned ntok) { PDL_WAIT();
    const unsigned j = blockIdx.z, t0 = blockIdx.y * CT, warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned nt = min(CT, ntok - t0);
    const unsigned rank = gridDim.x * 8u, r = blockIdx.x * 8u + warp;
    const float4* wa = reinterpret_cast<const float4*>(c.wd[j] + (u64)r * 2u * dim);
    const float4* wb = reinterpret_cast<const float4*>(c.wd[j] + (u64)r * 2u * dim + dim);
    const float4* xr[CT];
    const float4* mr[CT];
#pragma unroll
    for (unsigned t = 0; t < CT; ++t) {
        const unsigned tt = t0 + min(t, nt - 1u);
        xr[t] = reinterpret_cast<const float4*>(c.x[j] + (u64)tt * x_stride);
        mr[t] = reinterpret_cast<const float4*>(c.msg[j] + (u64)tt * msg_stride);
    }
    float sx[CT], sm[CT], dx[CT], dm[CT];
#pragma unroll
    for (unsigned t = 0; t < CT; ++t) sx[t] = sm[t] = dx[t] = dm[t] = 0.0f;
    for (unsigned i = lane; i < (dim >> 2); i += 32u) {
        const float4 a = __ldg(wa + i), b = __ldg(wb + i);
#pragma unroll
        for (unsigned t = 0; t < CT; ++t) {
            const float4 x = __ldg(xr[t] + i), m = __ldg(mr[t] + i);
            sx[t] = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, sx[t]))));
            sm[t] = fmaf(m.x, m.x, fmaf(m.y, m.y, fmaf(m.z, m.z, fmaf(m.w, m.w, sm[t]))));
            dx[t] = fmaf(a.x, x.x, fmaf(a.y, x.y, fmaf(a.z, x.z, fmaf(a.w, x.w, dx[t]))));
            dm[t] = fmaf(b.x, m.x, fmaf(b.y, m.y, fmaf(b.z, m.z, fmaf(b.w, m.w, dm[t]))));
        }
    }
#pragma unroll
    for (unsigned t = 0; t < CT; ++t) {
        const float vsx = warp_sum(sx[t]), vsm = warp_sum(sm[t]), vdx = warp_sum(dx[t]), vdm = warp_sum(dm[t]);
        if (lane == 0 && t < nt) c.cd[j][(u64)(t0 + t) * rank + r] = vdx * rsqrtf(vsx / (float)dim + 1e-6f) + vdm * rsqrtf(vsm / (float)dim + 1e-6f);
    }
}
// combine: thread n of block x covers output n = 256 * blockIdx.x + threadIdx.x for 8 tokens;
// the tokens' rank vectors sit in shared memory. rank % 4 == 0, rank <= 64.
// grid (ceil(dim / 256), ceil(ntok / 8), jobs).
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_combine3_b(Ctrl3 c, unsigned x_stride, unsigned msg_stride, unsigned dim,
                                                                             unsigned rank, unsigned ntok) { PDL_WAIT();
    const unsigned j = blockIdx.z, t0 = blockIdx.y * CT;
    const unsigned nt = min(CT, ntok - t0);
    __shared__ float cds[CT][64];
    for (unsigned i = threadIdx.x; i < CT * rank; i += blockDim.x) {
        const unsigned t = i / rank, q = i - t * rank;
        cds[t][q] = t < nt ? c.cd[j][(u64)(t0 + t) * rank + q] : 0.0f;
    }
    __syncthreads();
    const unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= dim) return;
    const float4* wg = reinterpret_cast<const float4*>(c.wgf[j] + (u64)n * rank);
    const float4* wf = reinterpret_cast<const float4*>(c.wgf[j] + (u64)(dim + n) * rank);
    float g[CT], f[CT];
#pragma unroll
    for (unsigned t = 0; t < CT; ++t) g[t] = f[t] = 0.0f;
    for (unsigned q = 0; q < (rank >> 2); ++q) {
        const float4 a = __ldg(wg + q), b = __ldg(wf + q);
#pragma unroll
        for (unsigned t = 0; t < CT; ++t) {
            const float4 v = *reinterpret_cast<const float4*>(&cds[t][4u * q]);
            g[t] = fmaf(a.x, v.x, fmaf(a.y, v.y, fmaf(a.z, v.z, fmaf(a.w, v.w, g[t]))));
            f[t] = fmaf(b.x, v.x, fmaf(b.y, v.y, fmaf(b.z, v.z, fmaf(b.w, v.w, f[t]))));
        }
    }
#pragma unroll
    for (unsigned t = 0; t < CT; ++t) {
        if (t < nt) {
            const unsigned tok = t0 + t;
            const float xv = c.x[j][(u64)tok * x_stride + n];
            c.out[j][(u64)tok * dim + n] = (xv - tanhf(f[t]) * xv) + tanhf(g[t]) * c.msg[j][(u64)tok * msg_stride + n];
        }
    }
}

// Up to three rms_q8 jobs (blockIdx.y = job): y_j = rms_norm(x_j) * w_j, quantised. Rows of x_j at
// t * x_stride; grid (ntok, jobs).
struct Rms3 {
    const float* x[3];
    const float* w[3];
    float* y[3];
    signed char* xq[3];
    float* xd[3];
    __half* xh[3];
};
extern "C" __global__ void __launch_bounds__(1024) astral_ink_rms_q8_3(Rms3 r, unsigned x_stride, unsigned dim, float eps) { PDL_WAIT();
    const unsigned t = blockIdx.x, j = blockIdx.y;
    const float* x = r.x[j] + (u64)t * x_stride;
    float* y = r.y[j] ? r.y[j] + (u64)t * dim : nullptr;
    signed char* xq = r.xq[j] + (u64)t * dim;
    float* xd = r.xd[j] + (u64)t * (dim >> 5);
    __half* xh = r.xh[j] ? r.xh[j] + (u64)t * dim : nullptr;
    const float* w = r.w[j];
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) ss = fmaf(x[i], x[i], ss);
    const float inv = rsqrtf(block_sum(ss) / (float)dim + eps);
    const unsigned lane = threadIdx.x & 31u, nw = blockDim.x >> 5;
    for (unsigned b = threadIdx.x >> 5; b < (dim >> 5); b += nw) {
        const unsigned i = b * 32u + lane;
        const float v = x[i] * inv * w[i];
        if (y) y[i] = v;
        q8_store(v, xq, xd, b, lane, xh);
    }
}

// x *= sigmoid(g), then quantised to int8 (n % 32 == 0). One warp per 32-value block.
extern "C" __global__ void __launch_bounds__(256) astral_ink_gate_q8(float* __restrict__ x, const float* __restrict__ g,
                                                                     signed char* __restrict__ xq, float* __restrict__ xd,
                                                                     __half* __restrict__ xh, unsigned n) { PDL_WAIT();
    const unsigned b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (b >= (n >> 5)) return;
    const unsigned i = b * 32u + lane;
    const float v = x[i] * (1.0f / (1.0f + expf(-g[i])));
    x[i] = v;
    q8_store(v, xq, xd, b, lane, xh);
}

// SITU over `slots` gate_up rows, quantised to int8 per 32 values (n_ff % 32 == 0).
extern "C" __global__ void __launch_bounds__(256) astral_ink_situ_q8(const float* __restrict__ gu, signed char* __restrict__ xq,
                                                                     float* __restrict__ xd, __half* __restrict__ xh, unsigned n_ff,
                                                                     unsigned slots, float beta, float lb) { PDL_WAIT();
    const unsigned b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (b >= slots * (n_ff >> 5)) return;
    const unsigned i = b * 32u + lane;
    const unsigned s = i / n_ff, c = i - s * n_ff;
    const float g = gu[(u64)s * 2 * n_ff + c];
    const float u = gu[(u64)s * 2 * n_ff + n_ff + c];
    const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
    q8_store(a * (lb > 0.0f ? lb * tanhf(u / lb) : u), xq, xd, b, lane, xh);
}

// One token of astral_ink_conv_residual for channel c, with every rounding explicit (no
// compiler contraction) so the single-sequence and batched forms agree bit for bit.
__device__ __forceinline__ void conv_res_tok(const float* __restrict__ x, const float* __restrict__ moe_w, unsigned slots, unsigned t,
                                             unsigned c, unsigned channels, float& in0, float& in1, float& in2, float w0, float w1,
                                             float w2, float w3, float* __restrict__ cur, float* __restrict__ bank_row) {
    float xin;
    if (moe_w) {
        xin = 0.0f;
        for (unsigned s = 0; s < slots; ++s) xin = fmaf(moe_w[(u64)t * slots + s], x[((u64)t * slots + s) * channels + c], xin);
    } else {
        xin = x[(u64)t * channels + c];
    }
    float out = __fmul_rn(in0, w0);
    out = fmaf(in1, w1, out);
    out = fmaf(in2, w2, out);
    out = fmaf(xin, w3, out);
    in0 = in1;
    in1 = in2;
    in2 = xin;
    const float y = __fmul_rn(__fadd_rn(xin, out), rsqrtf(5.0f));
    const u64 o = (u64)t * channels + c;
    if (bank_row) {
        bank_row[o] = cur[o];
        cur[o] = y;
    } else {
        cur[o] = __fadd_rn(cur[o], y);
    }
}

// Short conv (no dynamic taps) of `x` into the residual stream:
//   y = conv(x)   (x = sum_s w[s] y_s[c] when `moe_w` is set: the MoE combine first)
//   bank_row ? (bank_row = cur, cur = y) : (cur += y)
extern "C" __global__ void astral_ink_conv_residual(const float* __restrict__ x, const float* __restrict__ moe_w, unsigned slots,
                                                    unsigned ntok, float* __restrict__ state, const float* __restrict__ weight,
                                                    float* __restrict__ cur, float* __restrict__ bank_row, unsigned channels,
                                                    const int* __restrict__ seq_p, unsigned n_seqs, unsigned conv_stride) { PDL_WAIT();
    // Over `ntok` consecutive tokens (token-major x / cur / bank_row), one thread per channel.
    // With moe_w: x is [t][slot][c] and the input is sum_s moe_w[t * slots + s] x[t][s][c].
    // With seq_p (batched decode): row t is one token of its own sequence, with its own state.
    const unsigned t0 = seq_p ? blockIdx.y : 0u, t1 = seq_p ? blockIdx.y + 1u : ntok;
    float* st = seq_p ? state + row_seq(seq_p, n_seqs, blockIdx.y) * conv_stride : state;
    for (unsigned c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        float in0 = st[c * 3 + 0], in1 = st[c * 3 + 1], in2 = st[c * 3 + 2];
        const float* w = weight + (u64)c * 4;
        const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
        for (unsigned t = t0; t < t1; ++t) conv_res_tok(x, moe_w, slots, t, c, channels, in0, in1, in2, w0, w1, w2, w3, cur, bank_row);
        st[c * 3 + 0] = in0;
        st[c * 3 + 1] = in1;
        st[c * 3 + 2] = in2;
    }
}

// =============================================================================================
// Split-K decode attention (flash-decode). Same maths as astral_ink_attn_decode:
//   logit_j = scale * q.k_j + bias(h, pos - j) - sink_h, softmax with an extra zero sink logit.
// Pass 1: grid (heads, max_splits); block = 4 warps over one chunk of ATT_CHUNK positions, one
// warp per key (lane = 4 dims, so hd must be 128), online softmax per warp, merged per block
// into part[h][split] = {m, l, acc[hd]}. Splits past the last position exit at once (the grid
// is fixed so the launch can live in a CUDA graph). Pass 2 merges the splits and the sink.
#define ATT_CHUNK 64
extern "C" __global__ void __launch_bounds__(128) astral_ink_attn_split(
    const float* __restrict__ q, const float* __restrict__ r, const float* __restrict__ sink,
    const __half* __restrict__ kc, const __half* __restrict__ vc, const float* __restrict__ rel_bias,
    float* __restrict__ part, const int* __restrict__ pos_p, unsigned heads, unsigned kv_heads, unsigned n_rel,
    unsigned extent, unsigned window, float scale, unsigned capacity, unsigned max_splits,
    const int* __restrict__ seq_p, unsigned n_seqs) { PDL_WAIT();
    // blockIdx.z = query row z (see row_pos; causal: it sees positions <= its own).
    constexpr unsigned HD = 128;
    const unsigned h = blockIdx.x, sp = blockIdx.y, z = blockIdx.z;
    q += (u64)z * heads * HD;
    r += (u64)z * heads * n_rel;
    sink += (u64)z * heads;
    part += (u64)z * heads * max_splits * (HD + 2);
    {
        const u64 so = row_seq(seq_p, n_seqs, z) * capacity * kv_heads * HD;
        kc += so;
        vc += so;
    }
    const unsigned kvh = h / (heads / kv_heads);
    const unsigned pos = min((unsigned)row_pos(pos_p, seq_p, z), min(capacity, (unsigned)MAX_CTX) - 1u);
    const unsigned lo = (window && pos + 1 > window) ? pos + 1 - window : 0u;
    // Split sp takes chunks sp, sp + gridDim.y, ... (a fixed split count serves any context).
    if (lo + sp * ATT_CHUNK > pos) return;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned width = kv_heads * HD;
    const float4 qv = reinterpret_cast<const float4*>(q + (u64)h * HD)[lane];
    const float rs = lane < n_rel ? r[(u64)h * n_rel + lane] : 0.0f;
    const float sk = sink[h];
    float m = -INFINITY, l = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    // Keys warp, warp + 4, ...: processed 4 at a time as independent chains (loads, dots and
    // reductions interleaved), then one online-softmax update per group.
    constexpr unsigned GK = 4;
    for (unsigned start = lo + sp * ATT_CHUNK; start <= pos; start += gridDim.y * ATT_CHUNK) {
    const unsigned end = min(pos + 1, start + ATT_CHUNK);
    for (unsigned j0 = start + warp; j0 < end; j0 += 4u * GK) {
        float sc[GK];
        uint2 vr[GK];
#pragma unroll
        for (unsigned u = 0; u < GK; ++u) {
            const unsigned j = j0 + 4u * u;
            const bool ok = j < end;
            const unsigned jj = ok ? j : j0;
            const uint2 kr = reinterpret_cast<const uint2*>(kc + (u64)jj * width + (u64)kvh * HD)[lane];
            vr[u] = reinterpret_cast<const uint2*>(vc + (u64)jj * width + (u64)kvh * HD)[lane];
            const float2 k01 = __half22float2(*reinterpret_cast<const __half2*>(&kr.x));
            const float2 k23 = __half22float2(*reinterpret_cast<const __half2*>(&kr.y));
            float tt = scale * (qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y);
            const unsigned dist = pos - jj;
            if (lane < n_rel && dist < extent) tt = fmaf(rel_bias[(u64)dist * n_rel + lane], rs, tt);
            sc[u] = tt;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
            for (unsigned u = 0; u < GK; ++u) sc[u] += __shfl_xor_sync(0xffffffffu, sc[u], o);
        }
        float mx = m;
#pragma unroll
        for (unsigned u = 0; u < GK; ++u) {
            sc[u] = (j0 + 4u * u < end) ? sc[u] - sk : -INFINITY;
            mx = fmaxf(mx, sc[u]);
        }
        const float c = expf(m - mx);
        l *= c; a0 *= c; a1 *= c; a2 *= c; a3 *= c;
#pragma unroll
        for (unsigned u = 0; u < GK; ++u) {
            const float p = expf(sc[u] - mx);
            const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&vr[u].x));
            const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&vr[u].y));
            l += p;
            a0 = fmaf(p, v01.x, a0); a1 = fmaf(p, v01.y, a1);
            a2 = fmaf(p, v23.x, a2); a3 = fmaf(p, v23.y, a3);
        }
        m = mx;
    }
    }
    __shared__ float wm[4], wl[4], wacc[4][HD];
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    wacc[warp][lane * 4 + 0] = a0; wacc[warp][lane * 4 + 1] = a1;
    wacc[warp][lane * 4 + 2] = a2; wacc[warp][lane * 4 + 3] = a3;
    __syncthreads();
    const float M = fmaxf(fmaxf(wm[0], wm[1]), fmaxf(wm[2], wm[3]));
    float* out = part + ((u64)h * max_splits + sp) * (HD + 2);
    const unsigned d = threadIdx.x;   // 128 threads = HD
    float acc = 0.0f, L = 0.0f;
#pragma unroll
    for (unsigned w = 0; w < 4; ++w) {
        const float f = wm[w] == -INFINITY ? 0.0f : expf(wm[w] - M);
        acc = fmaf(wacc[w][d], f, acc);
        L = fmaf(wl[w], f, L);
    }
    out[2 + d] = acc;
    if (d == 0) { out[0] = M; out[1] = L; }
}

extern "C" __global__ void __launch_bounds__(128) astral_ink_attn_split1(
    const float* __restrict__ q, const float* __restrict__ r, const float* __restrict__ sink,
    const __half* __restrict__ kc, const __half* __restrict__ vc, const float* __restrict__ rel_bias,
    float* __restrict__ part, const int* __restrict__ pos_p, unsigned heads, unsigned kv_heads, unsigned n_rel,
    unsigned extent, unsigned window, float scale, unsigned capacity, unsigned max_splits,
    const int* __restrict__ seq_p, unsigned n_seqs) { PDL_WAIT();
    // blockIdx.z = query row z (see row_pos; causal: it sees positions <= its own).
    constexpr unsigned HD = 128;
    const unsigned h = blockIdx.x, sp = blockIdx.y, z = blockIdx.z;
    q += (u64)z * heads * HD;
    r += (u64)z * heads * n_rel;
    sink += (u64)z * heads;
    part += (u64)z * heads * max_splits * (HD + 2);
    {
        const u64 so = row_seq(seq_p, n_seqs, z) * capacity * kv_heads * HD;
        kc += so;
        vc += so;
    }
    const unsigned kvh = h / (heads / kv_heads);
    const unsigned pos = min((unsigned)row_pos(pos_p, seq_p, z), min(capacity, (unsigned)MAX_CTX) - 1u);
    const unsigned lo = (window && pos + 1 > window) ? pos + 1 - window : 0u;
    // One 64-position chunk per split, max_splits of them for the whole range (decode at full
    // capacity, exact-size prefill); a block handles splits sp, sp + gridDim.y, ... (each
    // written as its own partial, so the grid size does not change the result).
    for (unsigned spl = sp; spl < max_splits; spl += gridDim.y) {
    const unsigned start = lo + spl * ATT_CHUNK;
    if (start > pos) return;
    const unsigned end = min(pos + 1, start + ATT_CHUNK);
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned width = kv_heads * HD;
    const float4 qv = reinterpret_cast<const float4*>(q + (u64)h * HD)[lane];
    const float rs = lane < n_rel ? r[(u64)h * n_rel + lane] : 0.0f;
    const float sk = sink[h];
    float m = -INFINITY, l = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    // Keys warp, warp + 4, ...: processed 4 at a time as independent chains (loads, dots and
    // reductions interleaved), then one online-softmax update per group.
    constexpr unsigned GK = 4;
    {
    for (unsigned j0 = start + warp; j0 < end; j0 += 4u * GK) {
        float sc[GK];
        uint2 vr[GK];
#pragma unroll
        for (unsigned u = 0; u < GK; ++u) {
            const unsigned j = j0 + 4u * u;
            const bool ok = j < end;
            const unsigned jj = ok ? j : j0;
            const uint2 kr = reinterpret_cast<const uint2*>(kc + (u64)jj * width + (u64)kvh * HD)[lane];
            vr[u] = reinterpret_cast<const uint2*>(vc + (u64)jj * width + (u64)kvh * HD)[lane];
            const float2 k01 = __half22float2(*reinterpret_cast<const __half2*>(&kr.x));
            const float2 k23 = __half22float2(*reinterpret_cast<const __half2*>(&kr.y));
            float tt = scale * (qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y);
            const unsigned dist = pos - jj;
            if (lane < n_rel && dist < extent) tt = fmaf(rel_bias[(u64)dist * n_rel + lane], rs, tt);
            sc[u] = tt;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
            for (unsigned u = 0; u < GK; ++u) sc[u] += __shfl_xor_sync(0xffffffffu, sc[u], o);
        }
        float mx = m;
#pragma unroll
        for (unsigned u = 0; u < GK; ++u) {
            sc[u] = (j0 + 4u * u < end) ? sc[u] - sk : -INFINITY;
            mx = fmaxf(mx, sc[u]);
        }
        const float c = expf(m - mx);
        l *= c; a0 *= c; a1 *= c; a2 *= c; a3 *= c;
#pragma unroll
        for (unsigned u = 0; u < GK; ++u) {
            const float p = expf(sc[u] - mx);
            const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&vr[u].x));
            const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&vr[u].y));
            l += p;
            a0 = fmaf(p, v01.x, a0); a1 = fmaf(p, v01.y, a1);
            a2 = fmaf(p, v23.x, a2); a3 = fmaf(p, v23.y, a3);
        }
        m = mx;
    }
    }
    __shared__ float wm[4], wl[4], wacc[4][HD];
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    wacc[warp][lane * 4 + 0] = a0; wacc[warp][lane * 4 + 1] = a1;
    wacc[warp][lane * 4 + 2] = a2; wacc[warp][lane * 4 + 3] = a3;
    __syncthreads();
    const float M = fmaxf(fmaxf(wm[0], wm[1]), fmaxf(wm[2], wm[3]));
    float* out = part + ((u64)h * max_splits + spl) * (HD + 2);
    const unsigned d = threadIdx.x;   // 128 threads = HD
    float acc = 0.0f, L = 0.0f;
#pragma unroll
    for (unsigned w = 0; w < 4; ++w) {
        const float f = wm[w] == -INFINITY ? 0.0f : expf(wm[w] - M);
        acc = fmaf(wacc[w][d], f, acc);
        L = fmaf(wl[w], f, L);
    }
    out[2 + d] = acc;
    if (d == 0) { out[0] = M; out[1] = L; }
    __syncthreads();   // the shared partials are rewritten by the next split
    }
}

extern "C" __global__ void __launch_bounds__(128) astral_ink_attn_merge(const float* __restrict__ part, float* __restrict__ out,
                                                                        const int* __restrict__ pos_p, unsigned window,
                                                                        unsigned capacity, unsigned max_splits,
                                                                        const int* __restrict__ seq_p) { PDL_WAIT();
    // blockIdx.x = head, blockIdx.y = query row.
    constexpr unsigned HD = 128;
    const unsigned h = blockIdx.x, z = blockIdx.y, d = threadIdx.x, heads = gridDim.x;
    const unsigned pos = min((unsigned)row_pos(pos_p, seq_p, z), min(capacity, (unsigned)MAX_CTX) - 1u);
    const unsigned lo = (window && pos + 1 > window) ? pos + 1 - window : 0u;
    const unsigned n_splits = min((pos + 1 - lo + ATT_CHUNK - 1) / ATT_CHUNK, max_splits);
    const float* p = part + ((u64)z * heads + h) * max_splits * (HD + 2);
    float M = 0.0f;   // the sink logit
    for (unsigned s = 0; s < n_splits; ++s) M = fmaxf(M, p[(u64)s * (HD + 2)]);
    float L = expf(0.0f - M), acc = 0.0f;
    for (unsigned s = 0; s < n_splits; ++s) {
        const float* ps = p + (u64)s * (HD + 2);
        const float f = expf(ps[0] - M);
        L = fmaf(ps[1], f, L);
        acc = fmaf(ps[2 + d], f, acc);
    }
    out[((u64)z * heads + h) * HD + d] = acc / L;
}

// attn_merge fused with the output gate and its quantisation (gate_q8): v = merged *
// sigmoid(g), stored to out and quantised per 32 values into (xq, xd, xh) for the O projection.
extern "C" __global__ void __launch_bounds__(128) astral_ink_attn_merge_q8(const float* __restrict__ part, float* __restrict__ out,
                                                                           const int* __restrict__ pos_p, unsigned window,
                                                                           unsigned capacity, unsigned max_splits,
                                                                           const float* __restrict__ g, signed char* __restrict__ xq,
                                                                           float* __restrict__ xd, __half* __restrict__ xh,
                                                                           const int* __restrict__ seq_p) { PDL_WAIT();
    constexpr unsigned HD = 128;
    const unsigned h = blockIdx.x, z = blockIdx.y, d = threadIdx.x, heads = gridDim.x;
    const unsigned pos = min((unsigned)row_pos(pos_p, seq_p, z), min(capacity, (unsigned)MAX_CTX) - 1u);
    const unsigned lo = (window && pos + 1 > window) ? pos + 1 - window : 0u;
    const unsigned n_splits = min((pos + 1 - lo + ATT_CHUNK - 1) / ATT_CHUNK, max_splits);
    const float* p = part + ((u64)z * heads + h) * max_splits * (HD + 2);
    float M = 0.0f;
    for (unsigned s = 0; s < n_splits; ++s) M = fmaxf(M, p[(u64)s * (HD + 2)]);
    float L = expf(0.0f - M), acc = 0.0f;
    for (unsigned s = 0; s < n_splits; ++s) {
        const float* ps = p + (u64)s * (HD + 2);
        const float f = expf(ps[0] - M);
        L = fmaf(ps[1], f, L);
        acc = fmaf(ps[2 + d], f, acc);
    }
    const u64 i = ((u64)z * heads + h) * HD + d;
    const float v = acc / L * (1.0f / (1.0f + expf(-g[i])));
    out[i] = v;
    q8_store(v, xq, xd, (unsigned)(i >> 5), d & 31u, xh);
}

// =============================================================================================
// Parallel prefill convs. A 4-tap causal conv has no recurrence: output t needs inputs
// t-3..t, taken from the saved state for t < 3. So every (token, channel) runs in parallel,
// reading `src` and writing `dst` (never in place); astral_ink_conv_state then stores the last
// three inputs as the new state (it must run after the conv, which reads the old state).

__device__ __forceinline__ float conv_in(const float* __restrict__ src, unsigned stride, const float* __restrict__ state,
                                         int t, unsigned c) {
    // t >= 0: token t of src; t = -1..-3: state slots 2..0 (state[c*3 + 2] is the newest).
    return t >= 0 ? src[(u64)t * stride + c] : state[c * 3 + (unsigned)(3 + t)];
}

// dst[t*dst_stride + c] = (x_t + conv + taps) * norm; optional per-token dynamic taps.
extern "C" __global__ void astral_ink_conv_par(const float* __restrict__ src, unsigned src_stride, float* __restrict__ dst,
                                               unsigned dst_stride, unsigned ntok, const float* __restrict__ state,
                                               const float* __restrict__ weight, const float* __restrict__ dyn, unsigned dyn_stride,
                                               unsigned dyn_heads, unsigned channels, float* __restrict__ hist) { PDL_WAIT();
    const float norm = rsqrtf(dyn ? 9.0f : 5.0f);
    const unsigned per_head = dyn_heads ? channels / dyn_heads : 1u;
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= channels || t >= ntok) return;
    const int ti = (int)t;
    const float in0 = conv_in(src, src_stride, state, ti - 3, c), in1 = conv_in(src, src_stride, state, ti - 2, c);
    const float in2 = conv_in(src, src_stride, state, ti - 1, c), cur = src[(u64)t * src_stride + c];
    if (hist) hist[(u64)t * channels + c] = cur;   // verify passes: the conv input, for the rollback
    const float* w = weight + (u64)c * 4;
    float out = in0 * w[0];
    out += in1 * w[1];
    out += in2 * w[2];
    out += cur * w[3];
    if (dyn) {
        const float* tp = dyn + (u64)t * dyn_stride + (u64)(c / per_head) * 4;
        out += cur * tp[0];
        out += in2 * tp[1];
        out += in1 * tp[2];
        out += in0 * tp[3];
    }
    dst[(u64)t * dst_stride + c] = (cur + out) * norm;
}

// Residual form: y = conv(src); bank_row ? (bank_row = cur, cur = y) : (cur += y). Token-major.
extern "C" __global__ void astral_ink_conv_res_par(const float* __restrict__ src, unsigned ntok, const float* __restrict__ state,
                                                   const float* __restrict__ weight, float* __restrict__ cur,
                                                   float* __restrict__ bank_row, unsigned channels, float* __restrict__ hist) { PDL_WAIT();
    const float norm = rsqrtf(5.0f);
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= channels || t >= ntok) return;
    const int ti = (int)t;
    const float in0 = conv_in(src, channels, state, ti - 3, c), in1 = conv_in(src, channels, state, ti - 2, c);
    const float in2 = conv_in(src, channels, state, ti - 1, c), x = src[(u64)t * channels + c];
    if (hist) hist[(u64)t * channels + c] = x;   // verify passes: the conv input, for the rollback
    const float* w = weight + (u64)c * 4;
    float out = in0 * w[0];
    out += in1 * w[1];
    out += in2 * w[2];
    out += x * w[3];
    const float y = (x + out) * norm;
    const u64 o = (u64)t * channels + c;
    if (bank_row) {
        bank_row[o] = cur[o];
        cur[o] = y;
    } else {
        cur[o] = cur[o] + y;
    }
}

// New conv state after `ntok` tokens of src: the last three inputs (older state shifts in
// when ntok < 3).
extern "C" __global__ void astral_ink_conv_state(const float* __restrict__ src, unsigned src_stride, unsigned ntok,
                                                 float* __restrict__ state, unsigned channels) { PDL_WAIT();
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int n = (int)ntok;
    const float s0 = conv_in(src, src_stride, state, n - 3, c), s1 = conv_in(src, src_stride, state, n - 2, c);
    const float s2 = conv_in(src, src_stride, state, n - 1, c);
    state[c * 3 + 0] = s0;
    state[c * 3 + 1] = s1;
    state[c * 3 + 2] = s2;
}

// out[t][c] = sum_s w[t*slots + s] * y[(t*slots + s)][c]   (MoE combine over ntok tokens)
extern "C" __global__ void astral_ink_moe_combine_t(const float* __restrict__ y, const float* __restrict__ w, float* __restrict__ out,
                                                    unsigned dim, unsigned slots, unsigned ntok) { PDL_WAIT();
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= dim || t >= ntok) return;
    float acc = 0.0f;
    for (unsigned s = 0; s < slots; ++s) acc += w[(u64)t * slots + s] * y[((u64)t * slots + s) * dim + c];
    out[(u64)t * dim + c] = acc;
}

// Loop controller part 1, 8 tokens per block: cd[t][r] = rx_t * (Wx[r] . x_t) + rm_t * (Wm[r] . msg_t),
// one warp per rank row r (W row read once per 8 tokens); the RMS scales come from `rms`
// ([t][2]: x, msg), computed once per token by astral_ink_rms2.
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_down_t(const float* __restrict__ wd, const float* __restrict__ x,
                                                                         unsigned x_stride, const float* __restrict__ msg,
                                                                         unsigned msg_stride, const float* __restrict__ rms,
                                                                         float* __restrict__ cd, unsigned dim, unsigned rank,
                                                                         unsigned ntok) { PDL_WAIT();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, nw = blockDim.x >> 5;
    const unsigned t0 = blockIdx.y * 8u;
    for (unsigned r = blockIdx.x * nw + warp; r < rank; r += gridDim.x * nw) {
        const float* w = wd + (u64)r * 2u * dim;
        float dx[8], dm[8];
#pragma unroll
        for (unsigned b = 0; b < 8; ++b) { dx[b] = 0.0f; dm[b] = 0.0f; }
        for (unsigned i = lane; i < dim; i += 32u) {
            const float wx = w[i], wm = w[dim + i];
#pragma unroll
            for (unsigned b = 0; b < 8; ++b) {
                const unsigned t = min(t0 + b, ntok - 1u);
                dx[b] = fmaf(wx, x[(u64)t * x_stride + i], dx[b]);
                dm[b] = fmaf(wm, msg[(u64)t * msg_stride + i], dm[b]);
            }
        }
#pragma unroll
        for (unsigned b = 0; b < 8; ++b) {
            const float a = warp_sum(dx[b]), m = warp_sum(dm[b]);
            const unsigned t = t0 + b;
            if (lane == 0 && t < ntok) cd[(u64)t * rank + r] = a * rms[2 * t] + m * rms[2 * t + 1];
        }
    }
}

// rms[t] = {1/rms(x_t), 1/rms(msg_t)} (eps 1e-6). One block per token.
extern "C" __global__ void __launch_bounds__(256) astral_ink_rms2(const float* __restrict__ x, unsigned x_stride, const float* __restrict__ msg,
                                                                  unsigned msg_stride, float* __restrict__ rms, unsigned dim) { PDL_WAIT();
    const unsigned t = blockIdx.x;
    const float* a = x + (u64)t * x_stride;
    const float* b = msg + (u64)t * msg_stride;
    float sa = 0.0f, sb = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) { sa = fmaf(a[i], a[i], sa); sb = fmaf(b[i], b[i], sb); }
    sa = block_sum(sa);
    sb = block_sum(sb);
    if (threadIdx.x == 0) { rms[2 * t] = rsqrtf(sa / (float)dim + 1e-6f); rms[2 * t + 1] = rsqrtf(sb / (float)dim + 1e-6f); }
}

// Loop controller part 2, 8 tokens per block: one warp per output element n, its gain and
// forget rows read once for the block's 8 tokens.
extern "C" __global__ void __launch_bounds__(256) astral_ink_ctrl_combine_t(const float* __restrict__ wgf, const float* __restrict__ cd,
                                                                            const float* __restrict__ x, unsigned x_stride,
                                                                            const float* __restrict__ msg, unsigned msg_stride,
                                                                            float* __restrict__ out, unsigned dim, unsigned rank,
                                                                            unsigned ntok) { PDL_WAIT();
    const unsigned n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (n >= dim) return;
    const unsigned t0 = blockIdx.y * 8u;
    const float* wg = wgf + (u64)n * rank;
    const float* wf = wgf + (u64)(dim + n) * rank;
    float g[8], f[8];
#pragma unroll
    for (unsigned b = 0; b < 8; ++b) { g[b] = 0.0f; f[b] = 0.0f; }
    for (unsigned i = lane; i < rank; i += 32u) {
        const float a = wg[i], bb = wf[i];
#pragma unroll
        for (unsigned b = 0; b < 8; ++b) {
            const float c = cd[(u64)min(t0 + b, ntok - 1u) * rank + i];
            g[b] = fmaf(a, c, g[b]);
            f[b] = fmaf(bb, c, f[b]);
        }
    }
#pragma unroll
    for (unsigned b = 0; b < 8; ++b) {
        const float gs = warp_sum(g[b]), fs = warp_sum(f[b]);
        const unsigned t = t0 + b;
        if (lane == 0 && t < ntok) {
            const float c = x[(u64)t * x_stride + n];
            out[(u64)t * dim + n] = (c - tanhf(fs) * c) + tanhf(gs) * msg[(u64)t * msg_stride + n];
        }
    }
}

// =============================================================================================
// Small f32 GEMM for the prefill loop controller: C[m][n] (+)= sum_k A(m, k) * W[n * ldw + k].
// A(m, k) = k < k_split ? a0[m * lda + k] * s(m, 0) : a1[m * lda1 + k - k_split] * s(m, 1), with
// s from `row_scale` ([m][2], or 1) -- so concat(rms(x), rms(msg)) is never materialised.
// 32x32 tiles, BK 32, 256 threads (2x2 outputs each); grid z splits K: atomic = 1 adds into C
// (which must start at zero), atomic = 2 writes split z's partial C to c + z * M * ldc (summed
// in a fixed order by astral_ink_sum_splits, so the result does not depend on timing).
extern "C" __global__ void __launch_bounds__(256) astral_ink_sgemm_tn(const float* __restrict__ a0, const float* __restrict__ a1,
                                                                      unsigned lda, unsigned lda1, unsigned k_split, const float* __restrict__ row_scale,
                                                                      const float* __restrict__ w, unsigned ldw, float* __restrict__ c,
                                                                      unsigned ldc, unsigned M, unsigned N, unsigned K,
                                                                      unsigned k_per_split, unsigned atomic) { PDL_WAIT();
    __shared__ float As[32][33], Ws[32][33];
    const unsigned tx = threadIdx.x & 15u, ty = threadIdx.x >> 4;   // 16 x 16 threads
    const unsigned m0 = blockIdx.y * 32u, n0 = blockIdx.x * 32u;
    const unsigned kbeg = blockIdx.z * k_per_split, kend = min(K, kbeg + k_per_split);
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    for (unsigned k0 = kbeg; k0 < kend; k0 += 32u) {
        for (unsigned i = threadIdx.x; i < 32u * 32u; i += blockDim.x) {
            const unsigned r = i >> 5, kk = i & 31u, k = k0 + kk;
            float av = 0.0f, wv = 0.0f;
            const unsigned m = m0 + r, n = n0 + r;
            if (m < M && k < kend) {
                if (k < k_split) av = a0[(u64)m * lda + k] * (row_scale ? row_scale[2 * m] : 1.0f);
                else av = a1[(u64)m * lda1 + (k - k_split)] * (row_scale ? row_scale[2 * m + 1] : 1.0f);
            }
            if (n < N && k < kend) wv = w[(u64)n * ldw + k];
            As[kk][r] = av;
            Ws[kk][r] = wv;
        }
        __syncthreads();
#pragma unroll 8
        for (unsigned kk = 0; kk < 32u; ++kk) {
            const float x0 = As[kk][ty * 2u], x1 = As[kk][ty * 2u + 1u];
            const float y0 = Ws[kk][tx * 2u], y1 = Ws[kk][tx * 2u + 1u];
            acc[0][0] = fmaf(x0, y0, acc[0][0]);
            acc[0][1] = fmaf(x0, y1, acc[0][1]);
            acc[1][0] = fmaf(x1, y0, acc[1][0]);
            acc[1][1] = fmaf(x1, y1, acc[1][1]);
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < 2; ++i) {
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const unsigned m = m0 + ty * 2u + i, n = n0 + tx * 2u + j;
            if (m < M && n < N) {
                if (atomic == 1u) atomicAdd(&c[(u64)m * ldc + n], acc[i][j]);
                else if (atomic == 2u) c[((u64)blockIdx.z * M + m) * ldc + n] = acc[i][j];   // per-split partials
                else c[(u64)m * ldc + n] = acc[i][j];
            }
        }
    }
}

// out[i] = ((p[0][i] + p[1][i]) + p[2][i]) + ... over `splits` partials of n values each.
extern "C" __global__ void astral_ink_sum_splits(const float* __restrict__ p, float* __restrict__ out, unsigned n, unsigned splits) { PDL_WAIT();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = p[i];
        for (unsigned z = 1; z < splits; ++z) v += p[(u64)z * n + i];
        out[i] = v;
    }
}

// out[t][n] = x - tanh(f) x + tanh(g) msg, with gf[t] = [gain (dim), forget (dim)].
extern "C" __global__ void astral_ink_ctrl_apply(const float* __restrict__ gf, const float* __restrict__ x, unsigned x_stride,
                                                 const float* __restrict__ msg, unsigned msg_stride, float* __restrict__ out,
                                                 unsigned dim, unsigned ntok) { PDL_WAIT();
    const unsigned n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= dim || t >= ntok) return;
    const float g = gf[(u64)t * 2u * dim + n], f = gf[(u64)t * 2u * dim + dim + n];
    const float c = x[(u64)t * x_stride + n];
    out[(u64)t * dim + n] = (c - tanhf(f) * c) + tanhf(g) * msg[(u64)t * msg_stride + n];
}

// =============================================================================================
// Flash-attention prefill (tensor cores, sm_80+). Same maths as astral_ink_attn_decode for
// every query token t (position pos[0] + t), causal over the cache:
//   logit = scale * q.k + bias(h, dist) - sink_{t,h},  softmax over keys plus a zero sink logit.
// Block = 4 warps = 64 queries of one head; keys in tiles of 64 staged in shared memory.
// QK^T and PV with mma m16n8k16 (f16 in, f32 accumulate; Q rounded to f16 as llama.cpp's flash
// attention does); the relative bias (n_rel <= 16) is a 16-term dot per (query, key) in f32.
// head_dim must be 128. grid = (ceil(ntok / 64), heads), 128 threads.
__device__ __forceinline__ void cp_async16(void* smem, const void* g) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sp), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

#define FA_Q 64
#define FA_K 64
#define FA_ROW 136   // halves per smem row: 128 + 8 pad (conflict-free fragment / ldmatrix reads)

__device__ __forceinline__ void mma_f16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

extern "C" __global__ void __launch_bounds__(128) astral_ink_attn_prefill(
    const float* __restrict__ q, const float* __restrict__ r, const float* __restrict__ sink,
    const __half* __restrict__ kc, const __half* __restrict__ vc, const float* __restrict__ rel_bias,
    float* __restrict__ out, const int* __restrict__ pos_p, unsigned ntok, unsigned heads, unsigned kv_heads,
    unsigned n_rel, unsigned extent, unsigned window, float scale, unsigned capacity) { PDL_WAIT();
    constexpr unsigned HD = 128;
    __shared__ __align__(16) __half Ks[FA_K * FA_ROW];
    __shared__ __align__(16) __half Vs[FA_K * FA_ROW];
    const unsigned h = blockIdx.y, qt = blockIdx.x;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned kvh = h / (heads / kv_heads), width = kv_heads * HD;
    const unsigned cap = min(capacity, (unsigned)MAX_CTX);
    const unsigned pos0 = (unsigned)pos_p[0];
    // This lane's two query rows (token indices), clamped for loads; `ok` marks real rows.
    const unsigned ta = qt * FA_Q + warp * 16u + g, tb = ta + 8u;
    const bool oka = ta < ntok, okb = tb < ntok;
    const unsigned la = min(ta, ntok - 1u), lb = min(tb, ntok - 1u);
    const unsigned pa = min(pos0 + la, cap - 1u), pb = min(pos0 + lb, cap - 1u);
    // Q fragments (A operand) for the 8 k-steps of 16 dims.
    unsigned qa[8][4];
    {
        const float* qra = q + ((u64)la * heads + h) * HD;
        const float* qrb = q + ((u64)lb * heads + h) * HD;
#pragma unroll
        for (unsigned ks = 0; ks < 8; ++ks) {
            const unsigned c = ks * 16u + 2u * t;
            qa[ks][0] = pack_h2(qra[c], qra[c + 1]);
            qa[ks][1] = pack_h2(qrb[c], qrb[c + 1]);
            qa[ks][2] = pack_h2(qra[c + 8], qra[c + 9]);
            qa[ks][3] = pack_h2(qrb[c + 8], qrb[c + 9]);
        }
    }
    float ra[16], rb[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) {
        ra[i] = i < n_rel ? r[((u64)la * heads + h) * n_rel + i] : 0.0f;
        rb[i] = i < n_rel ? r[((u64)lb * heads + h) * n_rel + i] : 0.0f;
    }
    const float ska = sink[(u64)la * heads + h], skb = sink[(u64)lb * heads + h];
    // Key range of the whole block.
    const unsigned first = min(pos0 + qt * FA_Q, cap - 1u);
    const unsigned last = min(pos0 + min(qt * FA_Q + FA_Q - 1u, ntok - 1u), cap - 1u);
    const unsigned kbeg = (window && first + 1u > window) ? first + 1u - window : 0u;
    float ma = -1e30f, mb = -1e30f, sa = 0.0f, sb = 0.0f;
    float o[16][4];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

    for (unsigned kb = kbeg & ~(FA_K - 1u); kb <= last; kb += FA_K) {
        __syncthreads();
        for (unsigned c = tid; c < FA_K * 16u; c += blockDim.x) {
            const unsigned row = c >> 4, part = c & 15u;
            const unsigned kp = min(kb + row, last);
            cp_async16(Ks + row * FA_ROW + part * 8u, kc + (u64)kp * width + (u64)kvh * HD + part * 8u);
            cp_async16(Vs + row * FA_ROW + part * 8u, vc + (u64)kp * width + (u64)kvh * HD + part * 8u);
        }
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
        // S = Q K^T for this warp's 16 queries x 64 keys.
        float s[8][4];
#pragma unroll
        for (unsigned nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            const __half* kr = Ks + (nt * 8u + g) * FA_ROW + 2u * t;
#pragma unroll
            for (unsigned ks = 0; ks < 8; ++ks) {
                const unsigned b0 = *reinterpret_cast<const unsigned*>(kr + ks * 16u);
                const unsigned b1 = *reinterpret_cast<const unsigned*>(kr + ks * 16u + 8u);
                mma_f16(s[nt], qa[ks], b0, b1);
            }
        }
        // Logits with bias, sink and masks; tile row maxima.
        float tma = -INFINITY, tmb = -INFINITY;
#pragma unroll
        for (unsigned nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const bool rowb = i >= 2;
                const unsigned key = kb + nt * 8u + 2u * t + (i & 1u);
                const unsigned qp = rowb ? pb : pa;
                const bool valid = (rowb ? okb : oka) && key <= qp && (!window || qp - key < window);
                float l = -INFINITY;
                if (valid) {
                    const unsigned dist = qp - key;
                    float bias = 0.0f;
                    if (dist < extent) {
                        const float4* br = reinterpret_cast<const float4*>(rel_bias + (u64)dist * n_rel);
                        const float* rr = rowb ? rb : ra;
#pragma unroll
                        for (unsigned v = 0; v < 4; ++v) {
                            if (v * 4u < n_rel) {
                                const float4 b4 = __ldg(br + v);
                                bias = fmaf(b4.x, rr[4 * v], bias);
                                bias = fmaf(b4.y, rr[4 * v + 1], bias);
                                bias = fmaf(b4.z, rr[4 * v + 2], bias);
                                bias = fmaf(b4.w, rr[4 * v + 3], bias);
                            }
                        }
                    }
                    l = fmaf(scale, s[nt][i], bias) - (rowb ? skb : ska);
                }
                s[nt][i] = l;
                if (rowb) tmb = fmaxf(tmb, l); else tma = fmaxf(tma, l);
            }
        }
        tma = fmaxf(tma, __shfl_xor_sync(0xffffffffu, tma, 1));
        tma = fmaxf(tma, __shfl_xor_sync(0xffffffffu, tma, 2));
        tmb = fmaxf(tmb, __shfl_xor_sync(0xffffffffu, tmb, 1));
        tmb = fmaxf(tmb, __shfl_xor_sync(0xffffffffu, tmb, 2));
        const float na = fmaxf(ma, tma), nb = fmaxf(mb, tmb);
        const float ca = expf(ma - na), cb = expf(mb - nb);
        ma = na;
        mb = nb;
        float psa = 0.0f, psb = 0.0f;
#pragma unroll
        for (unsigned nt = 0; nt < 8; ++nt) {
            s[nt][0] = expf(s[nt][0] - na); s[nt][1] = expf(s[nt][1] - na);
            s[nt][2] = expf(s[nt][2] - nb); s[nt][3] = expf(s[nt][3] - nb);
            psa += s[nt][0] + s[nt][1];
            psb += s[nt][2] + s[nt][3];
        }
        psa += __shfl_xor_sync(0xffffffffu, psa, 1);
        psa += __shfl_xor_sync(0xffffffffu, psa, 2);
        psb += __shfl_xor_sync(0xffffffffu, psb, 1);
        psb += __shfl_xor_sync(0xffffffffu, psb, 2);
        sa = sa * ca + psa;
        sb = sb * cb + psb;
#pragma unroll
        for (unsigned dn = 0; dn < 16; ++dn) { o[dn][0] *= ca; o[dn][1] *= ca; o[dn][2] *= cb; o[dn][3] *= cb; }
        // O += P V: P (16 x 64) as A fragments straight from the S accumulators.
#pragma unroll
        for (unsigned kk = 0; kk < 4; ++kk) {
            const unsigned pa_[4] = {pack_h2(s[2 * kk][0], s[2 * kk][1]), pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                     pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (unsigned dn = 0; dn < 16; ++dn) {
                unsigned b0, b1;
                const __half* vp = Vs + (kk * 16u + (lane & 15u)) * FA_ROW + dn * 8u;
                const unsigned sp = (unsigned)__cvta_generic_to_shared(vp);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(sp));
                mma_f16(o[dn], pa_, b0, b1);
            }
        }
    }
    // Normalise with the zero sink logit: out = o / (sum + exp(0 - m)).
    const float ia = 1.0f / (sa + expf(-ma)), ib = 1.0f / (sb + expf(-mb));
#pragma unroll
    for (unsigned dn = 0; dn < 16; ++dn) {
        const unsigned c = dn * 8u + 2u * t;
        if (oka) {
            float* orow = out + ((u64)ta * heads + h) * HD;
            orow[c] = o[dn][0] * ia;
            orow[c + 1] = o[dn][1] * ia;
        }
        if (okb) {
            float* orow = out + ((u64)tb * heads + h) * HD;
            orow[c] = o[dn][2] * ib;
            orow[c + 1] = o[dn][3] * ib;
        }
    }
}

// =============================================================================================
// Tokens per GEMM work item: `chunk` (64 = MMA_TOK of the mma.sync kernels, 128 for wgmma).
// MoE grouping on the GPU (prefill): sorts the P = ntok * k (token, expert) picks by expert and
// writes the tensor-core GEMM work items for gate_up (activation row = token) and down
// (activation row = pair), so no host sync is needed. Items are [e, start, count <= 64];
// entries past the last real item get count 0 (the GEMM skips them), so the GEMM grid can
// be the fixed maximum `max_items`. Order within an expert follows atomics (each output row
// is computed independently, so results do not depend on it). One block of 1024 threads;
// n_expert <= 256.
extern "C" __global__ void __launch_bounds__(1024) astral_ink_moe_group(const int* __restrict__ ids, unsigned P, unsigned k,
                                                                        unsigned n_expert, unsigned* __restrict__ items_gu,
                                                                        unsigned* __restrict__ xidx_gu, unsigned* __restrict__ yidx_gu,
                                                                        unsigned* __restrict__ items_dn, unsigned* __restrict__ xidx_dn,
                                                                        unsigned* __restrict__ yidx_dn, unsigned max_items, unsigned chunk) { PDL_WAIT();
    __shared__ unsigned cnt[256], off[256], ibase[256], cur[256];
    const unsigned tid = threadIdx.x;
    for (unsigned e = tid; e < 256u; e += blockDim.x) { cnt[e] = 0u; cur[e] = 0u; }
    __syncthreads();
    for (unsigned p = tid; p < P; p += blockDim.x) atomicAdd(&cnt[min((unsigned)max(ids[p], 0), n_expert - 1u)], 1u);
    __syncthreads();
    if (tid == 0) {
        unsigned o = 0, ib = 0;
        for (unsigned e = 0; e < n_expert; ++e) {
            off[e] = o;
            ibase[e] = ib;
            o += cnt[e];
            ib += (cnt[e] + chunk - 1u) / chunk;
        }
        cur[255] = ib;   // total items (stashed; cur[e] for e < n_expert <= 255 restarts at 0)
    }
    __syncthreads();
    const unsigned total_items = cur[255];
    __syncthreads();
    if (tid == 0) cur[255] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += blockDim.x) {
        const unsigned e = min((unsigned)max(ids[p], 0), n_expert - 1u);
        const unsigned slot = off[e] + atomicAdd(&cur[e], 1u);
        xidx_gu[slot] = p / k;
        yidx_gu[slot] = p;
        xidx_dn[slot] = p;
        yidx_dn[slot] = p;
    }
    for (unsigned e = tid; e < n_expert; e += blockDim.x) {
        const unsigned chunks = (cnt[e] + chunk - 1u) / chunk;
        for (unsigned c = 0; c < chunks; ++c) {
            const unsigned i = ibase[e] + c;
            if (i >= max_items) break;
            const unsigned st = off[e] + c * chunk, n = min(chunk, cnt[e] - c * chunk);
            items_gu[3 * i] = e; items_gu[3 * i + 1] = st; items_gu[3 * i + 2] = n;
            items_dn[3 * i] = e; items_dn[3 * i + 1] = st; items_dn[3 * i + 2] = n;
        }
    }
    for (unsigned i = total_items + tid; i < max_items; i += blockDim.x) {
        items_gu[3 * i] = 0u; items_gu[3 * i + 1] = 0u; items_gu[3 * i + 2] = 0u;
        items_dn[3 * i] = 0u; items_dn[3 * i + 1] = 0u; items_dn[3 * i + 2] = 0u;
    }
}

// Small-batch (verify) MoE grouping into the dp4a grouped-GEMV work items of gemv.cu
// (struct QItem: e, n, xi[8], yi[8]): the P = ntok * k picks in ids grouped by expert, up to
// 8 per item, so each used expert's weights are read once. gate_up items read the token's
// input (p / k) and write pair row p; down items read and write pair row p. Items from the
// last used one up to max_items get n = 0. One block of 1024 threads.
#define QITEM_NB 8u
#define QITEM_WORDS (2u + 2u * QITEM_NB)
extern "C" __global__ void __launch_bounds__(1024) astral_ink_moe_group_q(const int* __restrict__ ids, unsigned P, unsigned k,
                                                                          unsigned n_expert, unsigned* __restrict__ items_gu,
                                                                          unsigned* __restrict__ items_dn, unsigned max_items) { PDL_WAIT();
    __shared__ unsigned cnt[256], ibase[256], cur[256];
    __shared__ unsigned total;
    const unsigned tid = threadIdx.x;
    for (unsigned e = tid; e < 256u; e += blockDim.x) { cnt[e] = 0u; cur[e] = 0u; }
    __syncthreads();
    for (unsigned p = tid; p < P; p += blockDim.x) atomicAdd(&cnt[min((unsigned)max(ids[p], 0), n_expert - 1u)], 1u);
    __syncthreads();
    if (tid == 0) {
        unsigned ib = 0;
        for (unsigned e = 0; e < n_expert; ++e) {
            ibase[e] = ib;
            ib += (cnt[e] + QITEM_NB - 1u) / QITEM_NB;
        }
        total = ib;
    }
    __syncthreads();
    // Item headers (and empty items past the last one).
    for (unsigned e = tid; e < n_expert; e += blockDim.x) {
        const unsigned chunks = (cnt[e] + QITEM_NB - 1u) / QITEM_NB;
        for (unsigned c = 0; c < chunks; ++c) {
            const unsigned i = ibase[e] + c;
            if (i >= max_items) break;
            const unsigned n = min(QITEM_NB, cnt[e] - c * QITEM_NB);
            items_gu[QITEM_WORDS * i] = e; items_gu[QITEM_WORDS * i + 1] = n;
            items_dn[QITEM_WORDS * i] = e; items_dn[QITEM_WORDS * i + 1] = n;
        }
    }
    for (unsigned i = total + tid; i < max_items; i += blockDim.x) {
        items_gu[QITEM_WORDS * i] = 0u; items_gu[QITEM_WORDS * i + 1] = 0u;
        items_dn[QITEM_WORDS * i] = 0u; items_dn[QITEM_WORDS * i + 1] = 0u;
    }
    // Rows: pick p goes to slot cur[e]++ of its expert.
    for (unsigned p = tid; p < P; p += blockDim.x) {
        const unsigned e = min((unsigned)max(ids[p], 0), n_expert - 1u);
        const unsigned slot = atomicAdd(&cur[e], 1u);
        const unsigned i = ibase[e] + slot / QITEM_NB, b = slot % QITEM_NB;
        if (i >= max_items) continue;
        items_gu[QITEM_WORDS * i + 2 + b] = p / k; items_gu[QITEM_WORDS * i + 2 + QITEM_NB + b] = p;
        items_dn[QITEM_WORDS * i + 2 + b] = p;     items_dn[QITEM_WORDS * i + 2 + QITEM_NB + b] = p;
    }
}
