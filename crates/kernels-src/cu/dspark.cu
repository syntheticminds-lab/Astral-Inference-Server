// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//
// Inkling 2 DSpark speculative decoding: the checkpoint's draft model (`mtp.safetensors`) and
// the verify-side bookkeeping. The draft runs a block of `nblk` tokens (anchor + masks) through
// a small dense transformer that attends to K/V built from mixed target hidden states, then a
// recurrent Markov head picks the drafted tokens one by one. All activations are f32; the
// draft's KV cache is f32 `[capacity][kv_heads * 128]`.
// Parameter lists are mirrored by the launchers in crates/kernels/src/dspark.rs.

#include <cuda_fp16.h>

// Programmatic dependent launch (see inkling.cu): wait for the predecessor's results first.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if !defined(ASTRAL_PDL_LATE)
#define PDL_WAIT() asm volatile("griddepcontrol.wait;\n griddepcontrol.launch_dependents;" ::: "memory")
#else
#define PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#endif
#else
#define PDL_WAIT() do {} while (0)
#endif

typedef unsigned long long u64;

#define DSP_HD 128      // draft head dim
#define DSP_MAXG 4      // query heads per KV head handled by astral_dsp_attn

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
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

// int8 quantisation of one 32-value block held one value per lane (as inkling.cu's q8_store).
__device__ __forceinline__ void q8_store(float v, signed char* __restrict__ xq, float* __restrict__ xd, unsigned blk, unsigned lane,
                                         __half* __restrict__ xh) {
    if (xh) xh[blk * 32u + lane] = __float2half_rn(v);
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d = a * (1.0f / 127.0f);
    xq[blk * 32u + lane] = (signed char)(a == 0.0f ? 0 : __float2int_rn(v / d));
    if (lane == 0) xd[blk] = d;
}

// One block per row t: r = x + add (add optional), stored to res when given; y = rms_norm(r) * w
// (y optional) and quantised into (xq, xd, xh) when xq is given.
extern "C" __global__ void __launch_bounds__(1024) astral_dsp_add_rms_q8(
    const float* __restrict__ x, const float* __restrict__ add, float* res, const float* __restrict__ w, float* __restrict__ y,
    signed char* __restrict__ xq, float* __restrict__ xd, __half* __restrict__ xh, unsigned dim, float eps) { PDL_WAIT();
    const unsigned t = blockIdx.x;
    const u64 o = (u64)t * dim;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = x[o + i] + (add ? add[o + i] : 0.0f);
        if (res) res[o + i] = v;
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block_sum(ss) / (float)dim + eps);
    const unsigned lane = threadIdx.x & 31u, nw = blockDim.x >> 5;
    for (unsigned b = threadIdx.x >> 5; b < (dim >> 5); b += nw) {
        const unsigned i = b * 32u + lane;
        const float r = res ? res[o + i] : x[o + i] + (add ? add[o + i] : 0.0f);
        const float v = r * inv * w[i];
        if (y) y[o + i] = v;
        if (xq) q8_store(v, xq + o, xd + (o >> 5), b, lane, xh ? xh + o : nullptr);
    }
}

// One block per row: quantise row t of x (stride x_stride) into (xq, xd, xh), token-major.
extern "C" __global__ void __launch_bounds__(1024) astral_dsp_quant(const float* __restrict__ x, unsigned x_stride, signed char* __restrict__ xq,
                                                                    float* __restrict__ xd, __half* __restrict__ xh, unsigned dim) { PDL_WAIT();
    const unsigned t = blockIdx.x, lane = threadIdx.x & 31u, nw = blockDim.x >> 5;
    const u64 o = (u64)t * dim;
    for (unsigned b = threadIdx.x >> 5; b < (dim >> 5); b += nw) {
        const unsigned i = b * 32u + lane;
        q8_store(x[(u64)t * x_stride + i], xq + o, xd + (o >> 5), b, lane, xh ? xh + o : nullptr);
    }
}

// out[t][c] = sum_tap alpha[tap] * feat[tap * tap_stride + (row0 + t) * dim + c], with
// row0 = *row_p when given (else 0). Grid (ceil(dim / 256), ntok).
extern "C" __global__ void astral_dsp_mix4(const float* __restrict__ feat, u64 tap_stride, const int* __restrict__ row_p,
                                           float a0, float a1, float a2, float a3, float* __restrict__ out, unsigned dim) { PDL_WAIT();
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= dim) return;
    const u64 r = (u64)(t + (row_p ? (unsigned)max(row_p[0], 0) : 0u)) * dim + c;
    out[(u64)t * dim + c] = a0 * feat[r] + a1 * feat[tap_stride + r] + a2 * feat[2 * tap_stride + r] + a3 * feat[3 * tap_stride + r];
}

// Per (token t, head h) of 128 values at src[t * src_stride + h * 128]: optional RMSNorm with
// weight w, optional neox RoPE at position pos_p[pos_idx] + t (theta), written to dst row t
// (or row = position when to_cache) at column h * 128. Grid (ntok, heads), block 64.
extern "C" __global__ void __launch_bounds__(64) astral_dsp_head_prep(const float* src, unsigned src_stride, float* dst, unsigned dst_stride,
                                                                      const float* __restrict__ w, unsigned rope, const int* __restrict__ pos_p,
                                                                      unsigned pos_idx, unsigned to_cache, unsigned capacity, float theta, float eps) { PDL_WAIT();
    const unsigned t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const float* s = src + (u64)t * src_stride + h * DSP_HD;
    float x1 = s[i], x2 = s[i + 64];
    if (w) {
        const float inv = rsqrtf(block_sum(x1 * x1 + x2 * x2) / (float)DSP_HD + eps);
        x1 *= inv * w[i];
        x2 *= inv * w[i + 64];
    }
    const int pos = pos_p[pos_idx] + (int)t;
    if (rope) {
        const float inv_freq = 1.0f / powf(theta, (float)(2 * i) / (float)DSP_HD);
        float sn, cs;
        sincosf((float)pos * inv_freq, &sn, &cs);
        const float o1 = x1 * cs - x2 * sn, o2 = x2 * cs + x1 * sn;
        x1 = o1;
        x2 = o2;
    }
    const u64 row = to_cache ? (u64)min((unsigned)max(pos, 0), capacity - 1u) : (u64)t;
    float* d = dst + row * dst_stride + h * DSP_HD;
    d[i] = x1;
    d[i + 64] = x2;
}

// Block attention of the draft: query t (of nblk) of KV head kh, for its `g` query heads, over
// keys [lo, L) where L = pos_p[pos_idx] + nblk (the block's end) and lo = L - window (0 = all).
// No causal mask inside the block. Grid (nblk, kv_heads), block DSP_ATT_WARPS warps (split the keys).
#define DSP_ATT_WARPS 16
extern "C" __global__ void __launch_bounds__(32 * DSP_ATT_WARPS) astral_dsp_attn(const float* __restrict__ q, unsigned q_stride, const float* __restrict__ kc,
                                                                  const float* __restrict__ vc, float* __restrict__ out, const int* __restrict__ pos_p,
                                                                  unsigned pos_idx, unsigned nblk, unsigned heads, unsigned kv_heads,
                                                                  unsigned window, float scale, unsigned capacity) { PDL_WAIT();
    const unsigned t = blockIdx.x, kh = blockIdx.y, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned g = heads / kv_heads;
    const int L = min(pos_p[pos_idx] + (int)nblk, (int)capacity);
    const int lo = window ? max(0, L - (int)window) : 0;
    const unsigned kvw = kv_heads * DSP_HD;
    float qr[DSP_MAXG][4], m[DSP_MAXG], l[DSP_MAXG], acc[DSP_MAXG][4];
#pragma unroll
    for (unsigned j = 0; j < DSP_MAXG; j++) {
        m[j] = -INFINITY;
        l[j] = 0.0f;
        for (int e = 0; e < 4; e++) { acc[j][e] = 0.0f; qr[j][e] = 0.0f; }
        if (j < g) {
            const float4 v = *reinterpret_cast<const float4*>(q + (u64)t * q_stride + (kh * g + j) * DSP_HD + lane * 4);
            qr[j][0] = v.x * scale; qr[j][1] = v.y * scale; qr[j][2] = v.z * scale; qr[j][3] = v.w * scale;
        }
    }
    for (int key = lo + (int)warp; key < L; key += DSP_ATT_WARPS) {
        const u64 off = (u64)key * kvw + kh * DSP_HD + lane * 4;
        const float4 k = *reinterpret_cast<const float4*>(kc + off);
        const float4 v = *reinterpret_cast<const float4*>(vc + off);
#pragma unroll
        for (unsigned j = 0; j < DSP_MAXG; j++) {
            if (j >= g) break;
            const float s = warp_sum(qr[j][0] * k.x + qr[j][1] * k.y + qr[j][2] * k.z + qr[j][3] * k.w);
            const float mn = fmaxf(m[j], s);
            const float corr = __expf(m[j] - mn), p = __expf(s - mn);
            l[j] = l[j] * corr + p;
            acc[j][0] = acc[j][0] * corr + p * v.x;
            acc[j][1] = acc[j][1] * corr + p * v.y;
            acc[j][2] = acc[j][2] * corr + p * v.z;
            acc[j][3] = acc[j][3] * corr + p * v.w;
            m[j] = mn;
        }
    }
    __shared__ float sm[DSP_ATT_WARPS][DSP_MAXG], sl[DSP_ATT_WARPS][DSP_MAXG], sacc[DSP_ATT_WARPS][DSP_MAXG][DSP_HD];
#pragma unroll
    for (unsigned j = 0; j < DSP_MAXG; j++) {
        if (j >= g) break;
        if (lane == 0) { sm[warp][j] = m[j]; sl[warp][j] = l[j]; }
        for (int e = 0; e < 4; e++) sacc[warp][j][lane * 4 + e] = acc[j][e];
    }
    __syncthreads();
    const unsigned d = threadIdx.x;  // the first 128 threads = head dim
    if (d >= DSP_HD) return;
    for (unsigned j = 0; j < g; j++) {
        float mm = -INFINITY;
        for (int w = 0; w < DSP_ATT_WARPS; w++) mm = fmaxf(mm, sm[w][j]);
        float num = 0.0f, den = 0.0f;
        for (int w = 0; w < DSP_ATT_WARPS; w++) {
            if (sm[w][j] == -INFINITY) continue;
            const float c = __expf(sm[w][j] - mm);
            num += c * sacc[w][j][d];
            den += c * sl[w][j];
        }
        out[(u64)t * heads * DSP_HD + (kh * g + j) * DSP_HD + d] = den > 0.0f ? num / den : 0.0f;
    }
}

// astral_dsp_attn for B blocks at once (grid z = block), each over its own sequence's cache.
extern "C" __global__ void __launch_bounds__(32 * DSP_ATT_WARPS) astral_dsp_attn_b(const float* __restrict__ q, unsigned q_stride, const float* __restrict__ kc,
                                                                  const float* __restrict__ vc, float* __restrict__ out, const int* __restrict__ pos_p,
                                                                  unsigned pos_idx, unsigned nblk, unsigned heads, unsigned kv_heads,
                                                                  unsigned window, float scale, unsigned capacity, const int* __restrict__ seq_p) { PDL_WAIT();
    const unsigned t = blockIdx.x, kh = blockIdx.y, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    // Batched: blockIdx.z = block b of sequence seq_p[b], starting at pos_p[pos_idx + b].
    {
        const unsigned bz = blockIdx.z;
        q += (u64)bz * nblk * q_stride;
        const u64 so = (u64)max(seq_p[bz], 0) * capacity * kv_heads * DSP_HD;
        kc += so;
        vc += so;
        out += (u64)bz * nblk * heads * DSP_HD;
        pos_idx += bz;
    }
    const unsigned g = heads / kv_heads;
    const int L = min(pos_p[pos_idx] + (int)nblk, (int)capacity);
    const int lo = window ? max(0, L - (int)window) : 0;
    const unsigned kvw = kv_heads * DSP_HD;
    float qr[DSP_MAXG][4], m[DSP_MAXG], l[DSP_MAXG], acc[DSP_MAXG][4];
#pragma unroll
    for (unsigned j = 0; j < DSP_MAXG; j++) {
        m[j] = -INFINITY;
        l[j] = 0.0f;
        for (int e = 0; e < 4; e++) { acc[j][e] = 0.0f; qr[j][e] = 0.0f; }
        if (j < g) {
            const float4 v = *reinterpret_cast<const float4*>(q + (u64)t * q_stride + (kh * g + j) * DSP_HD + lane * 4);
            qr[j][0] = v.x * scale; qr[j][1] = v.y * scale; qr[j][2] = v.z * scale; qr[j][3] = v.w * scale;
        }
    }
    for (int key = lo + (int)warp; key < L; key += DSP_ATT_WARPS) {
        const u64 off = (u64)key * kvw + kh * DSP_HD + lane * 4;
        const float4 k = *reinterpret_cast<const float4*>(kc + off);
        const float4 v = *reinterpret_cast<const float4*>(vc + off);
#pragma unroll
        for (unsigned j = 0; j < DSP_MAXG; j++) {
            if (j >= g) break;
            const float s = warp_sum(qr[j][0] * k.x + qr[j][1] * k.y + qr[j][2] * k.z + qr[j][3] * k.w);
            const float mn = fmaxf(m[j], s);
            const float corr = __expf(m[j] - mn), p = __expf(s - mn);
            l[j] = l[j] * corr + p;
            acc[j][0] = acc[j][0] * corr + p * v.x;
            acc[j][1] = acc[j][1] * corr + p * v.y;
            acc[j][2] = acc[j][2] * corr + p * v.z;
            acc[j][3] = acc[j][3] * corr + p * v.w;
            m[j] = mn;
        }
    }
    __shared__ float sm[DSP_ATT_WARPS][DSP_MAXG], sl[DSP_ATT_WARPS][DSP_MAXG], sacc[DSP_ATT_WARPS][DSP_MAXG][DSP_HD];
#pragma unroll
    for (unsigned j = 0; j < DSP_MAXG; j++) {
        if (j >= g) break;
        if (lane == 0) { sm[warp][j] = m[j]; sl[warp][j] = l[j]; }
        for (int e = 0; e < 4; e++) sacc[warp][j][lane * 4 + e] = acc[j][e];
    }
    __syncthreads();
    const unsigned d = threadIdx.x;  // the first 128 threads = head dim
    if (d >= DSP_HD) return;
    for (unsigned j = 0; j < g; j++) {
        float mm = -INFINITY;
        for (int w = 0; w < DSP_ATT_WARPS; w++) mm = fmaxf(mm, sm[w][j]);
        float num = 0.0f, den = 0.0f;
        for (int w = 0; w < DSP_ATT_WARPS; w++) {
            if (sm[w][j] == -INFINITY) continue;
            const float c = __expf(sm[w][j] - mm);
            num += c * sacc[w][j][d];
            den += c * sl[w][j];
        }
        out[(u64)t * heads * DSP_HD + (kh * g + j) * DSP_HD + d] = den > 0.0f ? num / den : 0.0f;
    }
}

// Batched draft helpers (row t of B blocks of rps rows: block b = t / rps, row i = t % rps).
// head_prep_b: as astral_dsp_head_prep with position pos_p[b] + i and, to_cache, row
// seq_p[b] * capacity + position of dst. Grid (rows, heads), block 64.
extern "C" __global__ void __launch_bounds__(64) astral_dsp_head_prep_b(const float* src, unsigned src_stride, float* dst, unsigned dst_stride,
                                                                        const float* __restrict__ w, unsigned rope, const int* __restrict__ pos_p,
                                                                        const int* __restrict__ seq_p, unsigned rps, unsigned to_cache, unsigned capacity,
                                                                        float theta, float eps) { PDL_WAIT();
    const unsigned t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const unsigned b = t / rps, ri = t - b * rps;
    const float* s = src + (u64)t * src_stride + h * DSP_HD;
    float x1 = s[i], x2 = s[i + 64];
    if (w) {
        const float inv = rsqrtf(block_sum(x1 * x1 + x2 * x2) / (float)DSP_HD + eps);
        x1 *= inv * w[i];
        x2 *= inv * w[i + 64];
    }
    const int pos = pos_p[b] + (int)ri;
    if (rope) {
        const float inv_freq = 1.0f / powf(theta, (float)(2 * i) / (float)DSP_HD);
        float sn, cs;
        sincosf((float)pos * inv_freq, &sn, &cs);
        const float o1 = x1 * cs - x2 * sn, o2 = x2 * cs + x1 * sn;
        x1 = o1;
        x2 = o2;
    }
    const u64 row = to_cache ? (u64)max(seq_p[b], 0) * capacity + (u64)min((unsigned)max(pos, 0), capacity - 1u) : (u64)t;
    float* d = dst + row * dst_stride + h * DSP_HD;
    d[i] = x1;
    d[i + 64] = x2;
}

// out[t][c] = sum_tap alpha[tap] * feat[tap * tap_stride + rows_p[t] * dim + c]. Grid (ceil(dim / 256), rows).
extern "C" __global__ void astral_dsp_mix4_b(const float* __restrict__ feat, u64 tap_stride, const int* __restrict__ rows_p,
                                             float a0, float a1, float a2, float a3, float* __restrict__ out, unsigned dim) { PDL_WAIT();
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= dim) return;
    const u64 r = (u64)max(rows_p[t], 0) * dim + c;
    out[(u64)t * dim + c] = a0 * feat[r] + a1 * feat[tap_stride + r] + a2 * feat[2 * tap_stride + r] + a3 * feat[3 * tap_stride + r];
}

// astral_dsp_markov_in for B sequences: block b uses state row b, token prev[b], writes x row b.
extern "C" __global__ void astral_dsp_markov_in_b(const float* __restrict__ state, const float* __restrict__ w1, const int* __restrict__ prev,
                                                  unsigned vocab, unsigned first, float* __restrict__ x) { PDL_WAIT();
    const unsigned i = threadIdx.x, r = blockDim.x, b = blockIdx.x;
    const unsigned p = (unsigned)min(max(prev[b], 0), (int)vocab - 1);
    x[(u64)b * 2 * r + i] = first ? 0.0f : state[(u64)b * r + i];
    x[(u64)b * 2 * r + r + i] = w1[(u64)p * r + i];
}

// astral_dsp_markov_gate for B sequences: raw / cf rows b (3 rank), ch row b at ch + b * ch_stride,
// state row b; the latent is quantised into row b of xq / xd.
extern "C" __global__ void astral_dsp_markov_gate_b(const float* __restrict__ raw, const float* __restrict__ ch, unsigned ch_stride,
                                                    const float* __restrict__ cf, const float* __restrict__ bias, float* __restrict__ state,
                                                    unsigned first, signed char* __restrict__ xq, float* __restrict__ xd) { PDL_WAIT();
    const unsigned i = threadIdx.x, r = blockDim.x, b = blockIdx.x;
    raw += (u64)b * 3 * r;
    ch += (u64)b * ch_stride;
    cf += (u64)b * 3 * r;
    const float g = raw[i] + ch[i] + cf[i] + bias[i];
    const float c = raw[r + i] + ch[r + i] + cf[r + i] + bias[r + i];
    const float o = raw[2 * r + i] + ch[2 * r + i] + cf[2 * r + i] + bias[2 * r + i];
    const float gate = 1.0f / (1.0f + expf(-g));
    const float s0 = first ? 0.0f : state[(u64)b * r + i];
    state[(u64)b * r + i] = gate * s0 + (1.0f - gate) * tanhf(c);
    q8_store(tanhf(o), xq + (u64)b * r, xd + (u64)b * (r >> 5), i >> 5, i & 31u, nullptr);
}

// y[b][j] += base[b * base_stride + j] + fb[j] for j < n, B rows of y_stride. Grid (ceil(n / 256), B).
extern "C" __global__ void astral_dsp_add_rows(float* __restrict__ y, unsigned y_stride, const float* __restrict__ base, unsigned base_stride,
                                               const float* __restrict__ fb, unsigned n) { PDL_WAIT();
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    if (j >= n) return;
    y[(u64)b * y_stride + j] += base[(u64)b * base_stride + j] + fb[j];
}

// One block per row: act = silu(gate) * up over gu = [gate (n_ff) | up (n_ff)], quantised.
extern "C" __global__ void __launch_bounds__(1024) astral_dsp_swiglu_q8(const float* __restrict__ gu, unsigned n_ff, signed char* __restrict__ xq,
                                                                        float* __restrict__ xd, __half* __restrict__ xh) { PDL_WAIT();
    const unsigned t = blockIdx.x, lane = threadIdx.x & 31u, nw = blockDim.x >> 5;
    const float* g = gu + (u64)t * 2 * n_ff;
    const u64 o = (u64)t * n_ff;
    for (unsigned b = threadIdx.x >> 5; b < (n_ff >> 5); b += nw) {
        const unsigned i = b * 32u + lane;
        const float a = g[i], u = g[n_ff + i];
        q8_store(a / (1.0f + expf(-a)) * u, xq + o, xd + (o >> 5), b, lane, xh ? xh + o : nullptr);
    }
}

// Markov step input: x = [state (rank) | w1[prev] (rank)], state taken as 0 when first. Block = rank.
extern "C" __global__ void astral_dsp_markov_in(const float* __restrict__ state, const float* __restrict__ w1, const int* __restrict__ prev,
                                                unsigned vocab, unsigned first, float* __restrict__ x) { PDL_WAIT();
    const unsigned i = threadIdx.x, r = blockDim.x;
    const unsigned p = (unsigned)min(max(prev[0], 0), (int)vocab - 1);
    x[i] = first ? 0.0f : state[i];
    x[r + i] = w1[(u64)p * r + i];
}

// Markov gate: raw (3 * rank) += ch + cf + bias; g, c, o = split; state = sig(g) * state +
// (1 - sig(g)) * tanh(c) (state 0 when first); latent = tanh(o), quantised. Block = rank.
extern "C" __global__ void astral_dsp_markov_gate(const float* __restrict__ raw, const float* __restrict__ ch, const float* __restrict__ cf,
                                                  const float* __restrict__ bias, float* __restrict__ state, unsigned first,
                                                  signed char* __restrict__ xq, float* __restrict__ xd) { PDL_WAIT();
    const unsigned i = threadIdx.x, r = blockDim.x;
    const float g = raw[i] + ch[i] + cf[i] + bias[i];
    const float c = raw[r + i] + ch[r + i] + cf[r + i] + bias[r + i];
    const float o = raw[2 * r + i] + ch[2 * r + i] + cf[2 * r + i] + bias[2 * r + i];
    const float gate = 1.0f / (1.0f + expf(-g));
    const float s = first ? 0.0f : state[i];
    state[i] = gate * s + (1.0f - gate) * tanhf(c);
    q8_store(tanhf(o), xq, xd, i >> 5, i & 31u, nullptr);
}

// out[0] = argmax_i (y[i] + base[i] + fb[i]) over i < n. One block of 1024.
extern "C" __global__ void __launch_bounds__(1024) astral_dsp_add_argmax(const float* __restrict__ y, const float* __restrict__ base,
                                                                         const float* __restrict__ fb, unsigned n, int* __restrict__ out) { PDL_WAIT();
    __shared__ float bv[32];
    __shared__ unsigned bi[32];
    float best = -INFINITY;
    unsigned idx = 0xffffffffu;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = y[i] + base[i] + fb[i];
        if (v > best) { best = v; idx = i; }
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

// Greedy acceptance after verifying token[0..=k] (anchor + k drafts) at positions pos[0]..:
// j = leading drafts equal to the target's picks; emits out = [j + 1, d_1..d_j, g_j];
// token[0] = g_j (the next anchor); info = [n, j, n + j + 1]; pos[0] = n + j + 1. One thread.
// With `forced` (diagnostics), the reference is a fixed sequence by position instead:
// forced[n + i] is the token that must follow, and its next token is emitted.
extern "C" __global__ void astral_dsp_accept(const int* __restrict__ next_all, int* __restrict__ token, unsigned k, int* __restrict__ out,
                                             int* __restrict__ info, int* __restrict__ pos, const int* __restrict__ forced) { PDL_WAIT();
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int n = pos[0];
    unsigned j = 0;
    if (forced) {
        while (j < k && token[j + 1] == forced[n + 1 + j]) j++;
    } else {
        while (j < k && token[j + 1] == next_all[j]) j++;
    }
    const int next = forced ? forced[n + 1 + j] : next_all[j];
    out[0] = (int)j + 1;
    for (unsigned i = 0; i < j; i++) out[1 + i] = token[1 + i];
    out[1 + j] = next;
    token[0] = next;
    info[0] = n;
    info[1] = (int)j;
    info[2] = n + (int)j + 1;
    pos[0] = n + (int)j + 1;
}

// Short-conv state rollback after verification: for each entry (state, hist, channels) of the
// table, the new state holds the last three inputs of [old state; hist rows 0..info[1]] (the
// accepted tokens, info[1] + 1 of them). Grid (ceil(max_channels / 256), entries).
extern "C" __global__ void astral_dsp_conv_commit(const u64* __restrict__ table, const int* __restrict__ info) { PDL_WAIT();
    const u64* e = table + 3 * blockIdx.y;
    float* state = reinterpret_cast<float*>(e[0]);
    const float* hist = reinterpret_cast<const float*>(e[1]);
    const unsigned ch = (unsigned)e[2];
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const int n = info[1] + 1;
    float v[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
        const int t = n - 3 + i;
        v[i] = t >= 0 ? hist[(u64)t * ch + c] : state[c * 3 + (unsigned)(3 + t)];
    }
    state[c * 3 + 0] = v[0];
    state[c * 3 + 1] = v[1];
    state[c * 3 + 2] = v[2];
}

// ---------------------------------------------------------------------------------------------
// Multi-block argmax. Partial kernels leave each block's best (value, index) in part_v / part_i
// (row r at r * nparts); astral_dsp_argmax_fin reduces them to out[r]. Ties keep the lowest index.

__device__ __forceinline__ void best_merge(float& bv, unsigned& bi, float ov, unsigned oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// Block-wide (value, index) reduction of 256 threads; thread 0 returns the result.
__device__ __forceinline__ void block_best(float& best, unsigned& idx) {
    __shared__ float sv[8];
    __shared__ unsigned si[8];
    for (int o = 16; o > 0; o >>= 1) best_merge(best, idx, __shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, idx, o));
    if ((threadIdx.x & 31u) == 0) { sv[threadIdx.x >> 5] = best; si[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if (threadIdx.x < 32) {
        best = threadIdx.x < (blockDim.x >> 5) ? sv[threadIdx.x] : -INFINITY;
        idx = threadIdx.x < (blockDim.x >> 5) ? si[threadIdx.x] : 0xffffffffu;
        for (int o = 16; o > 0; o >>= 1) best_merge(best, idx, __shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, idx, o));
    }
}

// Rows of x (stride x_stride), first n values each. Grid (nparts, rows), 256 threads.
extern "C" __global__ void __launch_bounds__(256) astral_dsp_rowmax_part(const float* __restrict__ x, unsigned x_stride, unsigned n,
                                                                         float* __restrict__ part_v, unsigned* __restrict__ part_i) { PDL_WAIT();
    const unsigned r = blockIdx.y, np = gridDim.x;
    const unsigned chunk = (n + np - 1) / np, lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
    const float* xr = x + (u64)r * x_stride;
    float best = -INFINITY;
    unsigned idx = 0xffffffffu;
    for (unsigned i = lo + threadIdx.x; i < hi; i += blockDim.x) best_merge(best, idx, xr[i], i);
    block_best(best, idx);
    if (threadIdx.x == 0) { part_v[r * np + blockIdx.x] = best; part_i[r * np + blockIdx.x] = idx; }
}

// Markov head logits + partial argmax in one pass: v = fold[i] . latent + base[i] + fb[i] for
// i < n, fold Q8_0 rows of 256 (repacked: int8 [rows][256], f16 scales [rows][8]), latent int8
// (256) with 8 scales. 4 lanes per row. Grid (nparts), 256 threads.
extern "C" __global__ void __launch_bounds__(256) astral_dsp_fold_part(const unsigned char* __restrict__ fq, const unsigned short* __restrict__ fd,
                                                                       const signed char* __restrict__ lq, const float* __restrict__ ld,
                                                                       const float* __restrict__ base, const float* __restrict__ fb, unsigned n,
                                                                       float* __restrict__ part_v, unsigned* __restrict__ part_i) { PDL_WAIT();
    const unsigned np = gridDim.x, g = threadIdx.x & 3u;  // lane in the row's group: units 4g..4g+3
    const unsigned chunk = (n + np - 1) / np, lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
    uint4 la[4];
    float ls[2];
#pragma unroll
    for (int j = 0; j < 4; j++) la[j] = reinterpret_cast<const uint4*>(lq)[4 * g + j];
    ls[0] = ld[2 * g];
    ls[1] = ld[2 * g + 1];
    float best = -INFINITY;
    unsigned idx = 0xffffffffu;
    for (unsigned row0 = lo; row0 < hi; row0 += blockDim.x / 4) {
        const unsigned row = row0 + (threadIdx.x >> 2);
        const bool ok = row < hi;
        float acc = 0.0f;
        if (ok) {
            const uint4* w = reinterpret_cast<const uint4*>(fq + (u64)row * 256u) + 4 * g;
            const unsigned short* d = fd + (u64)row * 8u + 2 * g;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint4 wv = __ldg(w + j);
                int s = __dp4a((int)wv.x, (int)la[j].x, 0);
                s = __dp4a((int)wv.y, (int)la[j].y, s);
                s = __dp4a((int)wv.z, (int)la[j].z, s);
                s = __dp4a((int)wv.w, (int)la[j].w, s);
                acc = fmaf((float)s, __half2float(__ushort_as_half(__ldg(d + (j >> 1)))) * ls[j >> 1], acc);
            }
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        acc += __shfl_xor_sync(0xffffffffu, acc, 2);
        if (ok && g == 0) best_merge(best, idx, acc + base[row] + fb[row], row);
    }
    block_best(best, idx);
    if (threadIdx.x == 0) { part_v[blockIdx.x] = best; part_i[blockIdx.x] = idx; }
}

// out[r * out_stride] = best of row r's nparts partials. Grid (rows), 256 threads.
extern "C" __global__ void __launch_bounds__(256) astral_dsp_argmax_fin(const float* __restrict__ part_v, const unsigned* __restrict__ part_i,
                                                                        unsigned nparts, int* __restrict__ out, unsigned out_stride) { PDL_WAIT();
    const unsigned r = blockIdx.x;
    float best = -INFINITY;
    unsigned idx = 0xffffffffu;
    for (unsigned i = threadIdx.x; i < nparts; i += blockDim.x) best_merge(best, idx, part_v[r * nparts + i], part_i[r * nparts + i]);
    block_best(best, idx);
    if (threadIdx.x == 0) out[r * out_stride] = (int)idx;
}
