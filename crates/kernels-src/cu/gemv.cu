// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//
// General decode GEMV over the weight types of the Inkling 2 Mini GGUF (P1, correctness first):
//
//   y[s, n] = mat_scale[e] * sum_k W_e[n, k] * x[s, k],    e = ids ? min(ids[s], n_mats-1) : s
//
// One warp per output row; blockIdx.y selects the matrix "slot" s (an expert for MoE, or 0).
// Weight types (Astral repacked layouts, see astral_quant):
//   0 F32    row = K floats                       (router, loop controllers)
//   1 BF16   row = K bf16                         (sink and conv-tap projections)
//   2 Q8_0   qs: K bytes/row, aux: K/32 f16/row   (attention, embedding, lm head)
//   3 NVFP4  qs: K/2 bytes/row, aux: K/16 UE4M3   (experts; mat_scale = per-expert f32 scale)
// Decoding matches ggml bit for bit (dequantize_row_q8_0 / _nvfp4); astral_quant is the oracle.
// Parameter list mirrored by Gemv in crates/kernels/src/ops.rs.

#include <cuda_fp16.h>

// Programmatic dependent launch: on Hopper every kernel may be launched before its predecessor
// finishes (overlapping launch latency); griddepcontrol.wait blocks until the predecessor's
// results are visible, so it is the first thing every kernel does. A no-op otherwise.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if defined(ASTRAL_PDL_EARLY)  // multi-wave GEMV/GEMM: waiting successors would hold SM slots
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

__device__ __forceinline__ float bf16f(unsigned b) { return __uint_as_float(b << 16); }

__device__ __forceinline__ float e2m1_doubled(unsigned q) {
    const unsigned m = q & 7u;
    const float v = (m < 4u) ? (float)m : (m == 7u ? 12.0f : (float)(2u * m - 4u)); // 0 1 2 3 4 6 8 12
    return (q & 8u) ? -v : v;
}

// ggml_ue4m3_to_fp32: unsigned E4M3 (bias 7), 0x00 and 0x7F -> 0, result halved.
__device__ __forceinline__ float ue4m3_half(unsigned x) {
    if (x == 0u || x == 0x7fu) return 0.0f;
    const unsigned e = (x >> 3) & 0xfu, m = x & 7u;
    return e == 0u ? (float)m * 0.0009765625f                      // m * 2^-9 * 0.5
                   : __uint_as_float(((e + 119u) << 23) | (m << 20)); // (1 + m/8) * 2^(e-7) * 0.5
}

template <int WT>
__device__ __forceinline__ void gemv(const u8* __restrict__ qs, const u8* __restrict__ aux, const float* __restrict__ x,
                                     float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux, unsigned x_stride,
                                     const int* __restrict__ ids, const float* __restrict__ mat_scale, unsigned n_mats) {
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned n = blockIdx.x * (blockDim.x >> 5) + warp;
    if (n >= N) return;
    const unsigned s = blockIdx.y;
    // Device-side ids are clamped so a bad id can never read outside the weights.
    const unsigned e = ids ? min((unsigned)ids[s], n_mats - 1u) : s;
    const float* xs = x + (u64)s * x_stride;
    float acc = 0.0f;

    if constexpr (WT == 0) {  // F32, K % 4 == 0
        const float4* w = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(qs + e * mat_qs) + (u64)n * K);
        const float4* xv = reinterpret_cast<const float4*>(xs);
        for (unsigned i = lane; i < (K >> 2); i += 32u) {
            const float4 a = __ldg(w + i), b = __ldg(xv + i);
            acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc); acc = fmaf(a.z, b.z, acc); acc = fmaf(a.w, b.w, acc);
        }
    } else if constexpr (WT == 1) {  // BF16, K % 4 == 0
        const uint2* w = reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(qs + e * mat_qs) + (u64)n * K);
        const float4* xv = reinterpret_cast<const float4*>(xs);
        for (unsigned i = lane; i < (K >> 2); i += 32u) {
            const uint2 a = __ldg(w + i);
            const float4 b = __ldg(xv + i);
            acc = fmaf(bf16f(a.x & 0xffffu), b.x, acc); acc = fmaf(bf16f(a.x >> 16), b.y, acc);
            acc = fmaf(bf16f(a.y & 0xffffu), b.z, acc); acc = fmaf(bf16f(a.y >> 16), b.w, acc);
        }
    } else if constexpr (WT == 2) {  // Q8_0, K % 32 == 0: each lane takes 8 values (a quarter block)
        const u8* wq = qs + e * mat_qs + (u64)n * K;
        const u16* wd = reinterpret_cast<const u16*>(aux + e * mat_aux) + (u64)n * (K >> 5);
        for (unsigned i = lane; i < (K >> 3); i += 32u) {
            const uint2 q = __ldg(reinterpret_cast<const uint2*>(wq) + i);
            const float d = __half2float(__ushort_as_half(__ldg(wd + (i >> 2))));
            const float4 b0 = __ldg(reinterpret_cast<const float4*>(xs) + 2 * i);
            const float4 b1 = __ldg(reinterpret_cast<const float4*>(xs) + 2 * i + 1);
            float p = 0.0f;
            p = fmaf((float)(signed char)(q.x), b0.x, p);       p = fmaf((float)(signed char)(q.x >> 8), b0.y, p);
            p = fmaf((float)(signed char)(q.x >> 16), b0.z, p); p = fmaf((float)(signed char)(q.x >> 24), b0.w, p);
            p = fmaf((float)(signed char)(q.y), b1.x, p);       p = fmaf((float)(signed char)(q.y >> 8), b1.y, p);
            p = fmaf((float)(signed char)(q.y >> 16), b1.z, p); p = fmaf((float)(signed char)(q.y >> 24), b1.w, p);
            acc = fmaf(d, p, acc);
        }
    } else {  // NVFP4, K % 64 == 0: each lane takes one 16-element sub-block (8 bytes + 1 scale)
        const u8* wq = qs + e * mat_qs + (u64)n * (K >> 1);
        const u8* wd = aux + e * mat_aux + (u64)n * (K >> 4);
        for (unsigned sb = lane; sb < (K >> 4); sb += 32u) {
            const uint2 q = __ldg(reinterpret_cast<const uint2*>(wq) + sb);
            const float d = ue4m3_half(__ldg(wd + sb));
            const float* xb = xs + (u64)sb * 16u;
            const float4 lo0 = __ldg(reinterpret_cast<const float4*>(xb));
            const float4 lo1 = __ldg(reinterpret_cast<const float4*>(xb) + 1);
            const float4 hi0 = __ldg(reinterpret_cast<const float4*>(xb) + 2);
            const float4 hi1 = __ldg(reinterpret_cast<const float4*>(xb) + 3);
            const float lo[8] = {lo0.x, lo0.y, lo0.z, lo0.w, lo1.x, lo1.y, lo1.z, lo1.w};
            const float hi[8] = {hi0.x, hi0.y, hi0.z, hi0.w, hi1.x, hi1.y, hi1.z, hi1.w};
            const unsigned w2[2] = {q.x, q.y};
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned byte = (w2[j >> 2] >> ((j & 3) * 8)) & 0xffu;
                p = fmaf(e2m1_doubled(byte & 0xfu), lo[j], p);   // element j
                p = fmaf(e2m1_doubled(byte >> 4), hi[j], p);     // element j + 8
            }
            acc = fmaf(d, p, acc);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(u64)s * N + n] = mat_scale ? acc * mat_scale[e] : acc;
}

#define GEMV_ENTRY(NAME, WT)                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                    \
        const u8* __restrict__ qs, const u8* __restrict__ aux, const float* __restrict__ x, float* __restrict__ y, \
        unsigned N, unsigned K, u64 mat_qs, u64 mat_aux, unsigned x_stride, const int* __restrict__ ids,       \
        const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();                                                \
        gemv<WT>(qs, aux, x, y, N, K, mat_qs, mat_aux, x_stride, ids, mat_scale, n_mats);                      \
    }
GEMV_ENTRY(astral_gemv_f32, 0)
GEMV_ENTRY(astral_gemv_bf16, 1)
GEMV_ENTRY(astral_gemv_q8_0, 2)
GEMV_ENTRY(astral_gemv_nvfp4, 3)

// =============================================================================================
// int8-activation path (decode fast path). Activations are quantised per 32 values to int8
// with d = amax / 127 (llama.cpp's q8_1 scheme, as used by its MMVQ kernels), then Q8_0 and
// NVFP4 weights are multiplied with dp4a: 4 multiply-adds per instruction, 16-byte weight loads
// issued ahead of use. Mirrored by Gemv::{quantize, launch_q} in crates/kernels/src/ops.rs.

// One warp per 32-value block; total_blocks = slots * K / 32.
extern "C" __global__ void __launch_bounds__(256) astral_quantize_x_q8(const float* __restrict__ x, signed char* __restrict__ xq,
                                                                        float* __restrict__ xd, unsigned K, unsigned x_stride,
                                                                        unsigned total_blocks) { PDL_WAIT();
    const unsigned b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (b >= total_blocks) return;
    const unsigned per = K >> 5, s = b / per, kb = b - s * per;
    const float v = x[(u64)s * x_stride + kb * 32u + lane];
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d = a * (1.0f / 127.0f);
    const int q = a == 0.0f ? 0 : __float2int_rn(v / d);
    xq[(u64)s * K + kb * 32u + lane] = (signed char)q;
    if (lane == 0) xd[b] = d;
}

// 16-entry int8 table lookup of the 8 nibbles of q4: .x = low nibbles of bytes 0..3,
// .y = high nibbles (llama.cpp get_int_from_table_16, MIT).
__device__ __forceinline__ int2 table16(int q4, const unsigned* t) {
    unsigned tmp[2];
    const unsigned sel = 0x32103210u | (((unsigned)q4 & 0x88888888u) >> 1);
#pragma unroll
    for (unsigned i = 0; i < 2; ++i) {
        const unsigned sh = 16u * i;
        const unsigned lo = __byte_perm(t[0], t[1], (unsigned)q4 >> sh);
        const unsigned hi = __byte_perm(t[2], t[3], (unsigned)q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2((int)__byte_perm(tmp[0], tmp[1], 0x6420), (int)__byte_perm(tmp[0], tmp[1], 0x7531));
}

// E2M1 values doubled: 0 1 2 3 4 6 8 12, then negated (the NVFP4 scale carries the 0.5).
__constant__ unsigned k_e2m1x2[4] = {0x03020100u, 0x0c080604u, 0xfdfeff00u, 0xf4f8fafcu};

// Dot product of one weight row with int8 activations, computed by a group of G lanes (the
// group's lanes are `sub` = 0..G-1). Each lane keeps up to U 16-byte weight loads in flight,
// so a whole row is fetched in one memory round for the shapes of this model. The result is
// reduced over the group (`mask` = the group's lanes).
template <int WT, unsigned G, unsigned U>
__device__ __forceinline__ float q_dot(const u8* __restrict__ wq, const u8* __restrict__ wa, const uint4* __restrict__ xv,
                                       const float* __restrict__ xds, unsigned K, unsigned sub, unsigned mask) {
    float acc = 0.0f;
    const uint4* w = reinterpret_cast<const uint4*>(wq);
    const u16* wd = reinterpret_cast<const u16*>(wa);
    if constexpr (WT == 2) {    // Q8_0: unit = 16 weights (half a block)
        const unsigned units = K >> 4;
        for (unsigned base = 0; base < units; base += G * U) {
            uint4 wv[U];
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) wv[i] = __ldg(w + u);
            }
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) {
                    const uint4 a = __ldg(xv + u);
                    int sum = __dp4a((int)wv[i].x, (int)a.x, 0);
                    sum = __dp4a((int)wv[i].y, (int)a.y, sum);
                    sum = __dp4a((int)wv[i].z, (int)a.z, sum);
                    sum = __dp4a((int)wv[i].w, (int)a.w, sum);
                    acc = fmaf((float)sum, __half2float(__ushort_as_half(__ldg(wd + (u >> 1)))) * __ldg(xds + (u >> 1)), acc);
                }
            }
        }
    } else {                    // NVFP4: unit = 32 weights (two 16-value sub-blocks, one x block)
        const unsigned units = K >> 5;
        for (unsigned base = 0; base < units; base += G * U) {
            uint4 wv[U];
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) wv[i] = __ldg(w + u);
            }
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) {
                    const uint4 x0 = __ldg(xv + 2u * u), x1 = __ldg(xv + 2u * u + 1u);
                    const int2 a0 = table16((int)wv[i].x, k_e2m1x2), a1 = table16((int)wv[i].y, k_e2m1x2);
                    const int2 b0 = table16((int)wv[i].z, k_e2m1x2), b1 = table16((int)wv[i].w, k_e2m1x2);
                    int s0 = __dp4a(a0.x, (int)x0.x, 0);
                    s0 = __dp4a(a1.x, (int)x0.y, s0);
                    s0 = __dp4a(a0.y, (int)x0.z, s0);
                    s0 = __dp4a(a1.y, (int)x0.w, s0);
                    int s1 = __dp4a(b0.x, (int)x1.x, 0);
                    s1 = __dp4a(b1.x, (int)x1.y, s1);
                    s1 = __dp4a(b0.y, (int)x1.z, s1);
                    s1 = __dp4a(b1.y, (int)x1.w, s1);
                    // Two canonical UE4M3 scales as one half2: bits x << 7 give value * 2^-8, and
                    // ggml's extra 0.5 makes the factor 2^-9, applied once below.
                    const unsigned sc = __ldg(wd + u);
                    const float2 d2 = __half22float2(__halves2half2(__ushort_as_half((unsigned short)((sc & 0xffu) << 7)),
                                                                    __ushort_as_half((unsigned short)((sc >> 8) << 7))));
                    acc = fmaf(fmaf((float)s0, d2.x, (float)s1 * d2.y), __ldg(xds + u), acc);
                }
            }
        }
        acc *= 128.0f;   // 2^8 * 0.5
    }
#pragma unroll
    for (unsigned o = G / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(mask, acc, o);
    return acc;
}

// Rows per warp: Q8_0 1 (32 lanes per row), NVFP4 4 (8 lanes per row). Keep in sync with
// Gemv::q_rows_per_warp in ops.rs.
template <int WT> struct QShape {
    static constexpr unsigned R = WT == 2 ? 1u : 4u;
    static constexpr unsigned G = 32u / R;
    static constexpr unsigned U = WT == 2 ? 4u : 9u;
};

template <int WT, unsigned R = QShape<WT>::R, unsigned U = QShape<WT>::U>
__device__ __forceinline__ void gemv_q(const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,
                                       const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs,
                                       u64 mat_aux, unsigned xq_shared, const int* __restrict__ ids,
                                       const float* __restrict__ mat_scale, unsigned n_mats) {
    constexpr unsigned G = 32u / R;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned n = (blockIdx.x * (blockDim.x >> 5) + warp) * R + lane / G;
    if (n >= N) return;   // whole groups exit together (rows are group-aligned)
    const unsigned sub = lane % G, mask = (G == 32u ? 0xffffffffu : ((1u << G) - 1u) << (lane & ~(G - 1u)));
    const unsigned s = blockIdx.y;
    const unsigned e = ids ? min((unsigned)ids[s], n_mats - 1u) : s;
    const unsigned sx = xq_shared ? 0u : s;
    const uint4* xv = reinterpret_cast<const uint4*>(xq + (u64)sx * K);
    const float* xds = xd + (u64)sx * (K >> 5);
    const u64 rq = WT == 2 ? (u64)K : (u64)(K >> 1), ra = WT == 2 ? (u64)(K >> 5) * 2u : (u64)(K >> 4);
    const float acc = q_dot<WT, G, U>(qs + e * mat_qs + n * rq, aux + e * mat_aux + n * ra, xv, xds, K, sub, mask);
    if (sub == 0) y[(u64)s * N + n] = mat_scale ? acc * mat_scale[e] : acc;
}

#define GEMV_Q_ENTRY(NAME, WT)                                                                                     \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                        \
        const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,                 \
        const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux,      \
        unsigned xq_shared, const int* __restrict__ ids, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();   \
        gemv_q<WT>(qs, aux, xq, xd, y, N, K, mat_qs, mat_aux, xq_shared, ids, mat_scale, n_mats);                  \
    }
GEMV_Q_ENTRY(astral_gemv_q8_0_dp4a, 2)
GEMV_Q_ENTRY(astral_gemv_nvfp4_dp4a, 3)

// NVFP4 with a whole warp per row, for long rows (K >= 8192: e.g. a 17408-wide MLP down
// projection, where 4 rows per warp leave too few blocks and too many dependent rounds).
extern "C" __global__ void __launch_bounds__(256) astral_gemv_nvfp4_dp4a_r1(
    const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq, const float* __restrict__ xd,
    float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux, unsigned xq_shared, const int* __restrict__ ids,
    const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();
    gemv_q<3, 1, 9>(qs, aux, xq, xd, y, N, K, mat_qs, mat_aux, xq_shared, ids, mat_scale, n_mats);
}

// =============================================================================================
// Block-per-row variants of the F32 / BF16 GEMV for short, wide matrices (router 128 x 2304,
// loop-controller down 64 x 4608, sinks 24 x 2304), where one warp per row leaves the GPU
// nearly idle. Same parameters as the warp-per-row entries; grid = (N, slots), 256 threads.
template <int WT>
__device__ __forceinline__ void gemv_rowblock(const u8* __restrict__ qs, const float* __restrict__ x, float* __restrict__ y,
                                              unsigned N, unsigned K, u64 mat_qs, unsigned x_stride,
                                              const int* __restrict__ ids, const float* __restrict__ mat_scale, unsigned n_mats) {
    const unsigned n = blockIdx.x, s = blockIdx.y;
    const unsigned e = ids ? min((unsigned)ids[s], n_mats - 1u) : s;
    const float4* xv = reinterpret_cast<const float4*>(x + (u64)s * x_stride);
    float acc = 0.0f;
    if constexpr (WT == 0) {
        const float4* w = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(qs + e * mat_qs) + (u64)n * K);
        for (unsigned i = threadIdx.x; i < (K >> 2); i += blockDim.x) {
            const float4 a = __ldg(w + i), b = __ldg(xv + i);
            acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc); acc = fmaf(a.z, b.z, acc); acc = fmaf(a.w, b.w, acc);
        }
    } else {
        const uint2* w = reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(qs + e * mat_qs) + (u64)n * K);
        for (unsigned i = threadIdx.x; i < (K >> 2); i += blockDim.x) {
            const uint2 a = __ldg(w + i);
            const float4 b = __ldg(xv + i);
            acc = fmaf(bf16f(a.x & 0xffffu), b.x, acc); acc = fmaf(bf16f(a.x >> 16), b.y, acc);
            acc = fmaf(bf16f(a.y & 0xffffu), b.z, acc); acc = fmaf(bf16f(a.y >> 16), b.w, acc);
        }
    }
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31u) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        acc = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (threadIdx.x == 0) y[(u64)s * N + n] = mat_scale ? acc * mat_scale[e] : acc;
    }
}

#define GEMV_RB_ENTRY(NAME, WT)                                                                                \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                    \
        const u8* __restrict__ qs, const u8* __restrict__ aux, const float* __restrict__ x, float* __restrict__ y, \
        unsigned N, unsigned K, u64 mat_qs, u64 mat_aux, unsigned x_stride, const int* __restrict__ ids,       \
        const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();                                                \
        gemv_rowblock<WT>(qs, x, y, N, K, mat_qs, x_stride, ids, mat_scale, n_mats);                           \
    }
GEMV_RB_ENTRY(astral_gemv_f32_rowblock, 0)
GEMV_RB_ENTRY(astral_gemv_bf16_rowblock, 1)

// =============================================================================================
// Several Q8_0 GEMVs with the same K in one launch (int8 activations): an attention layer's
// Q, gate, R, K and V projections. Each launch costs a few microseconds of ramp-up and tail;
// one launch over ~20 MB of weights instead of five keeps the memory system busy throughout.
// Blocks [start_i, start_{i+1}) compute matrix i, 8 rows per block.
#define MULTI_MAX 8
struct MultiQ8 {
    const u8* qs[MULTI_MAX];
    const u8* aux[MULTI_MAX];
    const signed char* xq[MULTI_MAX];
    const float* xd[MULTI_MAX];
    float* y[MULTI_MAX];
    unsigned n[MULTI_MAX];
    unsigned start[MULTI_MAX + 1];
    unsigned count;
};

extern "C" __global__ void __launch_bounds__(256) astral_gemv_q8_0_multi(MultiQ8 m, unsigned K) { PDL_WAIT();
    constexpr unsigned R = QShape<2>::R, G = QShape<2>::G;
    const unsigned bx = blockIdx.x;
    unsigned i = 0;
    while (i + 1 < m.count && bx >= m.start[i + 1]) ++i;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned n = ((bx - m.start[i]) * (blockDim.x >> 5) + warp) * R + lane / G;
    if (n >= m.n[i]) return;
    const unsigned sub = lane % G, mask = (G == 32u ? 0xffffffffu : ((1u << G) - 1u) << (lane & ~(G - 1u)));
    const float acc = q_dot<2, G, QShape<2>::U>(m.qs[i] + (u64)n * K, m.aux[i] + (u64)n * (K >> 5) * 2u,
                                           reinterpret_cast<const uint4*>(m.xq[i]), m.xd[i], K, sub, mask);
    if (sub == 0) m.y[i][n] = acc;
}

// The same for T tokens at once (speculative verification, draft blocks): every weight row is
// read once and multiplied with the T int8 activation rows of its matrix (row t at
// xq + t * K); matrix i writes y[i] + t * n[i] + row. T is a template parameter so the T
// accumulators stay in registers.
template <unsigned T>
__device__ __forceinline__ void q8_multi_t(const MultiQ8& m, unsigned K) {
    constexpr unsigned G = 32u, U = QShape<2>::U;
    const unsigned bx = blockIdx.x;
    unsigned i = 0;
    while (i + 1 < m.count && bx >= m.start[i + 1]) ++i;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x;
    const unsigned n = (bx - m.start[i]) * (blockDim.x >> 5) + warp;
    if (n >= m.n[i]) return;
    const unsigned sub = lane & 31u, units = K >> 4;
    const uint4* w = reinterpret_cast<const uint4*>(m.qs[i] + (u64)n * K);
    const u16* wd = reinterpret_cast<const u16*>(m.aux[i] + (u64)n * (K >> 5) * 2u);
    const uint4* xv = reinterpret_cast<const uint4*>(m.xq[i]);
    const float* xd = m.xd[i];
    float acc[T];
#pragma unroll
    for (unsigned t = 0; t < T; ++t) acc[t] = 0.0f;
    for (unsigned base = 0; base < units; base += G * U) {
        uint4 wv[U];
#pragma unroll
        for (unsigned j = 0; j < U; ++j) {
            const unsigned u = base + sub + G * j;
            if (u < units) wv[j] = __ldg(w + u);
        }
#pragma unroll
        for (unsigned j = 0; j < U; ++j) {
            const unsigned u = base + sub + G * j;
            if (u < units) {
                const float dw = __half2float(__ushort_as_half(__ldg(wd + (u >> 1))));
#pragma unroll
                for (unsigned t = 0; t < T; ++t) {
                    const uint4 a = __ldg(xv + (u64)t * units + u);
                    int sum = __dp4a((int)wv[j].x, (int)a.x, 0);
                    sum = __dp4a((int)wv[j].y, (int)a.y, sum);
                    sum = __dp4a((int)wv[j].z, (int)a.z, sum);
                    sum = __dp4a((int)wv[j].w, (int)a.w, sum);
                    acc[t] = fmaf((float)sum, dw * __ldg(xd + (u64)t * (K >> 5) + (u >> 1)), acc[t]);
                }
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < T; ++t) {
        float a = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (sub == t) m.y[i][(u64)t * m.n[i] + n] = a;   // lane t stores token t
    }
}

// Tensor-core version for 1..8 tokens: int8 mma.sync m16n8k32 with the weights as A (16 rows x
// 32) and the T activation rows as B (32 x 8; columns >= T are zero). One Q8_0 scale block is
// exactly one k-step, so each step's int32 result is scaled by (weight scale x activation scale)
// into f32 accumulators. A block is 4 warps on the same 16 rows, each taking a quarter of K;
// the partial sums are added in shared memory. Same MultiQ8 as above (blocks per matrix =
// ceil(n / 16)); y_i[t * n_i + row].
__device__ __forceinline__ void mma_s8_16832(int (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
}

// NT tiles of 8 tokens (T <= 8 * NT): every weight fragment is loaded once and multiplied
// with each tile. A token's arithmetic (the K split, the order of the k-steps, the scaling and
// the final reduction) does not depend on NT, T or its tile, so results are the same for any
// batch size. U = k-steps of loads in flight (does not change the arithmetic).
template <unsigned NT, unsigned U>
__device__ __forceinline__ void q8_multi_mma(const MultiQ8& m, unsigned K, unsigned T) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned bx = blockIdx.x;
    unsigned i = 0;
    while (i + 1 < m.count && bx >= m.start[i + 1]) ++i;
    const unsigned N = m.n[i], row0 = (bx - m.start[i]) * 16u;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, g = lane >> 2, tg = lane & 3u;
    const unsigned KB = K >> 5;
    const unsigned r0 = min(row0 + g, N - 1u), r1 = min(row0 + g + 8u, N - 1u);
    const unsigned char* w0 = m.qs[i] + (u64)r0 * K + tg * 4u;
    const unsigned char* w1 = m.qs[i] + (u64)r1 * K + tg * 4u;
    const u16* s0 = reinterpret_cast<const u16*>(m.aux[i]) + (u64)r0 * KB;
    const u16* s1 = reinterpret_cast<const u16*>(m.aux[i]) + (u64)r1 * KB;
    const unsigned tiles = min((T + 7u) >> 3, NT);
    const signed char* xb[NT];
    const float* xd0[NT];
    const float* xd1[NT];
    bool tok_ok[NT], ok0[NT], ok1[NT];
#pragma unroll
    for (unsigned j = 0; j < NT; ++j) {
        const unsigned col = 8u * j + g, t0 = 8u * j + tg * 2u, t1 = t0 + 1u;
        tok_ok[j] = col < T;
        xb[j] = m.xq[i] + (u64)min(col, T - 1u) * K + tg * 4u;
        xd0[j] = m.xd[i] + (u64)min(t0, T - 1u) * KB;
        xd1[j] = m.xd[i] + (u64)min(t1, T - 1u) * KB;
        ok0[j] = t0 < T;
        ok1[j] = t1 < T;
    }
    const unsigned kb0 = warp * KB / 4u, kb1 = (warp + 1u) * KB / 4u;
    float acc[NT][4];
#pragma unroll
    for (unsigned j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    for (unsigned kb = kb0; kb < kb1; kb += U) {
        unsigned a[U][4], b[U][NT][2];
        float ws0[U], ws1[U], xs0[U][NT], xs1[U][NT];
#pragma unroll
        for (unsigned u = 0; u < U; ++u) {
            const unsigned k = min(kb + u, kb1 - 1u);
            const unsigned off = k * 32u;
            a[u][0] = __ldg(reinterpret_cast<const unsigned*>(w0 + off));
            a[u][1] = __ldg(reinterpret_cast<const unsigned*>(w1 + off));
            a[u][2] = __ldg(reinterpret_cast<const unsigned*>(w0 + off + 16u));
            a[u][3] = __ldg(reinterpret_cast<const unsigned*>(w1 + off + 16u));
            ws0[u] = __half2float(__ushort_as_half(__ldg(s0 + k)));
            ws1[u] = __half2float(__ushort_as_half(__ldg(s1 + k)));
#pragma unroll
            for (unsigned j = 0; j < NT; ++j) {
                if (j < tiles) {
                    b[u][j][0] = tok_ok[j] ? __ldg(reinterpret_cast<const unsigned*>(xb[j] + off)) : 0u;
                    b[u][j][1] = tok_ok[j] ? __ldg(reinterpret_cast<const unsigned*>(xb[j] + off + 16u)) : 0u;
                    xs0[u][j] = ok0[j] ? __ldg(xd0[j] + k) : 0.0f;
                    xs1[u][j] = ok1[j] ? __ldg(xd1[j] + k) : 0.0f;
                }
            }
        }
#pragma unroll
        for (unsigned u = 0; u < U; ++u) {
            if (kb + u < kb1) {
#pragma unroll
                for (unsigned j = 0; j < NT; ++j) {
                    if (j < tiles) {
                        int c[4];
                        mma_s8_16832(c, a[u], b[u][j][0], b[u][j][1]);
                        acc[j][0] = fmaf((float)c[0], ws0[u] * xs0[u][j], acc[j][0]);
                        acc[j][1] = fmaf((float)c[1], ws0[u] * xs1[u][j], acc[j][1]);
                        acc[j][2] = fmaf((float)c[2], ws1[u] * xs0[u][j], acc[j][2]);
                        acc[j][3] = fmaf((float)c[3], ws1[u] * xs1[u][j], acc[j][3]);
                    }
                }
            }
        }
    }
    __shared__ float red[4][32][4 * NT];
#pragma unroll
    for (unsigned j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) red[warp][lane][4 * j + e] = acc[j][e];
    __syncthreads();
    if (warp == 0) {
        const unsigned ra = row0 + g, rb = row0 + g + 8u;
        float* y = m.y[i];
#pragma unroll
        for (unsigned j = 0; j < NT; ++j) {
            if (j >= tiles) continue;
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) v[e] = red[0][lane][4 * j + e] + red[1][lane][4 * j + e] + red[2][lane][4 * j + e] + red[3][lane][4 * j + e];
            const unsigned t0 = 8u * j + tg * 2u, t1 = t0 + 1u;
            if (ra < N && ok0[j]) y[(u64)t0 * N + ra] = v[0];
            if (ra < N && ok1[j]) y[(u64)t1 * N + ra] = v[1];
            if (rb < N && ok0[j]) y[(u64)t0 * N + rb] = v[2];
            if (rb < N && ok1[j]) y[(u64)t1 * N + rb] = v[3];
        }
    }
#endif
}

extern "C" __global__ void __launch_bounds__(128) astral_gemv_q8_0_multi_mma(MultiQ8 m, unsigned K, unsigned T) { PDL_WAIT(); q8_multi_mma<1, 4>(m, K, T); }
extern "C" __global__ void __launch_bounds__(128) astral_gemv_q8_0_multi_mma_n2(MultiQ8 m, unsigned K, unsigned T) { PDL_WAIT(); q8_multi_mma<2, 4>(m, K, T); }
extern "C" __global__ void __launch_bounds__(128) astral_gemv_q8_0_multi_mma_n4(MultiQ8 m, unsigned K, unsigned T) { PDL_WAIT(); q8_multi_mma<4, 2>(m, K, T); }
extern "C" __global__ void __launch_bounds__(128) astral_gemv_q8_0_multi_mma_n8(MultiQ8 m, unsigned K, unsigned T) { PDL_WAIT(); q8_multi_mma<8, 2>(m, K, T); }

// q8_multi_t with a runtime token count <= T (rows t >= count are neither read nor written):
// the same per-token arithmetic as the single-token q_dot, for batched decode.
template <unsigned T>
__device__ __forceinline__ void q8_rows_t(const MultiQ8& m, unsigned K, unsigned count) {
    // A warp computes R weight rows at once (each activation load serves R rows); per row the
    // lane's units, their order and the reduction are those of q_dot<2, 32, 4>.
    constexpr unsigned G = 32u, U = 2u, R = 4u;
    const unsigned warp = threadIdx.x >> 5, sub = threadIdx.x & 31u;
    const unsigned N = m.n[0];
    const unsigned n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * R;
    if (n0 >= N) return;
    const unsigned units = K >> 4;
    const uint4* w[R];
    const u16* wd[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const unsigned n = min(n0 + r, N - 1u);
        w[r] = reinterpret_cast<const uint4*>(m.qs[0] + (u64)n * K);
        wd[r] = reinterpret_cast<const u16*>(m.aux[0] + (u64)n * (K >> 5) * 2u);
    }
    const uint4* xv = reinterpret_cast<const uint4*>(m.xq[0]);
    const float* xd = m.xd[0];
    float acc[R][T];
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
#pragma unroll
        for (unsigned t = 0; t < T; ++t) acc[r][t] = 0.0f;
    for (unsigned base = 0; base < units; base += G * U) {
        uint4 wv[U][R];
        float dw[U][R];
#pragma unroll
        for (unsigned j = 0; j < U; ++j) {
            const unsigned u = base + sub + G * j;
            if (u < units) {
#pragma unroll
                for (unsigned r = 0; r < R; ++r) {
                    wv[j][r] = __ldg(w[r] + u);
                    dw[j][r] = __half2float(__ushort_as_half(__ldg(wd[r] + (u >> 1))));
                }
            }
        }
#pragma unroll
        for (unsigned j = 0; j < U; ++j) {
            const unsigned u = base + sub + G * j;
            if (u < units) {
#pragma unroll
                for (unsigned t = 0; t < T; ++t) {
                    if (t < count) {
                        const uint4 a = __ldg(xv + (u64)t * units + u);
                        const float xs = __ldg(xd + (u64)t * (K >> 5) + (u >> 1));
#pragma unroll
                        for (unsigned r = 0; r < R; ++r) {
                            int sum = __dp4a((int)wv[j][r].x, (int)a.x, 0);
                            sum = __dp4a((int)wv[j][r].y, (int)a.y, sum);
                            sum = __dp4a((int)wv[j][r].z, (int)a.z, sum);
                            sum = __dp4a((int)wv[j][r].w, (int)a.w, sum);
                            acc[r][t] = fmaf((float)sum, dw[j][r] * xs, acc[r][t]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
            float a = acc[r][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (sub == t && t < count && n0 + r < N) m.y[0][(u64)t * N + n0 + r] = a;
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) astral_gemv_q8_0_rows_t16(MultiQ8 m, unsigned K, unsigned count) { PDL_WAIT(); q8_rows_t<16>(m, K, count); }
extern "C" __global__ void __launch_bounds__(256) astral_gemv_q8_0_rows_t32(MultiQ8 m, unsigned K, unsigned count) { PDL_WAIT(); q8_rows_t<32>(m, K, count); }

#define Q8_MULTI_T_ENTRY(T) \
    extern "C" __global__ void __launch_bounds__(256) astral_gemv_q8_0_multi_t##T(MultiQ8 m, unsigned K) { PDL_WAIT(); q8_multi_t<T>(m, K); }
Q8_MULTI_T_ENTRY(2)
Q8_MULTI_T_ENTRY(3)
Q8_MULTI_T_ENTRY(4)
Q8_MULTI_T_ENTRY(5)
Q8_MULTI_T_ENTRY(6)
Q8_MULTI_T_ENTRY(7)
Q8_MULTI_T_ENTRY(8)

// =============================================================================================
// Grouped int8-activation GEMM for prefill: work item i = (matrix e, up to NB activation rows
// xi[], output rows yi[]); every weight row is loaded once and multiplied with all of the
// item's activations. Dense layers use items of consecutive tokens; MoE layers group the
// (token, expert) pairs by expert. Indices are clamped to x_rows / y_rows on the device.
// grid = (row tiles, n_items), 256 threads. Mirrored by Gemv::launch_items in ops.rs.
#define NB 8
struct QItem {
    unsigned e, n, xi[NB], yi[NB];
};

template <int WT>
__device__ __forceinline__ void gemm_items(const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,
                                           const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs,
                                           u64 mat_aux, const QItem* __restrict__ items, unsigned x_rows, unsigned y_rows,
                                           const float* __restrict__ mat_scale, unsigned n_mats) {
    constexpr unsigned R = QShape<WT>::R, G = QShape<WT>::G, U = 4;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned n = (blockIdx.x * (blockDim.x >> 5) + warp) * R + lane / G;
    if (n >= N) return;
    const unsigned sub = lane % G, mask = (G == 32u ? 0xffffffffu : ((1u << G) - 1u) << (lane & ~(G - 1u)));
    const QItem& it = items[blockIdx.y];
    const unsigned e = min(it.e, n_mats - 1u), cnt = min(it.n, (unsigned)NB);
    if (cnt == 0) return;   // unused work item (device-built lists end with empty ones)
    unsigned xr[NB];
#pragma unroll
    for (unsigned b = 0; b < NB; ++b) xr[b] = min(it.xi[b], x_rows - 1u);
    float acc[NB];
#pragma unroll
    for (unsigned b = 0; b < NB; ++b) acc[b] = 0.0f;
    if constexpr (WT == 2) {    // Q8_0: unit = 16 weights
        const unsigned units = K >> 4;
        const uint4* w = reinterpret_cast<const uint4*>(qs + e * mat_qs + (u64)n * K);
        const u16* wd = reinterpret_cast<const u16*>(aux + e * mat_aux) + (u64)n * (K >> 5);
        for (unsigned base = 0; base < units; base += G * U) {
            uint4 wv[U];
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) wv[i] = __ldg(w + u);
            }
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) {
                    const float dw = __half2float(__ushort_as_half(__ldg(wd + (u >> 1))));
#pragma unroll
                    for (unsigned b = 0; b < NB; ++b) {
                        if (b < cnt) {
                            const uint4 a = __ldg(reinterpret_cast<const uint4*>(xq + (u64)xr[b] * K) + u);
                            int sum = __dp4a((int)wv[i].x, (int)a.x, 0);
                            sum = __dp4a((int)wv[i].y, (int)a.y, sum);
                            sum = __dp4a((int)wv[i].z, (int)a.z, sum);
                            sum = __dp4a((int)wv[i].w, (int)a.w, sum);
                            acc[b] = fmaf((float)sum, dw * __ldg(xd + (u64)xr[b] * (K >> 5) + (u >> 1)), acc[b]);
                        }
                    }
                }
            }
        }
    } else {                    // NVFP4: unit = 32 weights
        const unsigned units = K >> 5;
        const uint4* w = reinterpret_cast<const uint4*>(qs + e * mat_qs + (u64)n * (K >> 1));
        const u16* wd = reinterpret_cast<const u16*>(aux + e * mat_aux + (u64)n * (K >> 4));
        for (unsigned base = 0; base < units; base += G * U) {
            uint4 wv[U];
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) wv[i] = __ldg(w + u);
            }
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) {
                    const int2 a0 = table16((int)wv[i].x, k_e2m1x2), a1 = table16((int)wv[i].y, k_e2m1x2);
                    const int2 b0 = table16((int)wv[i].z, k_e2m1x2), b1 = table16((int)wv[i].w, k_e2m1x2);
                    const unsigned sc = __ldg(wd + u);
                    const float2 d2 = __half22float2(__halves2half2(__ushort_as_half((unsigned short)((sc & 0xffu) << 7)),
                                                                    __ushort_as_half((unsigned short)((sc >> 8) << 7))));
#pragma unroll
                    for (unsigned b = 0; b < NB; ++b) {
                        if (b < cnt) {
                            const uint4* xv = reinterpret_cast<const uint4*>(xq + (u64)xr[b] * K);
                            const uint4 x0 = __ldg(xv + 2u * u), x1 = __ldg(xv + 2u * u + 1u);
                            int s0 = __dp4a(a0.x, (int)x0.x, 0);
                            s0 = __dp4a(a1.x, (int)x0.y, s0);
                            s0 = __dp4a(a0.y, (int)x0.z, s0);
                            s0 = __dp4a(a1.y, (int)x0.w, s0);
                            int s1 = __dp4a(b0.x, (int)x1.x, 0);
                            s1 = __dp4a(b1.x, (int)x1.y, s1);
                            s1 = __dp4a(b0.y, (int)x1.z, s1);
                            s1 = __dp4a(b1.y, (int)x1.w, s1);
                            acc[b] = fmaf(fmaf((float)s0, d2.x, (float)s1 * d2.y), __ldg(xd + (u64)xr[b] * (K >> 5) + u), acc[b]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (unsigned b = 0; b < NB; ++b) acc[b] *= 128.0f;
    }
    const float ms = mat_scale ? mat_scale[e] : 1.0f;
#pragma unroll
    for (unsigned b = 0; b < NB; ++b) {
        float a = acc[b];
#pragma unroll
        for (unsigned o = G / 2; o > 0; o >>= 1) a += __shfl_xor_sync(mask, a, o);
        if (sub == 0 && b < cnt) y[(u64)min(it.yi[b], y_rows - 1u) * N + n] = a * ms;
    }
}

#define GEMM_ITEMS_ENTRY(NAME, WT)                                                                                  \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                         \
        const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,                  \
        const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux,       \
        const QItem* __restrict__ items, unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale,     \
        unsigned n_mats) { PDL_WAIT();                                                                                          \
        gemm_items<WT>(qs, aux, xq, xd, y, N, K, mat_qs, mat_aux, items, x_rows, y_rows, mat_scale, n_mats);        \
    }
GEMM_ITEMS_ENTRY(astral_gemm_q8_0_items, 2)
GEMM_ITEMS_ENTRY(astral_gemm_nvfp4_items, 3)

// =============================================================================================
// Tensor-core int8 GEMM for prefill (mma.sync, sm_80+). Same maths as gemm_items: per 32-value
// activation block, an integer dot of weights and int8 activations, scaled by both scales.
//   Q8_0:  mma m16n8k32 s8 per 32-wide weight block (one weight scale per block)
//   NVFP4: mma m16n8k16 s8 per 16-wide sub-block (E2M1 nibbles decoded to int8 in registers)
// A warp computes 16 weight rows x up to 64 tokens (8 n-tiles), so each weight fragment is
// loaded once per 64 tokens; 8 warps per block cover 128 rows. Work item i (3 words: matrix e,
// start, count <= 64) takes activation rows xidx[start..+count] to output rows yidx[..].
// N % 16 == 0, K % 32 == 0 (K % 64 for NVFP4). All indices are clamped on the device.
// grid = (ceil(N / 128), n_items), 256 threads. Mirrored by Gemv::launch_mma in ops.rs.
#define MMA_TOK 64

__device__ __forceinline__ void mma_k32(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
}
__device__ __forceinline__ void mma_k16(int (&c)[4], unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
}

template <int WT>
__device__ __forceinline__ void gemm_mma(const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,
                                         const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs,
                                         u64 mat_aux, const unsigned* __restrict__ items, const unsigned* __restrict__ xidx,
                                         const unsigned* __restrict__ yidx, unsigned idx_len, unsigned x_rows, unsigned y_rows,
                                         const float* __restrict__ mat_scale, unsigned n_mats) {
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned row0 = blockIdx.x * 128u + warp * 16u;
    if (row0 >= N) return;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], (unsigned)MMA_TOK), idx_len - start);
    if (count == 0) return;
    const unsigned ntiles = (count + 7u) / 8u;
    // B-fragment token (column g of each n-tile) and C-fragment tokens (columns 2t, 2t+1).
    unsigned xb[8], xc0[8], xc1[8];
#pragma unroll
    for (unsigned nt = 0; nt < 8; ++nt) {
        const unsigned jb = min(nt * 8u + g, count - 1u), j0 = min(nt * 8u + 2u * t, count - 1u), j1 = min(nt * 8u + 2u * t + 1u, count - 1u);
        xb[nt] = min(xidx[start + jb], x_rows - 1u);
        xc0[nt] = min(xidx[start + j0], x_rows - 1u);
        xc1[nt] = min(xidx[start + j1], x_rows - 1u);
    }
    float acc[8][4];
#pragma unroll
    for (unsigned nt = 0; nt < 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    const unsigned kb32 = K >> 5;
    const unsigned r_lo = row0 + g, r_hi = row0 + g + 8u;
    if constexpr (WT == 2) {
        const u8* w = qs + e * mat_qs;
        const u16* wd = reinterpret_cast<const u16*>(aux + e * mat_aux);
        for (unsigned kb = 0; kb < kb32; ++kb) {
            const u8* wl = w + (u64)r_lo * K + kb * 32u + 4u * t;
            const u8* wh = w + (u64)r_hi * K + kb * 32u + 4u * t;
            const unsigned a0 = __ldg(reinterpret_cast<const unsigned*>(wl)), a2 = __ldg(reinterpret_cast<const unsigned*>(wl + 16));
            const unsigned a1 = __ldg(reinterpret_cast<const unsigned*>(wh)), a3 = __ldg(reinterpret_cast<const unsigned*>(wh + 16));
            const float d_lo = __half2float(__ushort_as_half(__ldg(wd + (u64)r_lo * kb32 + kb)));
            const float d_hi = __half2float(__ushort_as_half(__ldg(wd + (u64)r_hi * kb32 + kb)));
#pragma unroll
            for (unsigned nt = 0; nt < 8; ++nt) {
                if (nt < ntiles) {
                    const signed char* xr = xq + (u64)xb[nt] * K + kb * 32u + 4u * t;
                    const unsigned b0 = __ldg(reinterpret_cast<const unsigned*>(xr)), b1 = __ldg(reinterpret_cast<const unsigned*>(xr + 16));
                    int c[4];
                    mma_k32(c, a0, a1, a2, a3, b0, b1);
                    const float x0 = __ldg(xd + (u64)xc0[nt] * kb32 + kb), x1 = __ldg(xd + (u64)xc1[nt] * kb32 + kb);
                    acc[nt][0] = fmaf((float)c[0], d_lo * x0, acc[nt][0]);
                    acc[nt][1] = fmaf((float)c[1], d_lo * x1, acc[nt][1]);
                    acc[nt][2] = fmaf((float)c[2], d_hi * x0, acc[nt][2]);
                    acc[nt][3] = fmaf((float)c[3], d_hi * x1, acc[nt][3]);
                }
            }
        }
    } else {
        // NVFP4 rows: K/2 bytes, sub-block sb (16 values) = 8 bytes: low nibble = value j,
        // high nibble = value j + 8. A-fragment columns 4t..4t+3 of a sub-block are the low
        // (t < 2) or high (t >= 2) nibbles of the sub-block's word (t & 1).
        const u8* w = qs + e * mat_qs;
        const u8* wd = aux + e * mat_aux;
        const unsigned ksb = K >> 4;
        for (unsigned sb = 0; sb < ksb; ++sb) {
            const unsigned wl = __ldg(reinterpret_cast<const unsigned*>(w + (u64)r_lo * (K >> 1) + sb * 8u) + (t & 1u));
            const unsigned wh = __ldg(reinterpret_cast<const unsigned*>(w + (u64)r_hi * (K >> 1) + sb * 8u) + (t & 1u));
            const int2 tl = table16((int)wl, k_e2m1x2), th = table16((int)wh, k_e2m1x2);
            const unsigned a0 = (unsigned)(t < 2u ? tl.x : tl.y), a1 = (unsigned)(t < 2u ? th.x : th.y);
            // Canonical UE4M3 as f16 bits x << 7 (= value * 2^-8); the 2^8 * 0.5 comes at the end.
            const float d_lo = __half2float(__ushort_as_half((unsigned short)((unsigned)__ldg(wd + (u64)r_lo * ksb + sb) << 7)));
            const float d_hi = __half2float(__ushort_as_half((unsigned short)((unsigned)__ldg(wd + (u64)r_hi * ksb + sb) << 7)));
            const unsigned kb = sb >> 1;
#pragma unroll
            for (unsigned nt = 0; nt < 8; ++nt) {
                if (nt < ntiles) {
                    const unsigned b0 = __ldg(reinterpret_cast<const unsigned*>(xq + (u64)xb[nt] * K + sb * 16u + 4u * t));
                    int c[4];
                    mma_k16(c, a0, a1, b0);
                    const float x0 = __ldg(xd + (u64)xc0[nt] * kb32 + kb), x1 = __ldg(xd + (u64)xc1[nt] * kb32 + kb);
                    acc[nt][0] = fmaf((float)c[0], d_lo * x0, acc[nt][0]);
                    acc[nt][1] = fmaf((float)c[1], d_lo * x1, acc[nt][1]);
                    acc[nt][2] = fmaf((float)c[2], d_hi * x0, acc[nt][2]);
                    acc[nt][3] = fmaf((float)c[3], d_hi * x1, acc[nt][3]);
                }
            }
        }
#pragma unroll
        for (unsigned nt = 0; nt < 8; ++nt) { acc[nt][0] *= 128.0f; acc[nt][1] *= 128.0f; acc[nt][2] *= 128.0f; acc[nt][3] *= 128.0f; }
    }
    const float ms = mat_scale ? mat_scale[e] : 1.0f;
#pragma unroll
    for (unsigned nt = 0; nt < 8; ++nt) {
        if (nt < ntiles) {
            const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
            if (j0 < count) {
                float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
                yr[r_lo] = acc[nt][0] * ms;
                yr[r_hi] = acc[nt][2] * ms;
            }
            if (j1 < count) {
                float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
                yr[r_lo] = acc[nt][1] * ms;
                yr[r_hi] = acc[nt][3] * ms;
            }
        }
    }
}

#define GEMM_MMA_ENTRY(NAME, WT)                                                                                    \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                         \
        const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,                  \
        const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux,       \
        const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx,   \
        unsigned idx_len, unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT(); \
        gemm_mma<WT>(qs, aux, xq, xd, y, N, K, mat_qs, mat_aux, items, xidx, yidx, idx_len, x_rows, y_rows,       \
                     mat_scale, n_mats);                                                                            \
    }
GEMM_MMA_ENTRY(astral_gemm_q8_0_mma, 2)
GEMM_MMA_ENTRY(astral_gemm_nvfp4_mma, 3)

// =============================================================================================
// Tensor-core int8 GEMM v2 (prefill): same maths and work items as gemm_mma, but staged through
// shared memory with a 4-stage cp.async pipeline (16-byte coalesced copies, activation rows
// gathered by xidx), so the next stages' loads are always in flight while the MMAs run.
// Block = 4 warps = 64 weight rows x 64 tokens; each stage covers 64 values of K.
// Requires N % 64 == 0 and K % 64 == 0. grid = (N / 64, n_items), 128 threads.
#define G2_STAGES 4
#define G2_ROWS 64
#define G2_ROWB 80   // padded smem row: 64 data bytes + 16 (conflict-free fragment reads)
#define G2_AROWB_FP4 48   // NVFP4 A row: 32 packed bytes + 16

__device__ __forceinline__ void cp_async16(void* smem, const void* g) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* g) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* g) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int WT>
struct G2Stage {
    static constexpr unsigned A_BYTES = G2_ROWS * (WT == 2 ? G2_ROWB : G2_AROWB_FP4);
    static constexpr unsigned DA_BYTES = G2_ROWS * 4;   // Q8_0: 2 f16 / row; NVFP4: 4 UE4M3 / row
    unsigned char a[A_BYTES];
    unsigned char da[DA_BYTES];
    signed char b[MMA_TOK * G2_ROWB];
    float db[MMA_TOK * 2];
};

template <int WT>
__device__ __forceinline__ void gemm_mma2(const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,
                                          const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs,
                                          u64 mat_aux, const unsigned* __restrict__ items, const unsigned* __restrict__ xidx,
                                          const unsigned* __restrict__ yidx, unsigned idx_len, unsigned x_rows, unsigned y_rows,
                                          const float* __restrict__ mat_scale, unsigned n_mats) {
    __shared__ __align__(16) G2Stage<WT> sm[G2_STAGES];
    __shared__ unsigned xrow[MMA_TOK];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned row0 = blockIdx.x * G2_ROWS;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], (unsigned)MMA_TOK), idx_len - start);
    if (count == 0 || row0 >= N) return;
    if (tid < MMA_TOK) xrow[tid] = min(xidx[start + min(tid, count - 1u)], x_rows - 1u);
    __syncthreads();
    const unsigned ntiles = (count + 7u) / 8u;
    const unsigned kb32 = K >> 5;
    const u8* wbase = qs + e * mat_qs;
    const u8* abase = aux + e * mat_aux;
    const unsigned KT = K / 64u;

    auto load = [&](unsigned stage, unsigned kt) {
        G2Stage<WT>& S = sm[stage];
        const unsigned k0 = kt * 64u;
        if constexpr (WT == 2) {
            for (unsigned c = tid; c < G2_ROWS * 4u; c += blockDim.x) {
                const unsigned r = c >> 2, part = c & 3u;
                cp_async16(S.a + r * G2_ROWB + part * 16u, wbase + (u64)(row0 + r) * K + k0 + part * 16u);
            }
            if (tid < G2_ROWS) cp_async4(S.da + tid * 4u, abase + ((u64)(row0 + tid) * kb32 + (k0 >> 5)) * 2u);
        } else {
            for (unsigned c = tid; c < G2_ROWS * 2u; c += blockDim.x) {
                const unsigned r = c >> 1, part = c & 1u;
                cp_async16(S.a + r * G2_AROWB_FP4 + part * 16u, wbase + (u64)(row0 + r) * (K >> 1) + (k0 >> 1) + part * 16u);
            }
            if (tid < G2_ROWS) cp_async4(S.da + tid * 4u, abase + (u64)(row0 + tid) * (K >> 4) + (k0 >> 4));
        }
        for (unsigned c = tid; c < MMA_TOK * 4u; c += blockDim.x) {
            const unsigned j = c >> 2, part = c & 3u;
            cp_async16(S.b + j * G2_ROWB + part * 16u, xq + (u64)xrow[j] * K + k0 + part * 16u);
        }
        if (tid >= 64u && tid < 64u + MMA_TOK) {
            const unsigned j = tid - 64u;
            cp_async8(S.db + j * 2u, xd + (u64)xrow[j] * kb32 + (k0 >> 5));
        }
    };

    float acc[8][4];
#pragma unroll
    for (unsigned nt = 0; nt < 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;

#pragma unroll
    for (unsigned s = 0; s < G2_STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    const unsigned rl = warp * 16u + g, rh = rl + 8u;   // this lane's rows within the tile
    for (unsigned kt = 0; kt < KT; ++kt) {
        cp_async_wait<G2_STAGES - 2>();
        __syncthreads();
        if (kt + G2_STAGES - 1 < KT) load((kt + G2_STAGES - 1) % G2_STAGES, kt + G2_STAGES - 1);
        cp_async_commit();
        const G2Stage<WT>& S = sm[kt % G2_STAGES];
        if constexpr (WT == 2) {
#pragma unroll
            for (unsigned kb = 0; kb < 2; ++kb) {
                const unsigned char* al = S.a + rl * G2_ROWB + kb * 32u + 4u * t;
                const unsigned char* ah = S.a + rh * G2_ROWB + kb * 32u + 4u * t;
                const unsigned a0 = *reinterpret_cast<const unsigned*>(al), a2 = *reinterpret_cast<const unsigned*>(al + 16);
                const unsigned a1 = *reinterpret_cast<const unsigned*>(ah), a3 = *reinterpret_cast<const unsigned*>(ah + 16);
                const float d_lo = __half2float(reinterpret_cast<const __half*>(S.da)[rl * 2u + kb]);
                const float d_hi = __half2float(reinterpret_cast<const __half*>(S.da)[rh * 2u + kb]);
#pragma unroll
                for (unsigned nt = 0; nt < 8; ++nt) {
                    if (nt < ntiles) {
                        const signed char* br = S.b + (nt * 8u + g) * G2_ROWB + kb * 32u + 4u * t;
                        int c[4];
                        mma_k32(c, a0, a1, a2, a3, *reinterpret_cast<const unsigned*>(br), *reinterpret_cast<const unsigned*>(br + 16));
                        const float x0 = S.db[(nt * 8u + 2u * t) * 2u + kb], x1 = S.db[(nt * 8u + 2u * t + 1u) * 2u + kb];
                        acc[nt][0] = fmaf((float)c[0], d_lo * x0, acc[nt][0]);
                        acc[nt][1] = fmaf((float)c[1], d_lo * x1, acc[nt][1]);
                        acc[nt][2] = fmaf((float)c[2], d_hi * x0, acc[nt][2]);
                        acc[nt][3] = fmaf((float)c[3], d_hi * x1, acc[nt][3]);
                    }
                }
            }
        } else {
#pragma unroll
            for (unsigned sb = 0; sb < 4; ++sb) {
                const unsigned wl = *reinterpret_cast<const unsigned*>(S.a + rl * G2_AROWB_FP4 + sb * 8u + (t & 1u) * 4u);
                const unsigned wh = *reinterpret_cast<const unsigned*>(S.a + rh * G2_AROWB_FP4 + sb * 8u + (t & 1u) * 4u);
                const int2 tl = table16((int)wl, k_e2m1x2), th = table16((int)wh, k_e2m1x2);
                const unsigned a0 = (unsigned)(t < 2u ? tl.x : tl.y), a1 = (unsigned)(t < 2u ? th.x : th.y);
                const float d_lo = __half2float(__ushort_as_half((unsigned short)((unsigned)S.da[rl * 4u + sb] << 7)));
                const float d_hi = __half2float(__ushort_as_half((unsigned short)((unsigned)S.da[rh * 4u + sb] << 7)));
                const unsigned kb = sb >> 1;
#pragma unroll
                for (unsigned nt = 0; nt < 8; ++nt) {
                    if (nt < ntiles) {
                        const unsigned b0 = *reinterpret_cast<const unsigned*>(S.b + (nt * 8u + g) * G2_ROWB + sb * 16u + 4u * t);
                        int c[4];
                        mma_k16(c, a0, a1, b0);
                        const float x0 = S.db[(nt * 8u + 2u * t) * 2u + kb], x1 = S.db[(nt * 8u + 2u * t + 1u) * 2u + kb];
                        acc[nt][0] = fmaf((float)c[0], d_lo * x0, acc[nt][0]);
                        acc[nt][1] = fmaf((float)c[1], d_lo * x1, acc[nt][1]);
                        acc[nt][2] = fmaf((float)c[2], d_hi * x0, acc[nt][2]);
                        acc[nt][3] = fmaf((float)c[3], d_hi * x1, acc[nt][3]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    const float ms = (mat_scale ? mat_scale[e] : 1.0f) * (WT == 2 ? 1.0f : 128.0f);
    const unsigned r_lo = row0 + rl, r_hi = row0 + rh;
#pragma unroll
    for (unsigned nt = 0; nt < 8; ++nt) {
        if (nt < ntiles) {
            const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
            if (j0 < count) {
                float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
                yr[r_lo] = acc[nt][0] * ms;
                yr[r_hi] = acc[nt][2] * ms;
            }
            if (j1 < count) {
                float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
                yr[r_lo] = acc[nt][1] * ms;
                yr[r_hi] = acc[nt][3] * ms;
            }
        }
    }
}

#define GEMM_MMA2_ENTRY(NAME, WT)                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) NAME(                                                         \
        const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,                  \
        const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux,       \
        const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx,   \
        unsigned idx_len, unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT(); \
        gemm_mma2<WT>(qs, aux, xq, xd, y, N, K, mat_qs, mat_aux, items, xidx, yidx, idx_len, x_rows, y_rows,      \
                      mat_scale, n_mats);                                                                           \
    }
GEMM_MMA2_ENTRY(astral_gemm_q8_0_mma2, 2)
GEMM_MMA2_ENTRY(astral_gemm_nvfp4_mma2, 3)

// =============================================================================================
// Tensor-core GEMM v3 (prefill): weights dequantised to f16 in registers, f16 activations, f16
// mma m16n8k16 accumulating in f32 straight through K (no per-block rescaling). Same tiles,
// pipeline and work items as v2 (3 stages: the f16 activation tile is twice the int8 one).
//   NVFP4: an E2M1 nibble n as f16 bits ((n & 8) << 12) | ((n & 7) << 9) is its value * 2^-14
//          (normal and subnormal alike), and a canonical UE4M3 byte x as f16 bits x << 7 is its
//          value * 2^-8, so w = (nib * 2^14) * (scale * 2^8) is exact in f16.
//   Q8_0:  int8 b as f16 = (1024 + (b + 128)) - 1152 exactly, times the f16 block scale (one
//          f16 rounding, ~2^-11 relative: far below the int8-activation error it replaces).
__device__ __forceinline__ void mma_f16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#define G3_STAGES 3
#define G3_BROWH 72   // f16 activation smem row: 64 + 8 pad (conflict-free fragment reads)

template <int WT>
struct G3Stage {
    static constexpr unsigned A_BYTES = G2_ROWS * (WT == 2 ? G2_ROWB : G2_AROWB_FP4);
    unsigned char a[A_BYTES];
    unsigned char da[G2_ROWS * 4];
    __half b[MMA_TOK * G3_BROWH];
};

__device__ __forceinline__ unsigned nib2_to_h2(unsigned x) {   // x: nibbles at bits 0..3 and 8..11
    const unsigned y = (x & 0xfu) | ((x & 0xf00u) << 8);
    return ((y & 0x00080008u) << 12) | ((y & 0x00070007u) << 9);
}
__device__ __forceinline__ unsigned i8x2_to_h2(unsigned v) {   // v: two int8 in bits 0..15
    const unsigned x = v ^ 0x8080u;
    const unsigned b = ((x & 0xffu) | ((x & 0xff00u) << 8)) | 0x64006400u;
    const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&b), __float2half2_rn(1152.0f));
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ unsigned hmul2u(unsigned a, __half2 s) {
    const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&a), s);
    return *reinterpret_cast<const unsigned*>(&r);
}

template <int WT>
__device__ __forceinline__ void gemm_mma3(const u8* __restrict__ qs, const u8* __restrict__ aux, const __half* __restrict__ xh,
                                          float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux,
                                          const unsigned* __restrict__ items, const unsigned* __restrict__ xidx,
                                          const unsigned* __restrict__ yidx, unsigned idx_len, unsigned x_rows, unsigned y_rows,
                                          const float* __restrict__ mat_scale, unsigned n_mats, unsigned kt_per_split) {
    // blockIdx.z selects K stages [z * kt_per_split, ..): with more than one split the partial
    // sums are atomically added into y, which the caller has zeroed.
    __shared__ __align__(16) G3Stage<WT> sm[G3_STAGES];
    __shared__ unsigned xrow[MMA_TOK];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned row0 = blockIdx.x * G2_ROWS;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], (unsigned)MMA_TOK), idx_len - start);
    if (count == 0 || row0 >= N) return;
    if (tid < MMA_TOK) xrow[tid] = min(xidx[start + min(tid, count - 1u)], x_rows - 1u);
    __syncthreads();
    const unsigned ntiles = (count + 7u) / 8u;
    const unsigned kb32 = K >> 5;
    const u8* wbase = qs + e * mat_qs;
    const u8* abase = aux + e * mat_aux;
    const unsigned KT_ALL = K / 64u;
    const unsigned kt0 = blockIdx.z * kt_per_split;
    if (kt0 >= KT_ALL) return;
    const unsigned KT = min(kt_per_split, KT_ALL - kt0);
    const bool split = gridDim.z > 1;

    auto load = [&](unsigned stage, unsigned ktl) {
        G3Stage<WT>& S = sm[stage];
        const unsigned kt = kt0 + ktl;
        const unsigned k0 = kt * 64u;
        if constexpr (WT == 2) {
            for (unsigned c = tid; c < G2_ROWS * 4u; c += blockDim.x) {
                const unsigned r = c >> 2, part = c & 3u;
                cp_async16(S.a + r * G2_ROWB + part * 16u, wbase + (u64)(row0 + r) * K + k0 + part * 16u);
            }
            if (tid < G2_ROWS) cp_async4(S.da + tid * 4u, abase + ((u64)(row0 + tid) * kb32 + (k0 >> 5)) * 2u);
        } else {
            for (unsigned c = tid; c < G2_ROWS * 2u; c += blockDim.x) {
                const unsigned r = c >> 1, part = c & 1u;
                cp_async16(S.a + r * G2_AROWB_FP4 + part * 16u, wbase + (u64)(row0 + r) * (K >> 1) + (k0 >> 1) + part * 16u);
            }
            if (tid < G2_ROWS) cp_async4(S.da + tid * 4u, abase + (u64)(row0 + tid) * (K >> 4) + (k0 >> 4));
        }
        for (unsigned c = tid; c < MMA_TOK * 8u; c += blockDim.x) {
            const unsigned j = c >> 3, part = c & 7u;
            cp_async16(S.b + j * G3_BROWH + part * 8u, xh + (u64)xrow[j] * K + k0 + part * 8u);
        }
    };

    float acc[8][4];
#pragma unroll
    for (unsigned nt = 0; nt < 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
#pragma unroll
    for (unsigned s = 0; s < G3_STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    const unsigned rl = warp * 16u + g, rh = rl + 8u;
    const __half2 k14 = __float2half2_rn(16384.0f), k8 = __float2half2_rn(256.0f);
    for (unsigned kt = 0; kt < KT; ++kt) {
        cp_async_wait<G3_STAGES - 2>();
        __syncthreads();
        if (kt + G3_STAGES - 1 < KT) load((kt + G3_STAGES - 1) % G3_STAGES, kt + G3_STAGES - 1);
        cp_async_commit();
        const G3Stage<WT>& S = sm[kt % G3_STAGES];
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            unsigned a[4];
            if constexpr (WT == 2) {
                const unsigned kb = ks >> 1;
                const __half2 dl = __half2half2(reinterpret_cast<const __half*>(S.da)[rl * 2u + kb]);
                const __half2 dh = __half2half2(reinterpret_cast<const __half*>(S.da)[rh * 2u + kb]);
                const unsigned char* al = S.a + rl * G2_ROWB + ks * 16u + 2u * t;
                const unsigned char* ah = S.a + rh * G2_ROWB + ks * 16u + 2u * t;
                a[0] = hmul2u(i8x2_to_h2(*reinterpret_cast<const unsigned short*>(al)), dl);
                a[1] = hmul2u(i8x2_to_h2(*reinterpret_cast<const unsigned short*>(ah)), dh);
                a[2] = hmul2u(i8x2_to_h2(*reinterpret_cast<const unsigned short*>(al + 8)), dl);
                a[3] = hmul2u(i8x2_to_h2(*reinterpret_cast<const unsigned short*>(ah + 8)), dh);
            } else {
                // Sub-block ks: bytes 2t, 2t+1 give columns 2t, 2t+1 (low nibbles) and 2t+8, 2t+9 (high).
                const unsigned wl = *reinterpret_cast<const unsigned*>(S.a + rl * G2_AROWB_FP4 + ks * 8u + (t >> 1) * 4u) >> ((t & 1u) * 16u);
                const unsigned wh = *reinterpret_cast<const unsigned*>(S.a + rh * G2_AROWB_FP4 + ks * 8u + (t >> 1) * 4u) >> ((t & 1u) * 16u);
                const unsigned short sl = (unsigned short)((unsigned)S.da[rl * 4u + ks] << 7), sh = (unsigned short)((unsigned)S.da[rh * 4u + ks] << 7);
                const __half2 dl = __hmul2(__half2half2(__ushort_as_half(sl)), k8), dh = __hmul2(__half2half2(__ushort_as_half(sh)), k8);
                a[0] = hmul2u(hmul2u(nib2_to_h2(wl & 0x0f0fu), k14), dl);
                a[1] = hmul2u(hmul2u(nib2_to_h2(wh & 0x0f0fu), k14), dh);
                a[2] = hmul2u(hmul2u(nib2_to_h2((wl >> 4) & 0x0f0fu), k14), dl);
                a[3] = hmul2u(hmul2u(nib2_to_h2((wh >> 4) & 0x0f0fu), k14), dh);
            }
#pragma unroll
            for (unsigned nt = 0; nt < 8; ++nt) {
                if (nt < ntiles) {
                    const __half* br = S.b + (nt * 8u + g) * G3_BROWH + ks * 16u + 2u * t;
                    mma_f16(acc[nt], a, *reinterpret_cast<const unsigned*>(br), *reinterpret_cast<const unsigned*>(br + 8));
                }
            }
        }
    }
    cp_async_wait<0>();
    const float ms = mat_scale ? mat_scale[e] : 1.0f;
    const unsigned r_lo = row0 + rl, r_hi = row0 + rh;
#pragma unroll
    for (unsigned nt = 0; nt < 8; ++nt) {
        if (nt < ntiles) {
            const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
            if (j0 < count) {
                float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
                if (split) { atomicAdd(yr + r_lo, acc[nt][0] * ms); atomicAdd(yr + r_hi, acc[nt][2] * ms); }
                else { yr[r_lo] = acc[nt][0] * ms; yr[r_hi] = acc[nt][2] * ms; }
            }
            if (j1 < count) {
                float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
                if (split) { atomicAdd(yr + r_lo, acc[nt][1] * ms); atomicAdd(yr + r_hi, acc[nt][3] * ms); }
                else { yr[r_lo] = acc[nt][1] * ms; yr[r_hi] = acc[nt][3] * ms; }
            }
        }
    }
}

extern "C" __global__ void astral_zero_f32(float* __restrict__ y, u64 n) { PDL_WAIT();
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) y[i] = 0.0f;
}

#define GEMM_MMA3_ENTRY(NAME, WT)                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) NAME(                                                         \
        const u8* __restrict__ qs, const u8* __restrict__ aux, const __half* __restrict__ xh,                       \
        float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux, const unsigned* __restrict__ items, \
        const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx, unsigned idx_len, unsigned x_rows,     \
        unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats, unsigned kt_per_split) { PDL_WAIT();             \
        gemm_mma3<WT>(qs, aux, xh, y, N, K, mat_qs, mat_aux, items, xidx, yidx, idx_len, x_rows, y_rows, mat_scale, \
                      n_mats, kt_per_split);                                                                        \
    }
GEMM_MMA3_ENTRY(astral_gemm_q8_0_mma3, 2)
GEMM_MMA3_ENTRY(astral_gemm_nvfp4_mma3, 3)

// =============================================================================================
// NVFP4 decode GEMV v2: the block's int8 activations (K bytes + K/32 scales) are staged in
// shared memory once; weights stream with ld.global.nc.L1::no_allocate (no L1 pollution).
// R rows per warp (G = 32 / R lanes per row), U 16-byte weight loads in flight per lane.
// Same parameters as GEMV_Q_ENTRY; dynamic shared memory = K + K/8 bytes.
__device__ __forceinline__ uint4 ld_stream16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ unsigned ld_stream_u16(const void* p) {
    unsigned short r;
    asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];\n" : "=h"(r) : "l"(p));
    return r;
}

template <unsigned R, unsigned U>
__device__ __forceinline__ void gemv_fp4_smem(const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,
                                              const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs,
                                              u64 mat_aux, unsigned xq_shared, const int* __restrict__ ids,
                                              const float* __restrict__ mat_scale, unsigned n_mats) {
    extern __shared__ uint4 smem4[];
    constexpr unsigned G = 32u / R;
    const unsigned s = blockIdx.y;
    const unsigned sx = xq_shared ? 0u : s;
    // Stage this slot's activations: K int8 then K/32 float scales.
    uint4* sq = smem4;
    float* sd = reinterpret_cast<float*>(reinterpret_cast<signed char*>(smem4) + K);
    {
        const uint4* gq = reinterpret_cast<const uint4*>(xq + (u64)sx * K);
        for (unsigned i = threadIdx.x; i < (K >> 4); i += blockDim.x) sq[i] = __ldg(gq + i);
        const float* gd = xd + (u64)sx * (K >> 5);
        for (unsigned i = threadIdx.x; i < (K >> 5); i += blockDim.x) sd[i] = __ldg(gd + i);
    }
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned n = (blockIdx.x * (blockDim.x >> 5) + warp) * R + lane / G;
    if (n >= N) return;
    const unsigned sub = lane % G, mask = (G == 32u ? 0xffffffffu : ((1u << G) - 1u) << (lane & ~(G - 1u)));
    const unsigned e = ids ? min((unsigned)ids[s], n_mats - 1u) : s;
    const u8* w = qs + e * mat_qs + (u64)n * (K >> 1);
    const u8* wd = aux + e * mat_aux + (u64)n * (K >> 4);
    const unsigned units = K >> 5;
    float acc = 0.0f;
    for (unsigned base = 0; base < units; base += G * U) {
        uint4 wv[U];
        unsigned sc[U];
#pragma unroll
        for (unsigned i = 0; i < U; ++i) {
            const unsigned u = base + sub + G * i;
            if (u < units) {
                wv[i] = ld_stream16(w + (u64)u * 16u);
                sc[i] = ld_stream_u16(wd + (u64)u * 2u);
            }
        }
#pragma unroll
        for (unsigned i = 0; i < U; ++i) {
            const unsigned u = base + sub + G * i;
            if (u < units) {
                const uint4 x0 = sq[2u * u], x1 = sq[2u * u + 1u];
                const int2 a0 = table16((int)wv[i].x, k_e2m1x2), a1 = table16((int)wv[i].y, k_e2m1x2);
                const int2 b0 = table16((int)wv[i].z, k_e2m1x2), b1 = table16((int)wv[i].w, k_e2m1x2);
                int s0 = __dp4a(a0.x, (int)x0.x, 0);
                s0 = __dp4a(a1.x, (int)x0.y, s0);
                s0 = __dp4a(a0.y, (int)x0.z, s0);
                s0 = __dp4a(a1.y, (int)x0.w, s0);
                int s1 = __dp4a(b0.x, (int)x1.x, 0);
                s1 = __dp4a(b1.x, (int)x1.y, s1);
                s1 = __dp4a(b0.y, (int)x1.z, s1);
                s1 = __dp4a(b1.y, (int)x1.w, s1);
                const float2 d2 = __half22float2(__halves2half2(__ushort_as_half((unsigned short)((sc[i] & 0xffu) << 7)),
                                                                __ushort_as_half((unsigned short)((sc[i] >> 8) << 7))));
                acc = fmaf(fmaf((float)s0, d2.x, (float)s1 * d2.y), sd[u], acc);
            }
        }
    }
    acc *= 128.0f;
#pragma unroll
    for (unsigned o = G / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(mask, acc, o);
    if (sub == 0) y[(u64)s * N + n] = mat_scale ? acc * mat_scale[e] : acc;
}

#define GEMV_FP4_SMEM_ENTRY(NAME, R, U)                                                                            \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                        \
        const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq,                 \
        const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux,      \
        unsigned xq_shared, const int* __restrict__ ids, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();   \
        gemv_fp4_smem<R, U>(qs, aux, xq, xd, y, N, K, mat_qs, mat_aux, xq_shared, ids, mat_scale, n_mats);         \
    }
GEMV_FP4_SMEM_ENTRY(astral_gemv_nvfp4_s1, 1, 6)
GEMV_FP4_SMEM_ENTRY(astral_gemv_nvfp4_s2, 2, 6)
GEMV_FP4_SMEM_ENTRY(astral_gemv_nvfp4_s4, 4, 6)
GEMV_FP4_SMEM_ENTRY(astral_gemv_nvfp4_s1u12, 1, 12)   // long rows (K >= 8192)
GEMV_FP4_SMEM_ENTRY(astral_gemv_nvfp4_s1u9, 1, 9)

// =============================================================================================
// Tensor-core GEMM v4 (prefill, dense layers): same dequant and maths as v3, bigger tiles.
// Block = 8 warps = 128 weight rows x 128 tokens; each warp owns 16 rows x all 128 tokens, so a
// dequantised A fragment feeds 16 MMAs, and B fragments come from ldmatrix.x4 (2 n-tiles per
// instruction) on an XOR-swizzled activation tile (16-byte chunk c of token row r stored at
// c ^ (r & 7): conflict-free without padding). 2-stage cp.async pipeline, 64 values of K per
// stage. Work items as v3 but up to 128 tokens each. N % 128 == 0, K % 64 == 0.
// grid = (N / 128, n_items), 256 threads.
#define G4_ROWS 128
#define G4_TOK 128
#define G4_STAGES 2

template <int WT>
struct G4Stage {
    static constexpr unsigned AROW = WT == 2 ? 64u : 32u;   // packed A bytes per row per stage
    unsigned char a[G4_ROWS * AROW];
    unsigned char da[G4_ROWS * 4];
    __half b[G4_TOK * 64];                                    // swizzled, 128 bytes per token row
};

__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

template <int WT>
__device__ __forceinline__ void gemm_mma4(const u8* __restrict__ qs, const u8* __restrict__ aux, const __half* __restrict__ xh,
                                          float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux,
                                          const unsigned* __restrict__ items, const unsigned* __restrict__ xidx,
                                          const unsigned* __restrict__ yidx, unsigned idx_len, unsigned x_rows, unsigned y_rows,
                                          const float* __restrict__ mat_scale, unsigned n_mats) {
    __shared__ __align__(128) G4Stage<WT> sm[G4_STAGES];
    __shared__ unsigned xrow[G4_TOK];
    constexpr unsigned AROW = G4Stage<WT>::AROW;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned row0 = blockIdx.x * G4_ROWS;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], (unsigned)G4_TOK), idx_len - start);
    if (count == 0 || row0 >= N) return;
    if (tid < G4_TOK) xrow[tid] = min(xidx[start + min(tid, count - 1u)], x_rows - 1u);
    __syncthreads();
    const unsigned npairs = (count + 15u) / 16u;   // pairs of n-tiles with real tokens
    const unsigned kb32 = K >> 5;
    const u8* wbase = qs + e * mat_qs;
    const u8* abase = aux + e * mat_aux;
    const unsigned KT = K / 64u;

    auto load = [&](unsigned stage, unsigned kt) {
        G4Stage<WT>& S = sm[stage];
        const unsigned k0 = kt * 64u;
        constexpr unsigned CPR = AROW / 16u;   // 16-byte chunks per A row
        for (unsigned c = tid; c < G4_ROWS * CPR; c += blockDim.x) {
            const unsigned r = c / CPR, part = c % CPR;
            const u8* src = WT == 2 ? wbase + (u64)(row0 + r) * K + k0 + part * 16u : wbase + (u64)(row0 + r) * (K >> 1) + (k0 >> 1) + part * 16u;
            cp_async16(S.a + r * AROW + part * 16u, src);
        }
        if (tid < G4_ROWS) {
            const u8* src = WT == 2 ? abase + ((u64)(row0 + tid) * kb32 + (k0 >> 5)) * 2u : abase + (u64)(row0 + tid) * (K >> 4) + (k0 >> 4);
            cp_async4(S.da + tid * 4u, src);
        }
        for (unsigned c = tid; c < G4_TOK * 8u; c += blockDim.x) {
            const unsigned j = c >> 3, part = c & 7u;
            cp_async16(S.b + j * 64u + ((part ^ (j & 7u)) * 8u), xh + (u64)xrow[j] * K + k0 + part * 8u);
        }
    };

    float acc[16][4];
#pragma unroll
    for (unsigned nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    load(0, 0);
    cp_async_commit();
    const unsigned rl = warp * 16u + g, rh = rl + 8u;
    const __half2 k14 = __float2half2_rn(16384.0f), k8 = __float2half2_rn(256.0f);
    // ldmatrix.x4 lane roles: matrix q = lane / 8 -> (n-tile offset q / 2, k half q % 2), row lane % 8.
    const unsigned lm_q = lane >> 3, lm_r = lane & 7u;
    for (unsigned kt = 0; kt < KT; ++kt) {
        if (kt + 1 < KT) load((kt + 1) % G4_STAGES, kt + 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const G4Stage<WT>& S = sm[kt % G4_STAGES];
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            unsigned a[4];
            if constexpr (WT == 2) {
                const unsigned kb = ks >> 1;
                const __half2 dl = __half2half2(reinterpret_cast<const __half*>(S.da)[rl * 2u + kb]);
                const __half2 dh = __half2half2(reinterpret_cast<const __half*>(S.da)[rh * 2u + kb]);
                const unsigned char* al = S.a + rl * AROW + ks * 16u + 2u * t;
                const unsigned char* ah = S.a + rh * AROW + ks * 16u + 2u * t;
                a[0] = hmul2u(i8x2_to_h2(*reinterpret_cast<const unsigned short*>(al)), dl);
                a[1] = hmul2u(i8x2_to_h2(*reinterpret_cast<const unsigned short*>(ah)), dh);
                a[2] = hmul2u(i8x2_to_h2(*reinterpret_cast<const unsigned short*>(al + 8)), dl);
                a[3] = hmul2u(i8x2_to_h2(*reinterpret_cast<const unsigned short*>(ah + 8)), dh);
            } else {
                const unsigned wl = *reinterpret_cast<const unsigned*>(S.a + rl * AROW + ks * 8u + (t >> 1) * 4u) >> ((t & 1u) * 16u);
                const unsigned wh = *reinterpret_cast<const unsigned*>(S.a + rh * AROW + ks * 8u + (t >> 1) * 4u) >> ((t & 1u) * 16u);
                const unsigned short sl = (unsigned short)((unsigned)S.da[rl * 4u + ks] << 7), sh = (unsigned short)((unsigned)S.da[rh * 4u + ks] << 7);
                const __half2 dl = __hmul2(__half2half2(__ushort_as_half(sl)), k8), dh = __hmul2(__half2half2(__ushort_as_half(sh)), k8);
                a[0] = hmul2u(hmul2u(nib2_to_h2(wl & 0x0f0fu), k14), dl);
                a[1] = hmul2u(hmul2u(nib2_to_h2(wh & 0x0f0fu), k14), dh);
                a[2] = hmul2u(hmul2u(nib2_to_h2((wl >> 4) & 0x0f0fu), k14), dl);
                a[3] = hmul2u(hmul2u(nib2_to_h2((wh >> 4) & 0x0f0fu), k14), dh);
            }
#pragma unroll
            for (unsigned np = 0; np < 8; ++np) {
                if (np < npairs) {
                    const unsigned tok = (np * 2u + (lm_q >> 1)) * 8u + lm_r;
                    const unsigned chunk = ks * 2u + (lm_q & 1u);
                    unsigned b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3, S.b + tok * 64u + ((chunk ^ (tok & 7u)) * 8u));
                    mma_f16(acc[2 * np], a, b0, b1);
                    mma_f16(acc[2 * np + 1], a, b2, b3);
                }
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();
    const float ms = mat_scale ? mat_scale[e] : 1.0f;
    const unsigned r_lo = row0 + rl, r_hi = row0 + rh;
#pragma unroll
    for (unsigned nt = 0; nt < 16; ++nt) {
        const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
        if (j0 < count) {
            float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
            yr[r_lo] = acc[nt][0] * ms;
            yr[r_hi] = acc[nt][2] * ms;
        }
        if (j1 < count) {
            float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
            yr[r_lo] = acc[nt][1] * ms;
            yr[r_hi] = acc[nt][3] * ms;
        }
    }
}

#define GEMM_MMA4_ENTRY(NAME, WT)                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                         \
        const u8* __restrict__ qs, const u8* __restrict__ aux, const __half* __restrict__ xh,                       \
        float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux, const unsigned* __restrict__ items, \
        const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx, unsigned idx_len, unsigned x_rows,     \
        unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();                                    \
        gemm_mma4<WT>(qs, aux, xh, y, N, K, mat_qs, mat_aux, items, xidx, yidx, idx_len, x_rows, y_rows, mat_scale, \
                      n_mats);                                                                                      \
    }
// (Q8_0 would need 49 KB of static shared memory: dense Q8_0 prefill stays on v3.)
GEMM_MMA4_ENTRY(astral_gemm_nvfp4_mma4, 3)

// =============================================================================================
// Tensor-core GEMM v5 (prefill, NVFP4 in the fragment layout of astral_quant::nvfp4_frag_layout).
// Block = 8 warps = 128 rows x 128 tokens; warp w owns row tile w (16 rows) x all 128 tokens.
// Per 64-column stage a lane reads one 16-byte word quad (its A fragments for 4 k-steps) and 8
// scale bytes; each k-step decodes 4 f16 pairs with 2-3 bit ops each and ONE hmul2 by the
// UE4M3 scale -- e2m1 * 2^-14 * scale is exact in f16 (an integer multiple of 2^-24, far below
// the f16 maximum) -- and the 2^14 is applied once to the f32 result. B as in v4.
// grid = (N / 128, n_items), 256 threads; N % 128 == 0, K % 64 == 0, items of <= 128 tokens.
#define G5_ROWS 128
#define G5_TOK 128

struct G5Stage {
    uint4 a[G5_ROWS / 16 * 32];   // 8 row tiles x 32 lanes x 16 bytes
    uint2 sc[G5_ROWS / 16 * 8];   // 8 row tiles x 8 groups x 8 bytes
    __half b[G5_TOK * 64];        // swizzled, 128 bytes per token row
};

__device__ __forceinline__ unsigned fp4pair(unsigned w) {   // codes at bits 12-15 and 28-31 -> f16 pair * 2^-14
    return (w & 0x80008000u) | ((w >> 3) & 0x0e000e00u);
}

extern "C" __global__ void __launch_bounds__(256) astral_gemm_nvfp4_mma5(
    const u8* __restrict__ frag, const u8* __restrict__ fscale, const __half* __restrict__ xh, float* __restrict__ y, unsigned N,
    unsigned K, const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx,
    unsigned idx_len, unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats, unsigned kc_per_split) { PDL_WAIT();
    // blockIdx.z selects K stages [z * kc_per_split, ..); with more than one split the partial sums
    // are atomically added into y, which the caller has zeroed.
    __shared__ __align__(128) G5Stage sm[2];
    __shared__ unsigned xrow[G5_TOK];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, t = lane & 3u, g = lane >> 2;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], (unsigned)G5_TOK), idx_len - start);
    const unsigned row0 = blockIdx.x * G5_ROWS;
    if (count == 0 || row0 >= N) return;
    if (tid < G5_TOK) xrow[tid] = min(xidx[start + min(tid, count - 1u)], x_rows - 1u);
    __syncthreads();
    const unsigned KC = K / 64u, NT = N / 16u;
    const unsigned kc0 = blockIdx.z * kc_per_split;
    if (kc0 >= KC) return;
    const unsigned KCS = min(kc_per_split, KC - kc0);
    const bool split = gridDim.z > 1;
    const uint4* fa = reinterpret_cast<const uint4*>(frag) + ((u64)e * NT + blockIdx.x * 8u) * KC * 32u;
    const uint2* fs = reinterpret_cast<const uint2*>(fscale) + ((u64)e * NT + blockIdx.x * 8u) * KC * 8u;

    auto load = [&](unsigned stage, unsigned kcl) {
        G5Stage& S = sm[stage];
        const unsigned kc = kc0 + kcl;
        // A: thread tid -> row tile tid / 32, lane tid % 32 (one 16-byte copy each).
        cp_async16(&S.a[tid], fa + ((u64)(tid >> 5) * KC + kc) * 32u + (tid & 31u));
        if (tid < 32) cp_async16(reinterpret_cast<uint4*>(S.sc) + tid, reinterpret_cast<const uint4*>(fs + ((u64)(tid >> 2) * KC + kc) * 8u) + (tid & 3u));
        const unsigned k0 = kc * 64u;
        for (unsigned c = tid; c < G5_TOK * 8u; c += blockDim.x) {
            const unsigned j = c >> 3, part = c & 7u;
            cp_async16(S.b + j * 64u + ((part ^ (j & 7u)) * 8u), xh + (u64)xrow[j] * K + k0 + part * 8u);
        }
    };

    float acc[16][4];
#pragma unroll
    for (unsigned nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    load(0, 0);
    cp_async_commit();
    // ldmatrix.x4 lane roles: matrix q = lane / 8 -> (n-tile offset q / 2, k half q % 2), row lane % 8.
    const unsigned lm_q = lane >> 3, lm_r = lane & 7u;
    const unsigned boff = ((lm_q >> 1) * 8u + lm_r) * 64u;   // halves: token row of this lane, pair 0
    unsigned bx[4];
#pragma unroll
    for (unsigned ks = 0; ks < 4; ++ks) bx[ks] = (((ks * 2u + (lm_q & 1u)) ^ lm_r) * 8u);
    const __half2 k8 = __float2half2_rn(256.0f);
    for (unsigned kc = 0; kc < KCS; ++kc) {
        if (kc + 1 < KCS) load((kc + 1) & 1u, kc + 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const G5Stage& S = sm[kc & 1u];
        const uint4 wq = S.a[warp * 32u + lane];
        const uint2 sb = S.sc[warp * 8u + g];
        const unsigned w4[4] = {wq.x, wq.y, wq.z, wq.w};
        const unsigned s4[2] = {sb.x, sb.y};
        const __half* bbase = S.b + boff;
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            // Scales of (row g, ks) and (row g + 8, ks) as an exact f16 pair.
            const unsigned sbytes = (s4[ks >> 1] >> ((ks & 1u) * 16u)) & 0xffffu;
            const unsigned sbits = ((sbytes & 0xffu) << 7) | ((sbytes >> 8) << 23);
            const __half2 s2 = __hmul2(*reinterpret_cast<const __half2*>(&sbits), k8);
            const __half2 dl = __low2half2(s2), dh = __high2half2(s2);
            const unsigned w = w4[ks];
            unsigned a[4];
            a[0] = hmul2u(fp4pair(w), dl);
            a[1] = hmul2u(fp4pair(w << 4), dh);
            a[2] = hmul2u(fp4pair(w << 8), dl);
            a[3] = hmul2u(fp4pair(w << 12), dh);
#pragma unroll
            for (unsigned np = 0; np < 8; ++np) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, bbase + np * 16u * 64u + bx[ks]);
                mma_f16(acc[2 * np], a, b0, b1);
                mma_f16(acc[2 * np + 1], a, b2, b3);
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();
    const float ms = (mat_scale ? mat_scale[e] : 1.0f) * 16384.0f;
    const unsigned r_lo = row0 + warp * 16u + g, r_hi = r_lo + 8u;
#pragma unroll
    for (unsigned nt = 0; nt < 16; ++nt) {
        const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
        if (j0 < count) {
            float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
            if (split) { atomicAdd(yr + r_lo, acc[nt][0] * ms); atomicAdd(yr + r_hi, acc[nt][2] * ms); }
            else { yr[r_lo] = acc[nt][0] * ms; yr[r_hi] = acc[nt][2] * ms; }
        }
        if (j1 < count) {
            float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
            if (split) { atomicAdd(yr + r_lo, acc[nt][1] * ms); atomicAdd(yr + r_hi, acc[nt][3] * ms); }
            else { yr[r_lo] = acc[nt][1] * ms; yr[r_hi] = acc[nt][3] * ms; }
        }
    }
}

// v6: v5 with a 32-row x 64-token warp tile (4 x 2 warps per 128 x 128 block). Each B fragment
// (ldmatrix.x4 = 2 n-tiles) feeds 4 MMAs (2 row tiles), halving shared-memory traffic per MMA;
// A fragments are decoded for 2 row tiles. Same parameters, layout and limits as v5.
extern "C" __global__ void __launch_bounds__(256) astral_gemm_nvfp4_mma6(
    const u8* __restrict__ frag, const u8* __restrict__ fscale, const __half* __restrict__ xh, float* __restrict__ y, unsigned N,
    unsigned K, const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx,
    unsigned idx_len, unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();
    __shared__ __align__(128) G5Stage sm[2];
    __shared__ unsigned xrow[G5_TOK];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, t = lane & 3u, g = lane >> 2;
    const unsigned wm = warp & 3u, wn = warp >> 2;   // row tiles 2wm, 2wm+1; tokens 64wn..64wn+63
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], (unsigned)G5_TOK), idx_len - start);
    const unsigned row0 = blockIdx.x * G5_ROWS;
    if (count == 0 || row0 >= N) return;
    if (tid < G5_TOK) xrow[tid] = min(xidx[start + min(tid, count - 1u)], x_rows - 1u);
    __syncthreads();
    const unsigned KC = K / 64u, NT = N / 16u;
    const uint4* fa = reinterpret_cast<const uint4*>(frag) + ((u64)e * NT + blockIdx.x * 8u) * KC * 32u;
    const uint2* fs = reinterpret_cast<const uint2*>(fscale) + ((u64)e * NT + blockIdx.x * 8u) * KC * 8u;
    auto load = [&](unsigned stage, unsigned kc) {
        G5Stage& S = sm[stage];
        cp_async16(&S.a[tid], fa + ((u64)(tid >> 5) * KC + kc) * 32u + (tid & 31u));
        if (tid < 32) cp_async16(reinterpret_cast<uint4*>(S.sc) + tid, reinterpret_cast<const uint4*>(fs + ((u64)(tid >> 2) * KC + kc) * 8u) + (tid & 3u));
        const unsigned k0 = kc * 64u;
        for (unsigned c = tid; c < G5_TOK * 8u; c += blockDim.x) {
            const unsigned j = c >> 3, part = c & 7u;
            cp_async16(S.b + j * 64u + ((part ^ (j & 7u)) * 8u), xh + (u64)xrow[j] * K + k0 + part * 8u);
        }
    };
    float acc[2][8][4];
#pragma unroll
    for (unsigned m = 0; m < 2; ++m)
#pragma unroll
        for (unsigned nt = 0; nt < 8; ++nt) acc[m][nt][0] = acc[m][nt][1] = acc[m][nt][2] = acc[m][nt][3] = 0.0f;
    load(0, 0);
    cp_async_commit();
    const unsigned lm_q = lane >> 3, lm_r = lane & 7u;
    const unsigned boff = (wn * 64u + (lm_q >> 1) * 8u + lm_r) * 64u;
    unsigned bx[4];
#pragma unroll
    for (unsigned ks = 0; ks < 4; ++ks) bx[ks] = (((ks * 2u + (lm_q & 1u)) ^ lm_r) * 8u);
    const __half2 k8 = __float2half2_rn(256.0f);
    for (unsigned kc = 0; kc < KC; ++kc) {
        if (kc + 1 < KC) load((kc + 1) & 1u, kc + 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const G5Stage& S = sm[kc & 1u];
        uint4 wq[2];
        uint2 sb[2];
#pragma unroll
        for (unsigned m = 0; m < 2; ++m) {
            wq[m] = S.a[(2u * wm + m) * 32u + lane];
            sb[m] = S.sc[(2u * wm + m) * 8u + g];
        }
        const __half* bbase = S.b + boff;
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            unsigned a[2][4];
#pragma unroll
            for (unsigned m = 0; m < 2; ++m) {
                const unsigned s2w = (ks < 2 ? sb[m].x : sb[m].y);
                const unsigned sbytes = (s2w >> ((ks & 1u) * 16u)) & 0xffffu;
                const unsigned sbits = ((sbytes & 0xffu) << 7) | ((sbytes >> 8) << 23);
                const __half2 s2 = __hmul2(*reinterpret_cast<const __half2*>(&sbits), k8);
                const __half2 dl = __low2half2(s2), dh = __high2half2(s2);
                const unsigned w = ks == 0 ? wq[m].x : ks == 1 ? wq[m].y : ks == 2 ? wq[m].z : wq[m].w;
                a[m][0] = hmul2u(fp4pair(w), dl);
                a[m][1] = hmul2u(fp4pair(w << 4), dh);
                a[m][2] = hmul2u(fp4pair(w << 8), dl);
                a[m][3] = hmul2u(fp4pair(w << 12), dh);
            }
#pragma unroll
            for (unsigned np = 0; np < 4; ++np) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, bbase + np * 16u * 64u + bx[ks]);
#pragma unroll
                for (unsigned m = 0; m < 2; ++m) {
                    mma_f16(acc[m][2 * np], a[m], b0, b1);
                    mma_f16(acc[m][2 * np + 1], a[m], b2, b3);
                }
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();
    const float ms = (mat_scale ? mat_scale[e] : 1.0f) * 16384.0f;
#pragma unroll
    for (unsigned m = 0; m < 2; ++m) {
        const unsigned r_lo = row0 + (2u * wm + m) * 16u + g, r_hi = r_lo + 8u;
#pragma unroll
        for (unsigned nt = 0; nt < 8; ++nt) {
            const unsigned j0 = wn * 64u + nt * 8u + 2u * t, j1 = j0 + 1u;
            if (j0 < count) {
                float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
                yr[r_lo] = acc[m][nt][0] * ms;
                yr[r_hi] = acc[m][nt][2] * ms;
            }
            if (j1 < count) {
                float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
                yr[r_lo] = acc[m][nt][1] * ms;
                yr[r_hi] = acc[m][nt][3] * ms;
            }
        }
    }
}

// =============================================================================================
// Native MXFP4 decode GEMV: values in the NVFP4 nibble layout (`qs`, K/2 bytes per row), one
// E8M0 scale byte per 32 values (`es`, K/32 bytes per row: the checkpoint's own scales). Same
// structure as gemv_fp4_smem (activations staged in shared memory, weights streamed): one scale
// per 16-byte unit, 2^(e - 128) from the exponent bits (the extra 2^-1 undoes the doubled
// E2M1 table). Same parameters as GEMV_Q_ENTRY (aux = es, mat_aux = n * K / 32); mat_scale
// is ignored (the scales are absolute).
__device__ __forceinline__ float warp_sum_f(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ unsigned ld_stream_u8(const void* p) {
    unsigned short r;
    asm volatile("ld.global.nc.L1::no_allocate.u8 %0, [%1];\n" : "=h"(r) : "l"(p));
    return r;
}

template <unsigned U>
__device__ __forceinline__ void gemv_mx4_smem(const u8* __restrict__ qs, const u8* __restrict__ es, const signed char* __restrict__ xq,
                                              const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs,
                                              u64 mat_es, unsigned xq_shared, const int* __restrict__ ids, unsigned n_mats) {
    extern __shared__ uint4 smem4[];
    const unsigned s = blockIdx.y;
    const unsigned sx = xq_shared ? 0u : s;
    uint4* sq = smem4;
    float* sd = reinterpret_cast<float*>(reinterpret_cast<signed char*>(smem4) + K);
    {
        const uint4* gq = reinterpret_cast<const uint4*>(xq + (u64)sx * K);
        for (unsigned i = threadIdx.x; i < (K >> 4); i += blockDim.x) sq[i] = __ldg(gq + i);
        const float* gd = xd + (u64)sx * (K >> 5);
        for (unsigned i = threadIdx.x; i < (K >> 5); i += blockDim.x) sd[i] = __ldg(gd + i);
    }
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned n = blockIdx.x * (blockDim.x >> 5) + warp;
    if (n >= N) return;
    const unsigned e = ids ? min((unsigned)ids[s], n_mats - 1u) : s;
    const u8* w = qs + e * mat_qs + (u64)n * (K >> 1);
    const u8* we = es + e * mat_es + (u64)n * (K >> 5);
    const unsigned units = K >> 5;
    float acc = 0.0f;
    for (unsigned base = 0; base < units; base += 32u * U) {
        uint4 wv[U];
        unsigned sc[U];
#pragma unroll
        for (unsigned i = 0; i < U; ++i) {
            const unsigned u = base + lane + 32u * i;
            if (u < units) {
                wv[i] = ld_stream16(w + (u64)u * 16u);
                sc[i] = ld_stream_u8(we + u);
            }
        }
#pragma unroll
        for (unsigned i = 0; i < U; ++i) {
            const unsigned u = base + lane + 32u * i;
            if (u < units) {
                const uint4 x0 = sq[2u * u], x1 = sq[2u * u + 1u];
                const int2 a0 = table16((int)wv[i].x, k_e2m1x2), a1 = table16((int)wv[i].y, k_e2m1x2);
                const int2 b0 = table16((int)wv[i].z, k_e2m1x2), b1 = table16((int)wv[i].w, k_e2m1x2);
                int t = __dp4a(a0.x, (int)x0.x, 0);
                t = __dp4a(a1.x, (int)x0.y, t);
                t = __dp4a(a0.y, (int)x0.z, t);
                t = __dp4a(a1.y, (int)x0.w, t);
                t = __dp4a(b0.x, (int)x1.x, t);
                t = __dp4a(b1.x, (int)x1.y, t);
                t = __dp4a(b0.y, (int)x1.z, t);
                t = __dp4a(b1.y, (int)x1.w, t);
                const float f = sc[i] >= 2u ? __int_as_float((int)((sc[i] - 1u) << 23)) : 0.0f;   // 2^(e - 128)
                acc = fmaf((float)t, f * sd[u], acc);
            }
        }
    }
    acc = warp_sum_f(acc);
    if (lane == 0) y[(u64)s * N + n] = acc;
}

extern "C" __global__ void __launch_bounds__(256) astral_gemv_mx4_s1(
    const u8* __restrict__ qs, const u8* __restrict__ es, const signed char* __restrict__ xq, const float* __restrict__ xd,
    float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_es, unsigned xq_shared, const int* __restrict__ ids,
    const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();
    (void)mat_scale;
    gemv_mx4_smem<6>(qs, es, xq, xd, y, N, K, mat_qs, mat_es, xq_shared, ids, n_mats);
}

// =============================================================================================
// Tensor-core GEMM v7 (prefill, NVFP4 fragment layout): Hopper warpgroup MMA (wgmma, sm_90a).
// Block = 2 warpgroups = 128 rows x 128 tokens. A (weights) is dequantised exactly as in v5 into
// the per-warp m16n8k16 A-fragment registers, which is also wgmma's register-A layout; B (f16
// activations) is read by the tensor cores straight from shared memory through a descriptor
// (K-major, 128-byte swizzle: the v5 tile layout, 1024-byte aligned). One
// wgmma.m64n128k16 per warpgroup per 16-column step: no ldmatrix, no per-warp MMA issue.
// Same parameters, layout and limits as v5. sm_90a only (traps elsewhere).
struct __align__(1024) G7Stage {
    __half b[G5_TOK * 64];        // swizzled, 128 bytes per token row (1024-byte aligned)
    uint4 a[G5_ROWS / 16 * 32];
    uint2 sc[G5_ROWS / 16 * 8];
};

__device__ __forceinline__ unsigned long long wgmma_desc(const void* smem) {
    const unsigned addr = (unsigned)__cvta_generic_to_shared(smem);
    unsigned long long d = (unsigned long long)((addr & 0x3ffffu) >> 4);
    d |= 1ull << 16;                    // leading byte offset (unused for swizzled K-major)
    d |= (unsigned long long)(1024 >> 4) << 32;   // stride byte offset: 8 rows x 128 B
    d |= 1ull << 62;                    // 128-byte swizzle
    return d;
}

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], const unsigned (&a)[4], unsigned long long db) {
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 0;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
#endif

extern "C" __global__ void __launch_bounds__(256) astral_gemm_nvfp4_wgmma(
    const u8* __restrict__ frag, const u8* __restrict__ fscale, const __half* __restrict__ xh, float* __restrict__ y, unsigned N,
    unsigned K, const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx,
    unsigned idx_len, unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats, unsigned kc_per_split) { PDL_WAIT();
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    __shared__ G7Stage sm[2];
    __shared__ unsigned xrow[G5_TOK];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, t = lane & 3u, g = lane >> 2;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], (unsigned)G5_TOK), idx_len - start);
    const unsigned row0 = blockIdx.x * G5_ROWS;
    if (count == 0 || row0 >= N) return;
    if (tid < G5_TOK) xrow[tid] = min(xidx[start + min(tid, count - 1u)], x_rows - 1u);
    __syncthreads();
    const unsigned KC = K / 64u, NT = N / 16u;
    const unsigned kc0 = blockIdx.z * kc_per_split;
    if (kc0 >= KC) return;
    const unsigned KCS = min(kc_per_split, KC - kc0);
    const bool split = gridDim.z > 1;
    const uint4* fa = reinterpret_cast<const uint4*>(frag) + ((u64)e * NT + blockIdx.x * 8u) * KC * 32u;
    const uint2* fs = reinterpret_cast<const uint2*>(fscale) + ((u64)e * NT + blockIdx.x * 8u) * KC * 8u;
    auto load = [&](unsigned stage, unsigned kcl) {
        G7Stage& S = sm[stage];
        const unsigned kc = kc0 + kcl;
        cp_async16(&S.a[tid], fa + ((u64)(tid >> 5) * KC + kc) * 32u + (tid & 31u));
        if (tid < 32) cp_async16(reinterpret_cast<uint4*>(S.sc) + tid, reinterpret_cast<const uint4*>(fs + ((u64)(tid >> 2) * KC + kc) * 8u) + (tid & 3u));
        const unsigned k0 = kc * 64u;
        for (unsigned c = tid; c < G5_TOK * 8u; c += blockDim.x) {
            const unsigned j = c >> 3, part = c & 7u;
            cp_async16(S.b + j * 64u + ((part ^ (j & 7u)) * 8u), xh + (u64)xrow[j] * K + k0 + part * 8u);
        }
    };
    float d[64];
#pragma unroll
    for (unsigned i = 0; i < 64; ++i) d[i] = 0.0f;
    load(0, 0);
    cp_async_commit();
    const __half2 k8 = __float2half2_rn(256.0f);
    for (unsigned kc = 0; kc < KCS; ++kc) {
        if (kc + 1 < KCS) load((kc + 1) & 1u, kc + 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // cp.async data -> async proxy (wgmma)
        const G7Stage& S = sm[kc & 1u];
        const uint4 wq = S.a[warp * 32u + lane];
        const uint2 sb = S.sc[warp * 8u + g];
        const unsigned w4[4] = {wq.x, wq.y, wq.z, wq.w};
        const unsigned s4[2] = {sb.x, sb.y};
        unsigned a[4][4];
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            const unsigned sbytes = (s4[ks >> 1] >> ((ks & 1u) * 16u)) & 0xffffu;
            const unsigned sbits = ((sbytes & 0xffu) << 7) | ((sbytes >> 8) << 23);
            const __half2 s2 = __hmul2(*reinterpret_cast<const __half2*>(&sbits), k8);
            const __half2 dl = __low2half2(s2), dh = __high2half2(s2);
            const unsigned w = w4[ks];
            a[ks][0] = hmul2u(fp4pair(w), dl);
            a[ks][1] = hmul2u(fp4pair(w << 4), dh);
            a[ks][2] = hmul2u(fp4pair(w << 8), dl);
            a[ks][3] = hmul2u(fp4pair(w << 12), dh);
        }
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const unsigned long long db = wgmma_desc(S.b);
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) wgmma_m64n128k16(d, a[ks], db + (unsigned long long)(ks * 2u));   // +32 bytes per k16
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        __syncthreads();
    }
    cp_async_wait<0>();
    const float ms = (mat_scale ? mat_scale[e] : 1.0f) * 16384.0f;
    const unsigned r_lo = row0 + warp * 16u + g, r_hi = r_lo + 8u;
#pragma unroll
    for (unsigned nt = 0; nt < 16; ++nt) {
        const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
        if (j0 < count) {
            float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
            if (split) { atomicAdd(yr + r_lo, d[4 * nt] * ms); atomicAdd(yr + r_hi, d[4 * nt + 2] * ms); }
            else { yr[r_lo] = d[4 * nt] * ms; yr[r_hi] = d[4 * nt + 2] * ms; }
        }
        if (j1 < count) {
            float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
            if (split) { atomicAdd(yr + r_lo, d[4 * nt + 1] * ms); atomicAdd(yr + r_hi, d[4 * nt + 3] * ms); }
            else { yr[r_lo] = d[4 * nt + 1] * ms; yr[r_hi] = d[4 * nt + 3] * ms; }
        }
    }
#else
    __trap();
#endif
}

// v8: v7 with one wgmma group in flight: 3 shared-memory stages (dynamic, 3 x 21 KB), A
// registers double-buffered (the K loop unrolled by 2), and wgmma.wait_group 1 before a stage
// is reused, so the next stage's dequant overlaps the tensor cores. Same parameters as v7.
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
__device__ __forceinline__ void g8_dequant(const G7Stage& S, unsigned warp, unsigned lane, unsigned g, unsigned (&a)[4][4]) {
    const __half2 k8 = __float2half2_rn(256.0f);
    const uint4 wq = S.a[warp * 32u + lane];
    const uint2 sb = S.sc[warp * 8u + g];
    const unsigned w4[4] = {wq.x, wq.y, wq.z, wq.w};
    const unsigned s4[2] = {sb.x, sb.y};
#pragma unroll
    for (unsigned ks = 0; ks < 4; ++ks) {
        const unsigned sbytes = (s4[ks >> 1] >> ((ks & 1u) * 16u)) & 0xffffu;
        const unsigned sbits = ((sbytes & 0xffu) << 7) | ((sbytes >> 8) << 23);
        const __half2 s2 = __hmul2(*reinterpret_cast<const __half2*>(&sbits), k8);
        const __half2 dl = __low2half2(s2), dh = __high2half2(s2);
        const unsigned w = w4[ks];
        a[ks][0] = hmul2u(fp4pair(w), dl);
        a[ks][1] = hmul2u(fp4pair(w << 4), dh);
        a[ks][2] = hmul2u(fp4pair(w << 8), dl);
        a[ks][3] = hmul2u(fp4pair(w << 12), dh);
    }
}
#endif

#ifndef G8_STAGES
#define G8_STAGES 5   // shared-memory stages of astral_gemm_nvfp4_wgmma2 (loads run G8_STAGES - 1 ahead)
#endif
extern "C" __global__ void __launch_bounds__(256) astral_gemm_nvfp4_wgmma2(
    const u8* __restrict__ frag, const u8* __restrict__ fscale, const __half* __restrict__ xh, float* __restrict__ y, unsigned N,
    unsigned K, const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx,
    unsigned idx_len, unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats, unsigned kc_per_split) { PDL_WAIT();
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    extern __shared__ __align__(1024) unsigned char g8smem[];
    G7Stage* sm = reinterpret_cast<G7Stage*>(g8smem);
    __shared__ unsigned xrow[G5_TOK];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, t = lane & 3u, g = lane >> 2;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], (unsigned)G5_TOK), idx_len - start);
    const unsigned row0 = blockIdx.x * G5_ROWS;
    if (count == 0 || row0 >= N) return;
    if (tid < G5_TOK) xrow[tid] = min(xidx[start + min(tid, count - 1u)], x_rows - 1u);
    __syncthreads();
    const unsigned KC = K / 64u, NT = N / 16u;
    const unsigned kc0 = blockIdx.z * kc_per_split;
    if (kc0 >= KC) return;
    const unsigned KCS = min(kc_per_split, KC - kc0);
    const bool split = gridDim.z > 1;
    const uint4* fa = reinterpret_cast<const uint4*>(frag) + ((u64)e * NT + blockIdx.x * 8u) * KC * 32u;
    const uint2* fs = reinterpret_cast<const uint2*>(fscale) + ((u64)e * NT + blockIdx.x * 8u) * KC * 8u;
    auto load = [&](unsigned stage, unsigned kcl) {
        G7Stage& S = sm[stage];
        const unsigned kc = kc0 + kcl;
        cp_async16(&S.a[tid], fa + ((u64)(tid >> 5) * KC + kc) * 32u + (tid & 31u));
        if (tid < 32) cp_async16(reinterpret_cast<uint4*>(S.sc) + tid, reinterpret_cast<const uint4*>(fs + ((u64)(tid >> 2) * KC + kc) * 8u) + (tid & 3u));
        const unsigned k0 = kc * 64u;
        for (unsigned c = tid; c < G5_TOK * 8u; c += blockDim.x) {
            const unsigned j = c >> 3, part = c & 7u;
            cp_async16(S.b + j * 64u + ((part ^ (j & 7u)) * 8u), xh + (u64)xrow[j] * K + k0 + part * 8u);
        }
    };
    float d[64];
#pragma unroll
    for (unsigned i = 0; i < 64; ++i) d[i] = 0.0f;
#pragma unroll
    for (unsigned p = 0; p < G8_STAGES - 1; ++p) {
        if (p < KCS) load(p, p);
        cp_async_commit();
    }
    unsigned a0[4][4], a1[4][4];
    // One K step: wait for its stage, dequant into A, issue its 4 wgmmas, keep 1 group in flight,
    // then refill the stage two steps ahead (the one the previous group just released).
    auto step = [&](unsigned kc, unsigned (&a)[4][4]) {
        cp_async_wait<G8_STAGES - 2>();
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const G7Stage& S = sm[kc % G8_STAGES];
        g8_dequant(S, warp, lane, g, a);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const unsigned long long db = wgmma_desc(S.b);
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) wgmma_m64n128k16(d, a[ks], db + (unsigned long long)(ks * 2u));
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        __syncthreads();   // both warpgroups released stage kc - 1
        if (kc + G8_STAGES - 1 < KCS) load((kc + G8_STAGES - 1) % G8_STAGES, kc + G8_STAGES - 1);
        cp_async_commit();
    };
    unsigned kc = 0;
    for (; kc + 1 < KCS; kc += 2) {
        step(kc, a0);
        step(kc + 1, a1);
    }
    if (kc < KCS) step(kc, a0);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    cp_async_wait<0>();
    const float ms = (mat_scale ? mat_scale[e] : 1.0f) * 16384.0f;
    const unsigned r_lo = row0 + warp * 16u + g, r_hi = r_lo + 8u;
#pragma unroll
    for (unsigned nt = 0; nt < 16; ++nt) {
        const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
        if (j0 < count) {
            float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
            if (split) { atomicAdd(yr + r_lo, d[4 * nt] * ms); atomicAdd(yr + r_hi, d[4 * nt + 2] * ms); }
            else { yr[r_lo] = d[4 * nt] * ms; yr[r_hi] = d[4 * nt + 2] * ms; }
        }
        if (j1 < count) {
            float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
            if (split) { atomicAdd(yr + r_lo, d[4 * nt + 1] * ms); atomicAdd(yr + r_hi, d[4 * nt + 3] * ms); }
            else { yr[r_lo] = d[4 * nt + 1] * ms; yr[r_hi] = d[4 * nt + 3] * ms; }
        }
    }
#else
    __trap();
#endif
}

// Native MXFP4 decode GEMV with each row's K split over WR warps (8 / WR rows per block),
// partial sums reduced in shared memory: more blocks and fewer dependent load rounds for short,
// wide matrices (N <= 8192, e.g. 5120 x 17408). Same parameters as astral_gemv_mx4_s1.
template <unsigned U, unsigned WR>
__device__ __forceinline__ void gemv_mx4_split(const u8* __restrict__ qs, const u8* __restrict__ es, const signed char* __restrict__ xq,
                                               const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs,
                                               u64 mat_es, unsigned xq_shared, const int* __restrict__ ids, unsigned n_mats) {
    extern __shared__ uint4 smem4[];
    __shared__ float part[8];
    const unsigned s = blockIdx.y;
    const unsigned sx = xq_shared ? 0u : s;
    uint4* sq = smem4;
    float* sd = reinterpret_cast<float*>(reinterpret_cast<signed char*>(smem4) + K);
    {
        const uint4* gq = reinterpret_cast<const uint4*>(xq + (u64)sx * K);
        for (unsigned i = threadIdx.x; i < (K >> 4); i += blockDim.x) sq[i] = __ldg(gq + i);
        const float* gd = xd + (u64)sx * (K >> 5);
        for (unsigned i = threadIdx.x; i < (K >> 5); i += blockDim.x) sd[i] = __ldg(gd + i);
    }
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned rpb = (blockDim.x >> 5) / WR;               // rows per block
    const unsigned r = warp / WR, piece = warp % WR;
    const unsigned n = blockIdx.x * rpb + r;
    const unsigned units = K >> 5;
    const unsigned u0 = piece * units / WR, u1 = (piece + 1) * units / WR;
    float acc = 0.0f;
    if (n < N) {
        const unsigned e = ids ? min((unsigned)ids[s], n_mats - 1u) : s;
        const u8* w = qs + e * mat_qs + (u64)n * (K >> 1);
        const u8* we = es + e * mat_es + (u64)n * (K >> 5);
        for (unsigned base = u0; base < u1; base += 32u * U) {
            uint4 wv[U];
            unsigned sc[U];
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + lane + 32u * i;
                if (u < u1) {
                    wv[i] = ld_stream16(w + (u64)u * 16u);
                    sc[i] = ld_stream_u8(we + u);
                }
            }
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + lane + 32u * i;
                if (u < u1) {
                    const uint4 x0 = sq[2u * u], x1 = sq[2u * u + 1u];
                    const int2 a0 = table16((int)wv[i].x, k_e2m1x2), a1 = table16((int)wv[i].y, k_e2m1x2);
                    const int2 b0 = table16((int)wv[i].z, k_e2m1x2), b1 = table16((int)wv[i].w, k_e2m1x2);
                    int t = __dp4a(a0.x, (int)x0.x, 0);
                    t = __dp4a(a1.x, (int)x0.y, t);
                    t = __dp4a(a0.y, (int)x0.z, t);
                    t = __dp4a(a1.y, (int)x0.w, t);
                    t = __dp4a(b0.x, (int)x1.x, t);
                    t = __dp4a(b1.x, (int)x1.y, t);
                    t = __dp4a(b0.y, (int)x1.z, t);
                    t = __dp4a(b1.y, (int)x1.w, t);
                    const float f = sc[i] >= 2u ? __int_as_float((int)((sc[i] - 1u) << 23)) : 0.0f;
                    acc = fmaf((float)t, f * sd[u], acc);
                }
            }
        }
    }
    acc = warp_sum_f(acc);
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (piece == 0 && lane == 0 && n < N) {
        float tot = 0.0f;
#pragma unroll
        for (unsigned q = 0; q < WR; ++q) tot += part[r * WR + q];
        y[(u64)s * N + n] = tot;
    }
}

#define GEMV_MX4_SPLIT_ENTRY(NAME, WR)                                                                              \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                         \
        const u8* __restrict__ qs, const u8* __restrict__ es, const signed char* __restrict__ xq,                  \
        const float* __restrict__ xd, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_es,        \
        unsigned xq_shared, const int* __restrict__ ids, const float* __restrict__ mat_scale, unsigned n_mats) {    \
        PDL_WAIT();                                                                                                 \
        (void)mat_scale;                                                                                            \
        gemv_mx4_split<6, WR>(qs, es, xq, xd, y, N, K, mat_qs, mat_es, xq_shared, ids, n_mats);                     \
    }
GEMV_MX4_SPLIT_ENTRY(astral_gemv_mx4_w2, 2)
GEMV_MX4_SPLIT_ENTRY(astral_gemv_mx4_w4, 4)

// =============================================================================================
// Grouped NVFP4 GEMV for batched decode: work item i (a QItem: matrix e, up to NB activation
// rows xi[], output rows yi[]) multiplies every weight row of matrix e once with each of the
// item's activation rows. The arithmetic per (row, activation) is exactly that of
// astral_gemv_nvfp4_s1 (32 lanes per row, U = 6 loads in flight, the same multiply-add order
// and reduction), so a token's result does not depend on how many tokens share the launch.
// Activations are staged in shared memory (NB rows of K int8 + K/32 scales).
// grid = (ceil(N / 64), n_items), 256 threads, NB * (K + K / 8) bytes of shared memory.
// Staged activations per column: the first 16 bytes of every 32-value unit, then the second
// 16 bytes, then the K / 32 scales (lanes of a warp read consecutive 16-byte words).
template <unsigned C>
__device__ __forceinline__ void fp4_items_rows(const u8* __restrict__ qs, const u8* __restrict__ aux, const uint4* __restrict__ sx,
                                               unsigned rs, float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux,
                                               const QItem& it, unsigned y_rows, const float* __restrict__ mat_scale, unsigned e) {
    constexpr unsigned G = 32u, U = 3u, R = 2u;
    const unsigned warp = threadIdx.x >> 5, sub = threadIdx.x & 31u;
    const unsigned units = K >> 5;
    for (unsigned rr = 0; rr < 8u; rr += R) {
        const unsigned n0 = blockIdx.x * 64u + warp * 8u + rr;
        if (n0 >= N) break;
        const u8* w[R];
        const u8* wd[R];
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            const unsigned n = min(n0 + r, N - 1u);
            w[r] = qs + e * mat_qs + (u64)n * (K >> 1);
            wd[r] = aux + e * mat_aux + (u64)n * (K >> 4);
        }
        float acc[R][C];
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
#pragma unroll
            for (unsigned t = 0; t < C; ++t) acc[r][t] = 0.0f;
        for (unsigned base = 0; base < units; base += G * U) {
            uint4 wv[U][R];
            unsigned sc[U][R];
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) {
#pragma unroll
                    for (unsigned r = 0; r < R; ++r) {
                        wv[i][r] = ld_stream16(w[r] + (u64)u * 16u);
                        sc[i][r] = ld_stream_u16(wd[r] + (u64)u * 2u);
                    }
                }
            }
#pragma unroll
            for (unsigned i = 0; i < U; ++i) {
                const unsigned u = base + sub + G * i;
                if (u < units) {
                    int2 a0[R], a1[R], b0[R], b1[R];
                    float2 d2[R];
#pragma unroll
                    for (unsigned r = 0; r < R; ++r) {
                        a0[r] = table16((int)wv[i][r].x, k_e2m1x2);
                        a1[r] = table16((int)wv[i][r].y, k_e2m1x2);
                        b0[r] = table16((int)wv[i][r].z, k_e2m1x2);
                        b1[r] = table16((int)wv[i][r].w, k_e2m1x2);
                        d2[r] = __half22float2(__halves2half2(__ushort_as_half((unsigned short)((sc[i][r] & 0xffu) << 7)),
                                                              __ushort_as_half((unsigned short)((sc[i][r] >> 8) << 7))));
                    }
#pragma unroll
                    for (unsigned t = 0; t < C; ++t) {
                        const uint4* sq = sx + (u64)t * rs;
                        const uint4 x0 = sq[u], x1 = sq[units + u];
                        const float xs = reinterpret_cast<const float*>(sq + 2u * units)[u];
#pragma unroll
                        for (unsigned r = 0; r < R; ++r) {
                            int s0 = __dp4a(a0[r].x, (int)x0.x, 0);
                            s0 = __dp4a(a1[r].x, (int)x0.y, s0);
                            s0 = __dp4a(a0[r].y, (int)x0.z, s0);
                            s0 = __dp4a(a1[r].y, (int)x0.w, s0);
                            int s1 = __dp4a(b0[r].x, (int)x1.x, 0);
                            s1 = __dp4a(b1[r].x, (int)x1.y, s1);
                            s1 = __dp4a(b0[r].y, (int)x1.z, s1);
                            s1 = __dp4a(b1[r].y, (int)x1.w, s1);
                            acc[r][t] = fmaf(fmaf((float)s0, d2[r].x, (float)s1 * d2[r].y), xs, acc[r][t]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            const unsigned n = n0 + r;
#pragma unroll
            for (unsigned t = 0; t < C; ++t) {
                float a = acc[r][t] * 128.0f;
#pragma unroll
                for (unsigned o = G / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (sub == 0 && n < N) y[(u64)min(it.yi[t], y_rows - 1u) * N + n] = mat_scale ? a * mat_scale[e] : a;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) astral_gemv_nvfp4_items_s1(
    const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq, const float* __restrict__ xd,
    float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux, const QItem* __restrict__ items, unsigned x_rows,
    unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();
    // A warp computes 2 rows at once, so each staged activation load serves both. Per row the
    // lane's units and their order are those of astral_gemv_nvfp4_s1 (U only groups loads);
    // the body is specialised on the item's token count (no predicated-off iterations).
    const QItem it = items[blockIdx.y];
    const unsigned cnt = min(it.n, (unsigned)NB);
    if (cnt == 0u) return;
    extern __shared__ uint4 smem4[];
    const unsigned units = K >> 5;
    const unsigned rs = 2u * units + (units + 3u) / 4u;   // uint4 per staged column
    for (unsigned t = 0; t < cnt; ++t) {
        const unsigned xr = min(it.xi[t], x_rows - 1u);
        uint4* sq = smem4 + (u64)t * rs;
        const uint4* gq = reinterpret_cast<const uint4*>(xq + (u64)xr * K);
        for (unsigned i = threadIdx.x; i < 2u * units; i += blockDim.x) sq[(i & 1u) * units + (i >> 1)] = __ldg(gq + i);
        float* sd = reinterpret_cast<float*>(sq + 2u * units);
        const float* gd = xd + (u64)xr * units;
        for (unsigned i = threadIdx.x; i < units; i += blockDim.x) sd[i] = __ldg(gd + i);
    }
    __syncthreads();
    const unsigned e = min(it.e, n_mats - 1u);
    switch (cnt) {
        case 1: fp4_items_rows<1>(qs, aux, smem4, rs, y, N, K, mat_qs, mat_aux, it, y_rows, mat_scale, e); break;
        case 2: fp4_items_rows<2>(qs, aux, smem4, rs, y, N, K, mat_qs, mat_aux, it, y_rows, mat_scale, e); break;
        case 3: fp4_items_rows<3>(qs, aux, smem4, rs, y, N, K, mat_qs, mat_aux, it, y_rows, mat_scale, e); break;
        case 4: fp4_items_rows<4>(qs, aux, smem4, rs, y, N, K, mat_qs, mat_aux, it, y_rows, mat_scale, e); break;
        case 5: fp4_items_rows<5>(qs, aux, smem4, rs, y, N, K, mat_qs, mat_aux, it, y_rows, mat_scale, e); break;
        case 6: fp4_items_rows<6>(qs, aux, smem4, rs, y, N, K, mat_qs, mat_aux, it, y_rows, mat_scale, e); break;
        case 7: fp4_items_rows<7>(qs, aux, smem4, rs, y, N, K, mat_qs, mat_aux, it, y_rows, mat_scale, e); break;
        default: fp4_items_rows<8>(qs, aux, smem4, rs, y, N, K, mat_qs, mat_aux, it, y_rows, mat_scale, e); break;
    }
}

// The Q8_0 form of astral_gemv_nvfp4_items_s1: each (row, activation) exactly as
// astral_gemv_q8_0_dp4a computes it (32 lanes per row, U = 4, same multiply-add order and
// reduction); activation rows are read from global memory (as the multi-token kernels do).
// grid = (ceil(N / 64), n_items), 256 threads.
extern "C" __global__ void __launch_bounds__(256) astral_gemv_q8_0_items_x(
    const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq, const float* __restrict__ xd,
    float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux, const QItem* __restrict__ items, unsigned x_rows,
    unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();
    constexpr unsigned G = 32u, U = QShape<2>::U;
    const QItem it = items[blockIdx.y];
    const unsigned cnt = min(it.n, (unsigned)NB);
    if (cnt == 0u) return;
    const unsigned warp = threadIdx.x >> 5, sub = threadIdx.x & 31u;
    const unsigned e = min(it.e, n_mats - 1u), units = K >> 4;
    for (unsigned rr = 0; rr < 64u; rr += blockDim.x >> 5) {
    const unsigned n = blockIdx.x * 64u + rr + warp;
    if (n >= N) break;
    const uint4* w = reinterpret_cast<const uint4*>(qs + e * mat_qs + (u64)n * K);
    const u16* wd = reinterpret_cast<const u16*>(aux + e * mat_aux + (u64)n * (K >> 5) * 2u);
    unsigned xr[NB];
#pragma unroll
    for (unsigned t = 0; t < NB; ++t) xr[t] = min(t < cnt ? it.xi[t] : 0u, x_rows - 1u);
    float acc[NB];
#pragma unroll
    for (unsigned t = 0; t < NB; ++t) acc[t] = 0.0f;
    for (unsigned base = 0; base < units; base += G * U) {
        uint4 wv[U];
#pragma unroll
        for (unsigned i = 0; i < U; ++i) {
            const unsigned u = base + sub + G * i;
            if (u < units) wv[i] = __ldg(w + u);
        }
#pragma unroll
        for (unsigned i = 0; i < U; ++i) {
            const unsigned u = base + sub + G * i;
            if (u < units) {
                const float dw = __half2float(__ushort_as_half(__ldg(wd + (u >> 1))));
#pragma unroll
                for (unsigned t = 0; t < NB; ++t) {
                    if (t < cnt) {
                        const uint4 a = __ldg(reinterpret_cast<const uint4*>(xq + (u64)xr[t] * K) + u);
                        int sum = __dp4a((int)wv[i].x, (int)a.x, 0);
                        sum = __dp4a((int)wv[i].y, (int)a.y, sum);
                        sum = __dp4a((int)wv[i].z, (int)a.z, sum);
                        sum = __dp4a((int)wv[i].w, (int)a.w, sum);
                        acc[t] = fmaf((float)sum, dw * __ldg(xd + (u64)xr[t] * (K >> 5) + (u >> 1)), acc[t]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < NB; ++t) {
        float a = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (t < cnt && sub == 0) y[(u64)min(it.yi[t], y_rows - 1u) * N + n] = mat_scale ? a * mat_scale[e] : a;
    }
    }
}

// =============================================================================================
// Batched decode GEMMs on tensor cores. A work item (QItem) is one matrix and up to 8 activation
// rows: the 8 columns of an m16n8 tile. A block is 4 warps x 16 weight rows, and streams its
// rows once per item. grid = (n_items, ceil(N / 64)): the items of one row block run next to
// each other, so items of the same matrix (dense rows in groups of 8, or an expert's tokens) read
// the weights from L2 after the first. The item's activation rows are staged in shared memory.

// NVFP4 in the fragment layout (astral_quant::nvfp4_frag_layout, see astral_gemm_nvfp4_mma5):
// per 16-row tile and 64-column stage, lane l holds one 16-byte word quad (its A fragments for 4
// k16-steps) and row group g = l / 4 one 8-byte scale pair. Activations are the f16 copy.
// N % 64 == 0, K % 64 == 0, dynamic shared memory 8 * (K + 8) * 2 bytes.
extern "C" __global__ void __launch_bounds__(512) astral_gemv_nvfp4_frag_items(
    const u8* __restrict__ frag, const u8* __restrict__ fscale, const __half* __restrict__ xh, float* __restrict__ y, unsigned N,
    unsigned K, const QItem* __restrict__ items, unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale,
    unsigned n_mats) { PDL_WAIT();
    const QItem* itp = items + blockIdx.x;
    const unsigned cnt = min(itp->n, (unsigned)NB);
    if (cnt == 0u) return;
    extern __shared__ uint4 xs4[];
    __half* xs = reinterpret_cast<__half*>(xs4);
    const unsigned stride = K + 8u;   // halves; 16 bytes of padding per row spreads the 8 rows over the banks
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned e = min(itp->e, n_mats - 1u);
    {
        const unsigned per_row = K >> 3;   // uint4 per activation row
        for (unsigned i = tid; i < NB * per_row; i += blockDim.x) {
            const unsigned r = i / per_row, c = i - r * per_row;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (r < cnt) v = __ldg(reinterpret_cast<const uint4*>(xh + (u64)min(itp->xi[r], x_rows - 1u) * K) + c);
            *reinterpret_cast<uint4*>(xs + r * stride + c * 8u) = v;
        }
    }
    __syncthreads();
    const unsigned KC = K >> 6, NT = N >> 4;
    const unsigned tile = blockIdx.y * (blockDim.x >> 5) + warp;
    if (tile >= NT) return;
    const uint4* fa = reinterpret_cast<const uint4*>(frag) + ((u64)e * NT + tile) * KC * 32u + lane;
    const uint2* fs = reinterpret_cast<const uint2*>(fscale) + ((u64)e * NT + tile) * KC * 8u + g;
    const __half* xb = xs + g * stride + 2u * t;
    const __half2 k8 = __float2half2_rn(256.0f);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    constexpr unsigned P = 4;   // stages of loads in flight
    uint4 wq[P];
    uint2 sq[P];
#pragma unroll
    for (unsigned p = 0; p < P; ++p) {
        if (p < KC) { wq[p] = __ldg(fa + p * 32u); sq[p] = __ldg(fs + p * 8u); }
    }
    for (unsigned kc0 = 0; kc0 < KC; kc0 += P) {
#pragma unroll
        for (unsigned p = 0; p < P; ++p) {
            const unsigned kc = kc0 + p;
            if (kc < KC) {
            const uint4 w = wq[p];
            const uint2 sb = sq[p];
            if (kc + P < KC) { wq[p] = __ldg(fa + (kc + P) * 32u); sq[p] = __ldg(fs + (kc + P) * 8u); }
            const unsigned w4[4] = {w.x, w.y, w.z, w.w};
            const unsigned s4[2] = {sb.x, sb.y};
#pragma unroll
            for (unsigned ks = 0; ks < 4; ++ks) {
                const unsigned sbytes = (s4[ks >> 1] >> ((ks & 1u) * 16u)) & 0xffffu;
                const unsigned sbits = ((sbytes & 0xffu) << 7) | ((sbytes >> 8) << 23);
                const __half2 s2 = __hmul2(*reinterpret_cast<const __half2*>(&sbits), k8);
                const __half2 dl = __low2half2(s2), dh = __high2half2(s2);
                unsigned a[4];
                a[0] = hmul2u(fp4pair(w4[ks]), dl);
                a[1] = hmul2u(fp4pair(w4[ks] << 4), dh);
                a[2] = hmul2u(fp4pair(w4[ks] << 8), dl);
                a[3] = hmul2u(fp4pair(w4[ks] << 12), dh);
                const __half* xk = xb + kc * 64u + ks * 16u;
                const unsigned b0 = *reinterpret_cast<const unsigned*>(xk);
                const unsigned b1 = *reinterpret_cast<const unsigned*>(xk + 8u);
                mma_f16(acc, a, b0, b1);
            }
            }
        }
    }
    const float ms = (mat_scale ? mat_scale[e] : 1.0f) * 16384.0f;
    const unsigned r_lo = tile * 16u + g, r_hi = r_lo + 8u;
    const unsigned j0 = 2u * t, j1 = j0 + 1u;
    if (j0 < cnt) {
        float* yr = y + (u64)min(itp->yi[j0], y_rows - 1u) * N;
        yr[r_lo] = acc[0] * ms;
        yr[r_hi] = acc[2] * ms;
    }
    if (j1 < cnt) {
        float* yr = y + (u64)min(itp->yi[j1], y_rows - 1u) * N;
        yr[r_lo] = acc[1] * ms;
        yr[r_hi] = acc[3] * ms;
    }
}

// Q8_0 (qs: K bytes per row, aux: K / 32 f16 scales per row; matrices mat_qs / mat_aux bytes
// apart) with int8 activations, on int8 mma.sync m16n8k32: one Q8_0 block per k-step, its int32
// result scaled into f32 by (weight scale x activation scale). Within a block a lane takes 8
// consecutive bytes of its rows and of its token (bytes 8t..8t+7 in the fragment positions of
// bytes 4t..4t+3 and 16+4t..16+4t+3): the same permutation of the 32 products on both sides, so
// the dot product is unchanged and each row's 32 bytes are read by one coalesced 8-byte load
// per lane. A block is 8 warps on 32 rows: warp w takes row tile w % 2 and the K quarter w / 2
// (many loads in flight even for few rows), and the quarters are added in shared memory.
// N any, K % 128 == 0.
extern "C" __global__ void __launch_bounds__(256) astral_gemv_q8_0_mma_items(
    const u8* __restrict__ qs, const u8* __restrict__ aux, const signed char* __restrict__ xq, const float* __restrict__ xd,
    float* __restrict__ y, unsigned N, unsigned K, u64 mat_qs, u64 mat_aux, const QItem* __restrict__ items, unsigned x_rows,
    unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const QItem* itp = items + blockIdx.x;
    const unsigned cnt = min(itp->n, (unsigned)NB);
    if (cnt == 0u) return;
    __shared__ float red[8][32][4];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned e = min(itp->e, n_mats - 1u);
    const unsigned KB = K >> 5, KQ = KB >> 2;
    const unsigned row0 = blockIdx.y * 32u + (warp & 1u) * 16u;
    const unsigned kb0 = (warp >> 1) * KQ;
    const unsigned ra = min(row0 + g, N - 1u), rb = min(row0 + g + 8u, N - 1u);
    const u8* wbase = qs + e * mat_qs;
    const uint2* wa = reinterpret_cast<const uint2*>(wbase + (u64)ra * K) + t + kb0 * 4u;
    const uint2* wb = reinterpret_cast<const uint2*>(wbase + (u64)rb * K) + t + kb0 * 4u;
    const u16* da = reinterpret_cast<const u16*>(aux + e * mat_aux) + (u64)ra * KB + kb0;
    const u16* db = reinterpret_cast<const u16*>(aux + e * mat_aux) + (u64)rb * KB + kb0;
    // Token g's activation bytes (B operand) and tokens 2t, 2t + 1's scales; tokens beyond the
    // item read row cnt - 1 and are not stored.
    const unsigned tg = min(g, cnt - 1u), t0 = min(2u * t, cnt - 1u), t1 = min(2u * t + 1u, cnt - 1u);
    const uint2* xa = reinterpret_cast<const uint2*>(xq + (u64)min(itp->xi[tg], x_rows - 1u) * K) + t + kb0 * 4u;
    const float* s0 = xd + (u64)min(itp->xi[t0], x_rows - 1u) * KB + kb0;
    const float* s1 = xd + (u64)min(itp->xi[t1], x_rows - 1u) * KB + kb0;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    constexpr unsigned P = 3;   // k-steps per batch of loads; KQ % P == 0 is not required
    for (unsigned k = 0; k < KQ; k += P) {
        uint2 qa[P], qb[P], bx[P];
        u16 sa[P], sb[P];
        float x0[P], x1[P];
#pragma unroll
        for (unsigned p = 0; p < P; ++p) {
            const unsigned kk = min(k + p, KQ - 1u);
            qa[p] = __ldg(wa + kk * 4u);
            qb[p] = __ldg(wb + kk * 4u);
            bx[p] = __ldg(xa + kk * 4u);
            sa[p] = __ldg(da + kk);
            sb[p] = __ldg(db + kk);
            x0[p] = __ldg(s0 + kk);
            x1[p] = __ldg(s1 + kk);
        }
#pragma unroll
        for (unsigned p = 0; p < P; ++p) {
            const unsigned a[4] = {qa[p].x, qb[p].x, qa[p].y, qb[p].y};
            int c[4];
            mma_s8_16832(c, a, bx[p].x, bx[p].y);
            // A repeated last step (KQ % P != 0) contributes nothing.
            const float live = (k + p < KQ) ? 1.0f : 0.0f;
            const float wsa = __half2float(__ushort_as_half(sa[p])) * live, wsb = __half2float(__ushort_as_half(sb[p])) * live;
            acc[0] = fmaf((float)c[0], wsa * x0[p], acc[0]);
            acc[1] = fmaf((float)c[1], wsa * x1[p], acc[1]);
            acc[2] = fmaf((float)c[2], wsb * x0[p], acc[2]);
            acc[3] = fmaf((float)c[3], wsb * x1[p], acc[3]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][i] = acc[i];
    __syncthreads();
    if (warp >= 2) return;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[warp][lane][i] + red[warp + 2][lane][i] + red[warp + 4][lane][i] + red[warp + 6][lane][i];
    const float ms = mat_scale ? mat_scale[e] : 1.0f;
    const unsigned r_lo = row0 + g, r_hi = r_lo + 8u;
    const unsigned j0 = 2u * t, j1 = j0 + 1u;
    if (j0 < cnt) {
        float* yr = y + (u64)min(itp->yi[j0], y_rows - 1u) * N;
        if (r_lo < N) yr[r_lo] = v[0] * ms;
        if (r_hi < N) yr[r_hi] = v[2] * ms;
    }
    if (j1 < cnt) {
        float* yr = y + (u64)min(itp->yi[j1], y_rows - 1u) * N;
        if (r_lo < N) yr[r_lo] = v[1] * ms;
        if (r_hi < N) yr[r_hi] = v[3] * ms;
    }
#endif
}

// Dense Q8_0 products for a batch of T tokens, every weight row read once per 8 * NT tokens:
// astral_gemv_q8_0_mma_items with NT tiles of 8 tokens per block (each weight fragment feeds NT
// MMAs) over up to 8 matrices (MultiQ8, blocks per matrix = ceil(n / 32)). grid = (blocks,
// ceil(T / (8 * NT))), 256 threads (2 row tiles x 4 K quarters), K % 128 == 0.
template <unsigned NT>
__device__ __forceinline__ void q8_dense_mma(const MultiQ8& m, unsigned K, unsigned T) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ float red[8][32][4 * NT];
    const unsigned bx = blockIdx.x;
    unsigned i = 0;
    while (i + 1 < m.count && bx >= m.start[i + 1]) ++i;
    const unsigned N = m.n[i];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned KB = K >> 5, KQ = KB >> 2;
    const unsigned row0 = (bx - m.start[i]) * 32u + (warp & 1u) * 16u;
    const unsigned kb0 = (warp >> 1) * KQ;
    const unsigned tb = blockIdx.y * 8u * NT;
    const unsigned ra = min(row0 + g, N - 1u), rb = min(row0 + g + 8u, N - 1u);
    const uint2* wa = reinterpret_cast<const uint2*>(m.qs[i] + (u64)ra * K) + t + kb0 * 4u;
    const uint2* wb = reinterpret_cast<const uint2*>(m.qs[i] + (u64)rb * K) + t + kb0 * 4u;
    const u16* da = reinterpret_cast<const u16*>(m.aux[i]) + (u64)ra * KB + kb0;
    const u16* db = reinterpret_cast<const u16*>(m.aux[i]) + (u64)rb * KB + kb0;
    const uint2* xa[NT];
    const float* s0[NT];
    const float* s1[NT];
#pragma unroll
    for (unsigned j = 0; j < NT; ++j) {
        const unsigned tg = min(tb + 8u * j + g, T - 1u), t0 = min(tb + 8u * j + 2u * t, T - 1u), t1 = min(tb + 8u * j + 2u * t + 1u, T - 1u);
        xa[j] = reinterpret_cast<const uint2*>(m.xq[i] + (u64)tg * K) + t + kb0 * 4u;
        s0[j] = m.xd[i] + (u64)t0 * KB + kb0;
        s1[j] = m.xd[i] + (u64)t1 * KB + kb0;
    }
    float acc[NT][4];
#pragma unroll
    for (unsigned j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    constexpr unsigned P = NT >= 4 ? 2 : 3;
    for (unsigned k = 0; k < KQ; k += P) {
        uint2 qa[P], qb[P], bxv[P][NT];
        u16 sa[P], sb[P];
        float x0[P][NT], x1[P][NT];
#pragma unroll
        for (unsigned p = 0; p < P; ++p) {
            const unsigned kk = min(k + p, KQ - 1u);
            qa[p] = __ldg(wa + kk * 4u);
            qb[p] = __ldg(wb + kk * 4u);
            sa[p] = __ldg(da + kk);
            sb[p] = __ldg(db + kk);
#pragma unroll
            for (unsigned j = 0; j < NT; ++j) {
                bxv[p][j] = __ldg(xa[j] + kk * 4u);
                x0[p][j] = __ldg(s0[j] + kk);
                x1[p][j] = __ldg(s1[j] + kk);
            }
        }
#pragma unroll
        for (unsigned p = 0; p < P; ++p) {
            const unsigned a[4] = {qa[p].x, qb[p].x, qa[p].y, qb[p].y};
            const float live = (k + p < KQ) ? 1.0f : 0.0f;
            const float wsa = __half2float(__ushort_as_half(sa[p])) * live, wsb = __half2float(__ushort_as_half(sb[p])) * live;
#pragma unroll
            for (unsigned j = 0; j < NT; ++j) {
                int c[4];
                mma_s8_16832(c, a, bxv[p][j].x, bxv[p][j].y);
                acc[j][0] = fmaf((float)c[0], wsa * x0[p][j], acc[j][0]);
                acc[j][1] = fmaf((float)c[1], wsa * x1[p][j], acc[j][1]);
                acc[j][2] = fmaf((float)c[2], wsb * x0[p][j], acc[j][2]);
                acc[j][3] = fmaf((float)c[3], wsb * x1[p][j], acc[j][3]);
            }
        }
    }
#pragma unroll
    for (unsigned j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) red[warp][lane][4 * j + q] = acc[j][q];
    __syncthreads();
    if (warp >= 2) return;
    const unsigned r_lo = row0 + g, r_hi = r_lo + 8u;
    float* y = m.y[i];
#pragma unroll
    for (unsigned j = 0; j < NT; ++j) {
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = red[warp][lane][4 * j + q] + red[warp + 2][lane][4 * j + q] + red[warp + 4][lane][4 * j + q] + red[warp + 6][lane][4 * j + q];
        const unsigned j0 = tb + 8u * j + 2u * t, j1 = j0 + 1u;
        if (j0 < T) {
            if (r_lo < N) y[(u64)j0 * N + r_lo] = v[0];
            if (r_hi < N) y[(u64)j0 * N + r_hi] = v[2];
        }
        if (j1 < T) {
            if (r_lo < N) y[(u64)j1 * N + r_lo] = v[1];
            if (r_hi < N) y[(u64)j1 * N + r_hi] = v[3];
        }
    }
#endif
}
extern "C" __global__ void __launch_bounds__(256) astral_gemv_q8_0_dense_n2(MultiQ8 m, unsigned K, unsigned T) { PDL_WAIT(); q8_dense_mma<2>(m, K, T); }
extern "C" __global__ void __launch_bounds__(256) astral_gemv_q8_0_dense_n4(MultiQ8 m, unsigned K, unsigned T) { PDL_WAIT(); q8_dense_mma<4>(m, K, T); }
extern "C" __global__ void __launch_bounds__(256) astral_gemv_q8_0_dense_n8(MultiQ8 m, unsigned K, unsigned T) { PDL_WAIT(); q8_dense_mma<8>(m, K, T); }

// =============================================================================================
// Warpgroup-MMA GEMM for dense matrices stored as f16 A fragments (astral_quant::f16_frag_layout:
// per 16-row tile and 64-column stage, lane l holds 64 bytes = its 4 m16n8k16 A fragments, so
// no dequantisation in the loop). Otherwise astral_gemm_nvfp4_wgmma2: 128 rows x 128 tokens per
// block (two warpgroups of 64 rows), B (f16 activations) K-major with the 128-byte swizzle, 3
// stages, one wgmma group in flight. grid = (N / 128, n_items, splits), 256 threads, N % 128 ==
// 0, K % 64 == 0, items of <= 128 tokens. Dynamic shared memory 3 * 32 KB.
struct __align__(1024) G9Stage {
    __half b[G5_TOK * 64];   // 16 KB, swizzled
    uint4 a[G5_ROWS / 16 * 32 * 4];   // 16 KB: 8 row tiles x 32 lanes x 64 bytes
};

// Up to 8 matrices with the same K in one launch: blockIdx.x runs over every matrix's row blocks
// (start[i] = first block of matrix i).
struct F16Multi {
    const uint4* frag[8];
    const __half* xh[8];
    float* y[8];
    unsigned n[8];
    unsigned start[9];
    unsigned count;
};
extern "C" __global__ void __launch_bounds__(256) astral_gemm_f16_wgmma(
    F16Multi mm, unsigned K,
    const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx, unsigned idx_len,
    unsigned x_rows, unsigned y_rows, unsigned kc_per_split) { PDL_WAIT();
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    unsigned mi = 0;
    while (mi + 1 < mm.count && blockIdx.x >= mm.start[mi + 1]) ++mi;
    const uint4* __restrict__ frag = mm.frag[mi];
    const __half* __restrict__ xh = mm.xh[mi];
    float* __restrict__ y = mm.y[mi];
    const unsigned N = mm.n[mi], bx = blockIdx.x - mm.start[mi];
    extern __shared__ __align__(1024) unsigned char g9smem[];
    G9Stage* sm = reinterpret_cast<G9Stage*>(g9smem);
    __shared__ unsigned xrow[G5_TOK];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, t = lane & 3u, g = lane >> 2;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], (unsigned)G5_TOK), idx_len - start);
    const unsigned row0 = bx * G5_ROWS;
    if (count == 0 || row0 >= N) return;
    if (tid < G5_TOK) xrow[tid] = min(xidx[start + min(tid, count - 1u)], x_rows - 1u);
    __syncthreads();
    const unsigned KC = K / 64u;
    const unsigned kc0 = blockIdx.z * kc_per_split;
    if (kc0 >= KC) return;
    const unsigned KCS = min(kc_per_split, KC - kc0);
    const bool split = gridDim.z > 1;
    const uint4* fa = frag + (u64)bx * 8u * KC * 128u;   // 8 row tiles of KC stages x 128 uint4
    auto load = [&](unsigned stage, unsigned kcl) {
        G9Stage& S = sm[stage];
        const unsigned kc = kc0 + kcl;
        const uint4* src = fa + ((u64)(tid >> 5) * KC + kc) * 128u + (tid & 31u) * 4u;
#pragma unroll
        for (unsigned q = 0; q < 4; ++q) cp_async16(&S.a[tid * 4u + q], src + q);
        const unsigned k0 = kc * 64u;
        for (unsigned c = tid; c < G5_TOK * 8u; c += blockDim.x) {
            const unsigned j = c >> 3, part = c & 7u;
            cp_async16(S.b + j * 64u + ((part ^ (j & 7u)) * 8u), xh + (u64)xrow[j] * K + k0 + part * 8u);
        }
    };
    float d[64];
#pragma unroll
    for (unsigned i = 0; i < 64; ++i) d[i] = 0.0f;
    load(0, 0);
    cp_async_commit();
    if (KCS > 1) load(1, 1);
    cp_async_commit();
    auto step = [&](unsigned kc) {
        cp_async_wait<1>();
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const G9Stage& S = sm[kc % 3u];
        unsigned a[4][4];
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            const uint4 v = S.a[tid * 4u + ks];
            a[ks][0] = v.x; a[ks][1] = v.y; a[ks][2] = v.z; a[ks][3] = v.w;
        }
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const unsigned long long db = wgmma_desc(S.b);
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) wgmma_m64n128k16(d, a[ks], db + (unsigned long long)(ks * 2u));
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        __syncthreads();
        if (kc + 2 < KCS) load((kc + 2) % 3u, kc + 2);
        cp_async_commit();
    };
    for (unsigned kc = 0; kc < KCS; ++kc) step(kc);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    cp_async_wait<0>();
    const unsigned r_lo = row0 + warp * 16u + g, r_hi = r_lo + 8u;
#pragma unroll
    for (unsigned nt = 0; nt < 16; ++nt) {
        const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
        if (j0 < count) {
            float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
            if (split) { atomicAdd(yr + r_lo, d[4 * nt]); atomicAdd(yr + r_hi, d[4 * nt + 2]); }
            else { yr[r_lo] = d[4 * nt]; yr[r_hi] = d[4 * nt + 2]; }
        }
        if (j1 < count) {
            float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
            if (split) { atomicAdd(yr + r_lo, d[4 * nt + 1]); atomicAdd(yr + r_hi, d[4 * nt + 3]); }
            else { yr[r_lo] = d[4 * nt + 1]; yr[r_hi] = d[4 * nt + 3]; }
        }
    }
#else
    __trap();
#endif
}

// =============================================================================================
// Grouped f16 GEMM on Hopper with TMA weights: y[yidx[j]] = mat_scale[e] * W_e x[xidx[j]] for the
// tokens j of each item (e, start, count <= 128), W_e row-major f16 [N][K] (all matrices stacked,
// [n_mats * N][K], read through the tensor map `tw` in 128 x 64 boxes with the 128-byte swizzle).
// Warpgroup 0 loads: one thread issues the weight tile's TMA, all 128 gather the item's activation
// rows (f16) with cp.async into the same swizzled layout; each stage's "full" mbarrier completes
// when both have landed. Warpgroups 1 and 2 (64 weight rows each) run m64n128k16 warpgroup MMAs
// with both operands in shared memory and release a stage once its MMAs retire. TF_STAGES
// stages of 32 KB. grid = (N / 128, n_items), 384 threads; N % 128 == 0, K % 64 == 0.
// The driver's CUtensorMap (opaque, 128 bytes, 64-byte aligned), passed by value.
struct __align__(64) CUtensorMapStub {
    unsigned long long opaque[16];
};
#define TF_STAGES 6

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
__device__ __forceinline__ void wgmma_ss_m64n128k16(float (&d)[64], unsigned long long da, unsigned long long db) {
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 0;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "l"(da), "l"(db));
}
__device__ __forceinline__ void tf_mbar_init(unsigned long long* b, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"((unsigned)__cvta_generic_to_shared(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void tf_mbar_arrive(unsigned long long* b) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"((unsigned)__cvta_generic_to_shared(b)) : "memory");
}
__device__ __forceinline__ void tf_mbar_expect_tx(unsigned long long* b, unsigned bytes) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" :: "r"((unsigned)__cvta_generic_to_shared(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tf_mbar_cp_async_arrive(unsigned long long* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"((unsigned)__cvta_generic_to_shared(b)) : "memory");
}
__device__ __forceinline__ void tf_mbar_wait(unsigned long long* b, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n TFW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra TFW_%=;\n}\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tf_tma_2d(void* dst, const void* tmap, int c0, int c1, unsigned long long* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(tmap), "r"(c0), "r"(c1), "r"((unsigned)__cvta_generic_to_shared(bar))
                 : "memory");
}
__device__ __forceinline__ void wgmma_ss_m64n256k16(float (&d)[128], unsigned long long da, unsigned long long db) {
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, 1, 1, 1, 0, 0;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                 : "l"(da), "l"(db));
}
#endif

template <unsigned BT>
__device__ __forceinline__ void gemm_f16_tma(
    const CUtensorMapStub& tw, const __half* __restrict__ xh, float* __restrict__ y, unsigned N, unsigned K,
    const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx, unsigned idx_len,
    unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    extern __shared__ __align__(1024) unsigned char tfsmem[];
    constexpr unsigned STAGES = BT == 256 ? 4u : TF_STAGES;
    constexpr unsigned STAGE_BYTES = 128u * 64u * 2u + BT * 64u * 2u;
    unsigned char* smb = tfsmem;
    __shared__ __align__(8) unsigned long long full[6], empty[6];
    __shared__ unsigned xrow[BT];
    const unsigned tid = threadIdx.x, wg = tid >> 7;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], BT), idx_len - start);
    const unsigned row0 = blockIdx.x * 128u;
    if (count == 0 || row0 >= N) return;
    const unsigned KC = K / 64u;
    for (unsigned q = tid; q < BT; q += blockDim.x) xrow[q] = min(xidx[start + min(q, count - 1u)], x_rows - 1u);
    if (tid == 0) {
        for (unsigned s = 0; s < STAGES; ++s) {
            tf_mbar_init(&full[s], 129);   // the TMA thread's expect_tx arrival + 128 cp.async arrivals
            tf_mbar_init(&empty[s], 8);    // one per consumer warp
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (wg == 0) {
        // Loader. Thread lt copies 8 of the 1024 16-byte activation pieces per stage.
        const unsigned lt = tid;
        constexpr unsigned PER = BT * 8u / 128u;   // 16-byte activation pieces per loader thread
        const __half* src[PER];
        unsigned dst[PER];
#pragma unroll
        for (unsigned q = 0; q < PER; ++q) {
            const unsigned c = lt + q * 128u, j = c >> 3, part = c & 7u;
            src[q] = xh + (u64)xrow[j] * K + part * 8u;
            dst[q] = j * 64u + ((part ^ (j & 7u)) * 8u);
        }
        const int wrow = (int)(e * N + row0);
        for (unsigned i = 0; i < KC; ++i) {
            const unsigned s = i % STAGES;
            if (i >= STAGES) tf_mbar_wait(&empty[s], ((i / STAGES) - 1u) & 1u);
            __half* sa = reinterpret_cast<__half*>(smb + s * STAGE_BYTES);
            __half* sbb = sa + 128u * 64u;
            if (lt == 0) {
                tf_mbar_expect_tx(&full[s], 128u * 64u * 2u);
                tf_tma_2d(sa, &tw, (int)(i * 64u), wrow, &full[s]);
            }
            const unsigned k0 = i * 64u;
#pragma unroll
            for (unsigned q = 0; q < PER; ++q) cp_async16(sbb + dst[q], src[q] + k0);
            tf_mbar_cp_async_arrive(&full[s]);
        }
        return;
    }

    const unsigned ct = tid - 128u, cw = ct >> 7, warp = ct >> 5, lane = ct & 31u, t = lane & 3u, g = lane >> 2;
    constexpr unsigned ND = BT / 2u;   // accumulators per thread
    float d[ND];
#pragma unroll
    for (unsigned i = 0; i < ND; ++i) d[i] = 0.0f;
    for (unsigned i = 0; i < KC; ++i) {
        const unsigned s = i % STAGES;
        tf_mbar_wait(&full[s], (i / STAGES) & 1u);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // cp.async data -> async proxy
        const __half* sa = reinterpret_cast<const __half*>(smb + s * STAGE_BYTES);
        const unsigned long long da = wgmma_desc(sa + cw * 64u * 64u), db = wgmma_desc(sa + 128u * 64u);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            if constexpr (BT == 256) wgmma_ss_m64n256k16(d, da + (unsigned long long)(ks * 2u), db + (unsigned long long)(ks * 2u));
            else wgmma_ss_m64n128k16(d, da + (unsigned long long)(ks * 2u), db + (unsigned long long)(ks * 2u));
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        if (i > 0) {
            __syncwarp();
            if (lane == 0) tf_mbar_arrive(&empty[(i - 1u) % STAGES]);
        }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const float ms = mat_scale ? mat_scale[e] : 1.0f;
    const unsigned r_lo = row0 + warp * 16u + g, r_hi = r_lo + 8u;
#pragma unroll
    for (unsigned nt = 0; nt < BT / 8u; ++nt) {
        const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
        if (j0 < count) {
            float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
            yr[r_lo] = d[4 * nt] * ms;
            yr[r_hi] = d[4 * nt + 2] * ms;
        }
        if (j1 < count) {
            float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
            yr[r_lo] = d[4 * nt + 1] * ms;
            yr[r_hi] = d[4 * nt + 3] * ms;
        }
    }
#else
    __trap();
#endif
}
extern "C" __global__ void __launch_bounds__(384, 1) astral_gemm_f16_tma(
    const __grid_constant__ CUtensorMapStub tw, const __half* __restrict__ xh, float* __restrict__ y, unsigned N, unsigned K,
    const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx, unsigned idx_len,
    unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT(); gemm_f16_tma<128>(tw, xh, y, N, K, items, xidx, yidx, idx_len, x_rows, y_rows, mat_scale, n_mats); }
extern "C" __global__ void __launch_bounds__(384, 1) astral_gemm_f16_tma_t256(
    const __grid_constant__ CUtensorMapStub tw, const __half* __restrict__ xh, float* __restrict__ y, unsigned N, unsigned K,
    const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx, unsigned idx_len,
    unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats) { PDL_WAIT(); gemm_f16_tma<256>(tw, xh, y, N, K, items, xidx, yidx, idx_len, x_rows, y_rows, mat_scale, n_mats); }

// =============================================================================================
// NVFP4 expert GEMM for prefill, warp-specialised with 256-row blocks (fragment layout, as
// astral_gemm_nvfp4_wgmma2). Warpgroup 0 copies each K stage (16 row tiles of A fragments and
// scales, and the item's <= 128 activation rows) into a ring of FW_STAGES stages with cp.async,
// completing the stage's "full" mbarrier. Warpgroups 1 and 2 each own 128 rows: every warp
// dequantises two 16-row tiles (rows 16w and 128 + 16w of the block) into registers (two sets,
// alternating stages, as the MMAs read them asynchronously) and issues two m64n128k16 per k16
// step on the same activation tile, then releases the stage once its MMAs retire.
// grid = (N / 256, n_items), 384 threads; N % 256 == 0, K % 64 == 0, items of <= 128 tokens.
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// One warp's A fragments of one 16-row tile for a 64-column stage (4 k16 steps), dequantised
// to f16 pairs times 2^-14 (see astral_gemm_nvfp4_mma5).
__device__ __forceinline__ void g10_dequant(const uint4 wq, const uint2 sb, unsigned (&a)[4][4]) {
    const __half2 k8 = __float2half2_rn(256.0f);
    const unsigned w4[4] = {wq.x, wq.y, wq.z, wq.w};
    const unsigned s4[2] = {sb.x, sb.y};
#pragma unroll
    for (unsigned ks = 0; ks < 4; ++ks) {
        const unsigned sbytes = (s4[ks >> 1] >> ((ks & 1u) * 16u)) & 0xffffu;
        const unsigned sbits = ((sbytes & 0xffu) << 7) | ((sbytes >> 8) << 23);
        const __half2 s2 = __hmul2(*reinterpret_cast<const __half2*>(&sbits), k8);
        const __half2 dl = __low2half2(s2), dh = __high2half2(s2);
        const unsigned w = w4[ks];
        a[ks][0] = hmul2u(fp4pair(w), dl);
        a[ks][1] = hmul2u(fp4pair(w << 4), dh);
        a[ks][2] = hmul2u(fp4pair(w << 8), dl);
        a[ks][3] = hmul2u(fp4pair(w << 12), dh);
    }
}
#endif
#define FW_STAGES 5
struct __align__(1024) FWStage {
    __half b[128 * 64];   // activations, 128-byte swizzle (16 KB)
    uint4 a[16 * 32];     // 16 row tiles x 32 lanes (8 KB)
    uint2 sc[16 * 8];     // 1 KB
};

extern "C" __global__ void __launch_bounds__(384, 1) astral_gemm_nvfp4_ws256(
    const u8* __restrict__ frag, const u8* __restrict__ fscale, const __half* __restrict__ xh, float* __restrict__ y, unsigned N,
    unsigned K, const unsigned* __restrict__ items, const unsigned* __restrict__ xidx, const unsigned* __restrict__ yidx,
    unsigned idx_len, unsigned x_rows, unsigned y_rows, const float* __restrict__ mat_scale, unsigned n_mats, unsigned unused) { PDL_WAIT();
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    extern __shared__ __align__(1024) unsigned char fwsmem[];
    FWStage* sm = reinterpret_cast<FWStage*>(fwsmem);
    __shared__ __align__(8) unsigned long long full[FW_STAGES], empty[FW_STAGES];
    __shared__ unsigned xrow[128];
    const unsigned tid = threadIdx.x, wg = tid >> 7;
    const unsigned* it = items + 3u * blockIdx.y;
    const unsigned e = min(it[0], n_mats - 1u);
    const unsigned start = min(it[1], idx_len);
    const unsigned count = min(min(it[2], 128u), idx_len - start);
    const unsigned row0 = blockIdx.x * 256u;
    if (count == 0 || row0 >= N) return;
    const unsigned KC = K / 64u, NT = N / 16u;
    if (tid < 128) xrow[tid] = min(xidx[start + min(tid, count - 1u)], x_rows - 1u);
    if (tid == 0) {
        for (unsigned s = 0; s < FW_STAGES; ++s) {
            tf_mbar_init(&full[s], 128);
            tf_mbar_init(&empty[s], 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    const uint4* fa = reinterpret_cast<const uint4*>(frag) + ((u64)e * NT + blockIdx.x * 16u) * KC * 32u;
    const uint2* fs = reinterpret_cast<const uint2*>(fscale) + ((u64)e * NT + blockIdx.x * 16u) * KC * 8u;

    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");   // the loader needs few registers
        const unsigned lt = tid;
        const uint4* a_src[4];
        unsigned a_dst[4];
#pragma unroll
        for (unsigned q = 0; q < 4; ++q) {
            const unsigned idx = lt + q * 128u;   // tile idx / 32, lane idx % 32
            a_src[q] = fa + (u64)(idx >> 5) * KC * 32u + (idx & 31u);
            a_dst[q] = idx;
        }
        const uint4* s_src = reinterpret_cast<const uint4*>(fs + (u64)((lt & 63u) >> 2) * KC * 8u) + (lt & 3u);
        const __half* b_src[8];
        unsigned b_dst[8];
#pragma unroll
        for (unsigned q = 0; q < 8; ++q) {
            const unsigned c = lt + q * 128u, j = c >> 3, part = c & 7u;
            b_src[q] = xh + (u64)xrow[j] * K + part * 8u;
            b_dst[q] = j * 64u + ((part ^ (j & 7u)) * 8u);
        }
        for (unsigned i = 0; i < KC; ++i) {
            const unsigned s = i % FW_STAGES;
            if (i >= FW_STAGES) tf_mbar_wait(&empty[s], ((i / FW_STAGES) - 1u) & 1u);
            FWStage& S = sm[s];
#pragma unroll
            for (unsigned q = 0; q < 4; ++q) cp_async16(&S.a[a_dst[q]], a_src[q] + i * 32u);
            if (lt < 64) cp_async16(reinterpret_cast<uint4*>(S.sc) + lt, s_src + i * 2u);
            const unsigned k0 = i * 64u;
#pragma unroll
            for (unsigned q = 0; q < 8; ++q) cp_async16(S.b + b_dst[q], b_src[q] + k0);
            tf_mbar_cp_async_arrive(&full[s]);
        }
        return;
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");   // two accumulator tiles + two A sets
    const unsigned ct = tid - 128u, warp = ct >> 5, lane = ct & 31u, t = lane & 3u, g = lane >> 2;
    // Warp w of the consumers: row tiles w (rows 16w) and w + 8 (rows 128 + 16w).
    float d0[64], d1[64];
#pragma unroll
    for (unsigned i = 0; i < 64; ++i) { d0[i] = 0.0f; d1[i] = 0.0f; }
    unsigned a0[2][4][4], a1[2][4][4];
    auto step = [&](unsigned i, unsigned (&a)[2][4][4]) {
        const unsigned s = i % FW_STAGES;
        tf_mbar_wait(&full[s], (i / FW_STAGES) & 1u);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const FWStage& S = sm[s];
        g10_dequant(S.a[warp * 32u + lane], S.sc[warp * 8u + g], a[0]);
        g10_dequant(S.a[(warp + 8u) * 32u + lane], S.sc[(warp + 8u) * 8u + g], a[1]);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const unsigned long long db = wgmma_desc(S.b);
#pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            wgmma_m64n128k16(d0, a[0][ks], db + (unsigned long long)(ks * 2u));
            wgmma_m64n128k16(d1, a[1][ks], db + (unsigned long long)(ks * 2u));
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        if (i > 0) {
            __syncwarp();
            if (lane == 0) tf_mbar_arrive(&empty[(i - 1u) % FW_STAGES]);
        }
    };
    unsigned i = 0;
    for (; i + 1 < KC; i += 2) {
        step(i, a0);
        step(i + 1, a1);
    }
    if (i < KC) step(i, a0);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const float ms = (mat_scale ? mat_scale[e] : 1.0f) * 16384.0f;
#pragma unroll
    for (unsigned h = 0; h < 2; ++h) {
        const float* d = h ? d1 : d0;
        const unsigned r_lo = row0 + h * 128u + warp * 16u + g, r_hi = r_lo + 8u;
#pragma unroll
        for (unsigned nt = 0; nt < 16; ++nt) {
            const unsigned j0 = nt * 8u + 2u * t, j1 = j0 + 1u;
            if (j0 < count) {
                float* yr = y + (u64)min(yidx[start + j0], y_rows - 1u) * N;
                yr[r_lo] = d[4 * nt] * ms;
                yr[r_hi] = d[4 * nt + 2] * ms;
            }
            if (j1 < count) {
                float* yr = y + (u64)min(yidx[start + j1], y_rows - 1u) * N;
                yr[r_lo] = d[4 * nt + 1] * ms;
                yr[r_hi] = d[4 * nt + 3] * ms;
            }
        }
    }
#else
    __trap();
#endif
}
