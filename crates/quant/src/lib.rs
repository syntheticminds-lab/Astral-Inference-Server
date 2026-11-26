// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-quant`: GGML tensor types and CPU reference dequantisation (spec 03 §3).
//!
//! The reference decoders are written from the GGML block-format definitions
//! (llama.cpp/ggml `ggml-common.h`, `ggml-quants.c`, `ggml-impl.h`; MIT License,
//! Copyright (c) 2023-2026 The ggml authors) and must agree bit-for-bit with them:
//! every GPU kernel for these formats is tested against the functions here.

#![forbid(unsafe_code)]

use astral_core::{AstralError, Result};

/// GGML tensor element types, by their on-disk `ggml_type` id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> Self {
        use GgmlType::*;
        match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            19 => Iq1S,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => Iq1M,
            30 => Bf16,
            34 => Tq1_0,
            35 => Tq2_0,
            39 => Mxfp4,
            40 => Nvfp4,
            other => Unknown(other),
        }
    }

    pub fn name(self) -> String {
        use GgmlType::*;
        match self {
            F32 => "f32",
            F16 => "f16",
            Q4_0 => "q4_0",
            Q4_1 => "q4_1",
            Q5_0 => "q5_0",
            Q5_1 => "q5_1",
            Q8_0 => "q8_0",
            Q8_1 => "q8_1",
            Q2K => "q2_K",
            Q3K => "q3_K",
            Q4K => "q4_K",
            Q5K => "q5_K",
            Q6K => "q6_K",
            Q8K => "q8_K",
            Iq2Xxs => "iq2_xxs",
            Iq2Xs => "iq2_xs",
            Iq3Xxs => "iq3_xxs",
            Iq1S => "iq1_s",
            Iq4Nl => "iq4_nl",
            Iq3S => "iq3_s",
            Iq2S => "iq2_s",
            Iq4Xs => "iq4_xs",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            F64 => "f64",
            Iq1M => "iq1_m",
            Bf16 => "bf16",
            Tq1_0 => "tq1_0",
            Tq2_0 => "tq2_0",
            Mxfp4 => "mxfp4",
            Nvfp4 => "nvfp4",
            Unknown(id) => return format!("unknown({id})"),
        }
        .to_string()
    }

    /// (elements per block, bytes per block). `None` for types whose layout Astral
    /// does not know; the loader rejects tensors of those types.
    pub fn layout(self) -> Option<(usize, usize)> {
        use GgmlType::*;
        Some(match self {
            F32 | I32 => (1, 4),
            F16 | Bf16 | I16 => (1, 2),
            I8 => (1, 1),
            I64 | F64 => (1, 8),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            Iq4Nl => (32, 18),
            Mxfp4 => (32, 17),
            Nvfp4 => (64, 36),
            Q2K => (256, 84),
            Q3K => (256, 110),
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Q8K => (256, 292),
            Iq4Xs => (256, 136),
            _ => return None,
        })
    }

    /// Whether [`dequantize`] supports this type.
    pub fn can_dequantize(self) -> bool {
        matches!(self, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 | GgmlType::Q8_0 | GgmlType::Nvfp4 | GgmlType::Mxfp4)
    }
}

/// IEEE 754 half precision to f32 (exact, including subnormals, inf and NaN).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: value = m * 2^-24. With p = index of m's top bit,
            // value = 2^(p-24) * (1 + rest), so the f32 exponent is p - 24 + 127.
            let p = 31 - m.leading_zeros();
            sign | ((p + 127 - 24) << 23) | ((m << (23 - p)) & 0x7f_ffff)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 to IEEE half, round-to-nearest-even (matches CUDA `__float2half_rn`), with
/// overflow to infinity, subnormals, and NaN preserved as a quiet NaN.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | if man == 0 { 0x7c00 } else { 0x7e00 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal (or zero) in half: shift the full significand into place, then round.
        if e < -10 {
            return sign;
        }
        let full = man | 0x80_0000;
        let shift = (14 - e) as u32; // bits dropped
        let mut h = full >> shift;
        let rem = full & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1; // may carry into the exponent, which is the correct rounding (incl. to inf)
    }
    sign | h as u16
}

/// Rounds an f32 to the nearest f16 value and back (what a kernel sees after `__float2half_rn`).
pub fn round_through_f16(f: f32) -> f32 {
    f16_to_f32(f32_to_f16(f))
}

/// bfloat16 to f32 (exact).
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// E2M1 values stored doubled, as integers, exactly as ggml's `kvalues_fp4`.
pub const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// NVFP4 sub-block scale, decoded exactly as ggml's `ggml_ue4m3_to_fp32`:
/// unsigned E4M3 (bias 7, 3 mantissa bits, sign bit ignored), `0x00` and `0x7F`
/// decode to zero, and the result is halved to pair with the doubled [`KVALUES_FP4`].
pub fn ue4m3_scale_ggml(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xf) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
    raw * 0.5
}

/// MXFP4 block scale: E8M0, halved to pair with the doubled table (ggml `GGML_E8M0_TO_FP32_HALF`).
pub fn e8m0_scale_half(e: u8) -> f32 {
    // 2^(e - 127) / 2 = 2^(e - 128); e = 0 gives 2^-128 (subnormal), computed exactly.
    let exp = e as i32 - 128;
    if exp >= -126 {
        f32::from_bits(((exp + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (exp + 149))
    }
}

/// Q8_0: `{ f16 d; i8 qs[32] }` -> `qs[j] * d`.
pub fn dequant_q8_0(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for j in 0..32 {
        out[j] = (block[2 + j] as i8) as f32 * d;
    }
}

/// NVFP4 (64 elements): `{ u8 d[4]; u8 qs[32] }`. Four 16-element sub-blocks, each with
/// its own UE4M3 scale. In sub-block `s`, byte `qs[s*8 + j]` holds element `s*16 + j`
/// in its low nibble and element `s*16 + j + 8` in its high nibble.
pub fn dequant_nvfp4(block: &[u8], out: &mut [f32]) {
    for s in 0..4 {
        let d = ue4m3_scale_ggml(block[s]);
        for j in 0..8 {
            let q = block[4 + s * 8 + j];
            out[s * 16 + j] = KVALUES_FP4[(q & 0x0f) as usize] as f32 * d;
            out[s * 16 + j + 8] = KVALUES_FP4[(q >> 4) as usize] as f32 * d;
        }
    }
}

/// MXFP4 (32 elements): `{ u8 e; u8 qs[16] }`; low nibble -> `j`, high nibble -> `j + 16`.
pub fn dequant_mxfp4(block: &[u8], out: &mut [f32]) {
    let d = e8m0_scale_half(block[0]);
    for j in 0..16 {
        let q = block[1 + j];
        out[j] = KVALUES_FP4[(q & 0x0f) as usize] as f32 * d;
        out[j + 16] = KVALUES_FP4[(q >> 4) as usize] as f32 * d;
    }
}

/// Repacks GGUF MXFP4 blocks (`{ e; qs[16] }`, 17 bytes, interleaved) into Astral's kernel
/// layout: a 16-byte-aligned value plane (`qs`, 16 bytes per block, same nibble order) and a
/// separate E8M0 scale plane (`es`, 1 byte per block). Load-time data movement only.
pub fn repack_mxfp4(blocks: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    if !blocks.len().is_multiple_of(17) {
        return Err(AstralError::Invalid(format!("mxfp4: {} bytes is not a whole number of 17-byte blocks", blocks.len())));
    }
    let n = blocks.len() / 17;
    let mut qs = Vec::with_capacity(n * 16);
    let mut es = Vec::with_capacity(n);
    for b in blocks.as_chunks::<17>().0 {
        if b[0] > 240 {
            // Scales of 2^114 and above: never produced by real quantizers, and beyond what
            // the GPU kernels decode exactly (they fold a 2^14 factor into the scale).
            return Err(AstralError::Unsupported(format!("mxfp4: block scale exponent {} > 240", b[0])));
        }
        es.push(b[0]);
        qs.extend_from_slice(&b[1..]);
    }
    Ok((qs, es))
}

/// Repacks GGUF Q8_0 blocks (`{ f16 d; i8 qs[32] }`, 34 bytes) into a 16-byte-aligned value
/// plane (32 bytes per block) and an f16 scale plane (2 bytes per block).
pub fn repack_q8_0(blocks: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    if !blocks.len().is_multiple_of(34) {
        return Err(AstralError::Invalid(format!("q8_0: {} bytes is not a whole number of 34-byte blocks", blocks.len())));
    }
    let n = blocks.len() / 34;
    let (mut qs, mut d) = (Vec::with_capacity(n * 32), Vec::with_capacity(n * 2));
    for b in blocks.as_chunks::<34>().0 {
        d.extend_from_slice(&b[..2]);
        qs.extend_from_slice(&b[2..]);
    }
    Ok((qs, d))
}

/// Repacks GGUF NVFP4 blocks (`{ u8 d[4]; u8 qs[32] }`, 36 bytes per 64 elements) into a
/// 16-byte-aligned value plane (32 bytes per block, same nibble order) and a scale plane
/// (4 UE4M3 bytes per block, one per 16-element sub-block).
pub fn repack_nvfp4(blocks: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    if !blocks.len().is_multiple_of(36) {
        return Err(AstralError::Invalid(format!("nvfp4: {} bytes is not a whole number of 36-byte blocks", blocks.len())));
    }
    let n = blocks.len() / 36;
    let (mut qs, mut d) = (Vec::with_capacity(n * 32), Vec::with_capacity(n * 4));
    for b in blocks.as_chunks::<36>().0 {
        // Canonical scales: ggml ignores bit 7 and decodes 0x7F as zero; store exactly that,
        // so a kernel may decode a scale as the f16 with bits `x << 7` (= value * 2^-8).
        d.extend(b[..4].iter().map(|&x| match x & 0x7f {
            0x7f => 0,
            v => v,
        }));
        qs.extend_from_slice(&b[4..]);
    }
    Ok((qs, d))
}

/// Tensor-core weight layout for MXFP4 (`mma.sync.m16n8k16`, A operand = weights).
///
/// The `n x k` matrix (`n % 16 == 0`, `k % 64 == 0`) is cut into tiles of 16 rows x 64
/// columns (two 32-element MXFP4 blocks). Each tile is 512 value bytes plus 32 scale bytes,
/// tiles ordered row-tile major. In a tile, lane `L = 4g + t` of a warp owns 16 contiguous
/// bytes: four words `[row g / block 0, row g+8 / block 0, row g / block 1, row g+8 / block 1]`,
/// each holding bytes `(2t, 2t+1, 2t+8, 2t+9)` of that row's 16-byte block. Low nibbles give
/// the A fragment for the block's columns 0..15, high nibbles for 16..31. Scales: byte
/// `4g + {0,1,2,3}` = E8M0 of (row g, blk 0), (row g+8, blk 0), (row g, blk 1), (row g+8, blk 1).
/// Input is GGUF MXFP4 blocks, row-major (`k / 32` blocks per row).
pub fn repack_mxfp4_mma(blocks: &[u8], n: usize, k: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    if !n.is_multiple_of(16) || !k.is_multiple_of(64) || blocks.len() != n * k / 32 * 17 {
        return Err(AstralError::Invalid(format!("mxfp4 mma repack: need n % 16 == 0, k % 64 == 0 and {} bytes (got n {n}, k {k}, {} bytes)", n * k / 32 * 17, blocks.len())));
    }
    let bpr = k / 32; // blocks per row
    let blk = |r: usize, b: usize| &blocks[(r * bpr + b) * 17..(r * bpr + b + 1) * 17];
    let (tiles_r, tiles_k) = (n / 16, k / 64);
    let mut q = vec![0u8; tiles_r * tiles_k * 512];
    let mut sc = vec![0u8; tiles_r * tiles_k * 32];
    for tr in 0..tiles_r {
        for tk in 0..tiles_k {
            let tile = tr * tiles_k + tk;
            for g in 0..8 {
                for (slot, (dr, db)) in [(0usize, 0usize), (8, 0), (0, 1), (8, 1)].iter().enumerate() {
                    let b = blk(tr * 16 + g + dr, tk * 2 + db);
                    if b[0] > 240 {
                        return Err(AstralError::Unsupported(format!("mxfp4: block scale exponent {} > 240", b[0])));
                    }
                    sc[tile * 32 + g * 4 + slot] = b[0];
                    for t in 0..4 {
                        let lane = g * 4 + t;
                        let dst = tile * 512 + lane * 16 + slot * 4;
                        let qs = &b[1..];
                        q[dst..dst + 4].copy_from_slice(&[qs[2 * t], qs[2 * t + 1], qs[2 * t + 8], qs[2 * t + 9]]);
                    }
                }
            }
        }
    }
    Ok((q, sc))
}

/// Dequantises `n` elements of type `ty` from `bytes` (which must hold exactly that many).
pub fn dequantize(ty: GgmlType, bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let (blk, sz) = ty.layout().ok_or_else(|| AstralError::Unsupported(format!("layout of {}", ty.name())))?;
    if !n.is_multiple_of(blk) || bytes.len() != n / blk * sz {
        return Err(AstralError::Invalid(format!("{}: {} bytes do not hold {n} elements", ty.name(), bytes.len())));
    }
    let mut out = vec![0f32; n];
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = f16_to_f32(u16::from_le_bytes(*c));
            }
        }
        GgmlType::Bf16 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = bf16_to_f32(u16::from_le_bytes(*c));
            }
        }
        GgmlType::Q8_0 => {
            for (o, b) in out.as_chunks_mut::<32>().0.iter_mut().zip(bytes.as_chunks::<34>().0) {
                dequant_q8_0(b, o);
            }
        }
        GgmlType::Nvfp4 => {
            for (o, b) in out.as_chunks_mut::<64>().0.iter_mut().zip(bytes.as_chunks::<36>().0) {
                dequant_nvfp4(b, o);
            }
        }
        GgmlType::Mxfp4 => {
            for (o, b) in out.as_chunks_mut::<32>().0.iter_mut().zip(bytes.as_chunks::<17>().0) {
                dequant_mxfp4(b, o);
            }
        }
        other => return Err(AstralError::Unsupported(format!("dequantize {}", other.name()))),
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24)); // largest subnormal
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14)); // smallest normal
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000); // -0
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn f16_exhaustive_against_formula() {
        // All 65,536 bit patterns against the IEEE definition evaluated in f64.
        for h in 0u32..=0xffff {
            let h = h as u16;
            let (s, e, m) = ((h >> 15) as i32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f64);
            let got = f16_to_f32(h);
            let sign = if s == 1 { -1.0 } else { 1.0 };
            match e {
                0x1f if m == 0.0 => assert!(got.is_infinite() && (got < 0.0) == (s == 1), "{h:#06x}"),
                0x1f => assert!(got.is_nan(), "{h:#06x}"),
                0 => assert_eq!(got as f64, sign * m * 2f64.powi(-24), "{h:#06x}"),
                _ => assert_eq!(got as f64, sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15), "{h:#06x}"),
            }
        }
    }

    #[test]
    fn bf16_known_values() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc040), -3.0);
    }

    #[test]
    fn ue4m3_matches_ggml_definition() {
        assert_eq!(ue4m3_scale_ggml(0x00), 0.0);
        assert_eq!(ue4m3_scale_ggml(0x7f), 0.0);
        assert_eq!(ue4m3_scale_ggml(0x38), 0.5); // exp 7, man 0: raw 1.0, halved
        assert_eq!(ue4m3_scale_ggml(0x01), 2f32.powi(-9) * 0.5); // subnormal
        assert_eq!(ue4m3_scale_ggml(0x7e), 224.0); // raw 448, halved
        assert_eq!(ue4m3_scale_ggml(0xb8), ue4m3_scale_ggml(0x38), "sign bit ignored");
        // Monotonic over the valid range.
        let mut prev = 0.0;
        for x in 1u8..0x7f {
            let v = ue4m3_scale_ggml(x);
            assert!(v > prev, "{x:#04x}");
            prev = v;
        }
    }

    #[test]
    fn nvfp4_nibble_order_and_scales() {
        // Sub-block scales 1.0, 2.0, 0.5, 4.0 (true values; ggml stores them halved).
        let mut b = [0u8; 36];
        b[0] = 0x38; // raw 1.0
        b[1] = 0x40; // raw 2.0
        b[2] = 0x30; // raw 0.5
        b[3] = 0x48; // raw 4.0
        for s in 0..4 {
            for j in 0..8 {
                // low nibble = index (j % 8) -> E2M1 {0,0.5,1,1.5,2,3,4,6}; high nibble = negative 6.0
                b[4 + s * 8 + j] = (j as u8) | (0xf << 4);
            }
        }
        let mut out = [0f32; 64];
        dequant_nvfp4(&b, &mut out);
        let e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let scales = [1.0, 2.0, 0.5, 4.0];
        for s in 0..4 {
            for j in 0..8 {
                assert_eq!(out[s * 16 + j], e2m1[j] * scales[s], "low s{s} j{j}");
                assert_eq!(out[s * 16 + j + 8], -6.0 * scales[s], "high s{s} j{j}");
            }
        }
    }

    #[test]
    fn q8_0_block() {
        let mut b = [0u8; 34];
        b[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
        for j in 0..32 {
            b[2 + j] = (j as i8 - 16) as u8;
        }
        let mut out = [0f32; 32];
        dequant_q8_0(&b, &mut out);
        for j in 0..32 {
            assert_eq!(out[j], (j as f32 - 16.0) * 0.5);
        }
    }

    #[test]
    fn mxfp4_block() {
        let mut b = [0u8; 17];
        b[0] = 127; // 2^0, halved -> 0.5 pairs with doubled table
        for j in 0..16 {
            b[1 + j] = 0x7 | (0x9 << 4); // +6.0 low, -0.5 high
        }
        let mut out = [0f32; 32];
        dequant_mxfp4(&b, &mut out);
        assert!(out[..16].iter().all(|&v| v == 6.0));
        assert!(out[16..].iter().all(|&v| v == -0.5));
        assert_eq!(e8m0_scale_half(0), 2f32.powi(-128));
        assert_eq!(e8m0_scale_half(254), 2f32.powi(126));
    }

    #[test]
    fn dequantize_checks_sizes() {
        assert!(dequantize(GgmlType::Q8_0, &[0u8; 34], 32).is_ok());
        assert!(dequantize(GgmlType::Q8_0, &[0u8; 33], 32).is_err());
        assert!(dequantize(GgmlType::Nvfp4, &[0u8; 36], 32).is_err(), "NVFP4 blocks are 64 elements");
        assert!(dequantize(GgmlType::Q4K, &[0u8; 144], 256).is_err(), "unsupported types error, not garbage");
    }

    #[test]
    fn repack_mxfp4_splits_planes() {
        let mut blocks = vec![0u8; 34];
        blocks[0] = 127;
        blocks[1..17].iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);
        blocks[17] = 130;
        blocks[18..34].iter_mut().for_each(|b| *b = 0xab);
        let (qs, es) = repack_mxfp4(&blocks).unwrap();
        assert_eq!(es, vec![127, 130]);
        assert_eq!(&qs[..16], &(0u8..16).collect::<Vec<_>>()[..]);
        assert!(qs[16..].iter().all(|&b| b == 0xab));
        assert!(repack_mxfp4(&blocks[..20]).is_err());
    }

    #[test]
    fn f32_to_f16_round_trips_every_half_and_rounds_to_even() {
        for h in 0u32..=0xffff {
            let h = h as u16;
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "{h:#06x}");
        }
        // Ties to even: 1 + 2^-11 is halfway between 1.0 (0x3c00) and the next half (0x3c01).
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(65520.0), 0x7c00, "rounds up to inf");
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000, "tie to even at the smallest subnormal");
        assert_eq!(f32_to_f16(1.5 * 2f32.powi(-24)), 0x0002);
    }

    #[test]
    fn mma_repack_places_every_nibble_and_scale() {
        // Give every block a unique scale and every byte a unique pattern, then check the
        // documented lane mapping for every element.
        let (n, k) = (32usize, 128usize);
        let bpr = k / 32;
        let mut blocks = vec![0u8; n * bpr * 17];
        for r in 0..n {
            for b in 0..bpr {
                let o = (r * bpr + b) * 17;
                blocks[o] = (100 + (r * bpr + b) % 100) as u8;
                for j in 0..16 {
                    blocks[o + 1 + j] = ((r * 7 + b * 13 + j * 31) % 256) as u8;
                }
            }
        }
        let (q, sc) = repack_mxfp4_mma(&blocks, n, k).unwrap();
        let tiles_k = k / 64;
        for r in 0..n {
            for c in 0..k {
                let (b, e) = (c / 32, c % 32);
                let byte = blocks[(r * bpr + b) * 17 + 1 + (e % 16)];
                let want = if e < 16 { byte & 0xf } else { byte >> 4 };
                // Locate through the layout.
                let (tr, tk) = (r / 16, c / 64);
                let (g, dr) = ((r % 16) % 8, (r % 16) / 8);
                let db = (c % 64) / 32;
                let col = e % 16; // column within the 16-wide mma k-tile
                let t = (col % 8) / 2;
                let pos = (col % 2) + if col >= 8 { 2 } else { 0 };
                let slot = dr + 2 * db;
                let got_byte = q[(tr * tiles_k + tk) * 512 + (g * 4 + t) * 16 + slot * 4 + pos];
                let got = if e < 16 { got_byte & 0xf } else { got_byte >> 4 };
                assert_eq!(got, want, "r{r} c{c}");
                assert_eq!(sc[(tr * tiles_k + tk) * 32 + g * 4 + slot], blocks[(r * bpr + b) * 17], "scale r{r} c{c}");
            }
        }
        assert!(repack_mxfp4_mma(&blocks, 24, k).is_err());
    }

    #[test]
    fn repack_q8_0_and_nvfp4_split_planes() {
        let mut q8 = vec![0u8; 68];
        q8[0..2].copy_from_slice(&[0x00, 0x3c]);
        q8[2..34].iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);
        q8[34..36].copy_from_slice(&[0x11, 0x22]);
        let (qs, d) = repack_q8_0(&q8).unwrap();
        assert_eq!(d, vec![0x00, 0x3c, 0x11, 0x22]);
        assert_eq!(&qs[..32], &(0u8..32).collect::<Vec<_>>()[..]);
        let mut nv = vec![0u8; 72];
        nv[0..4].copy_from_slice(&[1, 2, 3, 4]);
        nv[36..40].copy_from_slice(&[5, 6, 7, 8]);
        nv[40] = 0xee;
        let (qs, d) = repack_nvfp4(&nv).unwrap();
        assert_eq!(d, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!((qs.len(), qs[32]), (64, 0xee));
        assert!(repack_q8_0(&q8[..40]).is_err() && repack_nvfp4(&nv[..40]).is_err());
    }

    #[test]
    fn type_ids_round_trip_names() {
        assert_eq!(GgmlType::from_id(40), GgmlType::Nvfp4);
        assert_eq!(GgmlType::from_id(8).name(), "q8_0");
        assert_eq!(GgmlType::from_id(999), GgmlType::Unknown(999));
        assert_eq!(GgmlType::Nvfp4.layout(), Some((64, 36)));
    }
}

/// UE4M3 byte for 2^p, p in -9..=8 (subnormals below 2^-6). ggml decodes it with a 0.5
/// factor against doubled E2M1 values, so the decoded weight is exactly e2m1 * 2^p.
fn ue4m3_pow2(p: i32) -> u8 {
    match p {
        -9 => 1,
        -8 => 2,
        -7 => 4,
        -6..=8 => ((p + 7) as u8) << 3,
        _ => unreachable!("p {p} outside the UE4M3 power-of-two range"),
    }
}

/// E2M1 code nearest to `v` (round to nearest, ties to the even code).
fn e2m1_encode(v: f32) -> u8 {
    const VALS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let a = v.abs();
    let mut best = 0usize;
    for i in 1..8 {
        let (d, db) = ((a - VALS[i]).abs(), (a - VALS[best]).abs());
        if d < db || (d == db && i % 2 == 0) {
            best = i;
        }
    }
    (best as u8) | if v < 0.0 && best != 0 { 8 } else { 0 }
}

/// Result of [`mxfp4_ct_to_nvfp4`].
pub struct Mxfp4AsNvfp4 {
    /// NVFP4 value plane (`repack_nvfp4` layout: K/2 bytes per row, 8 bytes per 16 values,
    /// low nibble = value j, high = value j + 8).
    pub qs: Vec<u8>,
    /// UE4M3 scale per 16 values (K/16 bytes per row).
    pub aux: Vec<u8>,
    /// Per-matrix factor the kernels apply to every output (2^offset).
    pub mat_scale: f32,
    /// Blocks whose exponent was below the 18-binade UE4M3 window and were re-encoded.
    pub reencoded_blocks: usize,
}

/// Converts a compressed-tensors `mxfp4-pack-quantized` matrix (`packed` [n][k/2], value 2i in
/// the low nibble; `scales` [n][k/32] E8M0) to the NVFP4 layout of [`repack_nvfp4`].
///
/// Lossless when every block exponent is within 17 of the matrix maximum: an MXFP4 block is two
/// NVFP4 sub-blocks with the power-of-two scale 2^(e - emax + 8) (exact in UE4M3), and
/// `mat_scale = 2^(emax - 127 - 8)` restores the magnitude. Blocks below that window (rare) are
/// re-encoded at the smallest scale, rounding each value to the nearest E2M1 code.
pub fn mxfp4_ct_to_nvfp4(packed: &[u8], scales: &[u8], n: usize, k: usize) -> Result<Mxfp4AsNvfp4> {
    if !k.is_multiple_of(64) || packed.len() != n * k / 2 || scales.len() != n * k / 32 {
        return Err(AstralError::Invalid(format!("mxfp4 ct: n {n} k {k}: {} packed / {} scale bytes", packed.len(), scales.len())));
    }
    if scales.iter().any(|&e| e == 0xff) {
        return Err(AstralError::Invalid("mxfp4 ct: NaN (0xFF) block scale".into()));
    }
    let emax = *scales.iter().max().unwrap_or(&127) as i32;
    let mut qs = vec![0u8; n * k / 2];
    let mut aux = vec![0u8; n * k / 16];
    let mut reencoded = 0usize;
    let mut vals = [0u8; 32];
    for r in 0..n {
        for b in 0..k / 32 {
            let src = &packed[r * k / 2 + b * 16..r * k / 2 + b * 16 + 16];
            for (i, &byte) in src.iter().enumerate() {
                vals[2 * i] = byte & 0x0f;
                vals[2 * i + 1] = byte >> 4;
            }
            let mut p = scales[r * k / 32 + b] as i32 - emax + 8;
            if p < -9 {
                // Re-encode at 2^-9: each value shrinks by 2^(p + 9) (< 1).
                let f = 2f32.powi(p + 9);
                for v in vals.iter_mut() {
                    *v = e2m1_encode(e2m1_value(*v) * f);
                }
                p = -9;
                reencoded += 1;
            }
            let sc = ue4m3_pow2(p);
            for sb in 0..2 {
                let o = r * k / 2 + b * 16 + sb * 8;
                for j in 0..8 {
                    qs[o + j] = vals[sb * 16 + j] | (vals[sb * 16 + j + 8] << 4);
                }
                aux[r * k / 16 + b * 2 + sb] = sc;
            }
        }
    }
    Ok(Mxfp4AsNvfp4 { qs, aux, mat_scale: 2f32.powi(emax - 127 - 8), reencoded_blocks: reencoded })
}

/// The value of an E2M1 code.
pub fn e2m1_value(c: u8) -> f32 {
    const VALS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = VALS[(c & 7) as usize];
    if c & 8 != 0 {
        -v
    } else {
        v
    }
}

#[cfg(test)]
mod mxfp4_ct_tests {
    use super::*;

    #[test]
    fn mxfp4_ct_matches_nvfp4_decode() {
        // 2 rows x 64 values; exponents spanning the window, one below it.
        let (n, k) = (2usize, 64usize);
        let mut packed = vec![0u8; n * k / 2];
        for (i, b) in packed.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37) ^ 0x5a;
        }
        let scales = vec![120u8, 110, 127, 115];
        let c = mxfp4_ct_to_nvfp4(&packed, &scales, n, k).unwrap();
        assert_eq!(c.reencoded_blocks, 0);
        // Rebuild ggml NVFP4 blocks (4 x UE4M3 + 32 bytes per 64 values) and decode them.
        let mut blocks = Vec::new();
        for r in 0..n {
            blocks.extend_from_slice(&c.aux[r * 4..r * 4 + 4]);
            blocks.extend_from_slice(&c.qs[r * 32..r * 32 + 32]);
        }
        let mut got = vec![0f32; n * k];
        for (bi, blk) in blocks.chunks_exact(36).enumerate() {
            dequant_nvfp4(blk, &mut got[bi * 64..bi * 64 + 64]);
        }
        for r in 0..n {
            for j in 0..k {
                let byte = packed[r * k / 2 + j / 2];
                let code = if j % 2 == 0 { byte & 0xf } else { byte >> 4 };
                let want = e2m1_value(code) * 2f32.powi(scales[r * 2 + j / 32] as i32 - 127);
                assert_eq!(got[r * k + j] * c.mat_scale, want, "row {r} value {j}");
            }
        }
    }
}

/// f16 tensor-core fragment layout of a Q8_0 matrix (`repack_q8_0` planes: `qs` [n][k] int8,
/// `aux` [n][k/32] f16 scales) for the dense warpgroup-MMA GEMM (`astral_gemm_f16_wgmma`).
///
/// Tiles of 16 rows x 64 columns, ordered [row tile][column tile]. In a tile, lane `L` (g = L / 4,
/// t = L % 4) owns 32 f16 values: for each 16-column step `ks`, the 4 registers of its
/// `mma.m16n8k16` A fragment -- (row g, cols 2t, 2t+1), (row g+8, same), (row g, cols 2t+8, 2t+9),
/// (row g+8, same) -- each value `q * d` rounded once to f16 (as the mma.sync kernels compute it).
pub fn f16_frag_layout(qs: &[u8], aux: &[u8], n: usize, k: usize) -> Result<Vec<u16>> {
    if !n.is_multiple_of(16) || !k.is_multiple_of(64) || qs.len() != n * k || aux.len() != n * k / 16 {
        return Err(AstralError::Invalid(format!("f16 frag layout: n {n} k {k}: {} / {} bytes", qs.len(), aux.len())));
    }
    let (nt, kc_n) = (n / 16, k / 64);
    let mut out = vec![0u16; n * k];
    let val = |row: usize, col: usize| -> u16 {
        let q = qs[row * k + col] as i8 as f32;
        let o = (row * (k / 32) + col / 32) * 2;
        let d = f16_to_f32(u16::from_le_bytes([aux[o], aux[o + 1]]));
        f32_to_f16(q * d)
    };
    std::thread::scope(|sc| {
        let threads = std::thread::available_parallelism().map_or(8, |v| v.get()).min(32);
        let per = nt.div_ceil(threads);
        for (ci, chunk) in out.chunks_mut(per * kc_n * 32 * 32).enumerate() {
            let val = &val;
            sc.spawn(move || {
                for (i, slot) in chunk.chunks_mut(32 * 32).enumerate() {
                    let tile = ci * per * kc_n + i;
                    let (rt, kc) = (tile / kc_n, tile % kc_n);
                    for lane in 0..32 {
                        let (g, t) = (lane / 4, lane % 4);
                        let r = rt * 16 + g;
                        for ks in 0..4 {
                            let c0 = kc * 64 + ks * 16 + 2 * t;
                            let pairs = [(r, c0), (r + 8, c0), (r, c0 + 8), (r + 8, c0 + 8)];
                            for (p, &(row, col)) in pairs.iter().enumerate() {
                                let o = lane * 32 + ks * 8 + p * 2;
                                slot[o] = val(row, col);
                                slot[o + 1] = val(row, col + 1);
                            }
                        }
                    }
                }
            });
        }
    });
    Ok(out)
}

/// Tensor-core fragment layout of an NVFP4 matrix stack (`repack_nvfp4` planes: `qs`
/// [mats][n][k/2], `aux` [mats][n][k/16]) for the prefill GEMM (`astral_gemm_nvfp4_mma5`).
///
/// Tiles of 16 rows x 64 columns. In a tile, lane `L` (g = L / 4, t = L % 4) of a warp owns 16
/// bytes: one 32-bit word per 16-column step `ks`, holding the 8 E2M1 codes of its
/// `mma.m16n8k16` A fragment — pairs (row g, cols 2t, 2t+1), (row g+8, same), (row g,
/// cols 2t+8, 2t+9), (row g+8, same) — with pair p's two codes at bits 12-4p and 28-4p, so
/// `w << 4p` puts them where `(w & 0x80008000) | ((w >> 3) & 0x0E000E00)` turns them into an
/// f16 pair (value * 2^-14). Tiles are ordered [mat][row tile][column tile]; scales likewise,
/// 64 bytes per tile: byte `8g + 2ks + h` = UE4M3 of (row g + 8h, sub-block ks).
pub fn nvfp4_frag_layout(qs: &[u8], aux: &[u8], n: usize, k: usize, mats: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    if !n.is_multiple_of(16) || !k.is_multiple_of(64) || qs.len() != mats * n * k / 2 || aux.len() != mats * n * k / 16 {
        return Err(AstralError::Invalid(format!("nvfp4 frag layout: n {n} k {k} x{mats}: {} / {} bytes", qs.len(), aux.len())));
    }
    let (nt, kc_n) = (n / 16, k / 64);
    let mut frag = vec![0u8; mats * n * k / 2];
    let mut fs = vec![0u8; mats * n * k / 16];
    for e in 0..mats {
        let q = &qs[e * n * k / 2..(e + 1) * n * k / 2];
        let a = &aux[e * n * k / 16..(e + 1) * n * k / 16];
        let nib = |row: usize, col: usize| -> u32 {
            let sb = col / 16;
            let j = col % 16;
            let byte = q[row * (k / 2) + sb * 8 + (j % 8)];
            (if j < 8 { byte & 0x0f } else { byte >> 4 }) as u32
        };
        for rt in 0..nt {
            for kc in 0..kc_n {
                let tile = (e * nt + rt) * kc_n + kc;
                for lane in 0..32 {
                    let (g, t) = (lane / 4, lane % 4);
                    for ks in 0..4 {
                        let c0 = kc * 64 + ks * 16 + 2 * t;
                        let r = rt * 16 + g;
                        let pairs = [(r, c0), (r + 8, c0), (r, c0 + 8), (r + 8, c0 + 8)];
                        let mut w = 0u32;
                        for (p, &(row, col)) in pairs.iter().enumerate() {
                            w |= nib(row, col) << (12 - 4 * p);
                            w |= nib(row, col + 1) << (28 - 4 * p);
                        }
                        let o = (tile * 32 + lane) * 16 + ks * 4;
                        frag[o..o + 4].copy_from_slice(&w.to_le_bytes());
                    }
                }
                for g in 0..8 {
                    for ks in 0..4 {
                        for h in 0..2 {
                            fs[tile * 64 + g * 8 + ks * 2 + h] = a[(rt * 16 + g + 8 * h) * (k / 16) + kc * 4 + ks];
                        }
                    }
                }
            }
        }
    }
    Ok((frag, fs))
}

/// The values of a compressed-tensors MXFP4 matrix (`packed` [n][k/2], value 2i in the low nibble)
/// reordered to the NVFP4 nibble layout of [`repack_nvfp4`] (16-value sub-blocks, low nibble =
/// value j, high = value j + 8), unchanged otherwise: paired with the original E8M0 scales it is
/// exactly the checkpoint's MXFP4 matrix.
pub fn mxfp4_ct_reorder(packed: &[u8], n: usize, k: usize) -> Result<Vec<u8>> {
    if !k.is_multiple_of(32) || packed.len() != n * k / 2 {
        return Err(AstralError::Invalid(format!("mxfp4 reorder: n {n} k {k}: {} bytes", packed.len())));
    }
    let mut qs = vec![0u8; n * k / 2];
    for (blk, (src, dst)) in packed.chunks_exact(16).zip(qs.chunks_exact_mut(16)).enumerate() {
        let _ = blk;
        let mut vals = [0u8; 32];
        for (i, &b) in src.iter().enumerate() {
            vals[2 * i] = b & 0x0f;
            vals[2 * i + 1] = b >> 4;
        }
        for sb in 0..2 {
            for j in 0..8 {
                dst[sb * 8 + j] = vals[sb * 16 + j] | (vals[sb * 16 + j + 8] << 4);
            }
        }
    }
    Ok(qs)
}

/// Quantises f32 values (k % 64 == 0 per row; any number of rows) to GGML NVFP4 blocks (36
/// bytes per 64 values: four ue4m3 sub-block scales, then e2m1 nibbles over the doubled
/// [`KVALUES_FP4`] table), the input of [`repack_nvfp4`]. Each 16-value sub-block takes the
/// smallest scale that fits its largest magnitude; values round to the nearest level.
pub fn quantize_nvfp4(x: &[f32]) -> Result<Vec<u8>> {
    if !x.len().is_multiple_of(64) {
        return Err(AstralError::Invalid(format!("nvfp4: {} values is not a multiple of 64", x.len())));
    }
    // Scale codes by value (0x7F decodes to zero; skip it).
    let mut codes: Vec<(f32, u8)> = (1u8..0x7f).map(|c| (ue4m3_scale_ggml(c), c)).filter(|(v, _)| *v > 0.0).collect();
    codes.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::with_capacity(x.len() / 64 * 36);
    for blk in x.chunks_exact(64) {
        let mut scales = [0u8; 4];
        let mut qs = [0u8; 32];
        for sb in 0..4 {
            let v = &blk[sb * 16..sb * 16 + 16];
            let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            if amax == 0.0 {
                continue;
            }
            let want = amax / 12.0;
            let i = codes.partition_point(|(val, _)| *val < want).min(codes.len() - 1);
            let (d, code) = codes[i];
            scales[sb] = code;
            for (j, &xv) in v.iter().enumerate() {
                let t = xv / d;
                let mut best = 0usize;
                for (q, &kv) in KVALUES_FP4.iter().enumerate() {
                    if (t - kv as f32).abs() < (t - KVALUES_FP4[best] as f32).abs() {
                        best = q;
                    }
                }
                let byte = &mut qs[sb * 8 + (j % 8)];
                if j < 8 {
                    *byte |= best as u8;
                } else {
                    *byte |= (best as u8) << 4;
                }
            }
        }
        out.extend_from_slice(&scales);
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

/// Quantises f32 values to GGUF Q8_0 blocks (`{ f16 d; i8 qs[32] }`) exactly as ggml's
/// `quantize_row_q8_0_ref`: `d = amax / 127`, `q = round(x / d)` with the unrounded `d`, and
/// `d` stored as f16. Byte-identical to llama.cpp's output for the same input.
pub fn quantize_q8_0(x: &[f32]) -> Result<Vec<u8>> {
    if !x.len().is_multiple_of(32) {
        return Err(AstralError::Invalid(format!("q8_0: {} values is not a multiple of 32", x.len())));
    }
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend(blk.iter().map(|&v| ((v * id).round() as i8) as u8));
    }
    Ok(out)
}

/// Quantises f32 rows (`n` x `k`, k % 32 == 0) to Q8_0 in Astral's repacked layout (int8 values
/// [n][k], f16 block scales [n][k/32]; `d = amax / 127`, round to nearest), for a GEMV.
pub fn quantize_q8_0_rows(x: &[f32], n: usize, k: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    if !k.is_multiple_of(32) || x.len() != n * k {
        return Err(AstralError::Invalid(format!("q8_0 rows: n {n} k {k}: {} values", x.len())));
    }
    let (mut qs, mut ds) = (Vec::with_capacity(n * k), Vec::with_capacity(n * k / 16));
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let dh = f32_to_f16(d);
        let inv = if d > 0.0 { 1.0 / f16_to_f32(dh) } else { 0.0 };
        for &v in blk {
            qs.push(((v * inv).round().clamp(-127.0, 127.0) as i8) as u8);
        }
        ds.extend_from_slice(&dh.to_le_bytes());
    }
    Ok((qs, ds))
}

#[cfg(test)]
mod nvfp4_encode_tests {
    use super::*;

    #[test]
    fn q8_0_blocks_match_ggml() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 20.5) * 0.37).collect();
        let b = quantize_q8_0(&x).unwrap();
        assert_eq!(b.len(), 68);
        let mut y = vec![0f32; 64];
        for (blk, out) in b.chunks_exact(34).zip(y.chunks_exact_mut(32)) {
            dequant_q8_0(blk, out);
        }
        // Each value is within half a quantisation step (d / 2 = amax / 254) of the input.
        for (xb, yb) in x.chunks_exact(32).zip(y.chunks_exact(32)) {
            let half_step = xb.iter().fold(0f32, |m, v| m.max(v.abs())) / 254.0 * 1.001;
            for (a, c) in xb.iter().zip(yb) {
                assert!((a - c).abs() <= half_step, "{a} vs {c}");
            }
        }
        // The block's largest magnitude maps to +-127.
        assert_eq!(b[2] as i8, -127);
        assert!(quantize_q8_0(&x[..40]).is_err());
        assert_eq!(quantize_q8_0(&[0.0; 32]).unwrap(), vec![0u8; 34]);
    }

    #[test]
    fn nvfp4_round_trip_is_close() {
        // Pseudo-random values over a few orders of magnitude, and an all-zero sub-block.
        let mut x: Vec<f32> = (0..640).map(|i| ((i as f32 * 0.7311).sin() * 3.0) * 10f32.powi((i / 64) as i32 % 4 - 2)).collect();
        for v in &mut x[64..80] {
            *v = 0.0;
        }
        let blocks = quantize_nvfp4(&x).unwrap();
        assert_eq!(blocks.len(), x.len() / 64 * 36);
        let mut y = vec![0f32; x.len()];
        for (b, out) in blocks.chunks_exact(36).zip(y.chunks_exact_mut(64)) {
            dequant_nvfp4(b, out);
        }
        for sb in 0..x.len() / 16 {
            let (xs, ys) = (&x[sb * 16..sb * 16 + 16], &y[sb * 16..sb * 16 + 16]);
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            for (a, b) in xs.iter().zip(ys) {
                // e2m1 steps are at most 1/6 of the sub-block's range (plus scale rounding).
                assert!((a - b).abs() <= amax * 0.2 + 1e-30, "sub-block {sb}: {a} vs {b} (amax {amax})");
            }
        }
        assert!(y[64..80].iter().all(|&v| v == 0.0));
    }
}
