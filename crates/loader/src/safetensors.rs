// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Minimal, bounds-checked safetensors reader (single file): an 8-byte little-endian header
//! length, a JSON header `{name: {dtype, shape, data_offsets: [begin, end]}, ...}`, then the
//! data. Tensors are read on demand with positioned reads (no mmap).

use astral_core::{AstralError, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

fn bad(m: String) -> AstralError {
    AstralError::Invalid(format!("safetensors: {m}"))
}

#[derive(Debug, Clone)]
pub struct StTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute byte offset in the file and length.
    pub offset: u64,
    pub len: u64,
    /// Which file of the set holds it (always 0 for a single file).
    pub shard: usize,
}

impl StTensor {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

pub struct Safetensors {
    files: Vec<File>,
    tensors: BTreeMap<String, StTensor>,
    pub metadata: BTreeMap<String, String>,
}

impl Safetensors {
    /// A checkpoint directory: `model.safetensors`, or the shards listed by
    /// `model.safetensors.index.json` (or every `*.safetensors` file, `mtp.safetensors` aside).
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let single = dir.join("model.safetensors");
        if single.exists() {
            return Self::open(single);
        }
        let mut names: Vec<String> = Vec::new();
        if let Ok(b) = std::fs::read(dir.join("model.safetensors.index.json")) {
            let j: serde_json::Value = serde_json::from_slice(&b).map_err(|e| bad(format!("index JSON: {e}")))?;
            for f in j["weight_map"].as_object().ok_or_else(|| bad("index has no weight_map".into()))?.values().filter_map(|v| v.as_str()) {
                if !names.iter().any(|n| n == f) {
                    names.push(f.to_string());
                }
            }
        } else {
            for e in std::fs::read_dir(dir)? {
                let n = e?.file_name().to_string_lossy().to_string();
                if n.ends_with(".safetensors") && n != "mtp.safetensors" {
                    names.push(n);
                }
            }
            names.sort();
        }
        if names.is_empty() {
            return Err(bad(format!("no safetensors files in {}", dir.display())));
        }
        let mut all: Option<Self> = None;
        for n in &names {
            let mut one = Self::open(dir.join(n))?;
            match all.as_mut() {
                None => all = Some(one),
                Some(a) => {
                    let shard = a.files.len();
                    a.files.append(&mut one.files);
                    for (k, mut t) in std::mem::take(&mut one.tensors) {
                        t.shard = shard;
                        a.tensors.insert(k, t);
                    }
                }
            }
        }
        all.ok_or_else(|| bad("no safetensors files".into()))
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let size = file.metadata()?.len();
        let mut lenb = [0u8; 8];
        file.read_exact_at(&mut lenb, 0)?;
        let hlen = u64::from_le_bytes(lenb);
        if hlen > 512 << 20 || 8 + hlen > size {
            return Err(bad(format!("header of {hlen} bytes does not fit a {size}-byte file")));
        }
        let mut hb = vec![0u8; hlen as usize];
        file.read_exact_at(&mut hb, 8)?;
        let h: serde_json::Value = serde_json::from_slice(&hb).map_err(|e| bad(format!("header JSON: {e}")))?;
        let obj = h.as_object().ok_or_else(|| bad("header is not an object".into()))?;
        let base = 8 + hlen;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (k, x) in m {
                        metadata.insert(k.clone(), x.as_str().unwrap_or_default().to_string());
                    }
                }
                continue;
            }
            let dtype = v["dtype"].as_str().ok_or_else(|| bad(format!("{name}: no dtype")))?.to_string();
            let shape: Vec<u64> = v["shape"].as_array().ok_or_else(|| bad(format!("{name}: no shape")))?.iter().map(|x| x.as_u64().unwrap_or(u64::MAX)).collect();
            let off = v["data_offsets"].as_array().ok_or_else(|| bad(format!("{name}: no data_offsets")))?;
            let (a, b) = (off.first().and_then(|x| x.as_u64()), off.get(1).and_then(|x| x.as_u64()));
            let (Some(a), Some(b)) = (a, b) else { return Err(bad(format!("{name}: bad data_offsets"))) };
            if a > b || base + b > size || shape.contains(&u64::MAX) {
                return Err(bad(format!("{name}: data [{a}, {b}) outside the file")));
            }
            let esize = match dtype.as_str() {
                "F32" | "I32" | "U32" => 4,
                "F16" | "BF16" | "I16" | "U16" => 2,
                "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "BOOL" => 1,
                "F64" | "I64" | "U64" => 8,
                _ => 0,
            };
            let n: u64 = shape.iter().product();
            if esize != 0 && n.checked_mul(esize) != Some(b - a) {
                return Err(bad(format!("{name}: {} bytes for {n} x {dtype}", b - a)));
            }
            tensors.insert(name.clone(), StTensor { name: name.clone(), dtype, shape, offset: base + a, len: b - a, shard: 0 });
        }
        Ok(Self { files: vec![file], tensors, metadata })
    }

    pub fn tensor(&self, name: &str) -> Option<&StTensor> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    pub fn read(&self, t: &StTensor) -> Result<Vec<u8>> {
        let mut v = vec![0u8; t.len as usize];
        self.files[t.shard].read_exact_at(&mut v, t.offset)?;
        Ok(v)
    }

    /// A tensor converted to f32 (F32, BF16 or F16).
    pub fn read_f32(&self, t: &StTensor) -> Result<Vec<f32>> {
        let b = self.read(t)?;
        Ok(match t.dtype.as_str() {
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F16" => b.chunks_exact(2).map(|c| astral_quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            d => return Err(bad(format!("{}: dtype {d} is not a float type", t.name))),
        })
    }
}
