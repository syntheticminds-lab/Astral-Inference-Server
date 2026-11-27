// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-loader`: model weights from GGUF files and from safetensors checkpoints
//! (spec 06 §5, spec 08 §7).
//!
//! [`open_model`] takes either a `.gguf` file or a checkpoint directory. A checkpoint is
//! presented as a [`Gguf`] whose tensors are converted from safetensors as they are read, so
//! every model loads through the same tensor names, types and layouts.
//!
//! The loader is a parsing attack surface, so every length, count and offset read
//! from the file is bounds-checked before it is used: strings, arrays and tensor
//! counts have hard caps, sizes use checked arithmetic, and every tensor must lie
//! inside the file at an aligned offset without overlapping another tensor.
//! Tensor data is read with positioned reads; nothing is memory-mapped.

#![forbid(unsafe_code)]

pub mod hub;
pub mod inkling_checkpoint;
pub mod safetensors;

use astral_core::{AstralError, Result};
use astral_quant::GgmlType;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
const MAX_STRING: u64 = 64 << 20;
const MAX_ARRAY: u64 = 1 << 26;
const MAX_KV: u64 = 1 << 16;
const MAX_TENSORS: u64 = 1 << 20;
const MAX_DIMS: u32 = 8;
const DEFAULT_ALIGNMENT: u64 = 32;

/// A GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Short human-readable form; long arrays and strings are summarised.
    pub fn summary(&self) -> String {
        match self {
            Value::Str(s) if s.len() > 80 => format!("{:?}… ({} chars)", &s[..s.char_indices().nth(60).map_or(s.len(), |(i, _)| i)], s.chars().count()),
            Value::Str(s) => format!("{s:?}"),
            Value::Array(a) if a.len() > 32 => format!("[{} items, first {}]", a.len(), a[0].summary()),
            Value::Array(a) => format!("[{}]", a.iter().map(|v| v.summary()).collect::<Vec<_>>().join(", ")),
            v => format!("{v:?}").split('(').nth(1).map(|x| x.trim_end_matches(')').to_string()).unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: `dims[0]` is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the data section.
    pub offset: u64,
    pub n_elements: u64,
    pub n_bytes: u64,
}

impl TensorInfo {
    /// Bytes per row (`dims[0]` elements).
    pub fn row_bytes(&self) -> u64 {
        let (blk, sz) = self.ty.layout().expect("validated at load");
        self.dims[0] / blk as u64 * sz as u64
    }

    pub fn n_rows(&self) -> u64 {
        self.n_elements / self.dims[0].max(1)
    }
}

/// Produces a tensor's bytes on demand (GGUF layout), for models not read from a GGUF file.
pub type TensorFn = Box<dyn Fn(&TensorInfo) -> Result<Vec<u8>> + Send + Sync>;

enum Source {
    File(File),
    Converted(TensorFn),
}

/// A parsed GGUF file, or a checkpoint presented as one. Metadata and the tensor table are
/// held in memory; tensor data stays on disk until read.
pub struct Gguf {
    pub path: PathBuf,
    pub version: u32,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// Absolute file offset of the data section.
    pub data_offset: u64,
    pub file_len: u64,
    meta_index: HashMap<String, usize>,
    tensor_index: HashMap<String, usize>,
    source: Source,
}

fn bad(msg: impl Into<String>) -> AstralError {
    AstralError::Invalid(format!("gguf: {}", msg.into()))
}

struct Reader<R: Read> {
    r: R,
    pos: u64,
}

impl<R: Read> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b).map_err(|e| bad(format!("truncated at byte {}: {e}", self.pos)))?;
        self.pos += N as u64;
        Ok(b)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > MAX_STRING {
            return Err(bad(format!("string length {n} exceeds limit")));
        }
        let mut buf = vec![0u8; n as usize];
        self.r.read_exact(&mut buf).map_err(|e| bad(format!("truncated string at byte {}: {e}", self.pos)))?;
        self.pos += n;
        String::from_utf8(buf).map_err(|_| bad(format!("invalid UTF-8 in string at byte {}", self.pos)))
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(match self.u8()? {
                0 => false,
                1 => true,
                b => return Err(bad(format!("bool byte {b}"))),
            }),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 2 {
                    return Err(bad("arrays nested too deeply"));
                }
                let et = self.u32()?;
                let n = self.u64()?;
                if n > MAX_ARRAY {
                    return Err(bad(format!("array length {n} exceeds limit")));
                }
                let mut v = Vec::with_capacity(n.min(1 << 16) as usize);
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => return Err(bad(format!("unknown value type {t}"))),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        let mut r = Reader { r: BufReader::with_capacity(1 << 20, file.try_clone()?), pos: 0 };

        if &r.bytes::<4>()? != MAGIC {
            return Err(bad("not a GGUF file (bad magic)"));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(bad(format!("unsupported GGUF version {version}")));
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        if n_tensors > MAX_TENSORS || n_kv > MAX_KV {
            return Err(bad(format!("implausible counts: {n_tensors} tensors, {n_kv} keys")));
        }

        let mut metadata = Vec::with_capacity(n_kv as usize);
        let mut meta_index = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let v = r.value(ty, 0)?;
            if meta_index.insert(key.clone(), metadata.len()).is_some() {
                return Err(bad(format!("duplicate key {key}")));
            }
            metadata.push((key, v));
        }

        let alignment = match meta_index.get("general.alignment").map(|&i| &metadata[i].1) {
            None => DEFAULT_ALIGNMENT,
            Some(v) => match v.as_u64() {
                Some(a) if a.is_power_of_two() && a <= 1 << 20 => a,
                _ => return Err(bad("general.alignment must be a power of two")),
            },
        };

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut tensor_index = HashMap::new();
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()?;
            if nd == 0 || nd > MAX_DIMS {
                return Err(bad(format!("tensor {name}: {nd} dimensions")));
            }
            let dims = (0..nd).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let ty = GgmlType::from_id(r.u32()?);
            let offset = r.u64()?;
            let n_elements = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).ok_or_else(|| bad(format!("tensor {name}: element count overflows")))?;
            let (blk, sz) = ty.layout().ok_or_else(|| bad(format!("tensor {name}: unsupported type {}", ty.name())))?;
            if dims[0] % blk as u64 != 0 {
                return Err(bad(format!("tensor {name}: row of {} not a multiple of block {blk}", dims[0])));
            }
            let n_bytes = (n_elements / blk as u64).checked_mul(sz as u64).ok_or_else(|| bad(format!("tensor {name}: byte size overflows")))?;
            if offset % alignment != 0 {
                return Err(bad(format!("tensor {name}: offset {offset} not aligned to {alignment}")));
            }
            if tensor_index.insert(name.clone(), tensors.len()).is_some() {
                return Err(bad(format!("duplicate tensor {name}")));
            }
            tensors.push(TensorInfo { name, dims, ty, offset, n_elements, n_bytes });
        }

        let data_offset = r.pos.div_ceil(alignment) * alignment;
        // Every tensor inside the file, none overlapping.
        let mut spans: Vec<(u64, u64, usize)> = tensors.iter().enumerate().map(|(i, t)| (t.offset, t.offset + t.n_bytes, i)).collect();
        spans.sort_unstable();
        for w in spans.windows(2) {
            if w[0].1 > w[1].0 {
                return Err(bad(format!("tensors {} and {} overlap", tensors[w[0].2].name, tensors[w[1].2].name)));
            }
        }
        if let Some(&(_, end, i)) = spans.last()
            && data_offset.checked_add(end).is_none_or(|e| e > file_len)
        {
            return Err(bad(format!("tensor {} extends past end of file", tensors[i].name)));
        }

        Ok(Self { path, version, metadata, tensors, alignment, data_offset, file_len, meta_index, tensor_index, source: Source::File(file) })
    }

    /// A model whose tensors are produced by `read` (each a whole tensor in GGUF layout).
    /// `tensors` give names, GGML dims and types; offsets are assigned here.
    pub fn converted(path: PathBuf, metadata: Vec<(String, Value)>, tensors: Vec<(String, Vec<u64>, GgmlType)>, read: TensorFn) -> Result<Self> {
        let mut meta_index = HashMap::new();
        for (i, (k, _)) in metadata.iter().enumerate() {
            if meta_index.insert(k.clone(), i).is_some() {
                return Err(bad(format!("duplicate key {k}")));
            }
        }
        let (mut infos, mut tensor_index, mut offset) = (Vec::with_capacity(tensors.len()), HashMap::new(), 0u64);
        for (name, dims, ty) in tensors {
            let (blk, sz) = ty.layout().ok_or_else(|| bad(format!("tensor {name}: unsupported type {}", ty.name())))?;
            let n_elements: u64 = dims.iter().product();
            if dims.is_empty() || dims[0] % blk as u64 != 0 {
                return Err(bad(format!("tensor {name}: dims {dims:?} do not fit {}", ty.name())));
            }
            let n_bytes = n_elements / blk as u64 * sz as u64;
            if tensor_index.insert(name.clone(), infos.len()).is_some() {
                return Err(bad(format!("duplicate tensor {name}")));
            }
            infos.push(TensorInfo { name, dims, ty, offset, n_elements, n_bytes });
            offset += n_bytes.div_ceil(DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT;
        }
        Ok(Self { path, version: 3, metadata, tensors: infos, alignment: DEFAULT_ALIGNMENT, data_offset: 0, file_len: offset, meta_index, tensor_index, source: Source::Converted(read) })
    }

    /// Whether the tensors come from a checkpoint (converted on read) rather than a GGUF file.
    pub fn is_converted(&self) -> bool {
        matches!(self.source, Source::Converted(_))
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.meta_index.get(key).map(|&i| &self.metadata[i].1)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(|v| v.as_str())
    }

    /// Looks up `<architecture>.<suffix>`.
    pub fn arch_key(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.{suffix}"))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Reads `len` bytes starting `start` bytes into tensor `t`.
    pub fn read_tensor_range(&self, t: &TensorInfo, start: u64, len: u64) -> Result<Vec<u8>> {
        if start.checked_add(len).is_none_or(|e| e > t.n_bytes) {
            return Err(bad(format!("read [{start}, +{len}) outside tensor {} of {} bytes", t.name, t.n_bytes)));
        }
        match &self.source {
            Source::File(f) => {
                let mut buf = vec![0u8; len as usize];
                f.read_exact_at(&mut buf, self.data_offset + t.offset + start)?;
                Ok(buf)
            }
            Source::Converted(read) => {
                let mut all = read(t)?;
                if all.len() as u64 != t.n_bytes {
                    return Err(bad(format!("tensor {}: converter produced {} bytes, expected {}", t.name, all.len(), t.n_bytes)));
                }
                if start == 0 && len == t.n_bytes {
                    return Ok(all);
                }
                all.truncate((start + len) as usize);
                Ok(all.split_off(start as usize))
            }
        }
    }

    pub fn read_tensor(&self, t: &TensorInfo) -> Result<Vec<u8>> {
        self.read_tensor_range(t, 0, t.n_bytes)
    }

    /// Reads rows `[row, row + n)` (rows are `dims[0]` elements).
    pub fn read_rows(&self, t: &TensorInfo, row: u64, n: u64) -> Result<Vec<u8>> {
        let rb = t.row_bytes();
        self.read_tensor_range(t, row * rb, n * rb)
    }

    /// (tensor count, total bytes) per type name.
    pub fn type_tally(&self) -> BTreeMap<String, (usize, u64)> {
        let mut m = BTreeMap::new();
        for t in &self.tensors {
            let e = m.entry(t.ty.name()).or_insert((0, 0));
            e.0 += 1;
            e.1 += t.n_bytes;
        }
        m
    }

    /// Bytes in the data section not covered by any tensor (alignment padding), and the
    /// data-section length. A large gap means a type layout in `astral-quant` is wrong.
    pub fn coverage(&self) -> (u64, u64) {
        if self.is_converted() {
            let used: u64 = self.tensors.iter().map(|t| t.n_bytes).sum();
            return (0, used);
        }
        let data_len = self.file_len - self.data_offset;
        let used: u64 = self.tensors.iter().map(|t| t.n_bytes).sum();
        (data_len.saturating_sub(used), data_len)
    }
}

/// Opens model weights: a `.gguf` file, or a safetensors checkpoint directory (converted to
/// the GGUF tensor layout as it is read).
pub fn open_model(path: impl AsRef<Path>) -> Result<Gguf> {
    let path = path.as_ref();
    if path.is_dir() {
        if inkling_checkpoint::is_inkling_checkpoint(path) {
            return inkling_checkpoint::open(path);
        }
        return Err(bad(format!("{}: not a supported checkpoint (no config.json with a supported model_type)", path.display())));
    }
    Gguf::open(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a minimal GGUF v3 file in memory.
    fn build(kv: &[(&str, u32, Vec<u8>)], tensors: &[(&str, &[u64], u32, u64)], data: &[u8]) -> Vec<u8> {
        let mut f = Vec::new();
        f.extend_from_slice(MAGIC);
        f.extend_from_slice(&3u32.to_le_bytes());
        f.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        f.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        let s = |f: &mut Vec<u8>, x: &str| {
            f.extend_from_slice(&(x.len() as u64).to_le_bytes());
            f.extend_from_slice(x.as_bytes());
        };
        for (k, t, v) in kv {
            s(&mut f, k);
            f.extend_from_slice(&t.to_le_bytes());
            f.extend_from_slice(v);
        }
        for (name, dims, ty, off) in tensors {
            s(&mut f, name);
            f.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                f.extend_from_slice(&d.to_le_bytes());
            }
            f.extend_from_slice(&ty.to_le_bytes());
            f.extend_from_slice(&off.to_le_bytes());
        }
        while f.len() % 32 != 0 {
            f.push(0);
        }
        f.extend_from_slice(data);
        f
    }

    fn str_val(x: &str) -> Vec<u8> {
        let mut v = (x.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(x.as_bytes());
        v
    }

    fn write_tmp(bytes: &[u8], tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("astral-gguf-test-{}-{tag}.gguf", std::process::id()));
        std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
        p
    }

    #[test]
    fn parses_metadata_and_reads_tensors() {
        // One q8_0 tensor of 64 elements (2 blocks = 68 bytes) and one f32 [4].
        let mut data = vec![0u8; 68 + 28 + 16];
        data[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        data[2] = 5;
        let f32_off = 96u64; // 68 rounded up to 32
        for (i, v) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
            data[f32_off as usize + 4 * i..f32_off as usize + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        let file = build(&[("general.architecture", 8, str_val("test")), ("test.block_count", 4, 24u32.to_le_bytes().to_vec())], &[("w", &[64], 8, 0), ("b", &[4], 0, f32_off)], &data);
        let p = write_tmp(&file, "ok");
        let g = Gguf::open(&p).unwrap();
        assert_eq!(g.architecture(), Some("test"));
        assert_eq!(g.arch_key("block_count").and_then(|v| v.as_u64()), Some(24));
        let w = g.tensor("w").unwrap();
        assert_eq!((w.n_bytes, w.row_bytes()), (68, 68));
        let vals = astral_quant::dequantize(w.ty, &g.read_tensor(w).unwrap(), 64).unwrap();
        assert_eq!(vals[0], 5.0);
        let b = g.tensor("b").unwrap();
        assert_eq!(astral_quant::dequantize(b.ty, &g.read_tensor(b).unwrap(), 4).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(g.read_tensor_range(b, 8, 9).is_err(), "reads are bounded by the tensor");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn rejects_hostile_files() {
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("bad magic", b"GGUX\x03\0\0\0".to_vec()),
            ("truncated", build(&[], &[("w", &[64], 8, 0)], &[0u8; 10])),
            ("past eof", build(&[], &[("w", &[64], 8, 1 << 20)], &[0u8; 68])),
            ("overlap", build(&[], &[("a", &[64], 8, 0), ("b", &[64], 8, 32)], &[0u8; 200])),
            ("misaligned", build(&[], &[("w", &[64], 8, 4)], &[0u8; 100])),
            ("unknown type", build(&[], &[("w", &[64], 77, 0)], &[0u8; 100])),
            ("partial block", build(&[], &[("w", &[48], 8, 0)], &[0u8; 100])),
            ("dims overflow", build(&[], &[("w", &[1 << 40, 1 << 40], 0, 0)], &[0u8; 100])),
            ("huge string", {
                let mut f = build(&[], &[], &[]);
                f.truncate(24);
                f[16..24].copy_from_slice(&1u64.to_le_bytes()); // one key
                f.extend_from_slice(&u64::MAX.to_le_bytes()); // key length
                f
            }),
        ];
        for (what, bytes) in cases {
            let p = write_tmp(&bytes, &what.replace(' ', "_"));
            assert!(Gguf::open(&p).is_err(), "{what} must be rejected");
            std::fs::remove_file(p).ok();
        }
    }
}
