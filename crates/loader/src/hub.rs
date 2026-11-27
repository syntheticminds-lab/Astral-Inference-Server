// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Models named by a Hugging Face repository id.
//!
//! [`resolve`] turns a model argument into local paths:
//!
//! - a local `.gguf` file or checkpoint directory is used as is;
//! - `org/name` is a repository of safetensors weights: an existing snapshot in the Hugging Face
//!   cache is reused, otherwise the files Astral needs (configuration, tokenizer, chat template,
//!   safetensors shards) are downloaded;
//! - `org/name/file.gguf` is one GGUF file of a repository (plus its tokenizer files, if any).
//!
//! Downloads go to `$ASTRAL_CACHE` (default `~/.cache/astral/hub`) or a directory the caller
//! names, resume after an interruption, and send `$HF_TOKEN` (or `$HUGGING_FACE_HUB_TOKEN`) for
//! gated or private repositories. `$HF_ENDPOINT` overrides `https://huggingface.co`.

use astral_core::{AstralError, Result};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Where a model's files are once [`resolve`] returns.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// What to load: a `.gguf` file or a checkpoint directory.
    pub weights: PathBuf,
    /// A directory with `tokenizer.json` (and the chat template), if one was found.
    pub tokenizer_dir: Option<PathBuf>,
    /// A short name for the model (`name` of `org/name`, or the file / directory name).
    pub name: String,
}

/// Download options for [`resolve`].
#[derive(Debug, Clone, Default)]
pub struct HubOptions {
    /// Branch, tag or commit (default `main`).
    pub revision: Option<String>,
    /// Download directory (default `$ASTRAL_CACHE` or `~/.cache/astral/hub`).
    pub download_dir: Option<PathBuf>,
}

/// Resolves a model argument (local path or Hugging Face repository id) to local files,
/// downloading them if needed.
pub fn resolve(model: &str, opts: &HubOptions) -> Result<ResolvedModel> {
    resolve_with(model, opts, false)
}

/// Resolves the companion of a GGUF model (`--tokenizer`): like [`resolve`], but a repository
/// supplies only its tokenizer, chat template, configuration and draft (`mtp.safetensors`), not
/// its weights.
pub fn resolve_companion(model: &str, opts: &HubOptions) -> Result<ResolvedModel> {
    resolve_with(model, opts, true)
}

fn resolve_with(model: &str, opts: &HubOptions, companion: bool) -> Result<ResolvedModel> {
    let p = Path::new(model);
    if p.exists() {
        let name = p.file_stem().or_else(|| p.file_name()).map_or_else(|| model.to_string(), |s| s.to_string_lossy().into_owned());
        let tokenizer_dir = if p.is_dir() { Some(p.to_path_buf()) } else { p.parent().filter(|d| d.join("tokenizer.json").exists()).map(Path::to_path_buf) };
        return Ok(ResolvedModel { weights: p.to_path_buf(), tokenizer_dir, name });
    }
    let Some(repo) = RepoRef::parse(model) else {
        return Err(AstralError::Invalid(format!("model {model:?}: not a local file or directory, nor a Hugging Face repository id (org/name)")));
    };
    let rev = opts.revision.clone().unwrap_or_else(|| "main".into());
    if repo.file.is_none()
        && let Some(dir) = hf_cache_snapshot(&repo.id, &rev)
    {
        eprintln!("astral: using {} from the Hugging Face cache ({})", repo.id, dir.display());
        return Ok(ResolvedModel { tokenizer_dir: Some(dir.clone()), weights: dir, name: repo.name() });
    }
    let root = opts.download_dir.clone().unwrap_or_else(default_cache).join(repo.id.replace('/', "--")).join(&rev);
    let listing = list_files(&repo.id, &rev)?;
    let wanted: Vec<&RemoteFile> = listing
        .iter()
        .filter(|f| match &repo.file {
            Some(g) => f.path == *g || is_tokenizer_file(&f.path),
            None if companion => is_tokenizer_file(&f.path) || f.path == "mtp.safetensors",
            None => is_checkpoint_file(&f.path),
        })
        .collect();
    if let Some(g) = &repo.file
        && !wanted.iter().any(|f| f.path == *g)
    {
        return Err(AstralError::Invalid(format!("{}: no file {g} at revision {rev}", repo.id)));
    }
    if repo.file.is_none() && !companion && !wanted.iter().any(|f| f.path.ends_with(".safetensors")) {
        let ggufs: Vec<&str> = listing.iter().filter(|f| f.path.ends_with(".gguf")).map(|f| f.path.as_str()).collect();
        let hint = if ggufs.is_empty() { String::new() } else { format!("; for a GGUF file use {}/{}", repo.id, ggufs[0]) };
        return Err(AstralError::Invalid(format!("{}: no safetensors weights at revision {rev}{hint}", repo.id)));
    }
    let total: u64 = wanted.iter().map(|f| f.size).sum();
    eprintln!("astral: {} ({rev}): {} files, {:.2} GB -> {}", repo.id, wanted.len(), total as f64 / 1e9, root.display());
    for f in &wanted {
        download(&repo.id, &rev, f, &root)?;
    }
    let tokenizer_dir = root.join("tokenizer.json").exists().then(|| root.clone());
    let weights = match &repo.file {
        Some(g) => root.join(g),
        None => root.clone(),
    };
    Ok(ResolvedModel { weights, tokenizer_dir, name: repo.name() })
}

/// `org/name` or `org/name/path/to/file.gguf`.
struct RepoRef {
    id: String,
    file: Option<String>,
}

impl RepoRef {
    fn parse(s: &str) -> Option<Self> {
        let s = s.trim_start_matches("hf://").trim_start_matches("https://huggingface.co/");
        let parts: Vec<&str> = s.split('/').collect();
        let ok = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
        if parts.len() < 2 || !ok(parts[0]) || !ok(parts[1]) {
            return None;
        }
        let id = format!("{}/{}", parts[0], parts[1]);
        let file = (parts.len() > 2).then(|| parts[2..].join("/"));
        if file.as_ref().is_some_and(|f| !f.ends_with(".gguf")) {
            return None;
        }
        Some(Self { id, file })
    }

    fn name(&self) -> String {
        match &self.file {
            Some(f) => Path::new(f).file_stem().map_or_else(|| f.clone(), |s| s.to_string_lossy().into_owned()),
            None => self.id.rsplit('/').next().unwrap_or(&self.id).to_string(),
        }
    }
}

struct RemoteFile {
    path: String,
    size: u64,
}

/// Files a native checkpoint directory needs (weights, configuration, tokenizer, templates).
fn is_checkpoint_file(p: &str) -> bool {
    let name = p.rsplit('/').next().unwrap_or(p);
    if p.contains('/') && !p.starts_with("tokenizer/") {
        return false;
    }
    name.ends_with(".safetensors") || name.ends_with(".json") || name.ends_with(".jinja") || is_tokenizer_file(p)
}

fn is_tokenizer_file(p: &str) -> bool {
    let name = p.rsplit('/').next().unwrap_or(p);
    !p.contains('/') && (name.starts_with("tokenizer") || name.starts_with("chat_template") || name == "special_tokens_map.json" || name == "config.json" || name == "generation_config.json")
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into()).trim_end_matches('/').to_string()
}

fn token() -> Option<String> {
    std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")).ok().filter(|t| !t.is_empty())
}

fn get(url: &str) -> ureq::Request {
    let r = ureq::get(url).set("User-Agent", concat!("astral/", env!("CARGO_PKG_VERSION")));
    match token() {
        Some(t) => r.set("Authorization", &format!("Bearer {t}")),
        None => r,
    }
}

fn http_err(what: &str, e: ureq::Error) -> AstralError {
    match e {
        ureq::Error::Status(401 | 403, _) => AstralError::Invalid(format!("{what}: access denied (gated or private repository: set HF_TOKEN)")),
        ureq::Error::Status(404, _) => AstralError::Invalid(format!("{what}: not found")),
        e => AstralError::Invalid(format!("{what}: {e}")),
    }
}

fn list_files(repo: &str, rev: &str) -> Result<Vec<RemoteFile>> {
    let url = format!("{}/api/models/{repo}/tree/{rev}?recursive=true", endpoint());
    let body = get(&url).call().map_err(|e| http_err(repo, e))?.into_string().map_err(|e| AstralError::Invalid(format!("{repo}: {e}")))?;
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| AstralError::Invalid(format!("{repo}: bad file listing: {e}")))?;
    let items = v.as_array().ok_or_else(|| AstralError::Invalid(format!("{repo}: bad file listing")))?;
    Ok(items
        .iter()
        .filter(|f| f["type"] == "file")
        .filter_map(|f| Some(RemoteFile { path: f["path"].as_str()?.to_string(), size: f["lfs"]["size"].as_u64().or_else(|| f["size"].as_u64()).unwrap_or(0) }))
        .collect())
}

/// Downloads one file into `root`, resuming a partial download; skips complete files.
fn download(repo: &str, rev: &str, f: &RemoteFile, root: &Path) -> Result<()> {
    let dst = root.join(&f.path);
    if dst.metadata().is_ok_and(|m| m.len() == f.size) {
        return Ok(());
    }
    if let Some(d) = dst.parent() {
        std::fs::create_dir_all(d)?;
    }
    let part = dst.with_extension(format!("{}.part", dst.extension().map_or(String::new(), |e| e.to_string_lossy().into_owned())));
    let have = part.metadata().map_or(0, |m| m.len());
    let url = format!("{}/{repo}/resolve/{rev}/{}", endpoint(), f.path);
    let mut req = get(&url);
    if have > 0 && have < f.size {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = req.call().map_err(|e| http_err(&format!("{repo}/{}", f.path), e))?;
    let append = have > 0 && resp.status() == 206;
    let mut out = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(&part)?;
    let mut done = if append { have } else { 0 };
    let start = done;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let (t0, mut last) = (std::time::Instant::now(), std::time::Instant::now());
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        if f.size > (64 << 20) && last.elapsed().as_secs_f64() >= 2.0 {
            let rate = (done - start) as f64 / t0.elapsed().as_secs_f64().max(1e-3) / 1e6;
            eprintln!("astral:   {} {:.0} / {:.0} MB ({rate:.0} MB/s)", f.path, done as f64 / 1e6, f.size as f64 / 1e6);
            last = std::time::Instant::now();
        }
    }
    out.flush()?;
    drop(out);
    if f.size > 0 && done != f.size {
        return Err(AstralError::Invalid(format!("{repo}/{}: downloaded {done} of {} bytes (run again to resume)", f.path, f.size)));
    }
    std::fs::rename(&part, &dst)?;
    Ok(())
}

fn default_cache() -> PathBuf {
    if let Ok(d) = std::env::var("ASTRAL_CACHE") {
        return PathBuf::from(d);
    }
    home().join(".cache").join("astral").join("hub")
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

/// A complete snapshot of `repo` at `rev` in the Hugging Face cache (`$HF_HUB_CACHE`,
/// `$HF_HOME/hub` or `~/.cache/huggingface/hub`): its config, and every shard its safetensors
/// index lists (or at least one safetensors file).
fn hf_cache_snapshot(repo: &str, rev: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .unwrap_or_else(|| home().join(".cache").join("huggingface").join("hub"));
    let base = hub.join(format!("models--{}", repo.replace('/', "--")));
    let sha = std::fs::read_to_string(base.join("refs").join(rev)).ok().map(|s| s.trim().to_string()).unwrap_or_else(|| rev.to_string());
    let dir = base.join("snapshots").join(sha);
    if !dir.join("config.json").exists() || !dir.join("tokenizer.json").exists() {
        return None;
    }
    let shards: Vec<String> = match std::fs::read_to_string(dir.join("model.safetensors.index.json")) {
        Ok(s) => {
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            let mut names: Vec<String> = v["weight_map"].as_object()?.values().filter_map(|x| x.as_str().map(str::to_string)).collect();
            names.sort();
            names.dedup();
            names
        }
        Err(_) => std::fs::read_dir(&dir).ok()?.filter_map(|e| e.ok()?.file_name().into_string().ok()).filter(|n| n.ends_with(".safetensors")).collect(),
    };
    (!shards.is_empty() && shards.iter().all(|n| dir.join(n).exists())).then_some(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ids() {
        let r = RepoRef::parse("org/Model-2.5").unwrap();
        assert_eq!((r.id.as_str(), r.file.is_none(), r.name().as_str()), ("org/Model-2.5", true, "Model-2.5"));
        let r = RepoRef::parse("org/repo-GGUF/sub/model-Q4.gguf").unwrap();
        assert_eq!((r.id.as_str(), r.file.as_deref(), r.name().as_str()), ("org/repo-GGUF", Some("sub/model-Q4.gguf"), "model-Q4"));
        assert!(RepoRef::parse("https://huggingface.co/org/name").is_some());
        assert!(RepoRef::parse("just-a-name").is_none());
        assert!(RepoRef::parse("org/name/README.md").is_none());
        assert!(RepoRef::parse("../x/y").is_none());
    }

    #[test]
    fn checkpoint_files() {
        for f in ["model-00001-of-00002.safetensors", "config.json", "tokenizer.json", "chat_template.jinja", "mtp.safetensors"] {
            assert!(is_checkpoint_file(f), "{f}");
        }
        for f in ["model.gguf", "pytorch_model.bin", "original/consolidated.pth", "README.md", "images/a.png"] {
            assert!(!is_checkpoint_file(f), "{f}");
        }
    }
}
