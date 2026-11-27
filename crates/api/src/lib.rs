// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-api`: OpenAI-compatible HTTP server for Inkling 2 Mini.
//!
//! Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions`,
//! `POST /v1/chat/completions` (streaming with `"stream": true`, as Server-Sent Events) and
//! `POST /tokenize`. One engine thread owns the GPU and runs requests one at a time from a
//! bounded queue; with a draft (`--draft`), every request uses DSpark speculative decoding.
//!
//! Decoding is greedy: `temperature`, `top_p` and `seed` are accepted and ignored.

#![forbid(unsafe_code)]

pub mod chat;
pub mod engine;
pub mod http;
pub mod qwen_chat;
mod qwen_engine;

use astral_core::{AstralError, Result};
use chat::{OutputParser, Part, Specials};
use qwen_chat::QwenSpecials;
use engine::{Engine, EngineConfig, Event, Job};
use serde_json::{json, Value};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub engine: EngineConfig,
    /// Directory with `tokenizer.json` (the native checkpoint).
    pub tokenizer_dir: PathBuf,
    pub host: String,
    pub port: u16,
    pub model_name: String,
}

struct App {
    engine: Arc<Engine>,
    tok: tokenizers::Tokenizer,
    sp: Family,
    model: String,
    next_id: AtomicU64,
}

/// The model family served: its special tokens, chat template and output parsing.
enum Family {
    Inkling(Specials),
    Qwen(QwenSpecials),
}

impl Family {
    fn stop_ids(&self) -> Vec<u32> {
        match self {
            Family::Inkling(s) => s.stop_ids(),
            Family::Qwen(q) => q.stop_ids(),
        }
    }
}

/// Loads the model and serves until the process is stopped.
pub fn serve(cfg: ServerConfig) -> Result<()> {
    let tok = tokenizers::Tokenizer::from_file(cfg.tokenizer_dir.join("tokenizer.json")).map_err(|e| AstralError::Invalid(format!("tokenizer: {e}")))?;
    let sp = if cfg.engine.qwen { Family::Qwen(QwenSpecials::from_tokenizer(&tok).map_err(AstralError::Invalid)?) } else { Family::Inkling(Specials::from_tokenizer(&tok).map_err(AstralError::Invalid)?) };
    let engine = Engine::start(cfg.engine.clone())?;
    let app = Arc::new(App { engine, tok, sp, model: cfg.model_name.clone(), next_id: AtomicU64::new(1) });
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr).map_err(|e| AstralError::Invalid(format!("bind {addr}: {e}")))?;
    eprintln!(
        "astral: serving {} on http://{addr} (context {}, {})",
        cfg.model_name,
        app.engine.capacity,
        if app.engine.draft { format!("DSpark draft, k = {}", app.engine.k) } else { "no draft".into() }
    );
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let app = app.clone();
        let _ = std::thread::Builder::new().name("astral-http".into()).spawn(move || {
            let _ = handle(&app, stream);
        });
    }
    Ok(())
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn handle(app: &App, mut stream: TcpStream) -> std::io::Result<()> {
    let _ = stream.set_nodelay(true);
    let Some(req) = http::read_request(&stream)? else { return Ok(()) };
    if req.method == "OPTIONS" {
        return http::respond(&mut stream, 200, "text/plain", b"");
    }
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => http::respond(&mut stream, 200, "text/plain", b"ok"),
        ("GET", "/v1/models") => http::respond_json(
            &mut stream,
            200,
            &json!({ "object": "list", "data": [{ "id": app.model, "object": "model", "created": now(), "owned_by": "astral", "max_model_len": app.engine.capacity }] }),
        ),
        ("POST", p @ ("/v1/completions" | "/v1/chat/completions" | "/tokenize")) => {
            let body: Value = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return http::respond_error(&mut stream, 400, &format!("invalid JSON: {e}")),
            };
            match p {
                "/tokenize" => tokenize(app, &mut stream, &body),
                "/v1/completions" => generate(app, &mut stream, &body, false),
                _ => generate(app, &mut stream, &body, true),
            }
        }
        (_, "/health" | "/v1/models" | "/v1/completions" | "/v1/chat/completions" | "/tokenize") => http::respond_error(&mut stream, 405, "method not allowed"),
        _ => http::respond_error(&mut stream, 404, "not found"),
    }
}

/// The prompt token ids of a request.
fn prompt_ids(app: &App, body: &Value, chat_req: bool) -> std::result::Result<Vec<u32>, String> {
    let text = if chat_req || body.get("messages").is_some() {
        let msgs = body.get("messages").ok_or("missing messages")?;
        match &app.sp {
            Family::Inkling(_) => {
                let effort = body.get("reasoning_effort").or_else(|| body.get("chat_template_kwargs").and_then(|k| k.get("reasoning_effort")));
                chat::render(msgs, body.get("tools"), effort)?
            }
            Family::Qwen(_) => qwen_chat::render(msgs, body.get("tools"), qwen_chat::thinking_enabled(body))?,
        }
    } else {
        match body.get("prompt") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) if a.iter().all(|x| x.is_u64()) => return Ok(a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect()),
            Some(Value::Array(a)) if a.len() == 1 => match &a[0] {
                Value::String(s) => s.clone(),
                Value::Array(ids) if ids.iter().all(|x| x.is_u64()) => return Ok(ids.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect()),
                _ => return Err("prompt must be a string or a list of token ids".into()),
            },
            Some(Value::Array(_)) => return Err("batched prompts are not supported; send one prompt per request".into()),
            _ => return Err("missing prompt".into()),
        }
    };
    let enc = app.tok.encode(text.as_str(), false).map_err(|e| format!("tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

fn tokenize(app: &App, stream: &mut TcpStream, body: &Value) -> std::io::Result<()> {
    match prompt_ids(app, body, body.get("messages").is_some()) {
        Ok(ids) => http::respond_json(stream, 200, &json!({ "tokens": ids, "count": ids.len(), "max_model_len": app.engine.capacity })),
        Err(e) => http::respond_error(stream, 400, &e),
    }
}

fn stop_strings(body: &Value) -> Vec<String> {
    match body.get("stop") {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str()).filter(|s| !s.is_empty()).map(str::to_string).collect(),
        _ => Vec::new(),
    }
}

/// `/v1/completions` (`chat_req = false`) and `/v1/chat/completions`.
fn generate(app: &App, stream: &mut TcpStream, body: &Value, chat_req: bool) -> std::io::Result<()> {
    let prompt = match prompt_ids(app, body, chat_req) {
        Ok(p) => p,
        Err(e) => return http::respond_error(stream, 400, &e),
    };
    let cap = app.engine.capacity as usize;
    let room = cap.saturating_sub(prompt.len() + app.engine.margin());
    if room == 0 {
        return http::respond_error(stream, 400, &format!("prompt of {} tokens does not fit the context of {cap}", prompt.len()));
    }
    let asked = body.get("max_completion_tokens").or_else(|| body.get("max_tokens")).and_then(|v| v.as_u64()).map(|v| v as usize);
    if asked.is_some_and(|m| m > room) {
        return http::respond_error(stream, 400, &format!("max_tokens {} exceeds the room left in the context ({room} after a {}-token prompt)", asked.unwrap_or(0), prompt.len()));
    }
    let max_tokens = asked.unwrap_or(if chat_req { room } else { 16.min(room) });
    let ignore_eos = body.get("ignore_eos").and_then(|v| v.as_bool()).unwrap_or(false);
    let use_draft = body.get("speculative").and_then(|v| v.as_bool()).unwrap_or(true);
    let streaming = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let include_usage = body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(|v| v.as_bool()).unwrap_or(false);
    let stops = stop_strings(body);
    let (tx, rx) = mpsc::channel();
    let job = Job { prompt: prompt.clone(), max_tokens, stop_ids: if ignore_eos { Vec::new() } else { app.sp.stop_ids() }, use_draft, tx };
    if let Err(e) = app.engine.submit(job) {
        return http::respond_error(stream, 503, &e);
    }
    let id = format!("{}-{}", if chat_req { "chatcmpl" } else { "cmpl" }, app.next_id.fetch_add(1, Ordering::Relaxed));
    let created = now();
    let object = if chat_req { "chat.completion" } else { "text_completion" };
    let chunk_object = if chat_req { "chat.completion.chunk" } else { "text_completion" };
    if streaming {
        http::sse_start(stream)?;
        if chat_req {
            let first = json!({ "id": id, "object": chunk_object, "created": created, "model": app.model, "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "" }, "finish_reason": Value::Null }] });
            http::sse_send(stream, &first.to_string())?;
        }
    }
    let mut parser = match &app.sp {
        Family::Inkling(sp) => OutputParser::new(sp.clone()),
        Family::Qwen(q) => OutputParser::new_qwen(q.clone(), qwen_chat::thinking_enabled(body)),
    };
    let mut ids: Vec<u32> = Vec::new();
    let mut text_emitted = 0usize; // completions: bytes of decoded text already sent
    let mut finish = "length";
    let mut stopped_by_string = false;
    let mut error: Option<String> = None;
    for ev in rx.iter() {
        match ev {
            Event::Token(t) => {
                ids.push(t);
                // The new text of this token (reasoning or content for chat; all text otherwise).
                let delta: Option<(Part, String)> = if chat_req {
                    parser.push(t, &app.tok)
                } else {
                    let text = app.tok.decode(&ids, true).unwrap_or_default();
                    if text.len() > text_emitted && !text.ends_with('\u{FFFD}') && text.is_char_boundary(text_emitted) {
                        let d = text[text_emitted..].to_string();
                        text_emitted = text.len();
                        Some((Part::Content, d))
                    } else {
                        None
                    }
                };
                let Some((part, mut d)) = delta else { continue };
                // Stop strings apply to the answer text.
                if part == Part::Content && !stops.is_empty() {
                    let full = if chat_req { parser.content.clone() } else { app.tok.decode(&ids, true).unwrap_or_default() };
                    if let Some(cut) = stops.iter().filter_map(|s| full.find(s.as_str())).min() {
                        let before = full.len() - d.len();
                        d = if cut > before { full[before..cut].to_string() } else { String::new() };
                        if chat_req {
                            parser.content.truncate(cut);
                        }
                        stopped_by_string = true;
                    }
                }
                if streaming && !d.is_empty() {
                    let delta = if chat_req {
                        if part == Part::Reasoning {
                            json!({ "reasoning_content": d })
                        } else {
                            json!({ "content": d })
                        }
                    } else {
                        Value::Null
                    };
                    let chunk = if chat_req {
                        json!({ "id": id, "object": chunk_object, "created": created, "model": app.model, "choices": [{ "index": 0, "delta": delta, "finish_reason": Value::Null }] })
                    } else {
                        json!({ "id": id, "object": chunk_object, "created": created, "model": app.model, "choices": [{ "index": 0, "text": d, "logprobs": Value::Null, "finish_reason": Value::Null }] })
                    };
                    if http::sse_send(stream, &chunk.to_string()).is_err() {
                        return Ok(()); // client gone: dropping rx cancels the request
                    }
                }
                if stopped_by_string {
                    finish = "stop";
                    break; // dropping rx cancels the rest of the generation
                }
            }
            Event::Done { finish: f } => {
                finish = f;
                break;
            }
            Event::Error(e) => {
                error = Some(e);
                break;
            }
        }
    }
    if let Some(e) = error {
        if streaming {
            let _ = http::sse_send(stream, &json!({ "error": { "message": e, "type": "server_error" } }).to_string());
            return http::sse_send(stream, "[DONE]");
        }
        return http::respond_error(stream, 500, &e);
    }
    let tool_calls = if chat_req { parser.tool_calls() } else { Vec::new() };
    if !tool_calls.is_empty() && finish == "stop" {
        finish = "tool_calls";
    }
    let usage = json!({ "prompt_tokens": prompt.len(), "completion_tokens": ids.len(), "total_tokens": prompt.len() + ids.len() });
    if streaming {
        let mut fin_delta = json!({});
        if !tool_calls.is_empty() {
            let calls: Vec<Value> = tool_calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let mut c = c.clone();
                    c["index"] = json!(i);
                    c
                })
                .collect();
            fin_delta = json!({ "tool_calls": calls });
        }
        let last = if chat_req {
            json!({ "id": id, "object": chunk_object, "created": created, "model": app.model, "choices": [{ "index": 0, "delta": fin_delta, "finish_reason": finish }] })
        } else {
            json!({ "id": id, "object": chunk_object, "created": created, "model": app.model, "choices": [{ "index": 0, "text": "", "logprobs": Value::Null, "finish_reason": finish }] })
        };
        http::sse_send(stream, &last.to_string())?;
        if include_usage {
            http::sse_send(stream, &json!({ "id": id, "object": chunk_object, "created": created, "model": app.model, "choices": [], "usage": usage }).to_string())?;
        }
        return http::sse_send(stream, "[DONE]");
    }
    let resp = if chat_req {
        let mut msg = json!({ "role": "assistant", "content": parser.content });
        if !parser.reasoning.is_empty() {
            msg["reasoning_content"] = json!(parser.reasoning);
        }
        if !tool_calls.is_empty() {
            msg["tool_calls"] = json!(tool_calls);
        }
        json!({ "id": id, "object": object, "created": created, "model": app.model, "choices": [{ "index": 0, "message": msg, "finish_reason": finish }], "usage": usage })
    } else {
        let mut text = app.tok.decode(&ids, true).unwrap_or_default();
        if let Some(cut) = stops.iter().filter_map(|s| text.find(s.as_str())).min() {
            text.truncate(cut);
        }
        json!({ "id": id, "object": object, "created": created, "model": app.model, "choices": [{ "index": 0, "text": text, "logprobs": Value::Null, "finish_reason": finish }], "usage": usage })
    };
    http::respond_json(stream, 200, &resp)
}
