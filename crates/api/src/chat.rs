// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Inkling 2's chat template (`chat_template.jinja` of the checkpoint) in Rust, and the parser
//! that splits generated tokens into reasoning, answer text and tool calls (as vLLM's
//! `inkling` reasoning parser does).
//!
//! The template uses Hugging Face-only Jinja extensions (`{% generation %}`, `tojson` with
//! Python keyword arguments), so it is ported here rather than interpreted.

use serde_json::{Map, Value};

pub const MESSAGE_USER: &str = "<|message_user|>";
pub const MESSAGE_MODEL: &str = "<|message_model|>";
pub const MESSAGE_SYSTEM: &str = "<|message_system|>";
pub const MESSAGE_TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";
const INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";

fn effort_text(effort: Option<&Value>) -> Result<String, String> {
    let num = match effort {
        None | Some(Value::Null) => 0.9,
        Some(Value::String(s)) => match s.trim() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            other => return Err(format!("Unknown reasoning_effort: {other}")),
        },
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.9),
        Some(v) => return Err(format!("reasoning_effort must be a string or number, got {v}")),
    };
    if !(0.0..=0.99).contains(&num) {
        return Err("reasoning_effort must be in [0.0, 0.99]".into());
    }
    let n = if num == 0.0 { "0".to_string() } else { format!("{num}") };
    Ok(format!("{MESSAGE_SYSTEM}{CONTENT_TEXT}Thinking effort level: {n}{END_MESSAGE}"))
}

fn str_of<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|x| x.as_str()).filter(|s| !s.is_empty())
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(_)) => true,
    }
}

/// Tool-call arguments as an object (OpenAI clients send a JSON string).
fn tool_args(f: &Value) -> Result<Map<String, Value>, String> {
    match f.get("arguments") {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(Map::new()),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(m)) => Ok(m),
            _ => Err("tool call arguments must be a JSON object".into()),
        },
        Some(Value::Object(m)) => Ok(m.clone()),
        Some(_) => Err("tool call arguments must be an object".into()),
    }
}

/// The prompt text for `messages` (OpenAI chat format), as the checkpoint's template renders it.
pub fn render(messages: &Value, tools: Option<&Value>, effort: Option<&Value>) -> Result<String, String> {
    let msgs = messages.as_array().ok_or("messages must be an array")?;
    let mut out = String::new();
    if let Some(Value::Array(ts)) = tools
        && !ts.is_empty()
    {
        let mut specs = Vec::new();
        for t in ts {
            let f = t.get("function").unwrap_or(t);
            let mut spec = Map::new();
            spec.insert("description".into(), Value::String(str_of(f, "description").unwrap_or("").into()));
            spec.insert("name".into(), f.get("name").cloned().unwrap_or(Value::Null));
            let params = f.get("parameters").filter(|p| truthy(Some(p))).cloned().unwrap_or_else(|| Value::Object(Map::new()));
            spec.insert("parameters".into(), params);
            spec.insert("type".into(), Value::String(str_of(t, "type").unwrap_or("function").into()));
            specs.push(Value::Object(spec));
        }
        // serde_json maps are sorted by key: the template's tojson(sort_keys=true), compact.
        out.push_str(&format!("{MESSAGE_SYSTEM}tool_declare<|content_xml|>{}{END_MESSAGE}", Value::Array(specs)));
    }
    let mut effort_done = false;
    for m in msgs {
        let role = m.get("role").and_then(|r| r.as_str()).ok_or("message without a role")?;
        let mut rtok = match role {
            "user" => MESSAGE_USER.to_string(),
            "assistant" => MESSAGE_MODEL.to_string(),
            "system" | "developer" => MESSAGE_SYSTEM.to_string(),
            "tool" => MESSAGE_TOOL.to_string(),
            r => return Err(format!("Unknown message role: {r}")),
        };
        if !effort_done && role != "system" && role != "developer" {
            out.push_str(&effort_text(effort)?);
            effort_done = true;
        }
        if role == "tool" {
            if let Some(n) = str_of(m, "name") {
                rtok.push_str(n);
            } else if let Some(id) = str_of(m, "tool_call_id") {
                for prev in msgs {
                    if prev.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                        for tc in prev.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten() {
                            if tc.get("id").and_then(|x| x.as_str()) == Some(id)
                                && let Some(n) = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str())
                            {
                                rtok.push_str(n);
                            }
                        }
                    }
                }
            }
        } else if role == "assistant"
            && let Some(n) = str_of(m, "name")
        {
            rtok.push_str(n);
        }
        let assistant = role == "assistant";
        let has_tools = truthy(m.get("tool_calls"));
        let has_reasoning = truthy(m.get("reasoning_content")) || truthy(m.get("reasoning"));
        if assistant && has_reasoning {
            let rc = str_of(m, "reasoning_content").or_else(|| str_of(m, "reasoning")).unwrap_or("");
            out.push_str(&format!("{rtok}{CONTENT_THINKING}{rc}{END_MESSAGE}"));
        }
        match m.get("content") {
            Some(Value::String(c)) => {
                if !c.is_empty() || !(assistant && (has_tools || has_reasoning)) {
                    out.push_str(&format!("{rtok}{CONTENT_TEXT}{c}{END_MESSAGE}"));
                }
            }
            Some(Value::Array(parts)) => {
                for p in parts {
                    match p {
                        Value::String(s) => out.push_str(&format!("{rtok}{CONTENT_TEXT}{s}{END_MESSAGE}")),
                        Value::Object(_) => {
                            let ty = p.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                            match ty {
                                "text" | "input_text" => {
                                    let t = p.get("text").and_then(|x| x.as_str()).unwrap_or("");
                                    out.push_str(&format!("{rtok}{CONTENT_TEXT}{t}{END_MESSAGE}"));
                                }
                                "image" | "input_image" | "image_url" => return Err("image input is not supported by this server yet".into()),
                                "audio" | "input_audio" | "audio_url" => return Err("Inkling 2 does not accept audio input".into()),
                                t => return Err(format!("Unsupported content part type: {t}")),
                            }
                        }
                        _ => return Err("bad content part".into()),
                    }
                }
            }
            // The template renders nothing for a missing / null content.
            None | Some(Value::Null) => {}
            Some(_) => return Err("content must be a string or an array".into()),
        }
        if assistant && has_tools {
            for tc in m.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten() {
                let f = tc.get("function").ok_or("tool call without a function")?;
                let name = f.get("name").and_then(|n| n.as_str()).ok_or("tool call function name must be a string")?;
                let args = tool_args(f)?;
                let body: Vec<String> = args.iter().map(|(k, v)| format!("{}:{}", Value::String(k.clone()), v)).collect();
                out.push_str(&format!("{MESSAGE_MODEL}{name}{INVOKE_TOOL_JSON}{{\"name\":{},\"args\":{{{}}}}}{END_MESSAGE}", Value::String(name.into()), body.join(",")));
            }
        }
        if assistant {
            out.push_str(END_SAMPLING);
        }
    }
    if !effort_done {
        out.push_str(&effort_text(effort)?);
    }
    Ok(out)
}

/// Which part of the answer the generated tokens belong to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    /// Role header (and name) of a model message: not shown.
    Header,
    Reasoning,
    Content,
    ToolCall,
}

/// Special token ids the output parser reacts to.
#[derive(Debug, Clone)]
pub struct Specials {
    pub message_model: u32,
    pub content_text: u32,
    pub content_thinking: u32,
    pub end_message: u32,
    pub end_sampling: u32,
    pub invoke_tool_json: Option<u32>,
    pub endoftext: u32,
}

impl Specials {
    pub fn from_tokenizer(t: &tokenizers::Tokenizer) -> Result<Self, String> {
        let id = |s: &str| t.token_to_id(s).ok_or_else(|| format!("tokenizer has no {s}"));
        Ok(Self {
            message_model: id(MESSAGE_MODEL)?,
            content_text: id(CONTENT_TEXT)?,
            content_thinking: id(CONTENT_THINKING)?,
            end_message: id(END_MESSAGE)?,
            end_sampling: id(END_SAMPLING)?,
            invoke_tool_json: t.token_to_id(INVOKE_TOOL_JSON),
            endoftext: id("<|endoftext|>")?,
        })
    }

    /// Tokens that end a generation.
    pub fn stop_ids(&self) -> Vec<u32> {
        vec![self.end_sampling, self.endoftext]
    }
}

/// Incremental parse of a chat answer: feed tokens, get text deltas per part.
pub struct OutputParser {
    sp: Specials,
    part: Part,
    seg: Vec<u32>,
    seg_emitted: usize,
    pub reasoning: String,
    pub content: String,
    tool_raw: Vec<String>,
    /// Qwen3.5: the answer starts in `part`; `</think>` switches from reasoning to content.
    qwen: Option<crate::qwen_chat::QwenSpecials>,
}

impl OutputParser {
    pub fn new(sp: Specials) -> Self {
        Self { sp, part: Part::Header, seg: Vec::new(), seg_emitted: 0, reasoning: String::new(), content: String::new(), tool_raw: Vec::new(), qwen: None }
    }

    /// A parser for Qwen3.5 output. A generation prompt that opened a thinking block starts in
    /// reasoning (`thinking`), otherwise in the answer.
    pub fn new_qwen(q: crate::qwen_chat::QwenSpecials, thinking: bool) -> Self {
        let mut p = Self::new(Specials { message_model: 0, content_text: 0, content_thinking: 0, end_message: 0, end_sampling: 0, invoke_tool_json: None, endoftext: q.endoftext });
        p.part = if thinking { Part::Reasoning } else { Part::Content };
        p.qwen = Some(q);
        p
    }

    /// Feeds one generated token; returns a text delta of the reasoning or the content.
    pub fn push(&mut self, tok: u32, tk: &tokenizers::Tokenizer) -> Option<(Part, String)> {
        if let Some(q) = &self.qwen {
            if tok == q.think_end || tok == q.think_start {
                self.part = if tok == q.think_end { Part::Content } else { Part::Reasoning };
                self.seg.clear();
                self.seg_emitted = 0;
                return None;
            }
        }
        let sp = &self.sp;
        let new_part = if self.qwen.is_some() {
            None
        } else if tok == sp.content_thinking {
            Some(Part::Reasoning)
        } else if tok == sp.content_text {
            Some(Part::Content)
        } else if Some(tok) == sp.invoke_tool_json {
            Some(Part::ToolCall)
        } else if tok == sp.message_model || tok == sp.end_message {
            Some(Part::Header)
        } else {
            None
        };
        if let Some(p) = new_part {
            self.flush_tool(tk);
            self.part = p;
            self.seg.clear();
            self.seg_emitted = 0;
            if p == Part::ToolCall {
                self.tool_raw.push(String::new());
            }
            return None;
        }
        if self.part == Part::Header {
            return None;
        }
        self.seg.push(tok);
        let text = tk.decode(&self.seg, true).ok()?;
        if text.ends_with('\u{FFFD}') || text.len() <= self.seg_emitted || !text.is_char_boundary(self.seg_emitted) {
            return None;
        }
        let mut delta = text[self.seg_emitted..].to_string();
        self.seg_emitted = text.len();
        // Qwen: the blank lines after `<think>` and `</think>` are not part of the text.
        if self.qwen.is_some() {
            let empty = if self.part == Part::Reasoning { self.reasoning.is_empty() } else { self.content.is_empty() };
            if empty {
                delta = delta.trim_start().to_string();
                if delta.is_empty() {
                    return None;
                }
            }
        }
        match self.part {
            Part::Reasoning => self.reasoning.push_str(&delta),
            Part::Content => self.content.push_str(&delta),
            Part::ToolCall => {
                if let Some(t) = self.tool_raw.last_mut() {
                    t.push_str(&delta);
                }
                return None;
            }
            Part::Header => {}
        }
        Some((self.part, delta))
    }

    fn flush_tool(&mut self, _tk: &tokenizers::Tokenizer) {}

    /// Tool calls made by the answer, as OpenAI `tool_calls` entries.
    pub fn tool_calls(&self) -> Vec<Value> {
        let mut v = Vec::new();
        for (i, raw) in self.tool_raw.iter().enumerate() {
            let Ok(j) = serde_json::from_str::<Value>(raw.trim()) else { continue };
            let name = j.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
            let args = j.get("args").cloned().unwrap_or_else(|| Value::Object(Map::new()));
            v.push(serde_json::json!({
                "id": format!("call_{i}"),
                "type": "function",
                "function": { "name": name, "arguments": args.to_string() },
            }));
        }
        v
    }
}
