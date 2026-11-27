// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! The Qwen3.5 chat format (`chat_template.jinja` of the checkpoint) for plain chat: system,
//! user and assistant messages in ChatML (`<|im_start|>` / `<|im_end|>`), with the model's
//! thinking block (`<think>` ... `</think>`). Tools, tool results and image/video content are
//! not supported and are refused.

use serde_json::Value;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Special token ids of the Qwen3.5 tokenizer the server reacts to.
#[derive(Debug, Clone)]
pub struct QwenSpecials {
    pub think_start: u32,
    pub think_end: u32,
    pub im_end: u32,
    pub endoftext: u32,
}

impl QwenSpecials {
    pub fn from_tokenizer(t: &tokenizers::Tokenizer) -> Result<Self, String> {
        let id = |s: &str| t.token_to_id(s).ok_or_else(|| format!("tokenizer has no {s}"));
        Ok(Self { think_start: id("<think>")?, think_end: id("</think>")?, im_end: id(IM_END)?, endoftext: id("<|endoftext|>")? })
    }

    /// Tokens that end a generation.
    pub fn stop_ids(&self) -> Vec<u32> {
        vec![self.im_end, self.endoftext]
    }
}

/// Text of a message's `content` (a string, or a list of text parts).
fn content_text(m: &Value) -> Result<String, String> {
    match m.get("content") {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    Value::String(s) => out.push_str(s),
                    Value::Object(_) => match p.get("type").and_then(|t| t.as_str()).unwrap_or("text") {
                        "text" | "input_text" => out.push_str(p.get("text").and_then(|x| x.as_str()).unwrap_or("")),
                        "image" | "image_url" | "input_image" | "video" | "video_url" => return Err("image and video input are not supported by Astral for this model".into()),
                        t => return Err(format!("Unsupported content part type: {t}")),
                    },
                    _ => return Err("bad content part".into()),
                }
            }
            Ok(out)
        }
        Some(_) => Err("content must be a string or an array".into()),
    }
}

/// The prompt text for `messages`, ending with the assistant's generation prompt (a thinking
/// block is opened unless `enable_thinking` is false).
pub fn render(messages: &Value, tools: Option<&Value>, enable_thinking: bool) -> Result<String, String> {
    if tools.is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
        return Err("tools are not supported for this model by Astral yet".into());
    }
    let msgs = messages.as_array().filter(|a| !a.is_empty()).ok_or("messages must be a non-empty array")?;
    let role_of = |m: &Value| m.get("role").and_then(|r| r.as_str()).map(str::to_string).ok_or("message without a role");
    // The last user message: assistant turns after it keep their reasoning in the prompt.
    let mut last_query = None;
    for (i, m) in msgs.iter().enumerate().rev() {
        if role_of(m)? == "user" {
            last_query = Some(i);
            break;
        }
    }
    let last_query = last_query.ok_or("No user query found in messages.")?;
    let mut out = String::new();
    for (i, m) in msgs.iter().enumerate() {
        let role = role_of(m)?;
        let content = content_text(m)?;
        let content = content.trim();
        match role.as_str() {
            "system" | "developer" => {
                if i != 0 {
                    return Err("System message must be at the beginning.".into());
                }
                out.push_str(&format!("{IM_START}system\n{content}{IM_END}\n"));
            }
            "user" => out.push_str(&format!("{IM_START}user\n{content}{IM_END}\n")),
            "assistant" => {
                if m.get("tool_calls").and_then(|t| t.as_array()).is_some_and(|a| !a.is_empty()) {
                    return Err("tool calls are not supported for this model by Astral yet".into());
                }
                let mut reasoning = m.get("reasoning_content").and_then(|r| r.as_str()).map(str::to_string);
                let mut content = content.to_string();
                if reasoning.is_none()
                    && let Some((think, rest)) = content.split_once("</think>")
                {
                    reasoning = Some(think.trim_end_matches('\n').rsplit("<think>").next().unwrap_or("").trim_start_matches('\n').to_string());
                    content = rest.trim_start_matches('\n').to_string();
                }
                if i > last_query {
                    out.push_str(&format!("{IM_START}assistant\n<think>\n{}\n</think>\n\n{content}{IM_END}\n", reasoning.unwrap_or_default().trim()));
                } else {
                    out.push_str(&format!("{IM_START}assistant\n{content}{IM_END}\n"));
                }
            }
            "tool" => return Err("tool messages are not supported for this model by Astral yet".into()),
            _ => return Err("Unexpected message role.".into()),
        }
    }
    out.push_str(&format!("{IM_START}assistant\n"));
    out.push_str(if enable_thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" });
    Ok(out)
}

/// Whether the request asks for the thinking block (default yes): `chat_template_kwargs.enable_thinking`
/// or a top-level `enable_thinking`.
pub fn thinking_enabled(body: &Value) -> bool {
    body.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")).or_else(|| body.get("enable_thinking")).and_then(|v| v.as_bool()).unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_chatml_with_thinking_prompt() {
        let m = json!([{"role": "system", "content": " Be brief. "}, {"role": "user", "content": "Hi"}]);
        assert_eq!(render(&m, None, true).unwrap(), "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert!(render(&m, None, false).unwrap().ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn keeps_reasoning_only_after_the_last_user_message() {
        let m = json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "<think>\nx\n</think>\n\nb"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d", "reasoning_content": "why"},
            {"role": "user", "content": [{"type": "text", "text": "e"}]}
        ]);
        let p = render(&m, None, true).unwrap();
        assert!(p.contains("<|im_start|>assistant\nb<|im_end|>\n"));
        assert!(p.contains("<|im_start|>assistant\nd<|im_end|>\n"));
        let m2 = json!([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "why"}]);
        assert!(render(&m2, None, true).unwrap().contains("<|im_start|>assistant\n<think>\nwhy\n</think>\n\nb<|im_end|>\n"));
    }

    #[test]
    fn refuses_tools_and_images() {
        let m = json!([{"role": "user", "content": "a"}]);
        assert!(render(&m, Some(&json!([{"type": "function"}])), true).is_err());
        let img = json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]);
        assert!(render(&img, None, true).is_err());
        assert!(render(&json!([{"role": "tool", "content": "x"}, {"role": "user", "content": "a"}]), None, true).is_err());
    }
}
