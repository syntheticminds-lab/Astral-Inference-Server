# OpenAI-Compatible Server

`astral serve` exposes an OpenAI-compatible HTTP API. Existing OpenAI clients work by pointing their base URL at the server.

```bash
astral serve <model> [options]
```

`<model>` is a Hugging Face repository id, a local checkpoint directory or a GGUF file (see below).

## The model argument

`astral serve <model>` takes any of:

| `<model>` | Example |
|---|---|
| A Hugging Face repository with safetensors weights | `astral serve thinkingmachines/Inkling-2-Mini` |
| A local checkpoint directory | `astral serve ./Inkling-2-Mini/` |
| A local GGUF file | `astral serve model.gguf --tokenizer thinkingmachines/Inkling-2-Mini` |
| One GGUF file of a Hugging Face repository | `astral serve org/repo-GGUF/model-Q4_K_M.gguf --tokenizer thinkingmachines/Inkling-2-Mini` |

Repositories are downloaded on first use to `~/.cache/astral/hub` (or `--download-dir`, or `$ASTRAL_CACHE`), only the files serving needs, and an interrupted download resumes where it stopped. A complete copy already in the Hugging Face cache (`$HF_HUB_CACHE`, `$HF_HOME/hub` or `~/.cache/huggingface/hub`) is used without downloading. Set `HF_TOKEN` for gated or private repositories, and `HF_ENDPOINT` for a mirror.

A GGUF file has no chat template or draft, so it needs `--tokenizer`, a directory or repository with `tokenizer.json` (the native checkpoint). Speculative decoding runs when the checkpoint has its DSpark draft (`mtp.safetensors`).

`astral serve Qwen/Qwen3.5-4B` serves the Qwen3.5-4B text model: one request at a time, greedy, with the model's reasoning returned as `reasoning_content` (disable it per request with `"chat_template_kwargs": {"enable_thinking": false}`). Image input, tools and the `mtp` draft layer are not supported. Batching and draft options do not apply to it. See [supported models](../models/supported_models.md#qwen35-4b).

## Options

The names follow vLLM's, so existing launch scripts carry over. `--flag value` and `--flag=value` both work; `astral serve --help` lists everything.

| Option | Default | Description |
|---|---|---|
| `--tokenizer <dir\|org/name>` | the model's own | Tokenizer, chat template and draft source; required with a GGUF model |
| `--revision <rev>` | `main` | Branch, tag or commit of a repository |
| `--download-dir <dir>` | `~/.cache/astral/hub` | Where repositories are downloaded |
| `--served-model-name <id>` | the model's name | Model id reported by `/v1/models` |
| `--host <addr>` | `0.0.0.0` | Listen address |
| `--port <port>` | `8000` | Listen port |
| `--max-model-len <n>` | `8192` | Context length in tokens (prompt + output) of each request |
| `--max-num-seqs <n>` | as many as fit, up to 64 | Requests decoded together (continuous batching); each reserves `--max-model-len` positions of KV cache |
| `--max-num-batched-tokens <n>` | `2048` | Tokens per forward pass: every decoding request's next token plus prompt chunks (64 to 8192) |
| `--gpu-memory-utilization <f>` | all but a 6 GB margin | Fraction of GPU memory Astral may use for weights, KV cache and buffers |
| `--kv-cache-dtype <auto\|fp16>` | `auto` (fp16) | KV cache precision |
| `--max-queue <n>` | `64` | Requests that may wait beyond the running ones; more get `503` |
| `--no-draft` | off | Serve without speculative decoding |
| `--num-speculative-tokens <k>` | `6` | Most drafted tokens verified per round (1–7) |

The earlier spellings `--checkpoint`, `--ctx`, `--max-seqs`, `--name`, `--queue` and `--k` still work.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/v1/models` | The served model and its context length |
| `POST` | `/v1/completions` | Text completion; `prompt` is a string or a list of token ids |
| `POST` | `/v1/chat/completions` | Chat completion with the checkpoint's chat template |
| `POST` | `/tokenize` | Token ids of a `prompt` or of chat `messages` |

## Request parameters

| Parameter | Endpoints | Notes |
|---|---|---|
| `messages` | chat | System, user, assistant and tool messages; text content parts |
| `prompt` | completions | A string, a list of token ids, or a one-element list of either |
| `max_tokens` / `max_completion_tokens` | both | Defaults: chat, the rest of the context; completions, 16 |
| `stream` | both | Server-Sent Events, as the OpenAI API |
| `stream_options.include_usage` | both | A final chunk with token usage |
| `stop` | both | String or list of strings; the text stops before the first match |
| `ignore_eos` | both | Keep generating past the end-of-answer token |
| `reasoning_effort` | chat | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or a number in [0, 0.99] |
| `tools` | chat | Function tools; calls come back in `tool_calls` |
| `speculative` | both | `false` turns speculative decoding off for this request (see [continuous batching](continuous_batching.md) for when the draft runs) |

## Responses

Chat responses follow the OpenAI schema. The model's thinking is returned in `message.reasoning_content` (`delta.reasoning_content` when streaming), the answer in `content`, and tool calls in `tool_calls` with `finish_reason: "tool_calls"`. Every response carries `usage`.

## Current limitations

- Decoding is greedy. `temperature`, `top_p` and `seed` are accepted and ignored.
- Text input only; image parts are rejected.
- Requests are batched continuously (up to `--max-seqs` at a time, see [continuous batching](continuous_batching.md)); more wait in a bounded queue. A client that disconnects cancels its request.
- Each running request reserves a full `--ctx` of KV cache (there is no paged KV pool yet), so `--ctx` and `--max-seqs` trade against each other: 64 requests of 2,048 positions take about 25 GB.
- The DSpark draft runs only while a request is alone; with other requests running it decodes in the batch without it.
