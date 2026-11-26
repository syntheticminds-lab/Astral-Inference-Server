# OpenAI-Compatible Server

`astral serve` exposes an OpenAI-compatible HTTP API. Existing OpenAI clients work by pointing their base URL at the server.

```bash
astral serve <model> [options]
```

`<model>` is the model's native checkpoint directory (safetensors weights, tokenizer, configuration and draft) or a GGUF file.

## Options

| Option | Default | Description |
|---|---|---|
| `--checkpoint <dir>` | `<model>` | Native checkpoint directory: tokenizer, configuration and DSpark draft. Required when `<model>` is a GGUF file |
| `--no-draft` | off | Serve without speculative decoding |
| `--k <n>` | `6` | Most drafted tokens verified per round (1–7) |
| `--host <addr>` | `0.0.0.0` | Listen address |
| `--port <port>` | `8000` | Listen port |
| `--ctx <n>` | `8192` | Context length in tokens (prompt + output) of each request |
| `--max-seqs <n>` | as many as fit, up to 64 | Requests decoded together (continuous batching); each reserves `--ctx` positions of KV cache |
| `--name <id>` | `inkling-2-mini` | Model id reported by `/v1/models` |
| `--queue <n>` | `64` | Requests that may wait beyond the running ones; more get `503` |

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
