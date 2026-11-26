# Quickstart

This guide serves Inkling 2 Mini with its DSpark draft and sends requests to it. Install Astral first ([installation](installation.md)).

## 1. Get the model files

Astral runs Inkling 2 Mini from its native checkpoint directory: the safetensors weights, the tokenizer (`tokenizer.json`), the configuration (`config.json`, with `mtp_config`) and the DSpark draft (`mtp.safetensors`).

A GGUF file of the model works too: pass it as the model and the checkpoint directory with `--checkpoint`, for the tokenizer and the draft.

## 2. Start the server

```bash
astral serve Inkling-2-Mini/
```

Loading takes about half a minute: weights, draft, and CUDA graph capture. The server is ready when it prints:

```text
astral: serving inkling-2-mini on http://0.0.0.0:8000 (context 8192, DSpark draft, k = 6)
```

Add `--no-draft` to serve without speculative decoding. See the [server reference](../serving/openai_compatible_server.md) for every option.

## 3. Send a request

With `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "inkling-2-mini", "messages": [{"role": "user", "content": "What is 17 * 23?"}], "max_tokens": 400}'
```

With the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
    max_tokens=400,
)
print(resp.choices[0].message.content)
```

The model's thinking is returned separately, in `reasoning_content`. More examples are in [`examples/`](../../examples/).

## Offline generation

The CLI can also generate from token ids without a server:

```bash
./target/release/astral generate Inkling-2-Mini/ 64 <token ids...>
./target/release/astral spec-generate Inkling-2-Mini/ Inkling-2-Mini/ 6 512 @prompt.ids
```

`spec-generate` also runs plain decoding afterwards and reports how closely the two outputs agree.
