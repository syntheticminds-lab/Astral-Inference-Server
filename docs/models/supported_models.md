# Supported Models

Each architecture in Astral is implemented natively, kernel by kernel, and verified against a reference implementation.

| Model | Architecture | Weights | Server | Speculative decoding | Notes |
|---|---|---|---|---|---|
| Inkling 2 Mini | `inkling2_loop` (looped MoE, attention residuals, short convolutions) | Native safetensors checkpoint, or GGUF | ✅ | ✅ DSpark draft (`mtp.safetensors`) | Verified layer by layer against llama.cpp |
| Qwen3.5-4B | `qwen3_5` (hybrid gated delta-net + attention) | Dense BF16 safetensors checkpoint (`Qwen/Qwen3.5-4B`) | ✅ one request at a time | — | Greedy output matches a BF16 reference up to near-ties; 281 tok/s single-stream (512 in / 128 out, GH200) |
| Qwen3.8-27B | Hybrid gated delta-net + attention | Native safetensors checkpoint | CLI only | — | Greedy output matches vLLM |

Support for more models is coming soon.

## Inputs

- Text only. Image input is not supported yet.
- The checkpoint's own tokenizer and chat template are used. Astral's chat template port produces the same token ids as the reference implementation.

## Qwen3.5-4B

`astral serve Qwen/Qwen3.5-4B` serves the model's text decoder through `/v1/completions` and `/v1/chat/completions` (streaming included).

- The chat format is the checkpoint's: ChatML (`<|im_start|>` / `<|im_end|>`) with the thinking block. Generation stops on `<|im_end|>` or `<|endoftext|>`.
- Thinking is on by default. The model's reasoning is returned as `reasoning_content` (as for Inkling 2 Mini) and the answer as `content`; pass `"chat_template_kwargs": {"enable_thinking": false}` to skip the thinking block.
- The BF16 projection weights are quantised to 8 bits (Q8_0) when loaded, with 8-bit activations, so greedy output can differ from a BF16 reference where two tokens are nearly tied. The embedding and output head stay BF16.
- Requests run one at a time (no continuous batching yet), and decoding is greedy.
- Not supported: image and video input (the vision tower is not loaded), tool calls, and the multi-token-prediction (`mtp`) draft layer.
