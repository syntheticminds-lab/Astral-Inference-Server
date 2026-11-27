# Benchmarks: Inkling 2 Mini

Single-stream decode speed of Astral, vLLM and SGLang serving Inkling 2 Mini on one GPU.

## GH200: safetensors checkpoint with the MTP draft

Single-stream decode of Inkling 2 Mini (native safetensors checkpoint) on a GH200, 512-token prompts and 128 output tokens, through each engine's OpenAI-compatible server.

| Engine | Model | Config | Tokens/s |
|---|---|---|---|
| **Astral Inference Server** | Inkling 2 Mini (safetensors) | MTP draft, GH200 | **265** |
| vLLM | Inkling 2 Mini (safetensors) | MTP draft (DSpark, 7 tokens), GH200 | 162 |

Astral is 54% faster than vLLM with MTP enabled. vLLM without a draft reaches 159 tokens/s.

## H100 PCIe

## Setup

| | |
|---|---|
| GPU | NVIDIA H100 PCIe 80 GB, CUDA 12.8 driver |
| Astral | `astral serve`, Inkling 2 Mini, defaults (adaptive DSpark, at most 6 drafts) |
| vLLM | `vllm/vllm-openai:nightly` (0.34.0rc1), native checkpoint, `--reasoning-parser inkling`; Python frontend and Rust frontend (`VLLM_USE_RUST_FRONTEND=1`); MTP: official DSpark config, 7 drafts |
| SGLang | `lmsysorg/sglang:inkling-2-mini`, native checkpoint; MTP: `--speculative-algorithm DSPARK` (7 drafts) |
| Workload | One request at a time; streaming `/v1/completions` on the chat-template token ids; 512 output tokens; greedy; `ignore_eos`; context 8,192 |
| Prompts | Two sets of 16 chat prompts (code, prose, explanations, math, lists): [`prompts_set_a.json`](../../bench/serving/prompts_set_a.json), [`prompts_set_b.json`](../../bench/serving/prompts_set_b.json) |
| Metric | Decode tok/s from the first streamed token to the last, averaged over prompts |

Every engine runs on its own greedy text. With a draft, speed depends on how predictable that text is, so results are averaged over 32 prompts.

## Results

<p align="center">
  <img alt="Speedup vs vLLM (Python)" src="../../bench/charts/astral_vs_vllm_python.png" width=48%>
  <img alt="Speedup vs vLLM (Rust frontend)" src="../../bench/charts/astral_vs_vllm_rust.png" width=48%>
</p>

Mean decode tok/s over all 32 prompts:

| Engine | Without MTP | Speedup | With MTP | Speedup |
|---|---|---|---|---|
| vLLM (Python frontend) | 130.2 | 1.00× | 209.6 | 1.00× |
| vLLM (Rust frontend) | 130.4 | 1.00× | 209.7 | 1.00× |
| SGLang | 135.6 | 1.04× | 205.9 | 0.98× |
| **Astral** | **172.0** | **1.32×** | **233.3** | **1.11×** |

Speedup is relative to vLLM's Python frontend.

Per prompt set:

| Engine | Set A, no MTP | Set A, MTP | Set B, no MTP | Set B, MTP |
|---|---|---|---|---|
| vLLM (Python) | 130.2 | 218.0 | 130.3 | 201.2 |
| vLLM (Rust) | 130.4 | 218.2 | 130.4 | 201.2 |
| SGLang | 135.6 | 219.0 | 135.7 | 192.8 |
| **Astral** | **172.2** | **242.0** | **171.8** | **224.6** |

With MTP on set B, Astral is faster than vLLM on 14 of 16 prompts and faster than SGLang on 15 of 16.

Repeated runs of the same configuration agree to within 0.5%. The spread between prompts is much larger, since acceptance depends on the text, which is why two prompt sets are reported.

## Time to first token

On these 28–61-token prompts, Astral's server returns the first token in 64–78 ms. vLLM takes 38–54 ms and SGLang 53–60 ms. Short-prompt latency has not been optimised yet.

## Reproduce

```bash
astral serve Inkling-2-Mini/ &
python bench/serving/benchmark_serving.py --base-url http://localhost:8000 \
    --prompts bench/serving/prompts_set_a.json --save-ids set_a.ids.json --label astral
```

To run the same token ids against another server, pass `--ids set_a.ids.json`. [`bench/charts/make_charts.py`](../../bench/charts/make_charts.py) draws the charts and tables.
