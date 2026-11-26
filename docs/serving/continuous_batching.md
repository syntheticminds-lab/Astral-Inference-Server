# Continuous Batching

`astral serve` runs many requests at once. Each running request owns a sequence slot: its
own KV cache and short-convolution state. Every step, the engine:

1. **Admits** waiting requests while slots are free. Their prompts are prefilled together:
   prompts are packed into passes of up to 2,048 tokens, a long prompt continuing in the next
   pass, so the weights are read once per pass instead of once per prompt.
2. **Decodes** one token for every running request in a single batched forward pass. Each
   batch size runs as its own CUDA graph, captured when the server starts.
3. **Retires** requests on their own stop token, `max_tokens`, the end of the context, or a
   client disconnect. A freed slot is reused by the next admitted request.

New requests are admitted between decode steps, so a long-running request never blocks new
ones and new ones never wait for a batch to drain.

## Speculative decoding

The DSpark draft is used while a request runs alone: a request that starts with no other
request running or waiting decodes in speculative rounds. When another request arrives, it
continues without the draft, in the batch, for the rest of its generation (its KV cache and
short-conv state are exactly those of plain decoding after each round, so the switch costs
nothing). Speculation pays most at low concurrency, where it is still used.

## Results do not depend on the batch

A request's greedy output is the same whether it runs alone or batched with others, and
whatever it is batched with. Every batched kernel computes each token with exactly the
arithmetic of the single-token kernel (the batched kernels share weight loads across tokens,
not sums), kernel choices never depend on the number of rows, and nothing in a forward pass
depends on block timing (no atomic accumulation: split-K partial sums are added in a fixed
order). Prefill is likewise the same for a prompt alone or packed with others.

This holds against plain (non-speculative) decoding. Speculative rounds verify several
tokens per pass with their own kernels, so a request decoded with the draft can differ from
its plain greedy output after a near-tie; this is unchanged from single-request serving. A
request that starts with the draft and later joins a batch continues with batch-independent
plain decoding.

`astral batch-check <gguf> <n> <steps>` checks this on the GPU: it decodes `n` sequences
alone and then together (teacher-forced on the single-sequence tokens) and compares every
row's logits and pick. It passes with a maximum logit difference of exactly zero for
`n` = 1 to 64. Over the server, [`bench/serving/batch_parity.py`](../../bench/serving/batch_parity.py)
runs every prompt alone and then all at once and compares the texts.

## Memory

There is no paged KV pool yet: each slot reserves `--ctx` positions in all 48 cache slots
(about 196 KB per position, plus 2.4 MB of conv state). With `--max-seqs` unset the server
creates as many slots as fit in free GPU memory, up to 64, and prints the number at start-up.
For many concurrent short requests, lower `--ctx`:

```bash
astral serve Inkling-2-Mini/ --ctx 2048 --max-seqs 64
```

## Performance

Inkling 2 Mini on one H100 PCIe 80 GB, served with `--ctx 2048 --max-seqs 64`;
vLLM (`vllm/vllm-openai:nightly`) and SGLang (`lmsysorg/sglang:inkling-2-mini`) serve the
native checkpoint with their defaults. Load from `vllm bench serve`: random prompts, 512 input
and 128 output tokens, `ignore_eos`, 8 x concurrency prompts (at least 16).
[`bench/baseline_servers.sh`](../../bench/baseline_servers.sh) runs it.

| Concurrency | Astral | Astral + DSpark | vLLM | SGLang |
|---|---|---|---|---|
| 1 | 147.8 tok/s | **165.7 tok/s** | 113.9 tok/s | 123.3 tok/s |
| 8 | 410.6 tok/s | 409.7 tok/s | 316.7 tok/s | **509.8 tok/s** |
| 32 | 594.2 tok/s | 595.8 tok/s | **1203.3 tok/s** | 901.7 tok/s |
| 64 | 646.3 tok/s | 649.9 tok/s | **1781.5 tok/s** | 1195.8 tok/s |

Time to first token, p50 / p99:

| Concurrency | Astral | vLLM | SGLang |
|---|---|---|---|
| 1 | 112 / 759 ms | 57 / 1953 ms | 81 / 233 ms |
| 8 | 631 / 1310 ms | 229 / 6810 ms | 375 / 736 ms |
| 32 | 2422 / 6855 ms | 613 / 2909 ms | 1046 / 1443 ms |
| 64 | 4822 / 12657 ms | 607 / 3366 ms | 1839 / 2717 ms |

Astral leads at concurrency 1 and is ahead of vLLM at 8; from 32 up it is behind both. At
these concurrencies the workload is dominated by prefill (four prompt tokens per output
token), and Astral's prefill runs at about 5,000 tokens/s (512 tokens in 100 ms alone, 16 x
512 in 1.13 s packed), well below what the GPU allows. A batched decode step takes 5.5 ms
for 1 sequence, 13.9 ms for 9, 30.8 ms for 33 and 51 ms for 64. Faster prefill GEMMs and
batched decode kernels are the next work.
