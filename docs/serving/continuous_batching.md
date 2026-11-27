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

`astral batch-check <model> <n> <steps>` checks this on the GPU: it decodes `n` sequences
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
