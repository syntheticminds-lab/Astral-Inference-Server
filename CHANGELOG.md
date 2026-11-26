# Changelog

All notable changes to Astral Inference Server are documented here.

## [Unreleased]

### Changed

- Relicensed from AGPL-3.0-only to Apache-2.0. Releases up to and including 2.5 remain available under AGPL-3.0-only.

### Added

- `astral serve` takes a Hugging Face repository id (`org/name`, or `org/name/file.gguf`) and downloads it on first use, reusing the Hugging Face cache; new vLLM-style options `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--served-model-name`, `--tokenizer`, `--revision`, `--download-dir`, `--num-speculative-tokens`, and `--help`.
- Inkling 2 Mini loads straight from its native safetensors checkpoint (`astral serve <checkpoint dir>`), converted on load to the same tensors as the GGUF path.
- OpenAI-compatible server (`astral serve`): `/v1/completions`, `/v1/chat/completions` (streaming, stop strings, reasoning output, tool calls), `/v1/models`, `/tokenize`, `/health`.
- Inkling 2 Mini chat template and tokenizer, matching reference token ids.
- Speculative decoding with the Inkling 2 Mini DSpark draft: GPU-side acceptance, short-conv rollback, one CUDA graph per round, adaptive draft length.
- Int8 tensor-core GEMV for 1–8 tokens; grouped MoE GEMV for small batches.
- Fused kernels: depth-mix + RMSNorm + quantisation, attention merge + output gate, router GEMV + top-k, batched loop controllers.
- Programmatic dependent launch on compute capability 9.0+.
- Qwen3.8-27B: native 4-bit decode, wgmma prefill on Hopper.
- Benchmark client and prompt sets (`bench/serving/`).
- Continuous batching in `astral serve`: one sequence slot per running request, admission between decode steps, packed multi-prompt prefill, per-request stop and eviction, `--max-seqs`. Batched greedy output is bit-identical to single-sequence output (`astral batch-check`).
- Deterministic prefill: split-K partial sums are added in a fixed order (`ASTRAL_SPLITK=1` restores the atomic split).

### Performance (H100 PCIe, Inkling 2 Mini, single stream)

- Decode: 172.0 tok/s (1.32× vLLM) without a draft; 233.3 tok/s (1.11× vLLM) with the DSpark draft.

## [0.0.1]

### Added

- Workspace, GPU wrapper, memory arena, kernel build pipeline, hardware probing, self-tests.
