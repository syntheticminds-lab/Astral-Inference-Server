<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logos/synthetic-minds-logo-dark.png">
    <img alt="Synthetic Minds" src="docs/assets/logos/synthetic-minds-logo-light.png" width=50%>
  </picture>
</p>

<h1 align="center">Astral Inference Server</h1>

<h3 align="center">
Fast, lean LLM inference and serving in Rust and CUDA
</h3>

<p align="center">
| <a href="https://www.syntheticmindslab.com/astral"><b>Website</b></a> | <a href="docs/README.md"><b>Documentation</b></a> | <a href="docs/getting_started/quickstart.md"><b>Quickstart</b></a> | <a href="docs/benchmarks/inkling2_mini.md"><b>Benchmarks</b></a> | <a href="docs/models/supported_models.md"><b>Supported Models</b></a> | <a href="CONTRIBUTING.md"><b>Contributing</b></a> |
</p>

## Astral Inference Server Version: 2.6

Astral now runs Inkling 2 Mini directly from its native safetensors checkpoint, with the model's own MTP draft and a new set of custom CUDA kernels. Tested on a GH200 against vLLM on the same model, **Astral is 54% faster** than vLLM with MTP enabled; vLLM without a draft is slower still.

| Engine | Model | Config | Tokens/s |
|---|---|---|---|
| **Astral Inference Server** | Inkling 2 Mini (safetensors) | MTP draft, GH200 | **265** |
| vLLM | Inkling 2 Mini (safetensors) | MTP draft (DSpark, 7 tokens), GH200 | 162 |

Single-stream decode, 512-token prompts and 128 output tokens, through each engine's OpenAI-compatible server.


> **Status:** Version 2.6 is open source and ready for testing.

---

## Support

Astral Inference Server is free and open source. If it is useful to you, you can help fund its development:

<a href="https://buymeacoffee.com/syntheticminds_lab"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" height="48"></a>

## About

Astral Inference Server is an inference engine and OpenAI-compatible server for large language models. The host side is Rust and every kernel is hand-written CUDA. It ships as a single binary with no Python runtime.

Astral Inference Server is developed by [Synthetic Minds LTD](#contact-us).

Astral is fast with:

- Hand-written CUDA kernels for every operator, verified layer by layer against a reference implementation
- CUDA graphs for every decode step and every speculative round, with **programmatic dependent launch** overlapping kernel launches on Hopper
- Low-precision weights end to end, with int8 **tensor-core GEMVs** for small batches and wgmma GEMMs for prefill
- Fused kernels: depth-mix + RMSNorm + quantisation, attention merge + output gate, router GEMV + top-k, batched loop controllers
- **Speculative decoding** with the model's own DSpark draft, GPU-side acceptance, short-conv state rollback and an **adaptive draft length**
- Grouped MoE kernels that read each used expert once per batch

Astral is flexible and easy to use with:

- An OpenAI-compatible API server: `/v1/completions`, `/v1/chat/completions`, streaming, stop strings, reasoning output and tool calls
- [Continuous batching](docs/serving/continuous_batching.md) with packed multi-prompt prefill; a request's greedy output is bit-identical alone or batched
- The checkpoint's own tokenizer and chat template, matching reference token ids exactly
- Loads native safetensors checkpoints directly, as well as GGUF files
- A bounded request queue with cancel-on-disconnect

Astral currently supports:

- Inkling 2 Mini, with or without its DSpark draft
- Qwen3.8-27B, from the command line

Support for more models is coming soon. Find the full list of supported models [here](docs/models/supported_models.md).

## Getting Started

Install Astral with one command (Linux, an NVIDIA GPU, and the CUDA toolkit with `nvcc` 12.8+):

```bash
curl -fsSL https://raw.githubusercontent.com/syntheticminds-lab/Astral-Inference-Server/main/install.sh | sh
```

The script builds Astral for the GPUs in your machine, installs Rust if it is missing, links `astral` into `~/.local/bin` and runs the kernel self-test. To build by hand instead, see [installation](docs/getting_started/installation.md).

Serve Inkling 2 Mini with its DSpark draft, straight from Hugging Face (downloaded on first use; a local checkpoint directory or a GGUF file works too):

```bash
astral serve thinkingmachines/Inkling-2-Mini
```

Or serve one of our GGUF quantisations, with the tokenizer and draft taken from the original model:

```bash
astral serve synthetic-minds/Inkling-2-Mini-Astral-GGUF/Inkling-2-Mini-Astral-Q-Q8_0.gguf --tokenizer thinkingmachines/Inkling-2-Mini
```

The server is OpenAI-compatible at `http://localhost:8000/v1`.

Visit our [documentation](docs/README.md) to learn more.

- [Installation](docs/getting_started/installation.md)
- [Quickstart](docs/getting_started/quickstart.md)
- [Supported Models](docs/models/supported_models.md)

## Editions

| | Developer | Enterprise |
|---|---|---|
| Licence | Apache-2.0, free | Commercial |
| Who it is for | Individuals, researchers, and organisations running it internally | Organisations deploying at scale |
| Deployment | One node (one GPU today; multiple GPUs on one node planned) | Multi-node and disaggregated serving |
| Engine, kernels, OpenAI-compatible server, speculative decoding | ✅ | ✅ |
| Custom deployments | — | ✅ |
| Support | Community (GitHub) | Commercial support |

This repository is the Developer edition. For Enterprise, visit [syntheticmindslab.com](https://www.syntheticmindslab.com/astral).

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to Astral Inference Server](CONTRIBUTING.md) for how to get involved.

## Citation

If you use Astral Inference Server for your research, please cite it:

```bibtex
@software{astral2026,
  title  = {Astral Inference Server: Fast, Lean LLM Inference and Serving in Rust and CUDA},
  author = {{Synthetic Minds LTD}},
  year   = {2026},
  note   = {Version 0.0.1}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](../../issues)
- For discussions with other users, please use GitHub [Discussions](../../discussions)
- For coordinated security disclosures, please follow our [security policy](SECURITY.md)
- For commercial licensing and the beta programme, please visit [syntheticmindslab.com](https://www.syntheticmindslab.com/astral)
<!-- --8<-- [end:contact-us] -->

## License

Copyright (c) 2026 Synthetic Minds LTD. Astral Inference Server is licensed under the Apache License 2.0 (`Apache-2.0`); see [LICENSE](LICENSE) and [NOTICE](NOTICE). "Astral" and "Astral Inference Server" are trademarks of Synthetic Minds LTD.
