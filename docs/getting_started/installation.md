# Installation

Astral is built from source. It produces one binary, `astral`, with the CUDA kernels embedded as fatbins.

## One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/syntheticminds-lab/Astral-Inference-Server/main/install.sh | sh
```

The script checks the requirements below, installs Rust with `rustup` if `cargo` is missing, downloads the source to `~/.astral/src`, builds it for the GPUs it finds, links `astral` into `~/.local/bin` and runs `astral selftest`. Run it again to update to the latest version.

You can set these environment variables before `sh`:

| Variable | Default | Meaning |
|---|---|---|
| `ASTRAL_HOME` | `~/.astral` | Install directory (source, binary, self-test log) |
| `ASTRAL_BIN_DIR` | `~/.local/bin` | Where the `astral` command is linked |
| `ASTRAL_REF` | `main` | Branch or tag to build |
| `ASTRAL_CUDA_ARCHS` | detected | Kernel targets, e.g. `80,90a` |
| `ASTRAL_NO_SELFTEST` | unset | Set to `1` to skip the self-test |

For example, to build for both Ampere and Hopper:

```bash
curl -fsSL https://raw.githubusercontent.com/syntheticminds-lab/Astral-Inference-Server/main/install.sh | ASTRAL_CUDA_ARCHS=80,90a sh
```

To uninstall, remove `~/.astral` and `~/.local/bin/astral`.

## Requirements

- Linux (x86_64)
- An NVIDIA GPU and driver with CUDA 12 support (see [supported hardware](../hardware/supported_hardware.md))
- The CUDA toolkit (`nvcc` 12.8 or newer) to compile the kernels
- Rust (the repository pins its version in `rust-toolchain.toml`; `rustup` installs it automatically)
- A C compiler (the tokenizer library builds a small C dependency)

Without `nvcc` the workspace still builds, but with no GPU kernels.

## Build by hand

```bash
cargo build --release
```

Kernels are compiled for the targets in `ASTRAL_CUDA_ARCHS` (default `80`). Build for every GPU generation you will run on. For Hopper, include `90a` to enable the wgmma prefill kernels:

```bash
ASTRAL_CUDA_ARCHS=80,90a cargo build --release
```

Each target is compiled to SASS ahead of time (`-gencode arch=compute_X,code=sm_X`), so the binary carries no PTX JIT step.

## Check the installation

```bash
./target/release/astral version        # build info and kernel targets
./target/release/astral hw             # GPU, driver and capability detection
./target/release/astral hw --bench     # plus a memory-bandwidth microbenchmark
./target/release/astral selftest       # every kernel against its CPU reference
```

`selftest` should report every case as passed. If a kernel module has no binary for your GPU, rebuild with the matching `ASTRAL_CUDA_ARCHS` entry.
