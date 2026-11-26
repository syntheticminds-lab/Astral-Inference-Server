#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
#
# Installs Astral Inference Server from source:
#
#   curl -fsSL https://raw.githubusercontent.com/syntheticminds-lab/Astral-Inference-Server/main/install.sh | sh
#
# Settings (environment variables):
#   ASTRAL_HOME        install root (default ~/.astral); the source goes to $ASTRAL_HOME/src
#   ASTRAL_BIN_DIR     where the `astral` command is linked (default ~/.local/bin)
#   ASTRAL_REF         branch or tag to build (default main)
#   ASTRAL_CUDA_ARCHS  kernel targets (default: detected from the installed GPUs)
#   ASTRAL_REPO        git URL to build from
#   ASTRAL_NO_SELFTEST set to 1 to skip the kernel self-test after the build
#
# Requirements: Linux, an NVIDIA driver, the CUDA toolkit (nvcc 12.8 or newer), git and a C
# compiler. Rust is installed with rustup when cargo is missing.

set -eu

main() {
    repo="${ASTRAL_REPO:-https://github.com/syntheticminds-lab/Astral-Inference-Server.git}"
    ref="${ASTRAL_REF:-main}"
    home_dir="${ASTRAL_HOME:-$HOME/.astral}"
    bin_dir="${ASTRAL_BIN_DIR:-$HOME/.local/bin}"
    src="$home_dir/src"

    say "Astral Inference Server installer"

    [ "$(uname -s)" = Linux ] || die "Astral runs on Linux with an NVIDIA GPU (this is $(uname -s))."
    case "$(uname -m)" in
        x86_64 | aarch64) ;;
        *) die "unsupported CPU architecture: $(uname -m)" ;;
    esac
    need git
    if ! has cc && ! has gcc && ! has clang; then
        die "a C compiler is required (for example: sudo apt install build-essential)"
    fi

    find_nvcc
    archs="${ASTRAL_CUDA_ARCHS:-$(detect_archs)}"
    say "kernel targets: $archs"

    if ! has cargo; then
        if [ -x "$HOME/.cargo/bin/cargo" ]; then
            PATH="$HOME/.cargo/bin:$PATH"
        else
            say "installing Rust with rustup"
            need curl
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
                sh -s -- -y --profile minimal --default-toolchain none --no-modify-path >/dev/null
            PATH="$HOME/.cargo/bin:$PATH"
        fi
    fi

    mkdir -p "$home_dir" "$bin_dir"
    if [ -d "$src/.git" ]; then
        say "updating $src ($ref)"
        git -C "$src" fetch --quiet --depth 1 origin "$ref"
        git -C "$src" checkout --quiet --force FETCH_HEAD
    else
        say "downloading the source to $src ($ref)"
        rm -rf "$src"
        git clone --quiet --depth 1 --branch "$ref" "$repo" "$src"
    fi

    # The repository pins its Rust version in rust-toolchain.toml; rustup installs it on first use.
    say "building (the first build takes a few minutes)"
    (cd "$src" && ASTRAL_CUDA_ARCHS="$archs" cargo build --release --quiet --bin astral)

    mkdir -p "$home_dir/bin"
    cp "$src/target/release/astral" "$home_dir/bin/astral.new"
    mv "$home_dir/bin/astral.new" "$home_dir/bin/astral"
    ln -sf "$home_dir/bin/astral" "$bin_dir/astral"
    say "installed $bin_dir/astral"

    if [ "${ASTRAL_NO_SELFTEST:-0}" != 1 ] && has nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        say "running the kernel self-test"
        if "$home_dir/bin/astral" selftest >"$home_dir/selftest.log" 2>&1; then
            say "self-test passed"
        else
            warn "the self-test failed; see $home_dir/selftest.log"
        fi
    fi

    echo
    "$home_dir/bin/astral" version || true
    case ":$PATH:" in
        *":$bin_dir:"*) ;;
        *)
            echo
            echo "Add $bin_dir to your PATH to run astral from anywhere:"
            echo "  echo 'export PATH=\"$bin_dir:\$PATH\"' >> ~/.bashrc && . ~/.bashrc"
            ;;
    esac
    echo
    echo "Start serving with: astral serve /path/to/Inkling-2-Mini/"
}

# nvcc compiles the kernels. Look on PATH, then in the usual CUDA install locations.
find_nvcc() {
    if has nvcc; then
        return
    fi
    for d in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda /usr/local/cuda-*; do
        if [ -n "$d" ] && [ -x "$d/bin/nvcc" ]; then
            PATH="$d/bin:$PATH"
            say "using nvcc from $d"
            return
        fi
    done
    die "nvcc was not found. Install the CUDA toolkit 12.8 or newer: https://developer.nvidia.com/cuda-downloads"
}

# One target per GPU generation in the machine; Hopper and Blackwell get their arch-specific
# variants (90a, 100a, 120a), which the tensor-core kernels need.
detect_archs() {
    caps=""
    if has nvidia-smi; then
        caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' .' | sort -u)
    fi
    if [ -z "$caps" ]; then
        warn "no GPU detected; building for Ampere and Hopper (80,90a)"
        echo "80,90a"
        return
    fi
    out=""
    for c in $caps; do
        case "$c" in
            90 | 100 | 120) c="${c}a" ;;
        esac
        out="${out:+$out,}$c"
    done
    echo "$out"
}

has() { command -v "$1" >/dev/null 2>&1; }
need() { has "$1" || die "'$1' is required but was not found"; }
say() { printf '\033[1mastral:\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33mastral:\033[0m %s\n' "$*" >&2; }
die() {
    printf '\033[1;31mastral:\033[0m %s\n' "$*" >&2
    exit 1
}

# Everything runs from main, so a partially downloaded script does nothing.
main "$@"
