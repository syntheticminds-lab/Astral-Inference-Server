#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
#
# Server-level baseline (spec 08 §3): the same OpenAI-API load generator (`vllm bench serve`,
# random prompts, fixed input/output lengths, ignore_eos) against each engine's server.
# Each engine runs alone on the GPU. A failure to start is recorded, not hidden.
#
# MODEL selects the checkpoints (4-bit weights, 16-bit activations on the A100):
#   inkling (default)  llama.cpp: Unsloth UD-Q4_K_XL GGUF; vLLM/SGLang: native NVFP4/MXFP8 checkpoint
#   qwen38             llama.cpp: Unsloth Qwen3.8-27B UD-Q4_K_XL GGUF;
#                      vLLM/SGLang: RedHatAI/Qwen3.8-27B-INT4 (W4A16, group 128)
#
#   qwen38-mxfp4       vLLM/SGLang only: RedHatAI/Qwen3.8-27B-MXFP4
#   qwen38-fp8         vLLM/SGLang only: Qwen/Qwen3.8-27B-FP8
#
# Engines: llamacpp, vllm, sglang and astral (`astral serve` from ASTRAL_BIN, default
# $W/astral/target/release/astral, with ASTRAL_ARGS). VLLM_ARGS and SGL_ARGS add server flags
# (for example a speculative-decoding config). Results go to $W/results/<stamp>-servers-$MODEL$TAG.
#
# Usage: MODEL=qwen38 bench/baseline_servers.sh [engines...]   (default engines: llamacpp vllm sglang)
set -uo pipefail

W=${ASTRAL_WORK:-$HOME/astral-work}
MODEL=${MODEL:-inkling}
case $MODEL in
  inkling)
    GGUF="$W/models/Inkling-2-Mini-UD-Q4_K_XL.gguf"; NATIVE="$W/models/Inkling-2-Mini"
    SLOT_CTX=8192; VLLM_EXTRA=""; SGL_EXTRA="" ;;
  qwen38)
    GGUF="$W/models/qwen-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"; NATIVE="$W/models/Qwen3.8-27B-INT4"
    # 16 full-attention layers at 65.5 KB/token (bf16) plus ~150 MB GDN state per slot:
    # 32 slots x 8K would not fit next to 17.7 GB of weights on a 40 GB card.
    SLOT_CTX=2048; VLLM_EXTRA="--language-model-only --max-num-seqs 256"; SGL_EXTRA="" ;;
    # max-num-seqs: vLLM 0.34 defaults to 1024 and refuses to start on the H100 when only
    # 1023 Mamba state blocks fit; 256 is far above the benchmark's 32 concurrent requests.
  qwen38-mxfp4)  # RedHatAI/Qwen3.8-27B-MXFP4 (compressed-tensors mxfp4, group 32, E8M0 scales); no GGUF equivalent
    GGUF=""; NATIVE="$W/models/Qwen3.8-27B-MXFP4"
    SLOT_CTX=2048; VLLM_EXTRA="--language-model-only --max-num-seqs 256"; SGL_EXTRA="" ;;
  qwen38-fp8)    # Qwen/Qwen3.8-27B-FP8 (E4M3, 128x128 block scales); native FP8 on Hopper
    GGUF=""; NATIVE="$W/models/Qwen3.8-27B-FP8"
    SLOT_CTX=2048; VLLM_EXTRA="--language-model-only --max-num-seqs 256"; SGL_EXTRA="" ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
LLAMA="$W/baselines/llama.cpp/build/bin"
VLLM_IMG=${VLLM_IMG:-vllm/vllm-openai:latest}
SGL_IMG=${SGL_IMG:-lmsysorg/sglang:latest}
IN=${IN_LEN:-512}
OUTL=${OUT_LEN:-128}
CONCURRENCY=${CONCURRENCY:-"1 2 4 8 16 32"}
CTX=${CTX:-8192}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$W/results/$STAMP-servers-$MODEL${TAG:-}"
ASTRAL_BIN=${ASTRAL_BIN:-$W/astral/target/release/astral}
ASTRAL_ARGS=${ASTRAL_ARGS:-}
VLLM_EXTRA="$VLLM_EXTRA ${VLLM_ARGS:-}"
SGL_EXTRA="$SGL_EXTRA ${SGL_ARGS:-}"
mkdir -p "$OUT"
ENGINES=${*:-llamacpp vllm sglang}

digest() { sudo -n docker image inspect "$1" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo unknown; }

{
  echo "{"
  echo "  \"timestamp_utc\": \"$STAMP\","
  echo "  \"model\": \"$MODEL\","
  echo "  \"workload\": {\"input_len\": $IN, \"output_len\": $OUTL, \"concurrency\": \"$CONCURRENCY\", \"dataset\": \"random\", \"ignore_eos\": true},"
  echo "  \"llama_cpp_commit\": \"$(git -C "$W/baselines/llama.cpp" rev-parse HEAD)\","
  echo "  \"vllm_image\": \"$(digest "$VLLM_IMG")\","
  echo "  \"sglang_image\": \"$(digest "$SGL_IMG")\","
  echo "  \"gguf\": \"$GGUF\","
  echo "  \"native_checkpoint\": \"$NATIVE\","
  echo "  \"astral\": \"$("$ASTRAL_BIN" version 2>/dev/null | head -1)\","
  echo "  \"astral_args\": \"$ASTRAL_ARGS\", \"vllm_args\": \"$VLLM_EXTRA\", \"sglang_args\": \"$SGL_EXTRA\""
  echo "}"
} > "$OUT/env.json"

wait_ready() { # url tries [container]: stop waiting as soon as the server process/container dies
  for _ in $(seq 1 "$2"); do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    if [ -n "${3:-}" ] && [ "$(sudo -n docker inspect -f '{{.State.Running}}' "$3" 2>/dev/null)" != "true" ]; then
      echo "   container $3 exited before becoming ready"; return 1
    fi
    sleep 2
  done
  return 1
}

stop_all() {
  sudo -n docker rm -f astral-vllm astral-sglang >/dev/null 2>&1 || true
  pkill -f "llama-server .*--port 8080" 2>/dev/null || true
  pkill -f "^$ASTRAL_BIN serve" 2>/dev/null || true
  sleep 3
}

run_client() { # engine base_url served_model_name backend
  local eng=$1 url=$2 name=$3 backend=$4
  for c in $CONCURRENCY; do
    local n=$(( c * 8 < 16 ? 16 : c * 8 ))
    echo "   $eng concurrency=$c prompts=$n"
    sudo -n docker run --rm --network host -v "$NATIVE":/tok:ro -v "$OUT":/results \
      --entrypoint vllm "$VLLM_IMG" bench serve \
      --backend "$backend" --base-url "$url" --endpoint /v1/completions --model "$name" --tokenizer /tok \
      --dataset-name random --random-input-len "$IN" --random-output-len "$OUTL" \
      --num-prompts "$n" --max-concurrency "$c" --ignore-eos --seed 1 \
      --save-result --result-dir /results --result-filename "$eng-c$c.json" \
      > "$OUT/$eng-c$c.log" 2>&1 || echo "   (client failed, see $eng-c$c.log)"
  done
}

warm_gpu() { # same time-based warm-up rule as `astral hw --bench`
  "$ASTRAL_BIN" hw --bench-fresh > "$OUT/astral_hw_$1.json" 2>&1 || true
}

for eng in $ENGINES; do
  stop_all
  echo "== $eng"
  case $eng in
    llamacpp)
      "$LLAMA/llama-server" -m "$GGUF" -ngl 99 -c $((SLOT_CTX * 32)) -np 32 --port 8080 --host 127.0.0.1 \
        --alias model > "$OUT/llamacpp-server.log" 2>&1 &
      if wait_ready http://127.0.0.1:8080/health 300; then warm_gpu llamacpp; run_client llamacpp http://127.0.0.1:8080 model openai
      else echo "   llama-server did not become ready" | tee "$OUT/llamacpp-FAILED"; fi ;;
    vllm)
      sudo -n docker run -d --name astral-vllm --gpus all --ipc=host --network host -v "$NATIVE":/models/native:ro \
        --entrypoint vllm "$VLLM_IMG" serve /models/native --served-model-name model \
        --max-model-len "$CTX" --gpu-memory-utilization 0.90 --port 8000 $VLLM_EXTRA > /dev/null
      if wait_ready http://127.0.0.1:8000/health 600 astral-vllm; then warm_gpu vllm; run_client vllm http://127.0.0.1:8000 model vllm
      else echo "   vLLM did not become ready" | tee "$OUT/vllm-FAILED"; fi
      sudo -n docker logs astral-vllm > "$OUT/vllm-server.log" 2>&1 ;;
    sglang)
      sudo -n docker run -d --name astral-sglang --gpus all --ipc=host --shm-size 32g --network host \
        -v "$NATIVE":/models/native:ro -v "$(cd "$(dirname "$0")" && pwd)/serving/sglang_launch.py":/sglang_launch.py:ro "$SGL_IMG" python3 /sglang_launch.py \
        --model-path /models/native --served-model-name model --context-length "$CTX" \
        --mem-fraction-static 0.85 --host 127.0.0.1 --port 30000 $SGL_EXTRA > /dev/null
      if wait_ready http://127.0.0.1:30000/health 600 astral-sglang; then warm_gpu sglang; run_client sglang http://127.0.0.1:30000 model openai
      else echo "   SGLang did not become ready" | tee "$OUT/sglang-FAILED"; fi
      sudo -n docker logs astral-sglang > "$OUT/sglang-server.log" 2>&1 ;;
    astral)
      # shellcheck disable=SC2086
      "$ASTRAL_BIN" serve "$GGUF" --checkpoint "$NATIVE" --name model --host 127.0.0.1 --port 8001 $ASTRAL_ARGS > "$OUT/astral-server.log" 2>&1 &
      if wait_ready http://127.0.0.1:8001/health 600; then warm_gpu astral; run_client astral http://127.0.0.1:8001 model openai
      else echo "   astral did not become ready" | tee "$OUT/astral-FAILED"; fi ;;
  esac
done
stop_all
echo "$OUT"
