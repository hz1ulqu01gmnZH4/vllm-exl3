#!/usr/bin/env bash
# Local RTX 5090 profile for a prepared qwen38-vllm workspace.
# Requires its .venv, supervise.py, model overlay, FP8-capable vLLM fork,
# ExLlamaV3 runtime below, and a plugin with precision-lru expert caching.
# The workspace defaults to this repository's parent; override QWEN_EXL3_SETUP
# to use another location. Trailing vLLM arguments override the profile.
set -euo pipefail
QWEN_EXL3_SETUP="${QWEN_EXL3_SETUP:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
QWEN_EXL3_RUNTIME="$QWEN_EXL3_SETUP/experiments/2026-09-29_exl3-full-model-cached-serving/artifacts/runtime"

export PYTHONPATH="$QWEN_EXL3_SETUP/experimental/expert-pool/src:$QWEN_EXL3_RUNTIME"
export VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1
export VLLM_EXL3_MOE_KERNEL=exllamav3 VLLM_EXL3_EXPERT_CACHE_ROWS=160
export VLLM_EXL3_EXPERT_CACHE_POLICY=precision-lru
export VLLM_EXL3_COOP=1
export VLLM_EXL3_NGRAM_TABLE=pinned EXL3_EXPANDABLE_SEGMENTS=0
export VLLM_CACHE_ROOT="$QWEN_EXL3_SETUP/cache/exl3"
export PYTORCH_ALLOC_CONF=expandable_segments:True,pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

# Four sequences with up to 131072 tokens each, FP8 KV, and no MTP.
# Reserve 8 GiB for KV; full-context concurrency still needs runtime validation.
exec "$QWEN_EXL3_SETUP/.venv/bin/python" "$QWEN_EXL3_SETUP/supervise.py" -- \
  "$QWEN_EXL3_SETUP/.venv/bin/python" -m vllm.entrypoints.cli.main serve \
  "$QWEN_EXL3_SETUP/models/Qwen3.8-Flash-Next-EXL3-cache" \
  --served-model-name qwen3.8-exl3 --host 127.0.0.1 --port 8009 \
  --quantization exl3 --dtype bfloat16 --tensor-parallel-size 1 \
  --load-format safetensors --safetensors-load-strategy lazy \
  --kv-cache-dtype fp8_e4m3 \
  --language-model-only \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
  --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 8589934592 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  "$@"
