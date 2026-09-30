#!/usr/bin/env bash
# Text-only, single-GPU Qwen EXL3 expert-cache profile.
set -euo pipefail

if [[ "${1:-}" == "--help" || $# == 0 ]]; then
  cat <<'HELP'
Usage: scripts/serve-exl3.sh [--check-config] MODEL_OVERLAY [vLLM options...]

Activate the matching vLLM/PyTorch/ExLlamaV3 environment first.
EXL3_PYTHON selects its Python executable (default: python).
PYTHONPATH may select the patched vLLM source and a compatible EXL3 extension.
EXL3_CACHE_ROWS selects initial expert slots per layer (default: 64).
EXL3_CACHE_POLICY selects global-lru (default) or compact precision-lru banks.
EXL3_DECODE_GRAPHS=1 enables a decode CUDA graph for the default single sequence.

Defaults: localhost:8009, qwen3.8-exl3, context 2048, one sequence, eager,
batch 128, BF16 KV with a 1 GiB budget. Trailing vLLM options override defaults.
--check-config resolves the engine configuration without loading weights.
See docs/expert-cache.html for the required vLLM patch and checkpoint overlay.
HELP
  exit 0
fi

check_config=0
if [[ "$1" == "--check-config" ]]; then
  check_config=1
  shift
fi
if [[ $# == 0 || ! -d "$1" ]]; then
  echo 'Expected an existing prepared model overlay directory.' >&2
  exit 2
fi
model_dir="$1"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${EXL3_PYTHON:-python}"

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_EXL3_MOE_KERNEL=exllamav3
export VLLM_EXL3_EXPERT_CACHE_ROWS="${EXL3_CACHE_ROWS:-64}"
export VLLM_EXL3_EXPERT_CACHE_POLICY="${EXL3_CACHE_POLICY:-global-lru}"
case "$VLLM_EXL3_EXPERT_CACHE_POLICY" in
  global-lru|precision-lru) ;;
  *) echo 'EXL3_CACHE_POLICY must be global-lru or precision-lru.' >&2; exit 2 ;;
esac
export VLLM_EXL3_NGRAM_TABLE=pinned EXL3_EXPANDABLE_SEGMENTS=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1}"

case "${EXL3_DECODE_GRAPHS:-0}" in
  0) execution_args=(--enforce-eager) ;;
  1) execution_args=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}') ;;
  *) echo 'EXL3_DECODE_GRAPHS must be 0 or 1.' >&2; exit 2 ;;
esac

serve_args=("$model_dir"
  --served-model-name qwen3.8-exl3 --host 127.0.0.1 --port 8009
  --quantization exl3 --dtype bfloat16 --tensor-parallel-size 1
  --load-format safetensors --safetensors-load-strategy lazy
  --language-model-only "${execution_args[@]}"
  --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 128
  --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 1073741824
  --no-enable-prefix-caching --seed 42
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  "$@")

if [[ "$check_config" == 1 ]]; then
  exec "$python_bin" "$script_dir/check-exl3-config.py" "${serve_args[@]}"
fi
exec "$python_bin" -m vllm.entrypoints.cli.main serve "${serve_args[@]}"
