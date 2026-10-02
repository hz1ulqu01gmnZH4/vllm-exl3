#!/usr/bin/env bash
set -euo pipefail
cd /home/ak/qwen38-vllm/vllm-exl3
export CACHE_TRIAL_LABEL="${1:?baseline or candidate required}"
export PYTHONPATH="$PWD/src:/home/ak/qwen38-vllm/experimental/expert-pool/src:/home/ak/qwen38-vllm/experiments/2026-09-29_exl3-full-model-cached-serving/artifacts/runtime"
export VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1
export VLLM_EXL3_MOE_KERNEL=exllamav3 VLLM_EXL3_EXPERT_CACHE_ROWS=160
export VLLM_EXL3_EXPERT_CACHE_POLICY=precision-lru VLLM_EXL3_COOP=1 VLLM_EXL3_COOP_DECODE_FAST=0
export VLLM_EXL3_NGRAM_TABLE=pinned EXL3_EXPANDABLE_SEGMENTS=0
export VLLM_CACHE_ROOT=/home/ak/qwen38-vllm/cache/exl3
export PYTORCH_ALLOC_CONF=expandable_segments:True,pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec timeout --signal=TERM --kill-after=25s 1080s /home/ak/qwen38-vllm/.venv/bin/python -u experiments/2026-10-02_exl3-byte-aware-cache-131k-concurrency4/config/trial.py
