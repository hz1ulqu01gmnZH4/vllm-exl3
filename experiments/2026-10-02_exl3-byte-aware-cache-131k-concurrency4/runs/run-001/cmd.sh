#!/usr/bin/env bash
# captured 2026-10-02T10:15:49Z
nvidia-smi; free -h; ps -eo pid,comm,args | rg "vllm|profile_events|profile_model" | head -20; cat ../.venv/lib/python3.12/site-packages/vllm/v1/core/sched/output.py | sed -n "226,260p"
