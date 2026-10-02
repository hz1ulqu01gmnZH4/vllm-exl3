#!/usr/bin/env bash
# captured 2026-10-02T09:05:02Z
nvidia-smi; ps -eo pid,comm,args | rg "vllm|strata|nsys" | head -30; cat ../serve-exl3.sh ../env.sh; rg --files ../experiments ../experimental | rg "profil|bench|strata|REPORT|report|\.py$" | head -100; command -v nsys; command -v ncu; free -h
