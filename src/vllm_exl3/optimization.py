"""Explicit serving optimization controls, read before graph capture."""

import os


def coop_decode_fast_enabled() -> bool:
    value = os.environ.get("VLLM_EXL3_COOP_DECODE_FAST", "0")
    if value not in ("0", "1"):
        raise ValueError("VLLM_EXL3_COOP_DECODE_FAST must be 0 or 1")
    return value == "1"
