"""Decode optimization controls reject mistyped choices before capture."""

import pytest

from vllm_exl3.optimization import coop_decode_fast_enabled


def test_unqualified_optimization_is_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_COOP_DECODE_FAST", raising=False)
    assert not coop_decode_fast_enabled()


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True)])
def test_explicit_control(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_EXL3_COOP_DECODE_FAST", value)
    assert coop_decode_fast_enabled() is expected


@pytest.mark.parametrize("value", ["", "auto", "true", "2"])
def test_bad_control_raises(monkeypatch, value):
    monkeypatch.setenv("VLLM_EXL3_COOP_DECODE_FAST", value)
    with pytest.raises(ValueError, match="COOP_DECODE_FAST"):
        coop_decode_fast_enabled()
