"""CPU contract tests for checkpoint loading into complete packed expert rows."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm_exl3 import exl3
from vllm_exl3.expert_cache import PackedExpertLayer
from vllm_exl3.expert_cache_loader import create_cached_weights, finalize_cached_weights


@pytest.fixture
def host_layer(monkeypatch):
    import vllm.config

    # Pinning itself is covered by the CUDA suite; this fixture exercises
    # loading bytes without needing a GPU driver or the optional NVFP4 fork.
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setenv("VLLM_EXL3_EXPERT_CACHE_ROWS", "2")
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        offload_config=SimpleNamespace(),
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(exl3, "_resolve_tp_geometry", lambda layer: (0, 1))
    layer = torch.nn.Module()
    method = SimpleNamespace(bits=2, moe=SimpleNamespace(experts_per_token=2))
    create_cached_weights(method, layer, 4, 128, 128)
    return layer, method, config


def _expert(seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        projection: {
            "trellis": torch.randint(
                -100, 100, (8, 8, 32), generator=generator, dtype=torch.int16
            ),
            "suh": torch.randn(128, generator=generator, dtype=torch.float16),
            "svh": torch.randn(128, generator=generator, dtype=torch.float16),
            "mul1": torch.tensor(-2082680531, dtype=torch.int32),
        }
        for projection in ("gate", "up", "down")
    }


def test_host_loader_matches_native_packer_without_nvfp4_extension(host_layer):
    layer, _, _ = host_layer
    experts = [_expert(i) for i in range(4)]
    for e, expert in enumerate(experts):
        for projection, shard in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            for attr, value in expert[projection].items():
                name = f"{'w2' if projection == 'down' else 'w13'}_{attr}"
                param = getattr(layer, name)
                param.weight_loader(param, value, name, shard_id=shard, expert_id=e)
    finalize_cached_weights(layer)
    reference = PackedExpertLayer.from_experts(experts)
    assert torch.equal(layer._exl3_host_source.data, reference.data)
    assert layer._exl3_host_source.shared_suh == reference.shared_suh


def test_host_loader_rejects_missing_and_duplicate_tensors(host_layer):
    layer, _, _ = host_layer
    with pytest.raises(RuntimeError, match="Incomplete EXL3 experts"):
        finalize_cached_weights(layer)
    param = layer.w13_mul1
    marker = torch.tensor(-2082680531, dtype=torch.int32)
    param.weight_loader(param, marker, "w13_mul1")
    with pytest.raises(ValueError, match="Duplicate"):
        param.weight_loader(param, marker, "w13_mul1")
    with pytest.raises(ValueError, match="layout mismatch"):
        layer.w13_trellis.weight_loader(
            layer.w13_trellis, torch.zeros(8, 8, 48, dtype=torch.int16), "w13_trellis"
        )


def test_host_loader_rejects_simultaneous_nvfp4_pool(host_layer):
    layer, method, config = host_layer
    config.offload_config.moe_expert_pool_rows = 2
    with pytest.raises(ValueError, match="NVFP4 pool"):
        create_cached_weights(method, layer, 4, 128, 128)
