"""Direct host loading for the single-GPU, eager EXL3 cache integration."""

import os

import torch
from torch.nn import Parameter

from .expert_cache import ATTRIBUTES, PROJECTIONS, Exl3ExpertCache, PackedExpertLayer


def cache_rows():
    rows = int(os.environ.get("VLLM_EXL3_EXPERT_CACHE_ROWS", "0"))
    if rows < 0:
        raise ValueError("VLLM_EXL3_EXPERT_CACHE_ROWS must be nonnegative")
    return rows


def create_cached_weights(method, layer, num_experts, hidden, intermediate):
    from vllm.config import get_current_vllm_config

    from .exl3 import _resolve_tp_geometry

    config = get_current_vllm_config()
    if not config.model_config.enforce_eager:
        raise ValueError("EXL3 cache loader currently requires --enforce-eager")
    if _resolve_tp_geometry(layer)[1] != 1 or getattr(layer, "use_ep", False):
        raise ValueError("EXL3 cache loader requires TP=EP=1")
    # The NVFP4 pool is an optional extension, absent from upstream vLLM.
    if (
        hasattr(config.offload_config, "moe_expert_pool_rows")
        and config.offload_config.moe_expert_pool_rows
    ):
        raise ValueError("Use VLLM_EXL3_EXPERT_CACHE_ROWS, not the NVFP4 pool flag")
    if not method.moe.experts_per_token <= cache_rows() < num_experts:
        raise ValueError("EXL3 cache rows must be >= top_k and < num_experts")

    # Derive the exact segment layout with the same validated packer as the
    # standalone cache, then allocate the complete source directly in host RAM.
    exemplar = {}
    for projection in PROJECTIONS:
        in_dim, out_dim = (
            (intermediate, hidden) if projection == "down" else (hidden, intermediate)
        )
        exemplar[projection] = {
            "trellis": torch.zeros(
                (in_dim // 16, out_dim // 16, method.bits * 16),
                dtype=torch.int16,
                device="cpu",
            ),
            "suh": torch.zeros(in_dim, dtype=torch.float16, device="cpu"),
            "svh": torch.zeros(out_dim, dtype=torch.float16, device="cpu"),
            "mul1": torch.tensor(-2082680531, dtype=torch.int32, device="cpu"),
        }
    source = PackedExpertLayer.from_experts([exemplar])
    source.data = torch.empty(
        (num_experts, source.data.shape[1]),
        dtype=torch.int32,
        device="cpu",
        pin_memory=True,
    )
    source.shared_suh = ()
    layer._exl3_host_source = source
    layer._exl3_host_loaded = set()
    layer._exl3_hidden_size = hidden
    layer._exl3_intermediate_local = intermediate
    layer._exl3_bits = method.bits
    layer._exl3_n_experts = num_experts
    layer._exl3_top_k = method.moe.experts_per_token

    def load(
        param,
        loaded_weight,
        weight_name,
        shard_id="w1",
        expert_id=0,
        return_success=False,
    ):
        del param
        projection = {"w1": "gate", "w3": "up", "w2": "down"}[shard_id]
        attr = weight_name.rsplit("_", 1)[-1]
        if attr not in (*ATTRIBUTES, "mul1") or not 0 <= expert_id < num_experts:
            raise ValueError(f"Unsupported cached expert tensor: {weight_name}")
        key = (expert_id, projection, attr)
        if key in layer._exl3_host_loaded:
            raise ValueError(f"Duplicate cached expert tensor: {key}")
        if attr == "mul1":
            if loaded_weight.numel() != 1 or int(loaded_weight) != -2082680531:
                raise ValueError("Cached experts require the mul1 codebook")
        else:
            segment = next(
                s for s in source.segments if s.name == f"{projection}_{attr}"
            )
            if (
                loaded_weight.dtype != segment.dtype
                or tuple(loaded_weight.shape) != segment.shape
            ):
                raise ValueError(
                    f"Cached expert layout mismatch: {key}, got {tuple(loaded_weight.shape)}"
                )
            source.view(source.data, expert_id, segment).copy_(loaded_weight)
        layer._exl3_host_loaded.add(key)
        return True if return_success else None

    for prefix in ("w13", "w2"):
        for attr in (*ATTRIBUTES, "mul1"):
            param = Parameter(torch.empty(0, device="cpu"), requires_grad=False)
            param.weight_loader = load
            layer.register_parameter(f"{prefix}_{attr}", param)
    from .exl3 import _exl3_routed_experts_loader

    layer.load_weights = _exl3_routed_experts_loader(layer)


def finalize_cached_weights(layer):
    source = layer._exl3_host_source
    expected = source.num_experts * len(PROJECTIONS) * 4
    if len(layer._exl3_host_loaded) != expected:
        raise RuntimeError(
            f"Incomplete EXL3 experts: {len(layer._exl3_host_loaded)}/{expected} tensors"
        )
    segments = {s.name: s for s in source.segments}
    source.shared_suh = tuple(
        torch.equal(
            source.view(source.data, e, segments["gate_suh"]),
            source.view(source.data, e, segments["up_suh"]),
        )
        for e in range(source.num_experts)
    )
    del layer._exl3_host_loaded


def install_expert_cache(model, device, max_decode_tokens):
    layers = [m for m in model.modules() if hasattr(m, "_exl3_host_source")]
    if not layers:
        raise RuntimeError("EXL3 cache requested but no host-loaded layers found")
    cache = Exl3ExpertCache(
        [m._exl3_host_source for m in layers],
        slots_per_layer=cache_rows(),
        top_k=layers[0]._exl3_top_k,
        max_decode_tokens=max_decode_tokens,
        device=device,
    )
    cache.attach(layers)
    # This loader explicitly requires eager execution, so profiling can safely
    # update cache residency and there is no graph capture gate to reopen.
    cache.set_promotions(True)
    model._exl3_cache = cache
    print(f"EXL3_CACHE_INSTALLED layers={len(layers)} {cache.snapshot()}", flush=True)
