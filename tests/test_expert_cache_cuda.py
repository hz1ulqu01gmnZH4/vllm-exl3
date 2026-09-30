"""Real-checkpoint cache parity, including changed-routing graph replay.

Set EXL3_CACHE_TEST_MODEL to a native Qwen3.8-Flash-Next EXL3 K2/K3 pack.
The explicit test needs no vLLM process or full-model allocation.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from vllm_exl3 import exl3
from vllm_exl3.expert_cache import Exl3ExpertCache, PackedExpertLayer


def _load_experts(model, layer, count):
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {}
    result = [{"gate": {}, "up": {}, "down": {}} for _ in range(count)]
    for e in range(count):
        for proj in ("gate", "up", "down"):
            for attr in ("trellis", "suh", "svh", "mul1"):
                key = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}_proj.{attr}"
                wanted.setdefault(index[key], []).append((key, e, proj, attr))
    for shard, keys in wanted.items():
        with safe_open(model / shard, framework="pt", device="cpu") as f:
            for key, e, proj, attr in keys:
                result[e][proj][attr] = f.get_tensor(key).clone()
    return result


def _reference(experts, source):
    inners = []
    for expert in experts:
        inners.append({
            p: exl3.make_linear_exl3(**{k: v.cuda() for k, v in values.items()})
            for p, values in expert.items()
        })
    layer = SimpleNamespace(
        w13_suh=inners[0]["gate"].suh,
        _exl3_hidden_size=source.hidden,
        _exl3_intermediate_local=source.intermediate,
        _exl3_bits=source.bits, _exl3_codebook_flags=(False, True) * 3,
        _exl3_inners=inners,
    )
    exl3.build_exl3_fused_state(layer, inners)
    return layer


@pytest.fixture(scope="module")
def real_layers():
    model_path = os.environ.get("EXL3_CACHE_TEST_MODEL")
    if not model_path:
        pytest.skip("EXL3_CACHE_TEST_MODEL is required for real-checkpoint GPU parity")
    assert torch.cuda.is_available(), "GPU qualification requires a CUDA device"
    os.environ["VLLM_EXL3_MOE_KERNEL"] = "exllamav3"
    # The standalone kernel environment has no vLLM logging adapter. Only
    # logging is supplied here; loaders, EXL3 kernels and cache are real.
    if not exl3._VLLM_AVAILABLE:
        exl3.logger.info_once = exl3.logger.info
        exl3.logger.warning_once = exl3.logger.warning
    assert not exl3._COOP, "This qualification covers the standard EXL3 fused kernel"
    torch.manual_seed(20260929)
    experts = [_load_experts(Path(model_path), layer, 16) for layer in (0, 12)]
    sources = [PackedExpertLayer.from_experts(e) for e in experts]
    assert [s.bits for s in sources] == [3, 2]
    refs = [_reference(e, s) for e, s in zip(experts, sources)]
    cache = Exl3ExpertCache(sources, slots_per_layer=10, top_k=10, max_decode_tokens=4)
    bound = [SimpleNamespace(
        _exl3_hidden_size=s.hidden, _exl3_intermediate_local=s.intermediate,
        _exl3_bits=s.bits,
    ) for s in sources]
    cache.attach(bound)
    return cache, refs, bound


def _inputs(rows, start=0, *, dtype=torch.bfloat16):
    x = torch.randn(rows, 2560, device="cuda", dtype=dtype) * 0.2
    ids = ((torch.arange(10, device="cuda") + start) % 16).repeat(rows, 1).contiguous()
    weights = torch.rand(rows, 10, device="cuda")
    weights /= weights.sum(dim=1, keepdim=True)
    return x, ids, weights


def _apply(layer, args):
    x, ids, weights = args
    return exl3.apply_exl3_experts(x, ids, weights, layer, fused=True)


def _close(got, want):
    assert torch.isfinite(got).all()
    assert want.float().norm() > 0
    torch.testing.assert_close(got, want, rtol=0.02, atol=0.02)


def test_real_k2_k3_decode_eviction_prefill_and_graph_replay(real_layers):
    cache, refs, bound = real_layers
    worst = 0.0

    def compare(i, args):
        nonlocal worst
        expected = _apply(refs[i], args)
        actual = _apply(bound[i], args)
        _close(actual, expected)
        worst = max(worst, (actual.float() - expected.float()).abs().max().item())
        return actual

    # Frozen placement: cold experts execute from temporary staging slots.
    compare(0, _inputs(1, 6))
    assert int(cache.buffers[0].gather_count[0]) == 6
    cache.set_promotions(True)
    # Each token chooses distinct experts; rows can repeat a route across tokens.
    for i, start, rows in ((0, 6, 1), (1, 6, 4), (0, 0, 4), (1, 2, 1)):
        compare(i, _inputs(rows, start))
    cache.snapshot()
    # A warm hit performs no host-to-device expert copy.
    args = _inputs(1, 2)
    compare(1, args)
    assert int(cache.buffers[1].gather_count[0]) == 0

    placement = cache.tables.row_key.clone()
    for rows in (32, 320, 2200):
        for i in range(2):
            compare(i, _inputs(rows, 4))
    assert torch.equal(cache.tables.row_key, placement), "Prefill must not evict decode entries"
    compare(0, _inputs(1, 7, dtype=torch.float16))

    # Capture with promotions disabled, replay after opening the gate. Both
    # routing and input change; a captured pointer to the old expert is a bug.
    cache.set_promotions(False)
    static = _inputs(1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            for layer in bound:
                _apply(layer, static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = [_apply(layer, static) for layer in bound]
    cache.set_promotions(True)
    for start in (6, 0, 3, 6):
        fresh = _inputs(1, start)
        expected = [_apply(layer, fresh) for layer in refs]
        for dest, src in zip(static, fresh):
            dest.copy_(src)
        graph.replay()
        for got, want in zip(output, expected):
            _close(got, want)
    snapshot = cache.snapshot()
    # Verify every occupied slot against the immutable packed source, including
    # slots that changed from a K3 layer to K2 or in the opposite direction.
    for row, key in enumerate(cache.tables.row_key.tolist()):
        if key < 0:
            continue
        i, expert = divmod(key, 16)
        source = cache.sources[i]
        assert torch.equal(cache.bank[row, :source.data.shape[1]].cpu(), source.data[expert])
    print(json.dumps({"result": "PASS", "max_abs_error": worst, **snapshot}, sort_keys=True))


def test_cache_rejects_incompatible_dispatch_and_geometry(real_layers, monkeypatch):
    cache, _, bound = real_layers
    args = _inputs(1)
    with pytest.raises(ValueError, match="requires fused"):
        exl3.apply_exl3_experts(args[0], args[1], args[2], bound[0], fused=False)
    monkeypatch.setenv("VLLM_EXL3_MOE_KERNEL", "native")
    with pytest.raises(RuntimeError, match="VLLM_EXL3_MOE_KERNEL"):
        _apply(bound[0], args)
    monkeypatch.setenv("VLLM_EXL3_MOE_KERNEL", "exllamav3")
    with pytest.raises(ValueError, match="shape, dtype or device"):
        cache.apply(0, args[0][:, :128], args[1], args[2])
    with pytest.raises(ValueError, match="already has"):
        cache.attach(bound)
