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


@pytest.fixture(scope="module", params=["global-lru", "precision-lru"])
def real_layers(request):
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
    # Two distinct K3 layers exercise offsets and cross-layer eviction inside
    # one precision pool, alongside a separate K2 pool.
    experts = [_load_experts(Path(model_path), layer, 16) for layer in (0, 12, 1)]
    sources = [PackedExpertLayer.from_experts(e) for e in experts]
    assert [s.bits for s in sources] == [3, 2, 3]
    refs = [_reference(e, s) for e, s in zip(experts, sources)]
    cache = Exl3ExpertCache(
        sources, slots_per_layer=10, top_k=10, max_decode_tokens=6,
        cache_policy=request.param,
    )
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
    for i, start, rows in ((0, 6, 1), (1, 6, 4), (2, 6, 4), (0, 0, 4), (1, 2, 1)):
        compare(i, _inputs(rows, start))
    cache.snapshot()
    # A warm hit performs no host-to-device expert copy.
    args = _inputs(1, 2)
    compare(1, args)
    assert int(cache.buffers[1].gather_count[0]) == 0

    placement = [t.row_key.clone() for t in cache.policy_tables.values()]
    for rows in (32, 320, 2200):
        for i in range(len(bound)):
            compare(i, _inputs(rows, 4))
    after = [t.row_key for t in cache.policy_tables.values()]
    assert all(torch.equal(a, b) for a, b in zip(after, placement)), "Prefill evicted entries"
    compare(0, _inputs(1, 7, dtype=torch.float16))

    # Capture with promotions disabled, replay after opening the gate. Both
    # routing and input change; a captured pointer to the old expert is a bug.
    for rows in (1, 2, 4, 6):
        cache.set_promotions(False)
        static = _inputs(rows)
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
            fresh = _inputs(rows, start)
            # Different token routes exercise the union of experts in a
            # verification batch, rather than repeating one token's route.
            fresh[1].add_(torch.arange(rows, device="cuda")[:, None]).remainder_(16)
            expected = [_apply(layer, fresh) for layer in refs]
            for dest, src in zip(static, fresh):
                dest.copy_(src)
            graph.replay()
            for got, want in zip(output, expected):
                _close(got, want)
    snapshot = cache.snapshot()
    # Verify every occupied slot against the immutable packed source, including
    # slots that changed from a K3 layer to K2 or in the opposite direction.
    for i, (table, index) in enumerate(cache.layer_tables):
        source = cache.sources[i]
        for expert, row in enumerate(table.layer_slice(table.hot_phys, index).tolist()):
            if row >= 0:
                assert torch.equal(cache.layer_banks[i][row, :source.data.shape[1]].cpu(),
                                   source.data[expert])
    if cache.cache_policy == "precision-lru":
        assert all(b.shape[1] == s.data.shape[1]
                   for b, s in zip(cache.layer_banks, cache.sources))
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


def test_cooperative_decode_matches_standard_experts(real_layers, monkeypatch):
    cache, refs, bound = real_layers
    assert hasattr(cache.ext, "exl3_moe_coop")
    for rows in (1, 2, 4, 6):
        for i in range(len(bound)):
            args = _inputs(rows, 5)
            args[1].add_(torch.arange(rows, device="cuda")[:, None]).remainder_(16)
            monkeypatch.setattr(exl3, "_COOP", False)
            expected = _apply(refs[i], args)
            monkeypatch.setattr(exl3, "_COOP", True)
            _close(_apply(bound[i], args), expected)


def test_early_cooperative_decode_preserves_outputs_and_graph_routes(real_layers, monkeypatch):
    """Skipping generic preparation must preserve sentinel and replay semantics."""
    _, refs, _ = real_layers
    monkeypatch.setattr(exl3, "_COOP", True)
    for dtype in (torch.float16, torch.bfloat16):
        for rows in (1, 2, 4, 6):
            for layer in refs:
                static = _inputs(rows, 3, dtype=dtype)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                monkeypatch.setattr(exl3, "_COOP_DECODE_FAST", True)
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        _apply(layer, static)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = _apply(layer, static)
                for sentinels in (False, True):
                    args = _inputs(rows, 6, dtype=dtype)
                    args[1].add_(torch.arange(rows, device="cuda")[:, None]).remainder_(16)
                    if sentinels:
                        args[1][0].fill_(-1)
                        args[1][-1, -1] = 16  # non-local expert sentinel
                    monkeypatch.setattr(exl3, "_COOP_DECODE_FAST", False)
                    expected = _apply(layer, args)
                    for dst, src in zip(static, args):
                        dst.copy_(src)
                    graph.replay()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    assert torch.isfinite(actual).all()
