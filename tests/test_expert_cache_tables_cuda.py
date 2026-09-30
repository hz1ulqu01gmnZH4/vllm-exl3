"""GPU/reference parity for fused EXL3 validation and variable graph widths."""
import os
import subprocess
import sys
from dataclasses import fields

import pytest
import torch

from vllm_exl3 import expert_cache_tables as policy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _snapshot(obj):
    return {f.name: getattr(obj, f.name).cpu().clone() for f in fields(obj)
            if isinstance(getattr(obj, f.name), torch.Tensor)}


def _compare(gpu, cpu):
    for name, value in _snapshot(gpu).items():
        assert torch.equal(value, getattr(cpu, name)), name


@pytest.mark.parametrize('controls', [
    {'gate': 0}, {'gate': 1},
    {'gate': 1, 'promote_limit': 2, 'promote_interval': 2},
    {'gate': 1, 'promote_min_misses': 2, 'protect_recent': 1},
])
def test_plan_matches_reference_across_graph_widths(controls):
    gpu = policy.allocate_global_tables('cuda', 64, [10, 10, 10], 60)
    cpu = policy.allocate_global_tables('cpu', 64, [10, 10, 10], 60)
    gb = policy.allocate_step_buffers('cuda', 64, 64)
    cb = policy.allocate_step_buffers('cpu', 64, 64)
    policy.set_control(gpu, **controls)
    policy.set_control(cpu, **controls)
    graphs = {}
    for tokens in (1, 2, 4, 6):
        ids = torch.arange(10, device='cuda').repeat(tokens, 1)
        # Noncontiguous weights retain the original cache input contract.
        weights = torch.ones(10, tokens, device='cuda').t()
        policy.step(gpu, 0, ids, gb, weights)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            policy.step(gpu, 0, ids, gb, weights)
        graphs[tokens] = graph, ids, weights
    # Reset after compile/capture so only changed-input replays mutate both copies.
    for name, value in _snapshot(cpu).items():
        getattr(gpu, name).copy_(value)
    for name, value in _snapshot(cb).items():
        getattr(gb, name).copy_(value)
    for step, tokens in enumerate((6, 1, 4, 2, 1, 6, 2, 4, 1)):
        graph, ids, weights = graphs[tokens]
        fresh = ((torch.arange(10) + 7 * step)[None, :]
                 + torch.arange(tokens)[:, None] * 3) % 64
        if step % 3 == 0:
            fresh[:, -2:] = -1  # Duplicate padding is legal.
        ids.copy_(fresh)
        graph.replay()
        policy.step_reference(cpu, 0, fresh, cb)
        _compare(gpu, cpu)
        _compare(gb, cb)
        # Other layers force cross-layer evictions and exercise index specialization.
        for layer in (1, 2):
            policy.step(gpu, layer, ids, gb, weights)
            policy.step_reference(cpu, layer, fresh, cb)
            _compare(gpu, cpu)
            _compare(gb, cb)
    policy.check_global_tables(gpu)


@pytest.mark.parametrize('kind,code', [
    ('high_id', 1), ('negative_id', 1), ('negative_weight', 2),
    ('nan', 2), ('positive_inf', 2), ('negative_inf', 2), ('duplicate', 4),
])
def test_invalid_routes_do_not_mutate_cache(kind, code):
    table = policy.allocate_global_tables('cuda', 64, [10, 10], 60)
    buffer = policy.allocate_step_buffers('cuda', 64, 64)
    policy.set_control(table, gate=1)
    ids = torch.arange(10, device='cuda').repeat(2, 1)
    weights = torch.ones_like(ids, dtype=torch.float32)
    if kind == 'high_id':
        ids[0, 0] = 64
    elif kind == 'negative_id':
        ids[0, 0] = -2
    elif kind == 'duplicate':
        ids[0, 1] = ids[0, 0]
    else:
        values = {'negative_weight': -1., 'nan': float('nan'),
                  'positive_inf': float('inf'), 'negative_inf': -float('inf')}
        weights[0, 0] = values[kind]
    before = _snapshot(table)
    before_buffers = _snapshot(buffer)
    policy.step(table, 0, ids, buffer, weights)
    torch.cuda.synchronize()
    assert table.error.item() == code
    for name, value in before.items():
        if name != 'error':
            assert torch.equal(getattr(table, name).cpu(), value), name
    assert all(torch.equal(getattr(buffer, k).cpu(), v) for k, v in before_buffers.items())
    # An error stays sticky and prevents later calls from changing placement.
    ids.copy_(torch.arange(10, device='cuda').repeat(2, 1))
    weights.fill_(1)
    policy.step(table, 0, ids, buffer, weights)
    assert table.error.item() == code
    for name, value in before.items():
        if name != 'error':
            assert torch.equal(getattr(table, name).cpu(), value), name


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_padding_ignores_weights_and_cross_token_repeats(dtype):
    table = policy.allocate_global_tables('cuda', 64, [10], 60)
    buffer = policy.allocate_step_buffers('cuda', 64, 64)
    ids = torch.arange(10, device='cuda').repeat(2, 1)
    ids[:, -2:] = -1
    weights = torch.ones(ids.shape, device='cuda', dtype=dtype)
    weights[:, -2:] = float('nan')
    if dtype == torch.float64:
        weights[0, 0] = 1e200  # Finite float64 must not be narrowed for validation.
    policy.step(table, 0, ids, buffer, weights)
    assert table.error.item() == 0
    assert buffer.routes[8:10].tolist() == [-1, -1]
    assert (buffer.routes[20:] == -1).all()


def test_unweighted_api_keeps_invalid_id_semantics():
    gpu = policy.allocate_global_tables('cuda', 16, [4], 16)
    cpu = policy.allocate_global_tables('cpu', 16, [4], 16)
    gb = policy.allocate_step_buffers('cuda', 16, 16)
    cb = policy.allocate_step_buffers('cpu', 16, 16)
    policy.set_control(gpu, gate=1)
    policy.set_control(cpu, gate=1)
    ids = torch.tensor([1, 9, 9, -1, -2, 16])
    policy.step(gpu, 0, ids.cuda(), gb)
    policy.step_reference(cpu, 0, ids, cb)
    _compare(gpu, cpu)
    _compare(gb, cb)


def test_device_error_reaches_async_assertion():
    script = '''
import torch
from vllm_exl3 import expert_cache_tables as p
t = p.allocate_global_tables('cuda', 16, [4], 4)
b = p.allocate_step_buffers('cuda', 16, 4)
x = torch.tensor([[0, 0]], device='cuda')
p.step(t, 0, x, b, torch.ones_like(x, dtype=torch.float32))
torch._assert_async((t.error == 0).all(), 'Invalid EXL3 routes or cache planner failure')
torch.cuda.synchronize()
'''
    completed = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(),
                               capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode != 0
    assert 'device-side assert' in completed.stderr, completed.stderr


@pytest.mark.parametrize('capacity,routes', [(24, 20), (50, 40)])
def test_cuda_buffer_capacity_requires_power_of_two(capacity, routes):
    table = policy.allocate_global_tables('cuda', 64, [10], 60)
    buffer = policy.allocate_step_buffers('cuda', 64, capacity)
    ids = torch.arange(routes, device='cuda').reshape(1, -1)
    with pytest.raises(ValueError, match="power of two"):
        policy.step(table, 0, ids, buffer, torch.ones_like(ids, dtype=torch.float32))
