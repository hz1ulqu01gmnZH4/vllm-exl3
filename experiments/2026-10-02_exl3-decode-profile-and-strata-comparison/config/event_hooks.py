"""Process-local CUDA-event instrumentation. No production source modifications."""
import json
from pathlib import Path

import torch
import vllm_exl3.expert_cache as cache_module
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT/'artifacts/events-enabled'
PHASES = ('plan', 'route_assert', 'copy_misses', 'pointers', 'expert_compute')
ACTIVE = None
LAYERS = {}
STEPS = 0


def mark(index):
    if ACTIVE is not None:
        ACTIVE[index].record()


class TimedKernel:
    def __init__(self, kernel, before, after):
        self.kernel, self.before, self.after = kernel, before, after

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def timed(*args, **kwargs):
            if self.before is not None:
                mark(self.before)
            value = launch(*args, **kwargs)
            mark(self.after)
            return value

        return timed


ORIGINAL_STEP = cache_module.policy.step


def timed_plan(*args, **kwargs):
    value = ORIGINAL_STEP(*args, **kwargs)
    mark(1)
    return value


ORIGINAL_APPLY = cache_module.Exl3ExpertCache.apply


def timed_apply(self, layer_index, x, ids, weights, *, limit=None):
    global ACTIVE
    assert ACTIVE is None, 'Unexpected recursive cache application'
    if x.shape[0] > self.max_decode_tokens:
        return ORIGINAL_APPLY(self, layer_index, x, ids, weights, limit=limit)
    key = (id(self), layer_index, x.shape[0])
    if key not in LAYERS:
        events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(6)]
        LAYERS[key] = (self, layer_index, events)
    ACTIVE = LAYERS[key][2]
    mark(0)
    try:
        value = ORIGINAL_APPLY(self, layer_index, x, ids, weights, limit=limit)
        mark(5)
        return value
    finally:
        ACTIVE = None


ORIGINAL_EXECUTE = GPUModelRunner.execute_model


def timed_execute(self, scheduler_output, *args, **kwargs):
    global STEPS
    measure = (STEPS < 12 and not kwargs.get('dummy_run', False)
               and scheduler_output.total_num_scheduled_tokens == 1 and GATE.exists())
    if not measure:
        return ORIGINAL_EXECUTE(self, scheduler_output, *args, **kwargs)
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    value = ORIGINAL_EXECUTE(self, scheduler_output, *args, **kwargs)
    end.record()
    end.synchronize()
    total_ms = begin.elapsed_time(end)
    rows = []
    for key, (cache, layer, events) in sorted(LAYERS.items()):
        if key[2] != 1:
            continue
        start_ms = begin.elapsed_time(events[0])
        end_ms = begin.elapsed_time(events[-1])
        assert 0 <= start_ms < end_ms <= total_ms, (layer, start_ms, end_ms, total_ms)
        times = {name: events[i].elapsed_time(events[i+1]) for i, name in enumerate(PHASES)}
        assert all(t >= 0 for t in times.values()), times
        copies = int(cache.buffers[layer].gather_count[0].item())
        assert 0 <= copies <= cache.top_k, copies
        rows.append({'layer': layer, 'bits': cache.sources[layer].bits,
                     'copies': copies, 'copy_bytes': copies*cache.sources[layer].data.shape[1]*4,
                     'start_ms': start_ms, 'end_ms': end_ms, **times})
    assert len(rows) == 48, len(rows)
    record = {'step': STEPS, 'model_step_ms': total_ms, 'layers': rows}
    with (ROOT/'results/cuda-events.jsonl').open('a') as output:
        output.write(json.dumps(record)+'\n')
    print('CUDA_EVENT_STEP', STEPS, total_ms, sum(r['copy_misses'] for r in rows),
          sum(r['expert_compute'] for r in rows), flush=True)
    STEPS += 1
    return value


def install():
    cache_module.policy.step = timed_plan
    cache_module._copy_misses = TimedKernel(cache_module._copy_misses, 2, 3)
    cache_module._update_pointers = TimedKernel(cache_module._update_pointers, None, 4)
    cache_module.Exl3ExpertCache.apply = timed_apply
    GPUModelRunner.execute_model = timed_execute
