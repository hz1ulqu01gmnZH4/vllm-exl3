"""Record decode routing on-device without per-layer CPU synchronization."""
import json
import os
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
import vllm_exl3.expert_cache as cache_module
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

ROOT = Path(__file__).resolve().parents[1]
LABEL = os.environ['CACHE_TRIAL_LABEL']
CAPACITY = 1024
CACHE = None
ENABLED = False
HOST_STEPS = []
PROGRESS = 0
START = time.monotonic()


@triton.jit
def record_routes(ids, copies, out, counts, enabled, layer: tl.constexpr,
                  LANES: tl.constexpr, CAP: tl.constexpr, WIDTH: tl.constexpr):
    if tl.load(enabled) != 0:
        step = tl.load(counts + layer)
        if step < CAP:
            lane = tl.arange(0, WIDTH)
            route = tl.load(ids + lane, lane < LANES, other=-1).to(tl.int32)
            base = out + (layer * CAP + step) * 42
            tl.store(base + lane, route, lane < 40)
            tl.store(base + 40, LANES // 10)
            tl.store(base + 41, tl.load(copies))
        tl.store(counts + layer, step + 1)


ORIGINAL_INIT = cache_module.Exl3ExpertCache.__init__


def cache_init(self, sources, **kwargs):
    global CACHE
    assert CACHE is None, 'Only one target cache expected; MTP must stay disabled'
    if LABEL == 'candidate':
        selected = json.loads((ROOT/'results/selected.json').read_text())
        kwargs['slots_per_layer'] = selected['slots_per_layer']
    ORIGINAL_INIT(self, sources, **kwargs)
    if LABEL == 'candidate':
        for table in self.policy_tables.values():
            cache_module.policy.set_control(table, **selected['controls'])
    CACHE = self
    self.trial_routes = torch.full((len(sources), CAPACITY, 42), -1,
                                  dtype=torch.int32, device=self.device)
    self.trial_counts = torch.zeros(len(sources), dtype=torch.int32, device=self.device)
    self.trial_enabled = torch.zeros(1, dtype=torch.int32, device=self.device)


ORIGINAL_APPLY = cache_module.Exl3ExpertCache.apply


def cache_apply(self, layer_index, x, ids, weights, *, limit=None):
    value = ORIGINAL_APPLY(self, layer_index, x, ids, weights, limit=limit)
    if x.shape[0] <= self.max_decode_tokens:
        assert x.shape[0] <= 4 and ids.shape[1] == 10
        record_routes[(1,)](ids, self.buffers[layer_index].gather_count,
                           self.trial_routes, self.trial_counts, self.trial_enabled,
                           layer_index, LANES=ids.numel(), CAP=CAPACITY, WIDTH=64)
    return value


ORIGINAL_EXECUTE = GPUModelRunner.execute_model


def execute(self, scheduler_output, *args, **kwargs):
    global PROGRESS
    dummy = kwargs.get('dummy_run', False)
    if ENABLED and not dummy:
        scheduled = scheduler_output.num_scheduled_tokens
        n = scheduler_output.total_num_scheduled_tokens
        assert not scheduler_output.scheduled_spec_decode_tokens, 'MTP not allowed'
        if scheduler_output.preempted_req_ids:
            raise RuntimeError(f'Preemption prevents target workload: {scheduler_output.preempted_req_ids}')
        computed = dict(zip(scheduler_output.scheduled_cached_reqs.req_ids,
                            scheduler_output.scheduled_cached_reqs.num_computed_tokens))
        if 0 < n <= 4:
            HOST_STEPS.append({
                'time': time.monotonic(), 'scheduled': dict(scheduled),
                'computed': computed,
                'steady_four': len(scheduled) == 4 and n == 4
                and all(computed.get(req, 0) >= 130560 for req in scheduled),
            })
        PROGRESS += 1
        if PROGRESS % 32 == 0:
            print('TRIAL_PROGRESS', LABEL, PROGRESS, n, len(scheduled),
                  round(time.monotonic()-START, 1), flush=True)
    return ORIGINAL_EXECUTE(self, scheduler_output, *args, **kwargs)


def begin(worker):
    global ENABLED
    assert CACHE is not None and not ENABLED
    initial = {
        'bits': [s.bits for s in CACHE.sources],
        'row_bytes': [s.data.shape[1]*4 for s in CACHE.sources],
        'top_k': CACHE.top_k, 'experts': CACHE.num_experts,
        'max_decode_tokens': CACHE.max_decode_tokens,
        'bank_bytes': sum(b.numel()*b.element_size() for b in CACHE.banks.values()),
        'trace_bytes': CACHE.trial_routes.numel()*4,
        'pools': {},
    }
    assert initial['max_decode_tokens'] == 4
    assert initial['bank_bytes'] <= 11970048000
    for key, table in CACHE.policy_tables.items():
        initial['pools'][str(key*4)] = {
            'layers': [i for i, s in enumerate(CACHE.sources) if s.data.shape[1] == key],
            'pool_rows': table.pool_rows, 'controls': cache_module.policy.read_control(table),
            **{name: getattr(table, name).cpu().tolist() for name in
               ['row_key', 'row_use', 'hot_phys', 'miss_count', 'clock', 'forwards']},
        }
    (ROOT/f'results/{LABEL}-initial.json').write_text(json.dumps(initial)+'\n')
    CACHE.trial_enabled.fill_(1)
    ENABLED = True
    return {'bank_bytes': initial['bank_bytes'], 'max_decode_tokens': initial['max_decode_tokens']}


def finish(worker):
    global ENABLED
    ENABLED = False
    CACHE.trial_enabled.zero_()
    torch.cuda.synchronize()
    counts = CACHE.trial_counts.cpu().tolist()
    assert len(set(counts)) == 1, counts
    assert 0 < counts[0] <= CAPACITY, counts
    assert counts[0] == len(HOST_STEPS), (counts, len(HOST_STEPS))
    routes = CACHE.trial_routes[:, :counts[0]].cpu().numpy()
    import numpy as np
    np.save(ROOT/f'results/{LABEL}-routes.npy', routes)
    steady = [s for s in HOST_STEPS if s['steady_four']]
    (ROOT/f'results/{LABEL}-steps.json').write_text(json.dumps(HOST_STEPS, indent=2)+'\n')
    assert len(steady) >= 128, f'Only {len(steady)} true four-way decode steps'
    elapsed = steady[-1]['time']-steady[0]['time']
    result = {'recorded_steps': counts[0], 'steady_four_steps': len(steady),
              'steady_four_elapsed_s': elapsed,
              'steady_four_aggregate_tok_s': 4*(len(steady)-1)/elapsed,
              'steady_four_per_request_tok_s': (len(steady)-1)/elapsed,
              'min_computed_in_steady': min(min(s['computed'].values()) for s in steady),
              'max_computed_in_steady': max(max(s['computed'].values()) for s in steady),
              'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    (ROOT/f'results/{LABEL}-steady.json').write_text(json.dumps(result, indent=2)+'\n')
    print('TRIAL_STEADY', json.dumps(result), flush=True)
    return result


def install():
    cache_module.Exl3ExpertCache.__init__ = cache_init
    cache_module.Exl3ExpertCache.apply = cache_apply
    GPUModelRunner.execute_model = execute
    Worker.cache_trial_begin = begin
    Worker.cache_trial_finish = finish
