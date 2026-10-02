"""Summarize recorded measurements; never infer CUDA work from CPU profiler time."""
import gzip
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
serving = json.loads((ROOT/'results/serving.json').read_text())
rates = [r['decode_tok_s'] for r in serving['requests']]
trace_files = list((ROOT/'results/profile').glob('*.json.gz'))
assert len(trace_files) == 1, trace_files
with gzip.open(trace_files[0], 'rt') as handle:
    trace = json.load(handle)
categories = Counter(e.get('cat', 'uncategorized') for e in trace['traceEvents'])
result = {
    'unprofiled_decode_rates_tok_s': rates,
    'geomean_decode_tok_s': math.exp(statistics.mean(math.log(r) for r in rates)),
    'unprofiled_decode_ms_per_token': [1000/r for r in rates],
    'torch_trace_categories': dict(categories),
    'torch_cuda_kernel_events': categories['kernel'],
    'scope': 'current launcher, single active request, no MTP, 34/39 prompt tokens, 128 output tokens',
}
rows = [json.loads(line) for line in (ROOT/'results/cuda-events.jsonl').read_text().splitlines()]
assert len(rows) == 12, len(rows)
phases = ['plan', 'route_assert', 'copy_misses', 'pointers', 'expert_compute']
total = sum(r['model_step_ms'] for r in rows)
phase_sums = {p: sum(layer[p] for r in rows for layer in r['layers']) for p in phases}
assert sum(phase_sums.values()) <= total, (phase_sums, total)
phase_sums['rest_of_model_step'] = total-sum(phase_sums.values())
result['cuda_events'] = {
    'steps': len(rows), 'layers_per_step': 48,
    'mean_model_step_ms': total/len(rows),
    'median_model_step_ms': statistics.median(r['model_step_ms'] for r in rows),
    'min_model_step_ms': min(r['model_step_ms'] for r in rows),
    'max_model_step_ms': max(r['model_step_ms'] for r in rows),
    'phases': {p: {'mean_ms_per_step': value/len(rows), 'pct_of_model_step': 100*value/total}
               for p, value in phase_sums.items()},
    'misses_per_step': [sum(layer['copies'] for layer in r['layers']) for r in rows],
    'copy_bytes_per_step': [sum(layer['copy_bytes'] for layer in r['layers']) for r in rows],
    'method': 'external CUDA timing events retained in decode graphs; synchronization and counters read after each measured execute_model; includes marker overhead, excludes sampling and frontend',
}
copied_bytes = sum(result['cuda_events']['copy_bytes_per_step'])
copy_ms = phase_sums['copy_misses']
misses = sum(result['cuda_events']['misses_per_step'])
result['cuda_events']['effective_copy_GB_s'] = copied_bytes/(copy_ms*1e6)
result['cuda_events']['hit_fraction'] = 1-misses/(len(rows)*48*10)
result['cuda_events']['mean_copy_MiB_per_step'] = copied_bytes/(len(rows)*2**20)
by_layer = []
for layer in range(48):
    samples = [next(x for x in row['layers'] if x['layer'] == layer) for row in rows]
    by_layer.append({'layer': layer, 'bits': samples[0]['bits'],
                     'mean_copies': statistics.mean(x['copies'] for x in samples),
                     **{p: statistics.mean(x[p] for x in samples) for p in phases}})
result['cuda_events']['by_layer'] = by_layer
dominant = (phase_sums['copy_misses']+phase_sums['expert_compute'])/total
result['hypothesis_verdict'] = 'CONFIRMED' if dominant > 0.5 else 'REFUTED'
result['copy_plus_expert_fraction_of_step'] = dominant
result['upper_bound_speedups_for_instrumented_model_step'] = {
    'eliminate_copies': 1/(1-phase_sums['copy_misses']/total),
    'halve_copy_time': 1/(1-0.5*phase_sums['copy_misses']/total),
    'halve_expert_compute_time': 1/(1-0.5*phase_sums['expert_compute']/total),
}
event_output = json.loads((ROOT/'results/event-output.json').read_text())
result['instrumented_first_16_tokens_equal_unprofiled_code'] = (
    event_output['tokens'] == serving['requests'][0]['tokens'][:16])
result['limitations'] = [
    'Two short prompts, one unprofiled measurement each; no confidence interval.',
    'GPU-event profile samples 12 early decode steps of the code prompt after warmup and two prior requests.',
    'No CUPTI kernel trace: external event instrumentation measures phases, adds marker overhead, and synchronizes each step.',
    'CUDA event denominator is elapsed execute_model GPU timeline including dispatch gaps, not sum of hardware kernel durations; sampling/frontend excluded.',
    'Hit fraction uses 480 unique token-expert routes per step; measured only for batch one without MTP.',
    'No long-context, concurrent serving, prefill phase profile, or Strata throughput comparison.',
    'Optimization benefits are untested; Amdahl calculations are conditional model-step bounds, not serving forecasts.',
]
(ROOT/'results/summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'cuda_events'}, indent=2))
print(json.dumps({k: v for k, v in result['cuda_events'].items() if k != 'by_layer'}, indent=2))
