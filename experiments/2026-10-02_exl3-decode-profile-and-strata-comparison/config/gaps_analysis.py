"""Exact analysis used once after the event summary; no additional GPU run."""
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = [json.loads(s) for s in (root/'results/cuda-events.jsonl').read_text().splitlines()]
out = {'gap_mean_ms': {}, 'precision_groups': {}}
first, between, last = [], [], []
for row in rows:
    layers = sorted(row['layers'], key=lambda x: x['start_ms'])
    first.append(layers[0]['start_ms'])
    between.append(sum(b['start_ms']-a['end_ms'] for a, b in zip(layers, layers[1:])))
    last.append(row['model_step_ms']-layers[-1]['end_ms'])
for key, values in [('before_first_cache', first), ('between_caches', between), ('after_last_cache', last)]:
    out['gap_mean_ms'][key] = statistics.mean(values)
for bits in [2, 3]:
    samples = [layer for row in rows for layer in row['layers'] if layer['bits'] == bits]
    out['precision_groups'][str(bits)] = {
        'layers_per_step': len(samples)//len(rows),
        'misses_per_step': sum(layer['copies'] for layer in samples)/len(rows),
        'copy_MiB_per_step': sum(layer['copy_bytes'] for layer in samples)/len(rows)/2**20,
        'copy_ms_per_step': sum(layer['copy_misses'] for layer in samples)/len(rows),
        'hit_fraction': 1-sum(layer['copies'] for layer in samples)/(len(samples)*10),
    }
(root/'results/gaps-and-precision.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
