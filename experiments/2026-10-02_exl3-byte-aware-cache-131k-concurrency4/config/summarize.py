"""Final analysis of saved results; no new inference or tuning."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
replay = json.loads((ROOT/'results/replay-summary.json').read_text())
steady = json.loads((ROOT/'results/baseline-steady.json').read_text())
initial = json.loads((ROOT/'results/baseline-initial.json').read_text())
outputs = json.loads((ROOT/'results/baseline-outputs.json').read_text())
steps = json.loads((ROOT/'results/baseline-steps.json').read_text())
routes = np.load(ROOT/'results/baseline-routes.npy')
indices = [i for i,s in enumerate(steps) if s['steady_four']]
selected = routes[:,indices]
misses = int(selected[:,:,41].sum())
unique = sum(len(set(int(x) for x in row[:40] if x != -1))
             for layer in selected for row in layer)
copied = int((selected[:,:,41]*np.array(initial['row_bytes'])[:,None]).sum())
summary = {
    'verdict':replay['verdict'], 'selected':replay['selected'],
    'deployment_changed':False, 'mtp_enabled':False,
    'baseline':{**steady, 'prompt_tokens_each':130560, 'output_tokens_each':512,
                'total_tokens_each':131072, 'completed_requests':len(outputs['requests']),
                'full_generation_wall_s':outputs['wall_s'],
                'bank_bytes':initial['bank_bytes'],'kv_bytes':8589934592,
                'trace_buffer_bytes':initial['trace_bytes'],
                'steady_copy_MiB_per_step':copied/len(indices)/2**20,
                'steady_copy_MiB_per_output_token':copied/len(indices)/4/2**20,
                'steady_unique_experts_per_step':unique/len(indices),
                'steady_distinct_misses_per_step':misses/len(indices),
                'steady_unique_expert_hit_fraction':1-misses/unique},
    'candidate_comparison':[
        {'label':r['label'], 'train_transfer_change_percent':-100*r['train_byte_reduction'],
         'heldout_transfer_change_percent':-100*r['heldout_byte_reduction'],
         'bank_bytes':r['bank_bytes'],'pool_rows':r['pool_rows']}
        for r in replay['configurations']],
    'offline_gate_percent_reduction':20,
    'next_action':'Retain precision-lru 160 and existing admission controls; no full-model candidate run.',
    'limitations':[
        'Four synthetic repeated source/documentation prompts, not the user\'s actual workload corpus.',
        'One completed full-model run; speed is a diagnostic baseline, not a confidence estimate.',
        '319 steady four-request decode steps at near-full context; no 32K-output endurance test.',
        'GPU routing instrumentation adds one small kernel per cached layer and a 7.875 MiB trace buffer.',
        'Candidate byte counts come from frozen-routing replay, not live candidate serving speed.',
        'Global LRU baseline replay matched every observed miss count across all 48 layers and 511 recorded steps.',
        'Initial startup failed before requests due callable RPC serialization; the single allowed correction used named RPC methods.',
        'Cache budget fixed; larger cache capacity and other optimization families were not tested.',
    ],
}
(ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
