"""Bounded byte-budget replay of real routes; exact observed-miss agreement is required."""
import heapq
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INITIAL = json.loads((ROOT/'results/baseline-initial.json').read_text())
ROUTES = np.load(ROOT/'results/baseline-routes.npy')
STEPS = json.loads((ROOT/'results/baseline-steps.json').read_text())
BITS = INITIAL['bits']
ROW_BYTES = INITIAL['row_bytes']
BUDGET = INITIAL['bank_bytes']
STEADY = [i for i, step in enumerate(STEPS) if step['steady_four']]
assert len(STEADY) >= 128
TRAIN = STEADY[:len(STEADY)//2]
HELDOUT = STEADY[len(STEADY)//2:]
assert ROUTES.shape == (48, len(STEPS), 42), ROUTES.shape
DEFAULT = {'promote_limit':0, 'promote_interval':1, 'promote_min_misses':1, 'protect_recent':0}


class Pool:
    def __init__(self, layers, slots, initial, controls, use_snapshot):
        self.layers = layers
        self.local = {layer:i for i, layer in enumerate(layers)}
        self.nrows = sum(slots[layer] for layer in layers)
        self.controls = controls
        if use_snapshot:
            assert self.nrows == initial['pool_rows']
            self.keys = initial['row_key'][:self.nrows].copy()
            self.uses = initial['row_use'][:self.nrows].copy()
            self.misses = initial['miss_count'].copy()
            self.clock = initial['clock'][0]
            self.forwards = initial['forwards'][0]
        else:
            assert initial['clock'] == [0], 'Capacity replay requires initial cold state'
            self.keys = [i*512+e for i, layer in enumerate(layers) for e in range(slots[layer])]
            self.uses = [0]*self.nrows
            self.misses = [0]*(512*len(layers))
            self.clock = self.forwards = 0
        assert len(self.keys) == self.nrows and len(set(self.keys)) == self.nrows
        self.hot = {key:row for row,key in enumerate(self.keys)}
        self.heap = [(use,row) for row,use in enumerate(self.uses)]
        heapq.heapify(self.heap)

    def step(self, layer, raw):
        self.clock += 1
        local = self.local[layer]
        if local == 0:
            self.forwards += 1
        selected = list(dict.fromkeys(int(x) for x in raw if x != -1))
        assert all(0 <= e < 512 for e in selected)
        keys = [local*512+e for e in selected]
        for key in keys:
            if key in self.hot:
                row = self.hot[key]
                self.uses[row] = self.clock
                heapq.heappush(self.heap, (self.clock,row))
        copied = promoted = 0
        c = self.controls
        promote = (self.forwards-1) % c['promote_interval'] == 0
        for key in keys:
            if key in self.hot:
                continue
            copied += 1
            self.misses[key] += 1
            if not (promote and (c['promote_limit'] == 0 or promoted < c['promote_limit'])
                    and self.misses[key] >= c['promote_min_misses']):
                continue
            while self.heap and self.heap[0][0] != self.uses[self.heap[0][1]]:
                heapq.heappop(self.heap)
            assert self.heap
            use,row = self.heap[0]
            cutoff = self.clock-c['protect_recent']*len(self.layers)
            if use >= self.clock or (c['protect_recent'] and use >= cutoff):
                continue
            heapq.heappop(self.heap)
            del self.hot[self.keys[row]]
            self.hot[key] = row
            self.keys[row] = key
            self.uses[row] = self.clock
            self.misses[key] = 0
            heapq.heappush(self.heap, (self.clock,row))
            promoted += 1
        return copied


def replay(label, slots, controls):
    bank_bytes = sum(n*b for n,b in zip(slots,ROW_BYTES))+40*sum(set(ROW_BYTES))
    assert bank_bytes <= BUDGET and all(10 <= n < 512 for n in slots)
    pools = {}
    for b in sorted(set(ROW_BYTES)):
        info = INITIAL['pools'][str(b)]
        pools[b] = Pool(info['layers'], slots, info, controls, slots == [160]*48)
    counts = np.empty((48,len(STEPS)), dtype=np.int32)
    for step in range(len(STEPS)):
        for layer in range(48):
            counts[layer,step] = pools[ROW_BYTES[layer]].step(layer,ROUTES[layer,step,:40])
    if label == 'baseline':
        mismatch = np.argwhere(counts != ROUTES[:,:,41])
        assert len(mismatch) == 0, f'Observed baseline mismatch at {mismatch[:8].tolist()}'
    bytes_by_step = (counts*np.array(ROW_BYTES)[:,None]).sum(axis=0)
    result = {'label':label, 'slots_per_layer':slots, 'controls':controls,
              'bank_bytes':bank_bytes, 'train_bytes':int(bytes_by_step[TRAIN].sum()),
              'heldout_bytes':int(bytes_by_step[HELDOUT].sum()),
              'all_steady_bytes':int(bytes_by_step[STEADY].sum()),
              'pool_rows': {str(b):sum(slots[i] for i in range(48) if ROW_BYTES[i] == b)
                            for b in sorted(set(ROW_BYTES))}}
    print('REPLAY', label, result['train_bytes'], result['heldout_bytes'], bank_bytes, flush=True)
    return result


def split_slots(shift):
    b2 = ROW_BYTES[BITS.index(2)]
    b3 = ROW_BYTES[BITS.index(3)]
    resident = BUDGET-40*(b2+b3)
    rows3 = int((160*BITS.count(3)*b3+shift*resident)//b3)
    rows2 = (resident-rows3*b3)//b2
    slots = [0]*48
    for bits, rows in [(2,rows2),(3,rows3)]:
        layers = [i for i,b in enumerate(BITS) if b == bits]
        for order,layer in enumerate(layers):
            slots[layer] = rows//len(layers)+(order < rows % len(layers))
    return slots


def main():
    results = [replay('baseline',[160]*48,dict(DEFAULT))]
    for shift in [-0.2,-0.1,0.1,0.2]:
        results.append(replay(f'k3_shift_{shift:+.1f}',split_slots(shift),dict(DEFAULT)))
    for name,control in [('admit_after_two',{'promote_min_misses':2}),
                         ('protect_one_forward',{'protect_recent':1})]:
        results.append(replay(name,[160]*48,{**DEFAULT,**control}))
    best_split = min(results[1:5],key=lambda r:r['train_bytes'])
    best_control = min(results[5:7],key=lambda r:r['train_bytes'])
    if max(best_split['train_bytes'],best_control['train_bytes']) < results[0]['train_bytes']:
        results.append(replay('best_split_plus_admission',best_split['slots_per_layer'],best_control['controls']))
    selected = min(results,key=lambda r:r['train_bytes'])
    for result in results:
        result['train_byte_reduction'] = 1-result['train_bytes']/results[0]['train_bytes']
        result['heldout_byte_reduction'] = 1-result['heldout_bytes']/results[0]['heldout_bytes']
    passed = selected['label'] != 'baseline' and selected['heldout_byte_reduction'] >= 0.2
    summary = {'baseline_observed_miss_agreement':True, 'configurations':results,
               'selection_rule':'minimum first-half transfer bytes; never choose by heldout result',
               'trace_steps':len(STEPS),'steady_four_steps':len(STEADY),
               'train_steps':len(TRAIN), 'heldout_steps':len(HELDOUT),
               'selected':selected['label'],'offline_gate_pass':passed,
               'verdict':'PROCEED_TO_MATCHED_MODEL' if passed else 'REFUTED_AT_OFFLINE_GATE',
               'limitations':['One synthetic mixed-code/prose near-131K four-request trace.',
                              'Routing is frozen in replay; generated routing can change after cache layout changes.',
                              'Offline byte savings do not establish latency or serving speedup.']}
    (ROOT/'results/replay-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if passed:
        (ROOT/'results/selected.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('REPLAY_DECISION',summary['verdict'],selected['label'],selected['heldout_byte_reduction'],flush=True)


if __name__ == '__main__':
    main()
