# Lab Notebook — EXL3 byte-aware cache 131K concurrency4

> Append-only running log. Newest entries at the bottom. Use `expt.sh log`.
> Record EVERY action, observation, parameter change, dead-end, and decision —
> including failures and surprises. Context is not a record; this file is.

---

### 2026-10-02T10:15:01Z — experiment created
- dir: `/home/ak/qwen38-vllm/vllm-exl3/experiments/2026-10-02_exl3-byte-aware-cache-131k-concurrency4`

### 2026-10-02T10:15:50Z — run-001 (exit 0, 1s)
```
nvidia-smi; free -h; ps -eo pid,comm,args | rg "vllm|profile_events|profile_model" | head -20; cat ../.venv/lib/python3.12/site-packages/vllm/v1/core/sched/output.py | sed -n "226,260p"
```
- stdout: `runs/run-001/stdout.log` (61 lines)
- stderr: `runs/run-001/stderr.log` (0 lines)

### 2026-10-02T10:22:36Z — run-002 (exit 1, 189s)
```
bash experiments/2026-10-02_exl3-byte-aware-cache-131k-concurrency4/config/run.sh baseline
```
- stdout: `runs/run-002/stdout.log` (64 lines)
- stderr: `runs/run-002/stderr.log` (58 lines)
- ⚠️ NON-ZERO EXIT — investigate before continuing.

### 2026-10-02T10:23:17Z
Run-002 started baseline. Process-local routing hook records expert IDs and observed distinct copies on-device in captured decode graphs; no production edits and no per-layer CPU synchronization. All four prompts reserve exactly 131072 total tokens each (130560 input + 512 output), mixed code/prose source documented by hashes. Steady timing requires four scheduled requests with one decode row each and all computed contexts >=130560. Candidate selection uses first half only, holdout gate >=20% saved bytes; exact baseline replay agreement is mandatory. MTP excluded and KV/cache budgets fixed.

### 2026-10-02T10:24:19Z
Run-002 exit 1 before any requests: vLLM rejects callable RPC serialization. Used the single preregistered infrastructure correction: bind trace begin/finish as named Worker methods and send string RPC names. Serialization policy, prompts, memory budgets, and gates unchanged. Failed startup retained as evidence; correction allowance exhausted. No benchmark result from run-002.

### 2026-10-02T10:27:12Z
Run-003 named RPC correction succeeded. TRIAL_BEGIN confirms bank_bytes=11970048000 and max_decode_tokens=4. Prompts are being processed with configured KV capacity 582852 tokens and max_model_len131072/max_num_seqs4; actual concurrent decode remains to be observed. The initial table snapshot was saved for exact replay. Full trace/replay scripts are complete; waiting on target workload, not running additional probes.

### 2026-10-02T10:35:19Z — run-003 (exit 0, 660s)
```
bash experiments/2026-10-02_exl3-byte-aware-cache-131k-concurrency4/config/run.sh baseline
```
- stdout: `runs/run-003/stdout.log` (129 lines)
- stderr: `runs/run-003/stderr.log` (31 lines)

### 2026-10-02T10:35:35Z
Run-003 exited 0. All four requests completed 130560 input + 512 output tokens. Observed 319 true four-way decode steps over 11.6968 seconds: 108.7476 aggregate tok/s, 27.1869 tok/s per request, computed contexts 130560..131070. Total generation wall including four prefills: 504.0216 s. Recorded 511 small decode steps, 48 layers each. Peak torch allocated 28844213248 bytes / reserved 29735518208 bytes. Beginning preregistered CPU replay; no extra GPU run unless heldout byte gate passes.

### 2026-10-02T10:35:39Z — run-004 (exit 0, 4s)
```
../.venv/bin/python experiments/2026-10-02_exl3-byte-aware-cache-131k-concurrency4/config/replay.py
```
- stdout: `runs/run-004/stdout.log` (8 lines)
- stderr: `runs/run-004/stderr.log` (0 lines)

### 2026-10-02T10:37:28Z — run-005 (exit 0, 0s)
```
../.venv/bin/python experiments/2026-10-02_exl3-byte-aware-cache-131k-concurrency4/config/summarize.py
```
- stdout: `runs/run-005/stdout.log` (114 lines)
- stderr: `runs/run-005/stderr.log` (0 lines)

### 2026-10-02T10:37:56Z
Run-004 exited 0. Exact CPU replay matched all observed GPU miss counts over 48 layers x 511 decode steps. Selection/holdout: first159 vs last160 steady-four steps. Seven configurations: baseline; K3 resident-budget shifts +/-10% and +/-20%; admit after2 misses; protect1 forward. All candidates worsened selection-half bytes, so baseline wins. Heldout shifts increased traffic 2.91% to24.40%; admit-after2 saved2.191% heldout but worsened selection11.074%; protect1 saved only0.000893% heldout. None meets20% gate. Do not run candidate or change deployment. Summary run-005: baseline399 MiB/step,99.65 MiB/outputtoken,84.42% unique-expert hits. Fixed-budget cache hypothesis refuted for tested workload/configurations; larger capacity and other optimizations untested. Stop criterion met.

### 2026-10-02T10:37:56Z — metric
- **baseline_four_request_aggregate** = 108.7475767814572 tok/s

### 2026-10-02T10:37:57Z — metric
- **baseline_four_request_per_request** = 27.1868941953643 tok/s

### 2026-10-02T10:37:57Z — metric
- **best_observed_heldout_byte_reduction** = 2.191472713695364 percent

### 2026-10-02T10:37:57Z — experiment marked REFUTED

### 2026-10-02T10:39:37Z
Completion scope review: only experiment artifacts added; tracked production code, tests, docs and deployment launcher unchanged. Re-read user requirement: test cache idea 1, no MTP, retain 131K at four concurrency. Actual target completed; seven frozen-route candidates failed preregistered gate, so no live candidate or further parameter search. Full-model startup correction, successful run, replay and derived summary are retained.

### Publication preparation
Unrelated process listings and workflow details were omitted from the published record; the Strata provenance log also omits upstream commit trailers. Raw originals remain locally under ignored artifacts/publication-originals. Benchmark measurements and decisions are unchanged.
