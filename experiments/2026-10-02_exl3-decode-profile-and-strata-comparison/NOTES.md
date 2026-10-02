# Lab Notebook — EXL3 decode profile and Strata comparison

> Append-only running log. Newest entries at the bottom. Use `expt.sh log`.
> Record EVERY action, observation, parameter change, dead-end, and decision —
> including failures and surprises. Context is not a record; this file is.

---

### 2026-10-02T09:04:28Z — experiment created
- dir: `/home/ak/qwen38-vllm/vllm-exl3/experiments/2026-10-02_exl3-decode-profile-and-strata-comparison`

### 2026-10-02T09:05:02Z — run-001 (exit 0, 0s)
```
nvidia-smi; ps -eo pid,comm,args | rg "vllm|strata|nsys" | head -30; cat ../serve-exl3.sh ../env.sh; rg --files ../experiments ../experimental | rg "profil|bench|strata|REPORT|report|\.py$" | head -100; command -v nsys; command -v ncu; free -h
```
- stdout: `runs/run-001/stdout.log` (179 lines)
- stderr: `runs/run-001/stderr.log` (0 lines)

### 2026-10-02T09:05:25Z — run-002 (exit 0, 4s)
```
git clone --depth 1 https://github.com/Niko1221/Strata.git experiments/2026-10-02_exl3-decode-profile-and-strata-comparison/artifacts/Strata
```
- stdout: `runs/run-002/stdout.log` (0 lines)
- stderr: `runs/run-002/stderr.log` (1 lines)

### 2026-10-02T09:07:38Z
Current local launcher: precision-lru 160, cooperative enabled, no MTP, max_num_seqs=4, 8 GiB FP8 KV, FULL_DECODE_ONLY [1,2,4]. GPU idle at start. Full-model run-003 launched with those settings and one active request; separate unprofiled throughput and 12-step torch trace. Strata shallow checkout pinned to 1678de333d0e0711bc414ad992b640e1a37dd814. nsys/ncu not on PATH. No production code edits.

### 2026-10-02T09:09:08Z — run-004 (exit 0, 0s)
```
git rev-parse HEAD; git -C experiments/2026-10-02_exl3-decode-profile-and-strata-comparison/artifacts/Strata show -s --format=fuller HEAD; lscpu; sha256sum ../serve-exl3.sh src/vllm_exl3/expert_cache.py src/vllm_exl3/expert_cache_tables.py src/vllm_exl3/exl3.py ../experiments/2026-09-29_exl3-full-model-cached-serving/artifacts/runtime/exllamav3_ext.so; cat ../.venv/lib/python3.12/site-packages/__editable__.vllm_exl3-0.5.0.pth; cat ../.venv/lib/python3.12/site-packages/torch/version.py
```
- stdout: `runs/run-004/stdout.log` (74 lines)
- stderr: `runs/run-004/stderr.log` (0 lines)

### 2026-10-02T09:09:53Z
Saved source findings and immutable Strata revision in data/sources.json. Prior LFU96 reduced throughput from 62.42 to 49.38 tok/s; therefore do not recommend simply replacing LRU with LFU. Prior cooperative fast preparation failed its latency gate; HC prototype failed real-checkpoint parity. The current single-stream cache requires explicit staging lifetime and mapping ownership changes before overlap is safe. Strata CPU kernels use other weight formats; CPU hybrid is an architectural project, not a drop-in flag.

### 2026-10-02T09:10:13Z — run-003 (exit 0, 195s)
```
bash experiments/2026-10-02_exl3-decode-profile-and-strata-comparison/config/run_profile.sh
```
- stdout: `runs/run-003/stdout.log` (182 lines)
- stderr: `runs/run-003/stderr.log` (37 lines)

### 2026-10-02T09:11:49Z
Run-003 completed: unprofiled decode 66.67076875643643 and 67.41956021904008 tok/s, 128 output tokens each, prompt lengths 34/39. Worker installed all 48 precision-lru layers; bank 11970048000 bytes. CUPTI_ERROR_UNKNOWN (999), so trace has CPU activity only and cannot support GPU phase shares. One resolution attempt: run-005 will instrument GPU phases with external CUDA timing events retained in CUDA graphs, synchronize only during the bounded measurement window, collect 12 decode steps and miss counts. This is the second/final profiling execution; no driver changes or further retries. Phase times include marker overhead and are not a serving speed benchmark.

### 2026-10-02T09:15:47Z
Event probe implementation: process-local hooks only; external CUDA events before/after planner, route assertion, miss-copy kernel, pointer update and expert compute. Each measured one-token execute_model records total elapsed GPU timeline and reads cache gather counts after completion. All 48 intervals must lie inside the current model step, rejecting stale capture-only events. Match run-003 cache warmup history. Planned summary denominator is elapsed model-step timeline, including dispatch gaps and marker overhead; not pure GPU busy time.

### 2026-10-02T09:16:20Z — run-006 (exit 0, 1s)
```
python3 experiments/2026-10-02_exl3-decode-profile-and-strata-comparison/config/analyze.py
```
- stdout: `runs/run-006/stdout.log` (102 lines)
- stderr: `runs/run-006/stderr.log` (0 lines)

### 2026-10-02T09:16:25Z — run-005 (exit 0, 165s)
```
bash experiments/2026-10-02_exl3-decode-profile-and-strata-comparison/config/run_profile.sh profile_events.py
```
- stdout: `runs/run-005/stdout.log` (78 lines)
- stderr: `runs/run-005/stderr.log` (31 lines)

### 2026-10-02T09:19:40Z
Run-005 completed all 12 decode steps; CUDA event bounds validated across all 48 layers on each step. Run-006 summary: 67.0441 tok/s unprofiled geomean. Instrumented model step 19.7987 ms mean (15.5522-25.2322), copies 7.5542 ms/38.155%, expert compute 2.8540 ms/14.415%, planner/assert/pointers 5.829%, rest 41.600%. Hypothesis confirmed narrowly for this instrumented workload (52.570% copy+expert); not a general busy-time or serving claim. Hit fraction 0.742882, 183.705 MiB/token, effective copy 25.4995 GB/s. First 16 instrumented output tokens match prior code output. Additional analysis of existing timestamps only (saved config/gaps_analysis.py, no new GPU run): rest contains 6.5484 ms between caches, 1.6652 ms before first cache; K3 copy traffic 111.60 MiB, K2 72.11 MiB per step. Suggestions in results/recommendations.json. Stop: two profiling executions used and hypothesis answered.

### 2026-10-02T09:19:40Z — metric
- **decode_geomean** = 67.04411912333741 tok/s

### 2026-10-02T09:19:40Z — metric
- **copy_share** = 38.155105820795654 percent

### 2026-10-02T09:19:40Z — experiment marked CONFIRMED

### 2026-10-02T09:21:27Z
Final source review confirms MTP3 with four requests implies 160 routes (4*4*10), exceeding MAX_DECODE_LANES=64; any MTP recommendation is a separately qualified one-request profile or a planner redesign. All tracked production files remain unchanged. No further measurements or tuning runs.

### 2026-10-02T09:25:20Z
Completion review: reread original profiling/Strata/recommendation request; inspected complete HTML report and experiment code additions. HTML validation passed: local links exist, monochrome colors, immutable Strata links. git diff confirms no tracked production source/tests/docs changes. Full-model executions 003/005 and analysis 006 exited 0. No broad tests needed for profiling-only artifacts; report makes no implementation-speedup claim.

### Publication preparation
Unrelated process listings and workflow details were omitted from the published record; the Strata provenance log also omits upstream commit trailers. Raw originals remain locally under ignored artifacts/publication-originals. Benchmark measurements and decisions are unchanged.
