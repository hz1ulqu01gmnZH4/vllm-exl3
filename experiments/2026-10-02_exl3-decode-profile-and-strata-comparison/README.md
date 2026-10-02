# Experiment: EXL3 decode profile and Strata comparison

| Field        | Value |
|--------------|-------|
| ID           | 2026-10-02_exl3-decode-profile-and-strata-comparison |
| Created      | 2026-10-02T09:04:28Z |
| Status       | CONFIRMED |
| Author       | unknown |

## Hypothesis
Expert-cache staging and packed expert compute dominate the current Qwen3.8 EXL3 decode critical path; a bounded profile can identify the highest-value next optimization and distinguish it from already rejected ideas.

## Independent variable(s)
Observation only: a representative short decode workload and, only if needed, one targeted discriminating probe. No production implementation changes.

## Dependent variable(s) / metrics
CPU/CUDA operator durations (ms), cache planning/copy/compute shares, output tokens/s if full serving is available, cache bytes, actual runtime identity, and Strata source revision.

## Controls / baseline
Current checkout and available local runtime/checkpoint. Existing 2026-09-30 and 2026-10-01 results are historical evidence, not a newly measured baseline. Seed 42 for newly generated synthetic data. Single GPU workload at a time.

## Success criteria
Confirm the hypothesis if cache staging plus packed expert compute account for over 50% of measured decode GPU duration; refute if at most 50%; inconclusive if representative profiling cannot execute or attribution is insufficient. Deliver a ranked, source-linked optimization report regardless of verdict, clearly separating measurements, prior evidence, and hypotheses.

## Kill criteria and budget
At most two new profiling/probe executions and 30 minutes of profiling/research work. Stop earlier when a dominant bottleneck is identified and recommendations can be ranked. If runtime/GPU access fails, make one concrete resolution attempt, then report the limitation without repeated retries. No broad tuning sweeps, model downloads, production patches, or full Strata build.
