# Experiment: EXL3 byte-aware cache 131K concurrency4

| Field        | Value |
|--------------|-------|
| ID           | 2026-10-02_exl3-byte-aware-cache-131k-concurrency4 |
| Created      | 2026-10-02T10:15:01Z |
| Status       | REFUTED |
| Author       | unknown |

## Hypothesis
At fixed expert-bank and KV memory, byte-aware precision-pool allocation or cautious admission reduces transferred expert bytes enough to improve actual four-request near-131K decode. MTP is excluded by the user.

## Independent variable(s)
Precision-pool capacity split (K3 resident-byte budget +/-10% and +/-20% of total resident bytes, K2 gets the remainder), admission after two misses, or one-forward recency protection. Keep existing global LRU within each precision pool. At most eight offline configurations including baseline and, only if useful, the best split plus best admission control.

## Dependent variable(s) / metrics
Actual routed expert IDs, distinct misses and copied bytes per decode step; replayed transfer-byte reduction; four-way steady decode steps/s and aggregate output tokens/s; per-request completion and output token identity. Actual four-request decode must be observed, not inferred from max_num_seqs.

## Controls / baseline
Current c24d861 runtime/checkpoint, cooperative path, no MTP, precision-lru 160 baseline, 11,970,048,000 expert-bank bytes including staging, 8 GiB FP8 KV, max_model_len=131072, max_num_seqs=4, max_num_batched_tokens=2048, decode graphs [1,2,4]. Four deterministic mixed code/prose prompts of 130560 tokens plus 512 outputs each, seed 42, temperature 0, prefix caching disabled. Synthetic public repository content is a proxy because no user workload corpus was supplied. Total length per request = 131072.

## Success criteria
Offline gate: >=20% fewer transferred bytes in the latter half of a held-out real four-request routing trace at no higher bank memory; exact replay of observed baseline miss counts is the single replay correctness check. Select parameters on the first half only. Only if this gate passes, execute one candidate full-model comparison. Full-model gate: >=10% faster steady four-way decode, no more than 3% per-request decode regression in the common steady window, fixed KV/context/concurrency and accepted checkpoint parity. Record all token differences; do not claim quality parity from short answer checks.

## Kill criteria and budget
45-minute overall investigation budget; at most two full-model starts (baseline trace and one candidate), at most eight CPU replay candidates and one targeted GPU parity check if a candidate qualifies. Each full-model run capped at 18 minutes. Stop on insufficient four-way decode (<128 steps), OOM/preemption preventing the target, failed replay agreement, or no candidate reaching the offline gate. One concrete correction permitted for an infrastructure failure; no parameter fishing or MTP. Do not modify deployment defaults during the test. Save an HTML report even for a negative result.
