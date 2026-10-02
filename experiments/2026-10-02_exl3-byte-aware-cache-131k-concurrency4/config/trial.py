"""Four near-131K requests; trace and compare fixed-budget cache variants."""
import hashlib
import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams
import trace_hooks

trace_hooks.install()
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
LABEL = os.environ['CACHE_TRIAL_LABEL']


def prompts(tokenizer):
    path = ROOT/'data/prompts.json'
    if path.exists():
        return json.loads(path.read_text())
    corpus_paths = [REPO/'src/vllm_exl3/expert_cache.py',
                    REPO/'src/vllm_exl3/expert_cache_tables.py',
                    REPO/'README.md', REPO/'docs/expert-cache.html']
    source_records, result = [], []
    questions = [
        'Write a detailed engineering review of this cache implementation. Explain correctness, ownership, eviction, CUDA graphs and tests, with concrete code examples.',
        'Design a serving performance experiment for this implementation. Give a detailed measurement plan, pseudocode and example result analysis for four concurrent long requests.',
        'Explain this EXL3 serving design to a new engineer. Discuss memory layout, routing, decode, prefill, and failure handling in depth.',
        'Review the documentation and source below, then propose a detailed debugging guide for long-context concurrent inference, with practical Python examples.',
    ]
    for i, source in enumerate(corpus_paths):
        content = source.read_text()
        source_records.append({'path': str(source), 'sha256': hashlib.sha256(content.encode()).hexdigest()})
        body = tokenizer.encode(content, add_special_tokens=False)
        prefix = tokenizer.apply_chat_template([{'role':'user','content':questions[i]+'\nReference material:\n'}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        # Assemble token IDs directly with a complete chat suffix, preserving the exact total length.
        head = tokenizer.encode(prefix.rsplit('<|im_end|>', 1)[0], add_special_tokens=False)
        suffix = tokenizer.encode('\nEnd of reference. '+questions[i]+'\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
        count = 130560-len(head)-len(suffix)
        assert count > 0 and body
        ids = head+(body*((count+len(body)-1)//len(body)))[:count]+suffix
        assert len(ids) == 130560
        result.append(ids)
    path.write_text(json.dumps(result)+'\n')
    (ROOT/'data/prompt-provenance.json').write_text(json.dumps({
        'kind': 'synthetic repeated public serving source/documentation; four distinct tasks',
        'sources': source_records, 'prompt_tokens_each':130560, 'output_tokens_each':512,
        'sha256': [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in result]}, indent=2)+'\n')
    return result


def main():
    assert LABEL in ('baseline', 'candidate')
    engine = LLM(
        model='/home/ak/qwen38-vllm/models/Qwen3.8-Flash-Next-EXL3-cache',
        quantization='exl3', dtype='bfloat16', tensor_parallel_size=1,
        load_format='safetensors', safetensors_load_strategy='lazy',
        language_model_only=True, max_model_len=131072,
        max_num_seqs=4, max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85, kv_cache_memory_bytes=8589934592,
        kv_cache_dtype='fp8_e4m3', enable_prefix_caching=False,
        seed=42, disable_log_stats=False,
        compilation_config={'mode':0, 'cudagraph_mode':'FULL_DECODE_ONLY',
                            'cudagraph_capture_sizes':[1,2,4]},
    )
    requests = [{'prompt_token_ids': ids} for ids in prompts(engine.get_tokenizer())]
    print('TRIAL_BEGIN', LABEL, engine.collective_rpc('cache_trial_begin'), flush=True)
    started = time.monotonic()
    outputs = engine.generate(requests, SamplingParams(
        temperature=0, max_tokens=512, ignore_eos=True), use_tqdm=False)
    result = {'label':LABEL, 'wall_s':time.monotonic()-started, 'requests':[]}
    for output in outputs:
        assert len(output.prompt_token_ids) == 130560
        assert len(output.outputs[0].token_ids) == 512
        metrics = output.metrics
        assert metrics is not None
        result['requests'].append({
            'request_id':output.request_id, 'prompt_tokens':len(output.prompt_token_ids),
            'output_tokens':list(output.outputs[0].token_ids), 'text':output.outputs[0].text,
            'first_token_ts':metrics.first_token_ts, 'last_token_ts':metrics.last_token_ts,
            'decode_tok_s_including_mixed_prefill':511/(metrics.last_token_ts-metrics.first_token_ts)})
    (ROOT/f'results/{LABEL}-outputs.json').write_text(json.dumps(result, indent=2)+'\n')
    result['steady'] = engine.collective_rpc('cache_trial_finish')
    print('TRIAL_COMPLETE', LABEL, result['wall_s'], flush=True)


if __name__ == '__main__':
    main()
