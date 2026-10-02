"""Bounded full-model decode profile using the current local launch configuration."""
import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    engine = LLM(
        model='/home/ak/qwen38-vllm/models/Qwen3.8-Flash-Next-EXL3-cache',
        quantization='exl3', dtype='bfloat16', tensor_parallel_size=1,
        load_format='safetensors', safetensors_load_strategy='lazy',
        language_model_only=True, max_model_len=131072,
        max_num_seqs=4, max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85, kv_cache_memory_bytes=8589934592,
        kv_cache_dtype='fp8_e4m3', enable_prefix_caching=False,
        seed=42, disable_log_stats=False,
        compilation_config={'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                            'cudagraph_capture_sizes': [1, 2, 4]},
        profiler_config={
            'profiler': 'torch', 'torch_profiler_dir': str(ROOT/'results/profile'),
            'torch_profiler_with_stack': False, 'ignore_frontend': True,
            'delay_iterations': 1, 'max_iterations': 12,
        },
    )
    result = {'startup_s': time.monotonic()-started,
              'scope': 'one active request; current four-sequence-capacity launcher; no MTP',
              'environment': {k: v for k, v in os.environ.items()
                              if k.startswith(('VLLM_EXL3_', 'EXL3_'))},
              'requests': []}
    tokenizer = engine.get_tokenizer()
    messages = [
        'Write a Python function that merges two sorted lists without modifying them. Explain its time complexity and include unit tests.',
        'Explain how a CPU cache works, including cache lines, locality, and why sequential memory access is often faster. Use concrete examples.',
    ]
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': message}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for message in messages]
    engine.generate([prompts[0]], SamplingParams(temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False)
    for prompt in prompts:
        started = time.monotonic()
        output = engine.generate([prompt], SamplingParams(
            temperature=0, max_tokens=128, ignore_eos=True), use_tqdm=False)[0]
        wall = time.monotonic()-started
        metrics = output.metrics
        assert metrics is not None, 'Request metrics missing'
        tokens = list(output.outputs[0].token_ids)
        assert len(tokens) == 128, len(tokens)
        decode_s = metrics.last_token_ts-metrics.first_token_ts
        assert decode_s > 0, decode_s
        row = {'prompt_tokens': len(output.prompt_token_ids), 'tokens': tokens,
               'text': output.outputs[0].text, 'wall_s': wall,
               'decode_s': decode_s, 'decode_tok_s': 127/decode_s}
        result['requests'].append(row)
        (ROOT/'results/serving.json').write_text(json.dumps(result, indent=2)+'\n')
        print('UNPROFILED_RESULT', row['prompt_tokens'], row['decode_tok_s'], flush=True)
    engine.start_profile()
    engine.generate([prompts[0]], SamplingParams(
        temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False)
    engine.stop_profile()
    print('PROFILE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
