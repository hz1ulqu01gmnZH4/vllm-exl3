"""Final profiling attempt: CUDA events work independently of CUPTI."""
import json
from pathlib import Path

from vllm import LLM, SamplingParams
import event_hooks

event_hooks.install()  # Also executed in the spawned worker before graph capture.
ROOT = Path(__file__).resolve().parents[1]


def main():
    assert not event_hooks.GATE.exists(), 'Measurement gate already exists'
    assert not (ROOT/'results/cuda-events.jsonl').exists(), 'Do not overwrite measurements'
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
    )
    tokenizer = engine.get_tokenizer()
    messages = [
        'Write a Python function that merges two sorted lists without modifying them. Explain its time complexity and include unit tests.',
        'Explain how a CPU cache works, including cache lines, locality, and why sequential memory access is often faster. Use concrete examples.',
    ]
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': message}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for message in messages]
    # Match cache history of run-003 before its trace, without collecting speed results.
    for prompt, count in [(prompts[0], 16), (prompts[0], 128), (prompts[1], 128)]:
        engine.generate([prompt], SamplingParams(temperature=0, max_tokens=count, ignore_eos=True), use_tqdm=False)
    event_hooks.GATE.write_text('12 decode steps; CUDA event instrumentation enabled\n')
    result = engine.generate([prompts[0]], SamplingParams(
        temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False)[0]
    (ROOT/'results/event-output.json').write_text(json.dumps({
        'tokens': list(result.outputs[0].token_ids), 'text': result.outputs[0].text,
        'prompt_tokens': len(result.prompt_token_ids)}, indent=2)+'\n')
    records = (ROOT/'results/cuda-events.jsonl').read_text().splitlines()
    assert len(records) == 12, len(records)
    print('CUDA_EVENTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
