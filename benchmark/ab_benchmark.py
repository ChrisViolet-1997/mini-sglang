"""
A/B Benchmark: Current (optimized) mini-sglang vs Baseline (original) mini-sglang.

Compares:
  - Prefill time (TTFT)
  - Per-token decode latency (ITI)
  - Total inference time
  - KV cache memory usage
  - Whether flash-forward skips decode steps for repeated segments
"""
import json
import os
import subprocess
import sys
import time

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_PATH = "/root/autodl-tmp/local_llm/dir"
PROMPT_PATH = "/root/autodl-tmp/mini-sglang/input_prompt_demo/demo3.txt"
MAX_NEW_TOKENS = 16384
MAX_SEQ_LEN = 41500  # Slightly over max_position_embeddings to fit input+output

CURRENT_DIR = "/root/autodl-tmp/mini-sglang"
BASELINE_DIR = "/root/autodl-tmp/mini-sglang-baseline"

OUTPUT_FILE = "/root/autodl-tmp/mini-sglang/benchmark/ab_benchmark_results.json"


def build_script(python_path: str, backend: str, label: str) -> str:
    """Build the inference script for a single run."""
    return f'''
import sys, os, time, json, gc
sys.path.insert(0, "{python_path}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
label = "{label}"

import torch
from minisgl.llm import LLM
from minisgl.core import SamplingParams
from transformers import AutoTokenizer

with open("{PROMPT_PATH}", "r") as f:
    raw_prompt = f.read()

# Apply chat template for proper instruction-following
tokenizer_for_template = AutoTokenizer.from_pretrained("{MODEL_PATH}", trust_remote_code=True)
messages = [{{"role": "user", "content": raw_prompt}}]
prompt = tokenizer_for_template.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(
    model_path="{MODEL_PATH}",
    attention_backend="{backend}",
    page_size=1,
    max_seq_len_override={MAX_SEQ_LEN},
    cuda_graph_max_bs=0,
)

# Get input token count
input_ids = llm.tokenizer.encode(prompt, return_tensors="pt").view(-1)
input_len = len(input_ids)

# Memory baseline
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
mem_model = torch.cuda.memory_allocated() / (1024**3)

# Run inference with timing
sampling_params = SamplingParams(max_tokens={MAX_NEW_TOKENS}, temperature=0.0, ignore_eos=True)

torch.cuda.synchronize()
t_start = time.perf_counter()
results = llm.generate(prompts=[prompt], sampling_params=sampling_params)
torch.cuda.synchronize()
t_end = time.perf_counter()

# Collect metrics
peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
output_tokens = len(results[0]["token_ids"])
output_text = results[0]["text"]
total_ms = (t_end - t_start) * 1000

# Estimate prefill vs decode (approximate via proportional split)
# Better: use internal scheduler metrics if available
# Heuristic: prefill processes input_len tokens in one pass (compute-bound)
# decode generates output_tokens one at a time (memory-bound)
# For a more accurate split, we time a prefill-only pass:
# But that requires modifying the scheduler. Use total / output_tokens as ITI estimate.

result_data = {{
    "label": "{label}",
    "backend": "{backend}",
    "input_tokens": input_len,
    "output_tokens": output_tokens,
    "total_time_ms": total_ms,
    "avg_per_token_ms": total_ms / max(output_tokens, 1),
    "throughput_tok_per_sec": (input_len + output_tokens) / (total_ms / 1000),
    "peak_memory_gb": peak_mem,
    "kv_memory_gb": peak_mem - mem_model,
    "model_memory_gb": mem_model,
    "output_preview": output_text,
}}

# Try to get prefill-only timing (run with max_tokens=1)
torch.cuda.synchronize()
t0 = time.perf_counter()
_ = llm.generate(prompts=[prompt], sampling_params=SamplingParams(max_tokens=1, temperature=0.0))
torch.cuda.synchronize()
t1 = time.perf_counter()
prefill_ms = (t1 - t0) * 1000
decode_ms = total_ms - prefill_ms

result_data["prefill_time_ms"] = prefill_ms
result_data["decode_time_ms"] = decode_ms
result_data["decode_per_token_ms"] = decode_ms / max(output_tokens - 1, 1)
result_data["prefill_tok_per_sec"] = input_len / (prefill_ms / 1000)
result_data["decode_tok_per_sec"] = output_tokens / (decode_ms / 1000) if decode_ms > 0 else 0

with open(f"/tmp/_ab_result_{{label.replace(' ', '_')}}.json", "w") as f:
    json.dump(result_data, f, ensure_ascii=False, indent=2)

print(f"[{{label}}] Done:")
print(f"  Input:    {{input_len}} tokens")
print(f"  Output:   {{output_tokens}} tokens")
print(f"  Prefill:  {{prefill_ms:.1f}} ms ({{input_len/(prefill_ms/1000):.0f}} tok/s)")
print(f"  Decode:   {{decode_ms:.1f}} ms ({{decode_ms/max(output_tokens-1,1):.1f}} ms/tok)")
print(f"  Total:    {{total_ms:.1f}} ms")
print(f"  Peak mem: {{peak_mem:.2f}} GiB")
'''


def run_config(python_path: str, backend: str, label: str) -> dict:
    """Run a benchmark configuration in a subprocess."""
    print(f"\n{'─' * 70}")
    print(f"  {label} (backend={backend})")
    print(f"{'─' * 70}")

    script = build_script(python_path, backend, label)
    result_file = f"/tmp/_ab_result_{label.replace(' ', '_')}.json"

    # Remove old result
    if os.path.exists(result_file):
        os.remove(result_file)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=False,
        timeout=600,
    )

    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)
    else:
        print(f"  [FAILED] exit code {proc.returncode}")
        return None


def main():
    print("=" * 70)
    print("  A/B BENCHMARK: Current (optimized) vs Baseline (original)")
    print("  GPU: RTX 5090 | Model: Qwen3-8B | Input: demo3.txt")
    print("=" * 70)

    results = {}

    # Run current version with flashinfer
    r = run_config(
        python_path=os.path.join(CURRENT_DIR, "python"),
        backend="fi",
        label="Current fi",
    )
    if r:
        results["current_fi"] = r

    # Run baseline version with flashinfer
    r = run_config(
        python_path=os.path.join(BASELINE_DIR, "python"),
        backend="fi",
        label="Baseline fi",
    )
    if r:
        results["baseline_fi"] = r

    # Print comparison
    if "current_fi" in results and "baseline_fi" in results:
        cur = results["current_fi"]
        base = results["baseline_fi"]

        print("\n" + "=" * 70)
        print("  COMPARISON: Current vs Baseline (both using flashinfer backend)")
        print("=" * 70)
        print(f"  {'Metric':<30} {'Current':<18} {'Baseline':<18} {'Speedup':<10}")
        print("  " + "-" * 76)

        rows = [
            ("Input tokens", f"{cur['input_tokens']}", f"{base['input_tokens']}", ""),
            ("Output tokens", f"{cur['output_tokens']}", f"{base['output_tokens']}", ""),
            ("Prefill (ms)", f"{cur['prefill_time_ms']:.1f}", f"{base['prefill_time_ms']:.1f}",
             f"{base['prefill_time_ms']/cur['prefill_time_ms']:.2f}x" if cur['prefill_time_ms'] > 0 else ""),
            ("Prefill (tok/s)", f"{cur['prefill_tok_per_sec']:.0f}", f"{base['prefill_tok_per_sec']:.0f}", ""),
            ("Decode total (ms)", f"{cur['decode_time_ms']:.1f}", f"{base['decode_time_ms']:.1f}",
             f"{base['decode_time_ms']/cur['decode_time_ms']:.2f}x" if cur['decode_time_ms'] > 0 else ""),
            ("Decode ITI (ms/tok)", f"{cur['decode_per_token_ms']:.2f}", f"{base['decode_per_token_ms']:.2f}",
             f"{base['decode_per_token_ms']/cur['decode_per_token_ms']:.2f}x" if cur['decode_per_token_ms'] > 0 else ""),
            ("Total time (ms)", f"{cur['total_time_ms']:.1f}", f"{base['total_time_ms']:.1f}",
             f"{base['total_time_ms']/cur['total_time_ms']:.2f}x" if cur['total_time_ms'] > 0 else ""),
            ("Peak memory (GiB)", f"{cur['peak_memory_gb']:.2f}", f"{base['peak_memory_gb']:.2f}", ""),
            ("KV memory (GiB)", f"{cur['kv_memory_gb']:.2f}", f"{base['kv_memory_gb']:.2f}", ""),
        ]

        for name, v1, v2, sp in rows:
            print(f"  {name:<30} {v1:<18} {v2:<18} {sp:<10}")

        print("\n  Output preview (current):")
        print(f"    {cur['output_preview'][:100]}...")
        print("  Output preview (baseline):")
        print(f"    {base['output_preview'][:100]}...")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
