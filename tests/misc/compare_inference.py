"""
Compare baseline vs flash-forward-optimized inference on the demo SQL prompt.

Usage:
    python tests/misc/compare_inference.py

This script:
1. Loads the demo prompt and tokenizes it
2. Runs baseline inference (no aliasing, standard decode)
3. Runs optimized inference (with aliasing guide for flash-forward)
4. Compares output quality, speed, and memory usage
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.llm import LLM
from minisgl.llm.llm import RequestAllFinished, RequestStatus
from minisgl.message import BaseBackendMsg, DetokenizeMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.tokenizer.aliasing import AliasingGuideTable, build_aliasing_guide

MODEL_PATH = "/root/autodl-tmp/local_llm/qwen3-8b"
DEMO_FILE = Path(__file__).resolve().parents[2] / "input_prompt" / "demo.txt"
MAX_OUTPUT_TOKENS = 4096
PAGE_SIZE = 1


class OptimizedLLM(LLM):
    """LLM that attaches aliasing guide to requests for flash-forward."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self._guides: Dict[int, AliasingGuideTable] = {}

    def generate_with_guide(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
        page_size: int = 1,
        min_match_pages: int = 4,
    ) -> List[Dict[str, str | List[int]]]:
        """Generate with aliasing guide attached to each request."""
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        self._guides = {}

        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        # Pre-compute aliasing guides
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            self.pending_requests.append((prompt, sp))
            input_ids = self._tokenize_one(prompt)
            guide = build_aliasing_guide(input_ids, page_size, min_match_pages)
            self._guides[i] = guide

        try:
            self.run_forever()
        except RequestAllFinished:
            pass

        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid = self.counter + added
            added += 1
            guide = self._guides.get(uid, AliasingGuideTable())
            results.append(
                UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params,
                        aliasing_guide=guide)
            )
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(input_ids.tolist() if isinstance(tokens_or_prompt, str)
                           else tokens_or_prompt),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results


def load_prompt():
    with open(DEMO_FILE, "r") as f:
        return f.read()


def run_baseline(prompt: str, sampling_params: SamplingParams):
    """Run standard inference without optimization."""
    print("=" * 70)
    print("BASELINE: Standard inference (no aliasing, no flash-forward)")
    print("=" * 70)

    llm = LLM(
        MODEL_PATH,
        page_size=PAGE_SIZE,
        max_seq_len_override=65536,
        max_extend_tokens=65536,
        cuda_graph_max_bs=0,
        attention_backend="fa",
    )

    # Warmup
    print("Warming up...")
    llm.generate(["Hello"], SamplingParams(max_tokens=8))

    # Actual run
    print("Running baseline inference...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    results = llm.generate([prompt], [sampling_params])
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    mem_peak = torch.cuda.max_memory_allocated()
    elapsed = t_end - t_start
    output_text = results[0]["text"]
    output_tokens = results[0]["token_ids"]
    num_output = len(output_tokens)

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output tokens: {num_output}")
    print(f"  Throughput: {num_output / elapsed:.1f} tok/s")
    print(f"  Peak GPU memory: {mem_peak / 1024**3:.2f} GB")
    print(f"  Output preview (first 300 chars):")
    print(f"    {output_text[:300]}")
    print()

    del llm
    torch.cuda.empty_cache()
    return {
        "time": elapsed,
        "output_tokens": num_output,
        "throughput": num_output / elapsed,
        "peak_memory": mem_peak,
        "output_text": output_text,
        "token_ids": output_tokens,
    }


def run_optimized(prompt: str, sampling_params: SamplingParams):
    """Run inference with aliasing guide and flash-forward."""
    print("=" * 70)
    print("OPTIMIZED: With aliasing guide + flash-forward")
    print("=" * 70)

    llm = OptimizedLLM(
        MODEL_PATH,
        page_size=PAGE_SIZE,
        max_seq_len_override=65536,
        max_extend_tokens=65536,
        cuda_graph_max_bs=0,
        attention_backend="fa",
    )

    # Warmup
    print("Warming up...")
    llm.generate(["Hello"], SamplingParams(max_tokens=8))

    # Show aliasing stats
    input_ids = llm._tokenize_one(prompt)
    guide = build_aliasing_guide(input_ids, PAGE_SIZE, min_match_pages=4)
    print(f"  Input tokens: {len(input_ids)}")
    print(f"  Alias entries: {len(guide.entries)}")
    saved_tokens = sum(e.num_pages for e in guide.entries) * PAGE_SIZE
    print(f"  Prefill tokens skippable: {saved_tokens} ({saved_tokens/len(input_ids)*100:.1f}%)")

    # Actual run
    print("Running optimized inference...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    results = llm.generate_with_guide([prompt], [sampling_params],
                                       page_size=PAGE_SIZE, min_match_pages=4)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    mem_peak = torch.cuda.max_memory_allocated()
    elapsed = t_end - t_start
    output_text = results[0]["text"]
    output_tokens = results[0]["token_ids"]
    num_output = len(output_tokens)

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output tokens: {num_output}")
    print(f"  Throughput: {num_output / elapsed:.1f} tok/s")
    print(f"  Peak GPU memory: {mem_peak / 1024**3:.2f} GB")
    print(f"  Output preview (first 300 chars):")
    print(f"    {output_text[:300]}")
    print()

    del llm
    torch.cuda.empty_cache()
    return {
        "time": elapsed,
        "output_tokens": num_output,
        "throughput": num_output / elapsed,
        "peak_memory": mem_peak,
        "output_text": output_text,
        "token_ids": output_tokens,
    }


def compare_results(baseline, optimized):
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()

    time_speedup = baseline["time"] / optimized["time"] if optimized["time"] > 0 else float("inf")
    mem_saved = (baseline["peak_memory"] - optimized["peak_memory"]) / 1024**2

    print(f"  {'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Delta':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'Time (s)':<25} {baseline['time']:>15.2f} {optimized['time']:>15.2f} {time_speedup:>14.2f}x")
    print(f"  {'Output tokens':<25} {baseline['output_tokens']:>15} {optimized['output_tokens']:>15} {'':>15}")
    print(f"  {'Throughput (tok/s)':<25} {baseline['throughput']:>15.1f} {optimized['throughput']:>15.1f} {optimized['throughput']/baseline['throughput']:>14.2f}x")
    print(f"  {'Peak memory (GB)':<25} {baseline['peak_memory']/1024**3:>15.2f} {optimized['peak_memory']/1024**3:>15.2f} {mem_saved:>+13.0f}MB")
    print()

    # Output quality comparison
    if baseline["token_ids"] == optimized["token_ids"]:
        print("  Output quality: IDENTICAL (token-for-token match)")
    else:
        min_len = min(len(baseline["token_ids"]), len(optimized["token_ids"]))
        first_diff = min_len
        for i in range(min_len):
            if baseline["token_ids"][i] != optimized["token_ids"][i]:
                first_diff = i
                break
        match_pct = first_diff / max(len(baseline["token_ids"]), 1) * 100
        print(f"  Output quality: DIVERGES at token {first_diff} ({match_pct:.1f}% prefix match)")
        print(f"    Baseline length:  {len(baseline['token_ids'])} tokens")
        print(f"    Optimized length: {len(optimized['token_ids'])} tokens")
    print()


def main():
    prompt = load_prompt()
    print(f"Demo prompt: {len(prompt)} chars")
    print(f"Model: {MODEL_PATH}")
    print(f"Page size: {PAGE_SIZE}")
    print(f"Max output tokens: {MAX_OUTPUT_TOKENS}")
    print()

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for reproducibility
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    baseline = run_baseline(prompt, sampling_params)
    optimized = run_optimized(prompt, sampling_params)
    compare_results(baseline, optimized)


if __name__ == "__main__":
    main()
