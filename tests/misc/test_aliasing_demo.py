"""Test aliasing token savings on real SQL prompts."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import pytest
from minisgl.tokenizer.aliasing import build_aliasing_guide


DEMO_FILE = Path(__file__).resolve().parents[2] / "input_prompt" / "demo.txt"
MODEL_PATH = os.environ.get("TOKENIZER_PATH", "/root/autodl-tmp/local_llm/qwen3-8b")


def _load_token_ids() -> torch.Tensor:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    with open(DEMO_FILE, "r") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.int32)


@pytest.fixture(scope="module")
def input_ids():
    if not DEMO_FILE.exists():
        pytest.skip(f"Demo file not found: {DEMO_FILE}")
    try:
        return _load_token_ids()
    except Exception as e:
        pytest.skip(f"Cannot load tokenizer: {e}")


def test_aliasing_savings_page1(input_ids):
    """page_size=1, min_match_pages=4: expect significant savings on SQL prompts."""
    guide = build_aliasing_guide(input_ids, page_size=1, min_match_pages=4)
    total_tokens = len(input_ids)
    saved_tokens = sum(e.num_pages for e in guide.entries)
    ratio = saved_tokens / total_tokens

    print(f"\n[page_size=1, min_match=4] {saved_tokens}/{total_tokens} tokens saved ({ratio*100:.1f}%)")
    print(f"  entries: {len(guide.entries)}")
    assert ratio > 0.30, f"Expected >30% savings, got {ratio*100:.1f}%"


def test_aliasing_savings_page16(input_ids):
    """page_size=16, min_match_pages=4: still meaningful savings with page alignment."""
    guide = build_aliasing_guide(input_ids, page_size=16, min_match_pages=4)
    total_tokens = len(input_ids)
    saved_tokens = sum(e.num_pages for e in guide.entries) * 16
    ratio = saved_tokens / total_tokens

    print(f"\n[page_size=16, min_match=4] {saved_tokens}/{total_tokens} tokens saved ({ratio*100:.1f}%)")
    print(f"  entries: {len(guide.entries)}")
    assert ratio > 0.04, f"Expected >4% savings, got {ratio*100:.1f}%"


def test_aliasing_report(input_ids):
    """Print a full report of aliasing savings across page sizes."""
    total_tokens = len(input_ids)
    print(f"\n{'='*60}")
    print(f"Demo file: {DEMO_FILE.name}")
    print(f"Total tokens: {total_tokens}")
    print(f"{'='*60}")
    print(f"{'page_size':>10} {'min_match':>10} {'entries':>8} {'saved_tok':>10} {'ratio':>8}")
    print(f"{'-'*10:>10} {'-'*10:>10} {'-'*8:>8} {'-'*10:>10} {'-'*8:>8}")

    for page_size in [1, 4, 8, 16, 32, 64]:
        for min_match in [2, 4, 8]:
            guide = build_aliasing_guide(input_ids, page_size, min_match_pages=min_match)
            if not guide.empty:
                saved = sum(e.num_pages for e in guide.entries) * page_size
                ratio = saved / total_tokens * 100
                print(f"{page_size:>10} {min_match:>10} {len(guide.entries):>8} {saved:>10} {ratio:>7.1f}%")
