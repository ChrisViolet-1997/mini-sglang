from __future__ import annotations

import torch
import pytest
from minisgl.tokenizer.aliasing import build_aliasing_guide, AliasingGuideTable


PAGE_SIZE = 4


def _make_ids(*segments: list[int]) -> torch.Tensor:
    return torch.tensor([t for seg in segments for t in seg], dtype=torch.int32)


def test_no_duplicates():
    ids = torch.arange(64, dtype=torch.int32)
    guide = build_aliasing_guide(ids, PAGE_SIZE)
    assert guide.empty


def test_single_duplicate_run():
    page = [1, 2, 3, 4]
    # layout: [unique_prefix x4 pages] [dup_src x4 pages] [filler x4 pages] [dup_dst x4 pages]
    prefix = list(range(100, 116))   # 4 pages, unique
    src    = page * 4                # 4 pages, repeated
    filler = list(range(200, 216))   # 4 pages, unique
    dst    = page * 4                # 4 pages, same as src

    ids = _make_ids(prefix, src, filler, dst)
    guide = build_aliasing_guide(ids, PAGE_SIZE, min_match_pages=4)

    assert not guide.empty
    assert len(guide.entries) == 1
    e = guide.entries[0]
    assert e.src_page == 4   # src starts at page 4 (after 4-page prefix)
    assert e.dst_page == 12  # dst starts at page 12 (4+4+4)
    assert e.num_pages == 4


def test_min_match_pages_threshold():
    page = [10, 20, 30, 40]
    prefix = list(range(50, 66))
    src    = page * 3   # only 3 pages
    filler = list(range(70, 86))
    dst    = page * 3

    ids = _make_ids(prefix, src, filler, dst)
    # require 4 pages minimum — should find nothing
    guide = build_aliasing_guide(ids, PAGE_SIZE, min_match_pages=4)
    assert guide.empty

    # require 3 pages minimum — should find it
    guide = build_aliasing_guide(ids, PAGE_SIZE, min_match_pages=3)
    assert not guide.empty


def test_entries_sorted_by_dst_page():
    page_a = [1, 2, 3, 4]
    page_b = [5, 6, 7, 8]
    # two independent duplicate runs
    seg = page_a * 4 + page_b * 4 + list(range(100, 132)) + page_b * 4 + page_a * 4
    ids = torch.tensor(seg, dtype=torch.int32)
    guide = build_aliasing_guide(ids, PAGE_SIZE, min_match_pages=4)

    dst_pages = [e.dst_page for e in guide.entries]
    assert dst_pages == sorted(dst_pages)


def test_no_self_alias():
    # a run that appears only once should not alias to itself
    page = [7, 8, 9, 10]
    ids = _make_ids(page * 8)
    guide = build_aliasing_guide(ids, PAGE_SIZE, min_match_pages=4)
    for e in guide.entries:
        assert e.src_page < e.dst_page


def test_short_sequence_returns_empty():
    ids = torch.arange(8, dtype=torch.int32)  # only 2 pages, below 2*min_match_pages
    guide = build_aliasing_guide(ids, PAGE_SIZE, min_match_pages=4)
    assert guide.empty
