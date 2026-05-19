from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class AliasEntry:
    src_page: int   # first occurrence page index (already computed KV)
    dst_page: int   # duplicate occurrence page index (to be aliased)
    num_pages: int  # number of consecutive matching pages


@dataclass
class AliasingGuideTable:
    entries: List[AliasEntry] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return len(self.entries) == 0


_HASH_BASE = 131
_HASH_MOD = (1 << 61) - 1  # Mersenne prime


def build_aliasing_guide(
    input_ids: torch.Tensor,
    page_size: int,
    min_match_pages: int = 4,
) -> AliasingGuideTable:
    """
    Scan input_ids for intra-sequence duplicate page runs and return an AliasingGuideTable.

    Only non-prefix duplicates are recorded (i.e. dst_page > src_page), since the
    RadixCache already handles shared-prefix reuse across requests.

    Args:
        input_ids: 1D int32 CPU tensor of token ids.
        page_size: number of tokens per KV cache page.
        min_match_pages: minimum consecutive matching pages to record an alias entry.
    """
    n_tokens = len(input_ids)
    n_pages = n_tokens // page_size  # only consider fully-aligned pages
    if n_pages < min_match_pages * 2:
        return AliasingGuideTable()

    page_hashes = _compute_page_hashes(input_ids, page_size, n_pages)
    return _find_duplicate_runs(page_hashes, min_match_pages)


def _compute_page_hashes(
    input_ids: torch.Tensor, page_size: int, n_pages: int
) -> List[int]:
    ids = input_ids[: n_pages * page_size].tolist()
    hashes: List[int] = []
    for p in range(n_pages):
        h = 0
        for t in ids[p * page_size : (p + 1) * page_size]:
            h = (h * _HASH_BASE + t + 1) % _HASH_MOD
        hashes.append(h)
    return hashes


def _find_duplicate_runs(
    page_hashes: List[int], min_match_pages: int
) -> AliasingGuideTable:
    n_pages = len(page_hashes)
    # Map hash -> list of page indices where it first appears
    first_seen: Dict[int, int] = {}
    entries: List[AliasEntry] = []

    p = 0
    while p < n_pages:
        h = page_hashes[p]
        if h not in first_seen:
            first_seen[h] = p
            p += 1
            continue

        src = first_seen[h]
        dst = p
        # src must strictly precede dst to avoid self-alias
        if src >= dst:
            p += 1
            continue

        # extend the run as far as hashes match and pages don't overlap
        run = 0
        while (
            dst + run < n_pages
            and src + run < dst  # no overlap
            and page_hashes[src + run] == page_hashes[dst + run]
        ):
            run += 1

        if run >= min_match_pages:
            entries.append(AliasEntry(src_page=src, dst_page=dst, num_pages=run))
            # skip past the matched dst run to avoid redundant sub-entries
            p = dst + run
        else:
            p += 1

    entries.sort(key=lambda e: e.dst_page)
    return AliasingGuideTable(entries=entries)
