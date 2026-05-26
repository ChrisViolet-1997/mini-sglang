"""
KV aliasing: after prefill, copy KV cache entries from src pages to dst pages
(with RoPE position adjustment) for intra-sequence duplicate page runs.

This enables the "skip suffix" optimization where duplicate token sequences
don't need to be computed during prefill — their KV can be copied from the
first occurrence.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
try:
    from minisgl.kernel.copy_kv_rope_triton import copy_kv_with_rope
except ImportError:
    from minisgl.kernel.copy_kv_rope import copy_kv_with_rope

if TYPE_CHECKING:
    from minisgl.core import Batch, Req
    from minisgl.kvcache.mha_pool import MHAKVCache


def compute_skippable_suffix(req: Req, page_size: int) -> int:
    """
    Compute how many tokens at the end of the request can be skipped during
    prefill because they are duplicates of earlier tokens (per aliasing guide).

    Returns the number of tokens that can be skipped (filled via KV copy instead).
    """
    guide = req.aliasing_guide
    if guide is None or guide.empty:
        return 0

    # The skippable suffix is the trailing portion covered by alias entries
    # that extends to the end of the input sequence.
    n_tokens = len(req.input_ids)
    n_pages = n_tokens // page_size
    if n_pages == 0:
        return 0

    # Find entries whose dst range reaches the end of the sequence
    skip_pages = 0
    for entry in reversed(guide.entries):
        entry_end = entry.dst_page + entry.num_pages
        if entry_end >= n_pages:
            # This entry covers up to the end
            skip_pages = entry.num_pages - max(0, entry_end - n_pages)
            break

    return skip_pages * page_size


def apply_kv_aliasing(
    batch: Batch,
    kv_cache: MHAKVCache,
    page_table: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    page_size: int,
) -> None:
    """
    For each request in the batch that has an aliasing guide, copy KV cache
    from src pages to dst pages with RoPE adjustment.
    """
    for req in batch.reqs:
        guide = req.aliasing_guide
        if guide is None or guide.empty:
            continue
        _apply_aliasing_for_req(req, kv_cache, page_table, cos_sin_cache, page_size)


def _apply_aliasing_for_req(
    req: Req,
    kv_cache: MHAKVCache,
    page_table: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    page_size: int,
) -> None:
    """Apply KV aliasing for a single request."""
    guide = req.aliasing_guide
    device = kv_cache.device

    for entry in guide.entries:
        N = entry.num_pages * page_size

        # Source and destination sequence positions
        src_seq_positions = torch.arange(
            entry.src_page * page_size,
            (entry.src_page + entry.num_pages) * page_size,
            dtype=torch.int32,
            device=device,
        )
        dst_seq_positions = torch.arange(
            entry.dst_page * page_size,
            (entry.dst_page + entry.num_pages) * page_size,
            dtype=torch.int32,
            device=device,
        )

        # Look up physical cache locations from page_table
        src_indices = page_table[req.table_idx, src_seq_positions.long()]
        dst_indices = page_table[req.table_idx, dst_seq_positions.long()]

        # Apply for each layer
        num_layers = kv_cache.num_layers
        storage_shape = kv_cache._storage_shape
        num_tokens, num_heads, head_dim = storage_shape

        for layer_id in range(num_layers):
            k_cache = kv_cache._k_buffer[layer_id].view(num_tokens, num_heads * head_dim)
            v_cache = kv_cache._v_buffer[layer_id].view(num_tokens, num_heads * head_dim)

            copy_kv_with_rope(
                k_cache=k_cache,
                v_cache=v_cache,
                src_indices=src_indices,
                dst_indices=dst_indices,
                cos_sin_cache=cos_sin_cache,
                src_positions=src_seq_positions,
                dst_positions=dst_seq_positions,
            )
