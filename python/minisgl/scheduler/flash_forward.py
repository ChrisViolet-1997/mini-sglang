"""
Decode Flash-Forward: detect repeated token sequences during decode and skip
ahead by copying KV cache instead of generating token-by-token.

When the model generates tokens that match a known candidate sequence (derived
from intra-sequence duplicates detected during prefill), we "fast-forward" by
writing the remaining tokens directly and filling KV cache via copy_kv_with_rope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch
try:
    from minisgl.kernel.copy_kv_rope_triton import copy_kv_with_rope
except ImportError:
    from minisgl.kernel.copy_kv_rope import copy_kv_with_rope
from minisgl.message import DetokenizeMsg

if TYPE_CHECKING:
    from minisgl.core import Req
    from minisgl.kvcache.mha_pool import MHAKVCache
    from minisgl.scheduler.cache import CacheManager
    from minisgl.tokenizer.aliasing import AliasingGuideTable


@dataclass
class FlashForwardCandidate:
    """A token block from input that may be regenerated during decode."""

    tokens: torch.Tensor  # 1D int32 CPU tensor
    src_start_pos: int  # absolute position of this block in the input sequence
    length: int  # number of tokens
    token_list: List[int]  # .tolist() cache for fast comparison


@dataclass
class FlashForwardState:
    """Per-request FSM state for flash-forward detection during decode."""

    candidates: List[FlashForwardCandidate]
    match_threshold: int = 4
    # Active matching state
    active_candidate_idx: int = -1
    matched_count: int = 0

    def reset(self) -> None:
        self.active_candidate_idx = -1
        self.matched_count = 0


class FlashForwardDetector:
    """Simplified FSM: detect when decode output matches a known candidate block."""

    def feed_token(self, state: FlashForwardState, token_id: int) -> None:
        """Feed a newly generated token into the FSM."""
        if state.active_candidate_idx >= 0:
            # We have an active candidate, check if the token continues the match
            cand = state.candidates[state.active_candidate_idx]
            if state.matched_count < cand.length and cand.token_list[state.matched_count] == token_id:
                state.matched_count += 1
                return
            else:
                # Mismatch: reset and try all candidates from scratch
                state.active_candidate_idx = -1
                state.matched_count = 0

        # Try to start a new match with any candidate
        for idx, cand in enumerate(state.candidates):
            if cand.token_list[0] == token_id:
                state.active_candidate_idx = idx
                state.matched_count = 1
                return

    def get_fast_forward_tokens(self, state: FlashForwardState) -> Optional[torch.Tensor]:
        """
        If the match threshold is reached, return the remaining tokens to fast-forward.

        Returns None if not triggered. Returns a 1D CPU int32 tensor of remaining
        tokens (excluding the already-matched prefix) if triggered.
        """
        if state.active_candidate_idx < 0:
            return None
        if state.matched_count < state.match_threshold:
            return None

        cand = state.candidates[state.active_candidate_idx]
        if state.matched_count >= cand.length:
            # Already matched the entire candidate, nothing to fast-forward
            return None

        # Return remaining tokens after the matched prefix
        remaining = cand.tokens[state.matched_count:]
        return remaining


def build_flash_forward_candidates(
    input_ids: torch.Tensor,
    aliasing_guide: AliasingGuideTable,
    page_size: int,
) -> List[FlashForwardCandidate]:
    """
    Build flash-forward candidates from the aliasing guide.

    Each AliasEntry(src_page, dst_page, num_pages) indicates that the src block
    appears again at dst. The src block's tokens are candidates that may be
    regenerated during decode.
    """
    if aliasing_guide.empty:
        return []

    candidates: List[FlashForwardCandidate] = []
    for entry in aliasing_guide.entries:
        src_start = entry.src_page * page_size
        length = entry.num_pages * page_size
        src_end = src_start + length

        if src_end > len(input_ids):
            # Truncate to available tokens
            length = len(input_ids) - src_start
            if length <= 0:
                continue

        tokens = input_ids[src_start : src_start + length].to(torch.int32)
        candidates.append(
            FlashForwardCandidate(
                tokens=tokens,
                src_start_pos=src_start,
                length=length,
                token_list=tokens.tolist(),
            )
        )

    return candidates


def execute_flash_forward(
    req: Req,
    ff_tokens: torch.Tensor,
    src_start_pos: int,
    matched_count: int,
    token_pool: torch.Tensor,
    page_table: torch.Tensor,
    kv_cache: MHAKVCache,
    cos_sin_cache: torch.Tensor,
    cache_manager: CacheManager,
    page_size: int,
) -> List[DetokenizeMsg]:
    """
    Execute flash-forward: write tokens, allocate pages, copy KV cache.

    Args:
        req: The request to fast-forward.
        ff_tokens: 1D int32 CPU tensor of tokens to append.
        src_start_pos: Absolute position of the source block in the input.
        matched_count: Number of tokens already matched (offset into src block).
        token_pool: [max_reqs, max_seq_len] token storage on device.
        page_table: [max_reqs, max_seq_len] page table on device.
        kv_cache: The MHA KV cache pool.
        cos_sin_cache: [max_position, head_dim] precomputed cos/sin for RoPE.
        cache_manager: Cache manager for page allocation.
        page_size: Tokens per page.

    Returns:
        List of DetokenizeMsg for the fast-forwarded tokens.
    """
    N = len(ff_tokens)
    if N == 0:
        return []

    old_device_len = req.device_len
    device = page_table.device

    # 1. Write ff_tokens into token_pool
    ff_tokens_device = ff_tokens.to(device)
    token_pool[req.table_idx, old_device_len : old_device_len + N] = ff_tokens_device

    # 2. Append to req.input_ids (host)
    req.input_ids = torch.cat([req.input_ids, ff_tokens.cpu()])

    # 3. Set cached_len and device_len for allocation
    req.cached_len = old_device_len
    req.device_len = old_device_len + N

    # 4. Allocate pages for new positions
    cache_manager.allocate_paged([req])

    # 5. Copy KV cache with RoPE correction for each layer
    # Source positions: src_start_pos + matched_count .. src_start_pos + matched_count + N - 1
    # Dest positions: old_device_len .. old_device_len + N - 1
    src_seq_positions = torch.arange(
        src_start_pos + matched_count,
        src_start_pos + matched_count + N,
        dtype=torch.int32,
        device=device,
    )
    dst_seq_positions = torch.arange(
        old_device_len,
        old_device_len + N,
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

    # 6. Mark KV as filled
    req.cached_len = req.device_len

    # 7. Build DetokenizeMsg list
    ff_token_list = ff_tokens.tolist()
    msgs: List[DetokenizeMsg] = []
    for tok in ff_token_list:
        msgs.append(DetokenizeMsg(uid=req.uid, next_token=tok, finished=False))

    return msgs
