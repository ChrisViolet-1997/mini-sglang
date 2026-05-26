"""
copy_kv_with_rope: Copy KV cache entries from src to dst positions,
adjusting K's RoPE embedding from src_positions to dst_positions.

This is a pure PyTorch fallback implementation for sm120 (Blackwell) GPUs
where sgl_kernel is not available.
"""
from __future__ import annotations

import torch


def copy_kv_with_rope(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    src_positions: torch.Tensor,
    dst_positions: torch.Tensor,
) -> None:
    """
    Copy KV cache from src_indices to dst_indices, applying RoPE rotation
    adjustment on K from src_positions to dst_positions.

    Args:
        k_cache: (num_tokens, hidden_size) flattened K cache for one layer
        v_cache: (num_tokens, hidden_size) flattened V cache for one layer
        src_indices: (N,) physical slot indices to copy from
        dst_indices: (N,) physical slot indices to copy to
        cos_sin_cache: (max_position, rotary_dim) interleaved [cos, sin] cache
        src_positions: (N,) sequence positions of source tokens
        dst_positions: (N,) sequence positions of destination tokens
    """
    # V cache: straight copy, no rotation needed
    v_cache[dst_indices] = v_cache[src_indices]

    # K cache: need to undo src RoPE and apply dst RoPE
    # RoPE: k_rotated = k * cos(pos) + rotate_half(k) * sin(pos)
    # To adjust from pos_src to pos_dst:
    #   k_dst = k_src * cos(pos_dst - pos_src) + rotate_half(k_src) * sin(pos_dst - pos_src)
    # But cos_sin_cache stores absolute positions, so we use the identity:
    #   k_at_dst = undo_rope(k_at_src, pos_src) then apply_rope(k_raw, pos_dst)
    # Which simplifies to applying a differential rotation.

    hidden_size = k_cache.shape[1]
    rotary_dim = cos_sin_cache.shape[1]  # This is rotary_dim (cos and sin interleaved)
    half_rotary = rotary_dim // 2

    k_src = k_cache[src_indices]  # (N, hidden_size)

    # Extract cos/sin for src and dst positions
    # cos_sin_cache shape: (max_pos, rotary_dim) where first half is cos, second half is sin
    cos_src = cos_sin_cache[src_positions.long(), :half_rotary]  # (N, half_rotary)
    sin_src = cos_sin_cache[src_positions.long(), half_rotary:]  # (N, half_rotary)
    cos_dst = cos_sin_cache[dst_positions.long(), :half_rotary]  # (N, half_rotary)
    sin_dst = cos_sin_cache[dst_positions.long(), half_rotary:]  # (N, half_rotary)

    # Undo src RoPE: k_raw = k_src * cos_src + rotate_half(k_src) * (-sin_src)
    # Then apply dst RoPE: k_dst = k_raw * cos_dst + rotate_half(k_raw) * sin_dst
    # Combined: use angle difference
    # cos(dst - src) = cos_dst * cos_src + sin_dst * sin_src
    # sin(dst - src) = sin_dst * cos_src - cos_dst * sin_src
    cos_diff = cos_dst * cos_src + sin_dst * sin_src  # (N, half_rotary)
    sin_diff = sin_dst * cos_src - cos_dst * sin_src  # (N, half_rotary)

    # Apply differential rotation to the rotary part of k
    k_rot = k_src[:, :rotary_dim]  # (N, rotary_dim)
    k_rot_first = k_rot[:, :half_rotary]  # (N, half_rotary)
    k_rot_second = k_rot[:, half_rotary:]  # (N, half_rotary)

    # rotate_half: [-second, first]
    k_new_first = k_rot_first * cos_diff - k_rot_second * sin_diff
    k_new_second = k_rot_second * cos_diff + k_rot_first * sin_diff

    # Assemble result
    k_dst = k_src.clone()
    k_dst[:, :half_rotary] = k_new_first
    k_dst[:, half_rotary:rotary_dim] = k_new_second
    # Non-rotary dimensions (if any) are copied as-is

    k_cache[dst_indices] = k_dst
