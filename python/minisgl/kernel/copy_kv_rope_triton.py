"""
Triton kernel for copy_kv_with_rope: fused KV cache copy with differential RoPE.

Each kernel launch handles one layer, parallelizing across N tokens.
The host loops over layers issuing lightweight launches with zero data copies.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_v_kernel(
    V_ptr,
    src_indices_ptr,
    dst_indices_ptr,
    hidden_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy V cache: one program per token, vectorized across hidden dim."""
    pid = tl.program_id(0)
    src_idx = tl.load(src_indices_ptr + pid)
    dst_idx = tl.load(dst_indices_ptr + pid)

    src_base = src_idx * hidden_size
    dst_base = dst_idx * hidden_size

    d_offs = tl.arange(0, BLOCK_D)
    for d_start in range(0, hidden_size, BLOCK_D):
        offs = d_offs + d_start
        mask = offs < hidden_size
        val = tl.load(V_ptr + src_base + offs, mask=mask)
        tl.store(V_ptr + dst_base + offs, val, mask=mask)


@triton.jit
def _copy_k_rope_kernel(
    K_ptr,
    src_indices_ptr,
    dst_indices_ptr,
    cos_sin_cache_ptr,
    src_positions_ptr,
    dst_positions_ptr,
    hidden_size: tl.constexpr,
    half_rotary: tl.constexpr,
    cos_sin_stride: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy K cache with differential RoPE: one program per token."""
    pid = tl.program_id(0)
    src_idx = tl.load(src_indices_ptr + pid)
    dst_idx = tl.load(dst_indices_ptr + pid)
    src_pos = tl.load(src_positions_ptr + pid)
    dst_pos = tl.load(dst_positions_ptr + pid)

    src_base = src_idx * hidden_size
    dst_base = dst_idx * hidden_size

    # Cos/sin cache offsets
    cos_src_base = src_pos * cos_sin_stride
    cos_dst_base = dst_pos * cos_sin_stride

    d_offs = tl.arange(0, BLOCK_D)

    # Process rotary dimensions in blocks
    for d_start in range(0, half_rotary, BLOCK_D):
        offs = d_offs + d_start
        mask = offs < half_rotary

        # Load cos/sin for differential rotation
        cos_s = tl.load(cos_sin_cache_ptr + cos_src_base + offs, mask=mask)
        sin_s = tl.load(cos_sin_cache_ptr + cos_src_base + half_rotary + offs, mask=mask)
        cos_d = tl.load(cos_sin_cache_ptr + cos_dst_base + offs, mask=mask)
        sin_d = tl.load(cos_sin_cache_ptr + cos_dst_base + half_rotary + offs, mask=mask)

        # cos(dst - src) = cos_dst * cos_src + sin_dst * sin_src
        # sin(dst - src) = sin_dst * cos_src - cos_dst * sin_src
        cos_diff = cos_d * cos_s + sin_d * sin_s
        sin_diff = sin_d * cos_s - cos_d * sin_s

        # Load K[src, :half_rotary] and K[src, half_rotary:rotary_dim]
        k_first = tl.load(K_ptr + src_base + offs, mask=mask)
        k_second = tl.load(K_ptr + src_base + half_rotary + offs, mask=mask)

        # Apply differential rotation
        k_new_first = k_first * cos_diff - k_second * sin_diff
        k_new_second = k_second * cos_diff + k_first * sin_diff

        # Store to dst
        tl.store(K_ptr + dst_base + offs, k_new_first, mask=mask)
        tl.store(K_ptr + dst_base + half_rotary + offs, k_new_second, mask=mask)

    # Copy non-rotary dimensions as-is
    rotary_dim = half_rotary * 2
    for d_start in range(rotary_dim, hidden_size, BLOCK_D):
        offs = d_offs + d_start
        mask = offs < hidden_size
        k_val = tl.load(K_ptr + src_base + offs, mask=mask)
        tl.store(K_ptr + dst_base + offs, k_val, mask=mask)


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
    Triton-accelerated KV copy with differential RoPE (single layer, in-place).

    Drop-in replacement for the pure PyTorch version in copy_kv_rope.py.

    Args:
        k_cache: (num_tokens, hidden_size) K cache for one layer
        v_cache: (num_tokens, hidden_size) V cache for one layer
        src_indices: (N,) physical slot indices to copy from
        dst_indices: (N,) physical slot indices to copy to
        cos_sin_cache: (max_position, rotary_dim) where first half is cos, second is sin
        src_positions: (N,) sequence positions of source tokens
        dst_positions: (N,) sequence positions of destination tokens
    """
    N = src_indices.shape[0]
    if N == 0:
        return

    hidden_size = k_cache.shape[1]
    rotary_dim = cos_sin_cache.shape[1]
    half_rotary = rotary_dim // 2

    BLOCK_D = min(128, triton.next_power_of_2(hidden_size))

    # Ensure contiguous and int32 for positions (Triton needs scalar-indexable)
    src_indices = src_indices.contiguous()
    dst_indices = dst_indices.contiguous()
    src_positions = src_positions.to(torch.int32).contiguous()
    dst_positions = dst_positions.to(torch.int32).contiguous()

    # Launch V copy kernel (N programs, one per token)
    _copy_v_kernel[(N,)](
        v_cache,
        src_indices, dst_indices,
        hidden_size=hidden_size,
        BLOCK_D=BLOCK_D,
    )

    # Launch K copy+rope kernel (N programs, one per token)
    _copy_k_rope_kernel[(N,)](
        k_cache,
        src_indices, dst_indices,
        cos_sin_cache,
        src_positions, dst_positions,
        hidden_size=hidden_size,
        half_rotary=half_rotary,
        cos_sin_stride=rotary_dim,
        BLOCK_D=BLOCK_D,
    )
