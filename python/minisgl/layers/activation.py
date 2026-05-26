from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def _can_use_flashinfer() -> bool:
    """Check if flashinfer kernels are usable on this GPU."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] > 7 or (cap[0] == 7 and cap[1] >= 5)


_USE_FLASHINFER = None


def _check_flashinfer():
    global _USE_FLASHINFER
    if _USE_FLASHINFER is None:
        if not _can_use_flashinfer():
            _USE_FLASHINFER = False
            return False
        try:
            from flashinfer import silu_and_mul  # noqa: F401
            _USE_FLASHINFER = True
        except (ImportError, RuntimeError):
            _USE_FLASHINFER = False
    return _USE_FLASHINFER


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if _check_flashinfer():
        from flashinfer import silu_and_mul as _silu_and_mul
        return _silu_and_mul(x, out=out)
    # Fallback: x is [*, 2*d], split into gate and up
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    result = torch.nn.functional.silu(gate) * up
    if out is not None:
        out.copy_(result)
        return out
    return result


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if _check_flashinfer():
        from flashinfer import gelu_and_mul as _gelu_and_mul
        return _gelu_and_mul(x, out=out)
    # Fallback
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    result = torch.nn.functional.gelu(gate) * up
    if out is not None:
        out.copy_(result)
        return out
    return result


__all__ = ["silu_and_mul", "gelu_and_mul"]
