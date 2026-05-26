from typing import Tuple

import torch

from .base import BaseOP


def _can_use_flashinfer_norm() -> bool:
    """Check if flashinfer norm kernels are usable on this GPU."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    if not (cap[0] > 7 or (cap[0] == 7 and cap[1] >= 5)):
        return False
    try:
        from flashinfer import rmsnorm  # noqa: F401
        return True
    except (ImportError, RuntimeError):
        return False


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure PyTorch RMSNorm fallback."""
    orig_dtype = x.dtype
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight).to(orig_dtype)


def _torch_rmsnorm_inplace(x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    """In-place pure PyTorch RMSNorm fallback."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x.mul_(torch.rsqrt(variance + eps))
    x.mul_(weight)


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        if _can_use_flashinfer_norm():
            from flashinfer import rmsnorm
            self._rmsnorm = rmsnorm
        else:
            self._rmsnorm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._rmsnorm is not None:
            return self._rmsnorm(x, self.weight, self.eps)
        return _torch_rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if self._rmsnorm is not None:
            self._rmsnorm(x, self.weight, self.eps, out=x)
        else:
            _torch_rmsnorm_inplace(x, self.weight, self.eps)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        if _can_use_flashinfer_norm():
            from flashinfer import fused_add_rmsnorm, rmsnorm
            self._rmsnorm = rmsnorm
            self._fused_add_rmsnorm = fused_add_rmsnorm
        else:
            self._rmsnorm = None
            self._fused_add_rmsnorm = None

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            if self._rmsnorm is not None:
                return self._rmsnorm(x, self.weight, self.eps), x
            return _torch_rmsnorm(x, self.weight, self.eps), x
        if self._fused_add_rmsnorm is not None:
            self._fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
        # Fallback: fused add + rmsnorm
        residual = residual + x
        x = _torch_rmsnorm(residual, self.weight, self.eps)
        return x, residual
