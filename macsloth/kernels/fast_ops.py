"""Thin calls into mlx.core.fast (C++/Metal kernels shipped with MLX)."""

from __future__ import annotations

import mlx.core as mx


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    """RMSNorm via ``mx.fast.rms_norm``."""
    return mx.fast.rms_norm(x, weight, eps)


def rope(  # noqa: PLR0913
    x: mx.array,
    dims: int,
    *,
    traditional: bool,
    base: float,
    scale: float,
    offset: int,
) -> mx.array:
    """RoPE via ``mx.fast.rope``."""
    return mx.fast.rope(
        x,
        dims,
        traditional=traditional,
        base=base,
        scale=scale,
        offset=offset,
    )


def cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    """Fused cross-entropy via ``mx.fast.cross_entropy`` (Metal falls back)."""
    return mx.fast.cross_entropy(logits, targets)
