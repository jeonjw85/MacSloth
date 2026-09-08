"""LoRA delta on MLX C++ Metal GEMM: scale * (x @ A @ B)."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class LoraDeltaShapeError(Exception):
    """Raised when x, A, B cannot form scale * (x @ A @ B)."""

    def __str__(self) -> str:
        """Return the error message."""
        return "LoRA shapes do not match x @ A @ B"


def _lora_delta_core(
    flat: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    scale: float,
) -> mx.array:
    """GEMM body: ``scale * (flat @ A @ B)``."""
    return mx.multiply(scale, (flat @ lora_a) @ lora_b)


_compiled_lora_delta_core = mx.compile(_lora_delta_core, shapeless=True)


def _lora_linear_vjp_core(
    x: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    cotan: mx.array,
    scale: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """VJP of ``base + scale * (x @ A @ B)`` w.r.t. x, base, A, B."""
    hidden = x @ lora_a
    scaled = mx.multiply(scale, cotan)
    grad_b = hidden.T @ scaled
    grad_hidden = scaled @ lora_b.T
    grad_a = x.T @ grad_hidden
    grad_x = grad_hidden @ lora_a.T
    return grad_x, cotan, grad_a, grad_b


_compiled_lora_linear_vjp = mx.compile(_lora_linear_vjp_core, shapeless=True)


@mx.custom_function
def _lora_linear_custom(
    x: mx.array,
    base_out: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    scale: float,
) -> mx.array:
    """Dropout=0 LoRA add with compiled GEMM forward."""
    delta = _compiled_lora_delta_core(x, lora_a, lora_b, scale).astype(x.dtype)
    return base_out + delta


@_lora_linear_custom.vjp
def _lora_linear_vjp(
    primals: tuple[mx.array, mx.array, mx.array, mx.array, float],
    cotan: mx.array,
    _output: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, float]:
    """Compiled VJP so autograd does not trace through the compiled forward."""
    x, _base_out, lora_a, lora_b, scale = primals
    grad_x, grad_base, grad_a, grad_b = _compiled_lora_linear_vjp(
        x, lora_a, lora_b, cotan, scale
    )
    return grad_x, grad_base, grad_a, grad_b, 0.0


def _lora_gemm_dims(
    x: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
) -> tuple[int, int]:
    """Return ``(K, N)`` after checking ``x @ A @ B`` is well-defined."""
    if x.ndim < 2:  # noqa: PLR2004
        raise LoraDeltaShapeError
    k_dim = int(lora_a.shape[0])
    rank = int(lora_a.shape[1])
    n_dim = int(lora_b.shape[1])
    if int(x.shape[-1]) != k_dim or int(lora_b.shape[0]) != rank:
        raise LoraDeltaShapeError
    return k_dim, n_dim


def lora_delta(
    x: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    scale: float,
) -> mx.array:
    """Return ``scale * (x @ lora_a @ lora_b)`` via MLX Metal GEMM.

    ``x`` is ``(..., K)``, ``lora_a`` is ``(K, R)``, ``lora_b`` is ``(R, N)``.
    Forward stays uncompiled so ``value_and_grad`` uses native matmul VJPs.
    """
    k_dim, n_dim = _lora_gemm_dims(x, lora_a, lora_b)
    prefix = x.shape[:-1]
    flat = x.reshape((-1, k_dim))
    out = _lora_delta_core(flat, lora_a, lora_b, scale).astype(x.dtype)
    return out.reshape(*prefix, n_dim)


def lora_linear_forward(
    x: mx.array,
    base_out: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    scale: float,
) -> mx.array:
    """Return ``base_out + scale * (x @ A @ B)`` on a compiled dropout=0 path.

    ``base_out`` is the frozen linear (or quantized linear) result. This does
    not replace that matmul; it only fuses the LoRA delta.
    """
    k_dim, n_dim = _lora_gemm_dims(x, lora_a, lora_b)
    prefix = x.shape[:-1]
    flat = x.reshape((-1, k_dim))
    base_flat = base_out.reshape((-1, n_dim))
    out = _lora_linear_custom(flat, base_flat, lora_a, lora_b, scale)
    return out.reshape(*prefix, n_dim)
