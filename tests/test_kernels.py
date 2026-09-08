"""Given/When/Then tests for mlx.fast wrappers and the Metal LoRA kernel."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")

import mlx.core as mx
from mlx_unsloth.kernels import (
    cross_entropy,
    lora_delta,
    lora_linear_forward,
    rms_norm,
    rope,
)


def test_lora_delta_matches_matmul() -> None:
    mx.random.seed(0)
    x = mx.random.normal((4, 8)).astype(mx.float32)
    lora_a = mx.random.normal((8, 2)).astype(mx.float32)
    lora_b = mx.random.normal((2, 6)).astype(mx.float32)
    scale = 0.5
    got = lora_delta(x, lora_a, lora_b, scale)
    ref = (scale * (x @ lora_a @ lora_b)).astype(mx.float32)
    mx.eval(got, ref)
    assert bool(mx.allclose(got, ref, atol=1e-5, rtol=1e-5).item())


def test_lora_linear_forward_matches_base_plus_delta() -> None:
    mx.random.seed(4)
    x = mx.random.normal((32, 64)).astype(mx.float32)
    weight = mx.random.normal((50, 64)).astype(mx.float32)
    lora_a = mx.random.normal((64, 8)).astype(mx.float32)
    lora_b = mx.random.normal((8, 50)).astype(mx.float32)
    scale = 0.5
    base_out = x @ weight.T
    got = lora_linear_forward(x, base_out, lora_a, lora_b, scale)
    ref = base_out + (scale * (x @ lora_a @ lora_b)).astype(mx.float32)
    mx.eval(got, ref)
    assert bool(mx.allclose(got, ref, atol=1e-5, rtol=1e-5).item())


def test_lora_linear_forward_reshapes_leading_dims() -> None:
    mx.random.seed(5)
    x = mx.random.normal((2, 16, 64)).astype(mx.float32)
    weight = mx.random.normal((50, 64)).astype(mx.float32)
    lora_a = mx.random.normal((64, 8)).astype(mx.float32)
    lora_b = mx.random.normal((8, 50)).astype(mx.float32)
    scale = 0.25
    base_out = x @ weight.T
    got = lora_linear_forward(x, base_out, lora_a, lora_b, scale)
    ref = base_out + (scale * (x @ lora_a @ lora_b)).astype(mx.float32)
    mx.eval(got, ref)
    assert got.shape == (2, 16, 50)
    assert bool(mx.allclose(got, ref, atol=1e-5, rtol=1e-5).item())


def test_lora_delta_grads_match_two_matmul() -> None:
    mx.random.seed(6)
    x = mx.random.normal((32, 64)).astype(mx.float32)
    lora_a = mx.random.normal((64, 8)).astype(mx.float32)
    lora_b = mx.random.normal((8, 50)).astype(mx.float32)
    scale = 0.5

    def naive(adapter_a: mx.array, adapter_b: mx.array) -> mx.array:
        return mx.sum(scale * (x @ adapter_a) @ adapter_b)

    def compiled(adapter_a: mx.array, adapter_b: mx.array) -> mx.array:
        return mx.sum(lora_delta(x, adapter_a, adapter_b, scale))

    _, naive_grads = mx.value_and_grad(naive, argnums=(0, 1))(lora_a, lora_b)
    _, compiled_grads = mx.value_and_grad(compiled, argnums=(0, 1))(lora_a, lora_b)
    mx.eval(*naive_grads, *compiled_grads)
    assert bool(
        mx.allclose(compiled_grads[0], naive_grads[0], atol=1e-5, rtol=1e-5).item()
    )
    assert bool(
        mx.allclose(compiled_grads[1], naive_grads[1], atol=1e-5, rtol=1e-5).item()
    )


def test_lora_linear_forward_grads_match_two_matmul() -> None:
    mx.random.seed(7)
    x = mx.random.normal((32, 64)).astype(mx.float32)
    weight = mx.random.normal((50, 64)).astype(mx.float32)
    lora_a = mx.random.normal((64, 8)).astype(mx.float32)
    lora_b = mx.random.normal((8, 50)).astype(mx.float32)
    scale = 0.5
    base_out = x @ weight.T

    def naive(adapter_a: mx.array, adapter_b: mx.array) -> mx.array:
        return mx.sum(base_out + scale * (x @ adapter_a) @ adapter_b)

    def compiled(adapter_a: mx.array, adapter_b: mx.array) -> mx.array:
        return mx.sum(lora_linear_forward(x, base_out, adapter_a, adapter_b, scale))

    _, naive_grads = mx.value_and_grad(naive, argnums=(0, 1))(lora_a, lora_b)
    _, compiled_grads = mx.value_and_grad(compiled, argnums=(0, 1))(lora_a, lora_b)
    mx.eval(*naive_grads, *compiled_grads)
    assert bool(
        mx.allclose(compiled_grads[0], naive_grads[0], atol=1e-5, rtol=1e-5).item()
    )
    assert bool(
        mx.allclose(compiled_grads[1], naive_grads[1], atol=1e-5, rtol=1e-5).item()
    )


def test_lora_delta_reshapes_leading_dims() -> None:
    mx.random.seed(1)
    x = mx.random.normal((2, 3, 8)).astype(mx.float32)
    lora_a = mx.random.normal((8, 2)).astype(mx.float32)
    lora_b = mx.random.normal((2, 6)).astype(mx.float32)
    got = lora_delta(x, lora_a, lora_b, 1.0)
    flat = x.reshape((6, 8))
    ref = (flat @ lora_a @ lora_b).reshape((2, 3, 6))
    mx.eval(got, ref)
    assert got.shape == (2, 3, 6)
    assert bool(mx.allclose(got, ref, atol=1e-5, rtol=1e-5).item())


def test_rms_norm_matches_manual() -> None:
    mx.random.seed(2)
    x = mx.random.normal((2, 4, 8)).astype(mx.float32)
    weight = mx.ones((8,), dtype=mx.float32)
    eps = 1e-5
    got = rms_norm(x, weight, eps)
    var = mx.mean(x * x, axis=-1, keepdims=True)
    ref = x * mx.rsqrt(var + eps) * weight
    mx.eval(got, ref)
    assert bool(mx.allclose(got, ref, atol=1e-5, rtol=1e-5).item())


def test_rope_matches_mx_fast() -> None:
    mx.random.seed(3)
    x = mx.random.normal((1, 2, 4, 8)).astype(mx.float32)
    got = rope(x, 8, traditional=True, base=10000.0, scale=1.0, offset=0)
    ref = mx.fast.rope(x, 8, traditional=True, base=10000.0, scale=1.0, offset=0)
    mx.eval(got, ref)
    assert bool(mx.allclose(got, ref, atol=1e-5, rtol=1e-5).item())


def test_cross_entropy_finite() -> None:
    logits = mx.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=mx.float32)
    targets = mx.array([2, 0], dtype=mx.int32)
    loss = cross_entropy(logits, targets)
    mx.eval(loss)
    assert loss.shape == (2,)
    assert bool(mx.isfinite(loss).all().item())
