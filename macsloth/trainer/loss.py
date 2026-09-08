"""Masked next-token CE matching mlx-lm ``default_loss``."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn


def masked_lm_loss(
    model: nn.Module,
    batch: mx.array,
    lengths: mx.array,
) -> tuple[mx.array, mx.array]:
    """Mean token CE with the same length mask as mlx-lm ``default_loss``.

    ``batch`` is token ids ``(B, T)``. ``lengths`` is ``(B, 2)`` of
    ``(prompt_offset, sequence_length)``. Returns ``(loss, ntoks)``.
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ce = mx.fast.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks
    return ce, ntoks
