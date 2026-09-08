"""Given/When/Then tests for masked LM loss vs mlx-lm default_loss."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx
from mlx import nn
from mlx_lm.tuner.trainer import default_loss
from mlx_unsloth.trainer.loss import masked_lm_loss


class _TinyLM(nn.Module):
    """Embedding + Linear stand-in for a causal LM head."""

    def __init__(self, vocab_size: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.head(self.embed(tokens))


def test_masked_lm_loss_matches_default_loss() -> None:
    vocab_size = 16
    hidden = 8
    batch_size = 4
    seq_len = 8
    mx.random.seed(0)
    model = _TinyLM(vocab_size, hidden)
    mx.eval(model.parameters())
    batch = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    lengths = mx.array([[0, 8], [1, 7], [2, 6], [0, 5]], dtype=mx.int32)

    ours_ce, ours_ntoks = masked_lm_loss(model, batch, lengths)
    ref_ce, ref_ntoks = default_loss(model, batch, lengths)
    mx.eval(ours_ce, ours_ntoks, ref_ce, ref_ntoks)

    assert bool(mx.allclose(ours_ce, ref_ce, atol=1e-4).item())
    assert bool(mx.array_equal(ours_ntoks, ref_ntoks).item())
