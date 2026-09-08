"""Given/When/Then tests for the native SFT loop."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx
import mlx.optimizers as optim
from macsloth.trainer import TrainingConfig, run_sft
from macsloth.trainer.errors import InvalidTrainingConfigError
from macsloth.trainer.loop import SftLoopResult
from mlx import nn
from mlx_lm.tuner.datasets import CacheDataset


class _TinyLM(nn.Module):
    """Embedding + Linear stand-in for a causal LM head."""

    def __init__(self, vocab_size: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = [nn.Linear(hidden, hidden, bias=False)]
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        hidden = self.embed(tokens)
        hidden = self.layers[0](hidden)
        return self.head(hidden)


class _TokenRows:
    def __init__(self, rows: tuple[tuple[int, ...], ...]) -> None:
        self._rows = rows

    def process(self, row: tuple[int, ...]) -> tuple[list[int], int]:
        return (list(row), 0)

    def __getitem__(self, idx: int) -> tuple[int, ...]:
        return self._rows[idx]

    def __len__(self) -> int:
        return len(self._rows)


def _cached_rows(batch_size: int, seq_len: int, vocab_size: int) -> CacheDataset:
    mx.random.seed(0)
    table = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    mx.eval(table)
    rows = tuple(
        tuple(int(table[row, col].item()) for col in range(seq_len))
        for row in range(batch_size)
    )
    return CacheDataset(_TokenRows(rows))


def test_run_sft_loss_is_finite_after_two_iters() -> None:
    vocab_size = 16
    hidden = 8
    batch_size = 2
    seq_len = 8
    mx.random.seed(0)
    model = _TinyLM(vocab_size, hidden)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=1e-4)
    dataset = _cached_rows(batch_size, seq_len, vocab_size)
    config = TrainingConfig(
        batch_size=batch_size,
        iters=2,
        max_seq_length=seq_len,
        grad_checkpoint=False,
    )

    result = run_sft(model, optimizer, dataset, config)

    assert isinstance(result, SftLoopResult)
    assert math.isfinite(result.last_loss)
    assert result.peak_memory_bytes >= 0


def test_run_sft_rejects_grad_accumulation_above_one() -> None:
    vocab_size = 16
    model = _TinyLM(vocab_size, 8)
    mx.eval(model.parameters())
    config = TrainingConfig(
        batch_size=2,
        iters=2,
        max_seq_length=8,
        grad_checkpoint=False,
        grad_accumulation_steps=2,
    )

    with pytest.raises(InvalidTrainingConfigError) as exc_info:
        run_sft(
            model,
            optim.Adam(learning_rate=1e-4),
            _cached_rows(2, 8, vocab_size),
            config,
        )

    assert exc_info.value.field == "grad_accumulation_steps"
    assert exc_info.value.value == 2


def test_run_sft_checkpoints_layers0_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[nn.Module] = []

    def fake_grad_checkpoint(layer: nn.Module) -> None:
        seen.append(layer)

    monkeypatch.setattr(
        "macsloth.trainer.loop.grad_checkpoint",
        fake_grad_checkpoint,
    )
    vocab_size = 16
    mx.random.seed(1)
    model = _TinyLM(vocab_size, 8)
    mx.eval(model.parameters())
    config = TrainingConfig(
        batch_size=2,
        iters=2,
        max_seq_length=8,
        grad_checkpoint=True,
    )

    run_sft(
        model,
        optim.Adam(learning_rate=1e-4),
        _cached_rows(2, 8, vocab_size),
        config,
    )

    assert seen == [model.layers[0]]
