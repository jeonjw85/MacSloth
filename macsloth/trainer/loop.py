"""Native SFT loop (does not call mlx_lm.tuner.trainer.train)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx_lm.tuner.trainer import grad_checkpoint, iterate_batches

from macsloth.trainer.errors import InvalidTrainingConfigError
from macsloth.trainer.loss import masked_lm_loss

if TYPE_CHECKING:
    from collections.abc import Callable

    from mlx_lm.tuner.datasets import CacheDataset

    from macsloth.trainer.lora import TrainingConfig


@dataclass(frozen=True, slots=True)
class SftLoopResult:
    """Loss and peak memory after ``run_sft``."""

    last_loss: float
    peak_memory_bytes: int


def run_sft(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataset: CacheDataset,
    config: TrainingConfig,
    *,
    after_step: Callable[[int], None] | None = None,
) -> SftLoopResult:
    """Run ``config.iters`` SFT steps with ``masked_lm_loss``.

    Skips validation. Gradient accumulation other than 1 is rejected.
    ``after_step`` is invoked with the 1-based iteration after ``mx.eval``.
    """
    if config.grad_accumulation_steps != 1:
        raise InvalidTrainingConfigError(
            field="grad_accumulation_steps",
            value=config.grad_accumulation_steps,
        )
    if config.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    loss_value_and_grad = nn.value_and_grad(model, masked_lm_loss)
    state = [model.state, optimizer.state, mx.random.state]
    model.train()
    loss_arr = mx.array(0.0)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch: tuple[mx.array, mx.array]) -> tuple[mx.array, mx.array]:
        (loss_inner, ntoks_inner), grad = loss_value_and_grad(model, *batch)
        optimizer.update(model, grad)
        return loss_inner, ntoks_inner

    batches = tuple(
        islice(
            iterate_batches(
                dataset=train_dataset,
                batch_size=config.batch_size,
                max_seq_length=config.max_seq_length,
                loop=True,
            ),
            config.iters,
        )
    )
    for batch in batches:
        mx.eval(*batch)
    for iteration, batch in enumerate(batches, start=1):
        loss_arr, ntoks = step(batch)
        mx.eval(state, loss_arr, ntoks)
        if mx.get_cache_memory() > config.clear_cache_threshold:
            mx.clear_cache()
        if after_step is not None:
            after_step(iteration)
    return SftLoopResult(
        last_loss=float(loss_arr.item()),
        peak_memory_bytes=int(mx.get_peak_memory()),
    )
