"""Unsloth-shaped SFT trainer on top of mlx-lm."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from typing_extensions import assert_never

from mlx_unsloth.trainer.errors import (
    EmptyTrainSetError,
    InsufficientUnifiedMemoryError,
    InvalidTrainingConfigError,
    MetalUnavailableError,
    MissingLoraSpecError,
    NotADatasetDirError,
    UnknownOptimizerError,
)
from mlx_unsloth.trainer.loop import run_sft

if TYPE_CHECKING:
    from mlx.nn import Module
    from mlx_lm.tokenizer_utils import TokenizerWrapper


class OptimizerName(str, Enum):
    """Optimizers this wrapper constructs."""

    ADAM = "adam"
    ADAMW = "adamw"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters forwarded to `run_sft` and the optimizer.

    `grad_checkpoint` defaults to False (faster; more memory). Set True
    to checkpoint activations.
    """

    batch_size: int = 4
    iters: int = 100
    learning_rate: float = 1e-5
    max_seq_length: int = 2048
    adapter_path: str = "adapters"
    mask_prompt: bool = False
    grad_checkpoint: bool = False
    val_batches: int = 25
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    grad_accumulation_steps: int = 1
    clear_cache_threshold: int = 0
    seed: int = 0
    optimizer: str = "adam"
    min_working_set_bytes: int | None = None

    def __post_init__(self) -> None:
        """Reject values mlx-lm or the optimizer cannot run with."""
        if self.batch_size < 1:
            raise InvalidTrainingConfigError(field="batch_size", value=self.batch_size)
        if self.iters < 1:
            raise InvalidTrainingConfigError(field="iters", value=self.iters)
        if self.learning_rate <= 0:
            raise InvalidTrainingConfigError(
                field="learning_rate", value=self.learning_rate
            )
        if self.min_working_set_bytes is not None and self.min_working_set_bytes < 1:
            raise InvalidTrainingConfigError(
                field="min_working_set_bytes", value=self.min_working_set_bytes
            )
        try:
            OptimizerName(self.optimizer)
        except ValueError:
            raise UnknownOptimizerError(name=self.optimizer) from None


def _write_adapter_config(model: Module, adapter_dir: Path) -> None:
    spec = getattr(model, "lora_spec", None)
    if spec is None:
        return
    payload = {
        "fine_tune_type": "lora",
        "num_layers": spec.num_layers,
        "lora_parameters": {
            "rank": spec.rank,
            "scale": spec.scale,
            "dropout": spec.dropout,
            "keys": list(spec.keys),
        },
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(payload, indent=4),
        encoding="utf-8",
    )


def _write_adapter_weights(model: Module, adapter_dir: Path) -> None:
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), weights)


def _apply_wired_limit() -> None:
    """Cap Metal wired memory at the device recommended working set."""
    if not mx.metal.is_available():
        return
    try:
        limit = mx.metal.device_info()["max_recommended_working_set_size"]
    except (KeyError, TypeError):
        return
    mx.set_wired_limit(limit)


def _optimizer(name: str, learning_rate: float) -> optim.Optimizer:
    kind = OptimizerName(name)
    match kind:
        case OptimizerName.ADAM:
            return optim.Adam(learning_rate=learning_rate)
        case OptimizerName.ADAMW:
            return optim.AdamW(learning_rate=learning_rate)
        case unreachable:
            assert_never(unreachable)


class SFTTrainer:
    """Load mlx-lm JSONL splits and run LoRA SFT.

    `data` is a directory with `train.jsonl` (and optional `valid.jsonl`)
    as written by `to_mlx_jsonl`.
    """

    def __init__(
        self,
        model: Module,
        tokenizer: TokenizerWrapper,
        data: Path | str,
        config: TrainingConfig | None = None,
    ) -> None:
        """Store the model, tokenizer, data directory, and config."""
        self._model = model
        self._tokenizer = tokenizer
        self._data = Path(data)
        self._config = TrainingConfig() if config is None else config
        self.last_peak_memory_bytes: int | None = None

    def train(self) -> None:
        """Run native SFT and write adapter config under `adapter_path`."""
        if not self._data.is_dir():
            raise NotADatasetDirError(path=self._data)
        if getattr(self._model, "lora_spec", None) is None:
            raise MissingLoraSpecError
        config = self._config
        if config.min_working_set_bytes is not None:
            if not mx.metal.is_available():
                raise MetalUnavailableError
            try:
                available = mx.metal.device_info()["max_recommended_working_set_size"]
            except (KeyError, TypeError) as exc:
                raise MetalUnavailableError from exc
            if available < config.min_working_set_bytes:
                raise InsufficientUnifiedMemoryError(
                    required_bytes=config.min_working_set_bytes,
                    available_bytes=available,
                )
        mx.random.seed(config.seed)
        train_set, _valid_set, _test_set = load_local_dataset(
            self._data,
            self._tokenizer,
            config,
        )
        if len(train_set) == 0:
            raise EmptyTrainSetError(path=self._data)
        adapter_dir = Path(config.adapter_path)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        _write_adapter_config(self._model, adapter_dir)
        mx.reset_peak_memory()
        _apply_wired_limit()
        result = run_sft(
            model=self._model,
            optimizer=_optimizer(config.optimizer, config.learning_rate),
            train_dataset=CacheDataset(train_set),
            config=config,
        )
        _write_adapter_weights(self._model, adapter_dir)
        self.last_peak_memory_bytes = result.peak_memory_bytes
