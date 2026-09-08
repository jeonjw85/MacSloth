"""Given/When/Then tests for SFTTrainer wrapping run_sft."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("mlx_lm")

import mlx.core as mx
from mlx.optimizers import Adam, AdamW
from mlx_lm.tuner.datasets import CacheDataset
from mlx_unsloth.models.loader import LoraSpec
from mlx_unsloth.trainer import SFTTrainer, TrainingConfig
from mlx_unsloth.trainer.errors import (
    EmptyTrainSetError,
    InsufficientUnifiedMemoryError,
    InvalidTrainingConfigError,
    MetalUnavailableError,
    MissingLoraSpecError,
    NotADatasetDirError,
    UnknownOptimizerError,
)
from mlx_unsloth.trainer.loop import SftLoopResult
from mlx_unsloth.trainer.lora import _write_adapter_weights

if TYPE_CHECKING:
    from pathlib import Path


class _Sized:
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size


def _sft_result(peak_memory_bytes: int = 0) -> SftLoopResult:
    return SftLoopResult(last_loss=0.0, peak_memory_bytes=peak_memory_bytes)


@pytest.fixture(autouse=True)
def _stub_adapter_weight_save(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora._write_adapter_weights",
        lambda _model, _adapter_dir: None,
    )


def test_train_calls_run_sft_with_cached_train_set_and_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_set = _Sized(8)
    valid_set = _Sized(4)
    model = SimpleNamespace(
        lora_spec=LoraSpec(
            num_layers=2,
            rank=16,
            scale=32.0,
            dropout=0.05,
            keys=("self_attn.q_proj",),
        )
    )
    tokenizer = SimpleNamespace()

    class _Load:
        config: TrainingConfig | None = None
        path: Path | None = None

    loaded = _Load()

    def fake_load_local_dataset(
        data_path: Path,
        tokenizer_arg: SimpleNamespace,
        config: TrainingConfig,
    ) -> tuple[_Sized, _Sized, tuple[()]]:
        loaded.path = data_path
        loaded.config = config
        assert tokenizer_arg is tokenizer
        return train_set, valid_set, ()

    class _Capture:
        model: SimpleNamespace | None = None
        optimizer: Adam | AdamW | None = None
        train_dataset: CacheDataset | None = None
        config: TrainingConfig | None = None

    captured = _Capture()

    def fake_run_sft(
        model: SimpleNamespace,
        optimizer: Adam | AdamW,
        train_dataset: CacheDataset,
        config: TrainingConfig,
    ) -> SftLoopResult:
        captured.model = model
        captured.optimizer = optimizer
        captured.train_dataset = train_dataset
        captured.config = config
        return _sft_result()

    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        fake_load_local_dataset,
    )
    monkeypatch.setattr("mlx_unsloth.trainer.lora.run_sft", fake_run_sft)

    config = TrainingConfig(
        batch_size=4,
        iters=20,
        learning_rate=2e-5,
        max_seq_length=512,
        adapter_path=str(tmp_path / "adapters"),
        mask_prompt=True,
        clear_cache_threshold=0,
    )
    trainer = SFTTrainer(model, tokenizer, tmp_path, config)
    trainer.train()

    assert loaded.path == tmp_path
    assert loaded.config is config
    assert captured.model is model
    assert isinstance(captured.optimizer, Adam)
    assert isinstance(captured.train_dataset, CacheDataset)
    assert len(captured.train_dataset) == 8
    assert captured.config is config
    assert captured.config.batch_size == 4
    assert captured.config.iters == 20
    assert captured.config.max_seq_length == 512
    assert captured.config.clear_cache_threshold == 0
    assert (tmp_path / "adapters").is_dir()
    written = json.loads(
        (tmp_path / "adapters" / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert written["fine_tune_type"] == "lora"
    assert written["num_layers"] == 2
    assert written["lora_parameters"]["rank"] == 16
    assert written["lora_parameters"]["keys"] == ["self_attn.q_proj"]


def test_train_writes_adapter_weights_after_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[Path] = []

    def fake_write(_model: SimpleNamespace, adapter_dir: Path) -> None:
        saved.append(adapter_dir)

    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora._write_adapter_weights",
        fake_write,
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: (_Sized(8), [], []),
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.run_sft",
        lambda *args, **kwargs: _sft_result(),
    )
    adapter = tmp_path / "adapters"
    SFTTrainer(
        SimpleNamespace(
            lora_spec=LoraSpec(
                num_layers=2,
                rank=8,
                scale=16.0,
                dropout=0.0,
                keys=("self_attn.q_proj",),
            )
        ),
        SimpleNamespace(),
        tmp_path,
        TrainingConfig(adapter_path=str(adapter), iters=1),
    ).train()

    assert saved == [adapter]


def test_write_adapter_weights_saves_trainable_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, dict[str, mx.array]]] = []

    def fake_save(path: str, weights: dict[str, mx.array]) -> None:
        captured.append((path, weights))

    monkeypatch.setattr("mlx_unsloth.trainer.lora.mx.save_safetensors", fake_save)

    class _Host:
        def trainable_parameters(self) -> dict[str, mx.array]:
            return {"lora_a": mx.ones((2, 2))}

    _write_adapter_weights(_Host(), tmp_path)

    assert captured[0][0] == str(tmp_path / "adapters.safetensors")
    assert list(captured[0][1]) == ["lora_a"]


def test_train_calls_run_sft_when_valid_split_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Capture:
        train_dataset: CacheDataset | None = None

    captured = _Capture()

    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: (_Sized(8), [], []),
    )

    def fake_run_sft(
        model: SimpleNamespace,
        optimizer: Adam | AdamW,
        train_dataset: CacheDataset,
        config: TrainingConfig,
    ) -> SftLoopResult:
        captured.train_dataset = train_dataset
        return _sft_result()

    monkeypatch.setattr("mlx_unsloth.trainer.lora.run_sft", fake_run_sft)

    SFTTrainer(
        SimpleNamespace(
            lora_spec=LoraSpec(
                num_layers=2,
                rank=16,
                scale=32.0,
                dropout=0.05,
                keys=("self_attn.q_proj",),
            )
        ),
        SimpleNamespace(),
        tmp_path,
    ).train()

    assert isinstance(captured.train_dataset, CacheDataset)
    assert len(captured.train_dataset) == 8


def test_adamw_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Capture:
        optimizer: Adam | AdamW | None = None

    captured = _Capture()
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: (_Sized(8), [], []),
    )

    def fake_run_sft(
        model: SimpleNamespace,
        optimizer: Adam | AdamW,
        train_dataset: CacheDataset,
        config: TrainingConfig,
    ) -> SftLoopResult:
        captured.optimizer = optimizer
        return _sft_result()

    monkeypatch.setattr("mlx_unsloth.trainer.lora.run_sft", fake_run_sft)

    SFTTrainer(
        SimpleNamespace(
            lora_spec=LoraSpec(
                num_layers=2,
                rank=16,
                scale=32.0,
                dropout=0.05,
                keys=("self_attn.q_proj",),
            )
        ),
        SimpleNamespace(),
        tmp_path,
        TrainingConfig(optimizer="adamw"),
    ).train()

    assert isinstance(captured.optimizer, AdamW)


def test_missing_lora_spec_raises_before_dataset_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []

    def fail_if_called(
        data_path: Path,
        tokenizer_arg: SimpleNamespace,
        config: TrainingConfig,
    ) -> tuple[_Sized, _Sized, tuple[()]]:
        called.append(True)
        return ([], [], ())

    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        fail_if_called,
    )

    with pytest.raises(MissingLoraSpecError) as exc_info:
        SFTTrainer(SimpleNamespace(), SimpleNamespace(), tmp_path).train()

    assert str(exc_info.value) == "get_peft_model is required before train"
    assert called == []


def test_empty_train_set_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: ([], [], []),
    )

    with pytest.raises(EmptyTrainSetError):
        SFTTrainer(
            SimpleNamespace(
                lora_spec=LoraSpec(
                    num_layers=2,
                    rank=16,
                    scale=32.0,
                    dropout=0.05,
                    keys=("self_attn.q_proj",),
                )
            ),
            SimpleNamespace(),
            tmp_path,
        ).train()


def test_training_config_grad_checkpoint_defaults_false() -> None:
    config = TrainingConfig()
    assert config.grad_checkpoint is False
    assert config.clear_cache_threshold == 0


def test_train_forwards_default_grad_checkpoint_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Capture:
        config: TrainingConfig | None = None

    captured = _Capture()
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: (_Sized(8), [], []),
    )

    def fake_run_sft(
        model: SimpleNamespace,
        optimizer: Adam | AdamW,
        train_dataset: CacheDataset,
        config: TrainingConfig,
    ) -> SftLoopResult:
        captured.config = config
        return _sft_result()

    monkeypatch.setattr("mlx_unsloth.trainer.lora.run_sft", fake_run_sft)

    SFTTrainer(
        SimpleNamespace(
            lora_spec=LoraSpec(
                num_layers=2,
                rank=16,
                scale=32.0,
                dropout=0.05,
                keys=("self_attn.q_proj",),
            )
        ),
        SimpleNamespace(),
        tmp_path,
    ).train()

    assert captured.config is not None
    assert captured.config.grad_checkpoint is False


def test_train_forwards_grad_checkpoint_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Capture:
        config: TrainingConfig | None = None

    captured = _Capture()
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: (_Sized(8), [], []),
    )

    def fake_run_sft(
        model: SimpleNamespace,
        optimizer: Adam | AdamW,
        train_dataset: CacheDataset,
        config: TrainingConfig,
    ) -> SftLoopResult:
        captured.config = config
        return _sft_result()

    monkeypatch.setattr("mlx_unsloth.trainer.lora.run_sft", fake_run_sft)

    SFTTrainer(
        SimpleNamespace(
            lora_spec=LoraSpec(
                num_layers=2,
                rank=16,
                scale=32.0,
                dropout=0.05,
                keys=("self_attn.q_proj",),
            )
        ),
        SimpleNamespace(),
        tmp_path,
        TrainingConfig(grad_checkpoint=False),
    ).train()

    assert captured.config is not None
    assert captured.config.grad_checkpoint is False


def test_unknown_optimizer_raises() -> None:
    with pytest.raises(UnknownOptimizerError):
        TrainingConfig(optimizer="sgd")


def test_file_instead_of_dir_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "train.jsonl"
    file_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(NotADatasetDirError):
        SFTTrainer(SimpleNamespace(), SimpleNamespace(), file_path).train()


def test_invalid_iters_raises() -> None:
    with pytest.raises(InvalidTrainingConfigError):
        TrainingConfig(iters=0)


def _lora_model() -> SimpleNamespace:
    return SimpleNamespace(
        lora_spec=LoraSpec(
            num_layers=2,
            rank=16,
            scale=32.0,
            dropout=0.05,
            keys=("self_attn.q_proj",),
        )
    )


def test_min_working_set_bytes_zero_raises() -> None:
    with pytest.raises(InvalidTrainingConfigError) as exc_info:
        TrainingConfig(min_working_set_bytes=0)

    assert exc_info.value.field == "min_working_set_bytes"
    assert exc_info.value.value == 0


def test_train_raises_metal_unavailable_when_metal_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []

    def fail_if_called(
        data_path: Path,
        tokenizer_arg: SimpleNamespace,
        config: TrainingConfig,
    ) -> tuple[_Sized, _Sized, tuple[()]]:
        called.append(True)
        return ([], [], ())

    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        fail_if_called,
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.is_available",
        lambda: False,
    )

    with pytest.raises(MetalUnavailableError) as exc_info:
        SFTTrainer(
            _lora_model(),
            SimpleNamespace(),
            tmp_path,
            TrainingConfig(min_working_set_bytes=1),
        ).train()

    assert str(exc_info.value) == "Metal is required when min_working_set_bytes is set"
    assert called == []


def test_train_raises_insufficient_unified_memory_when_working_set_too_small(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.device_info",
        lambda: {"max_recommended_working_set_size": 10},
    )

    with pytest.raises(InsufficientUnifiedMemoryError) as exc_info:
        SFTTrainer(
            _lora_model(),
            SimpleNamespace(),
            tmp_path,
            TrainingConfig(min_working_set_bytes=20),
        ).train()

    assert exc_info.value.required_bytes == 20
    assert exc_info.value.available_bytes == 10
    assert str(exc_info.value) == "unified memory working set 10 < required 20"


def test_train_proceeds_when_working_set_meets_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained: list[bool] = []
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.device_info",
        lambda: {"max_recommended_working_set_size": 20},
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: (_Sized(8), [], []),
    )

    def fake_run_sft(
        model: SimpleNamespace,
        optimizer: Adam | AdamW,
        train_dataset: CacheDataset,
        config: TrainingConfig,
    ) -> SftLoopResult:
        trained.append(True)
        return _sft_result()

    monkeypatch.setattr("mlx_unsloth.trainer.lora.run_sft", fake_run_sft)

    SFTTrainer(
        _lora_model(),
        SimpleNamespace(),
        tmp_path,
        TrainingConfig(min_working_set_bytes=10),
    ).train()

    assert trained == [True]


def test_train_raises_metal_unavailable_when_device_info_key_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.device_info",
        dict,
    )

    with pytest.raises(MetalUnavailableError) as exc_info:
        SFTTrainer(
            _lora_model(),
            SimpleNamespace(),
            tmp_path,
            TrainingConfig(min_working_set_bytes=1),
        ).train()

    assert str(exc_info.value) == "Metal is required when min_working_set_bytes is set"
    assert isinstance(exc_info.value.__cause__, KeyError)


def test_train_raises_metal_unavailable_when_device_info_not_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.device_info",
        lambda: None,
    )

    with pytest.raises(MetalUnavailableError) as exc_info:
        SFTTrainer(
            _lora_model(),
            SimpleNamespace(),
            tmp_path,
            TrainingConfig(min_working_set_bytes=1),
        ).train()

    assert str(exc_info.value) == "Metal is required when min_working_set_bytes is set"
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_train_records_peak_memory_bytes_from_run_sft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def fake_reset_peak_memory() -> None:
        order.append("reset")

    def fake_run_sft(
        model: SimpleNamespace,
        optimizer: Adam | AdamW,
        train_dataset: CacheDataset,
        config: TrainingConfig,
    ) -> SftLoopResult:
        order.append("run_sft")
        return _sft_result(peak_memory_bytes=4242)

    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: (_Sized(8), [], []),
    )
    monkeypatch.setattr("mlx_unsloth.trainer.lora.run_sft", fake_run_sft)
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.reset_peak_memory",
        fake_reset_peak_memory,
    )

    trainer = SFTTrainer(_lora_model(), SimpleNamespace(), tmp_path)
    assert trainer.last_peak_memory_bytes is None
    trainer.train()

    assert trainer.last_peak_memory_bytes == 4242
    assert order == ["reset", "run_sft"]


def test_train_sets_wired_limit_when_metal_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits: list[int] = []
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.metal.device_info",
        lambda: {"max_recommended_working_set_size": 99},
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.set_wired_limit",
        limits.append,
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: (_Sized(8), [], []),
    )
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.run_sft",
        lambda **kwargs: _sft_result(),
    )

    SFTTrainer(_lora_model(), SimpleNamespace(), tmp_path).train()

    assert limits == [99]


def test_train_leaves_peak_memory_none_when_run_sft_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run_sft(
        model: SimpleNamespace,
        optimizer: Adam | AdamW,
        train_dataset: CacheDataset,
        config: TrainingConfig,
    ) -> SftLoopResult:
        raise RuntimeError

    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.load_local_dataset",
        lambda path, tokenizer, config: (_Sized(8), [], []),
    )
    monkeypatch.setattr("mlx_unsloth.trainer.lora.run_sft", fail_run_sft)
    monkeypatch.setattr(
        "mlx_unsloth.trainer.lora.mx.reset_peak_memory",
        lambda: None,
    )

    trainer = SFTTrainer(_lora_model(), SimpleNamespace(), tmp_path)
    with pytest.raises(RuntimeError):
        trainer.train()

    assert trainer.last_peak_memory_bytes is None
