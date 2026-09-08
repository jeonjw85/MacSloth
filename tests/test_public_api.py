"""Given/When/Then tests for the Unsloth-shaped package exports."""

from __future__ import annotations

import mlx_unsloth
from mlx_unsloth import (
    FastLanguageModel,
    SFTTrainer,
    TrainingConfig,
    save_pretrained_merged,
    to_mlx_jsonl,
)


def test_package_exports_unsloth_entry_points() -> None:
    assert mlx_unsloth.FastLanguageModel is FastLanguageModel
    assert mlx_unsloth.SFTTrainer is SFTTrainer
    assert mlx_unsloth.TrainingConfig is TrainingConfig
    assert mlx_unsloth.to_mlx_jsonl is to_mlx_jsonl
    assert mlx_unsloth.save_pretrained_merged is save_pretrained_merged
    assert FastLanguageModel.save_pretrained_merged is save_pretrained_merged
    for name in (
        "FastLanguageModel",
        "SFTTrainer",
        "TrainingConfig",
        "to_mlx_jsonl",
        "save_pretrained_merged",
        "save_pretrained_gguf",
        "push_to_hub_merged",
        "write_ollama_modelfile",
    ):
        assert name in mlx_unsloth.__all__
