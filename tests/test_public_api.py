"""Given/When/Then tests for the Unsloth-shaped package exports."""

from __future__ import annotations

import macsloth
from macsloth import (
    FastLanguageModel,
    SFTTrainer,
    TrainingConfig,
    save_pretrained_merged,
    to_mlx_jsonl,
)


def test_package_exports_unsloth_entry_points() -> None:
    assert macsloth.FastLanguageModel is FastLanguageModel
    assert macsloth.SFTTrainer is SFTTrainer
    assert macsloth.TrainingConfig is TrainingConfig
    assert macsloth.to_mlx_jsonl is to_mlx_jsonl
    assert macsloth.save_pretrained_merged is save_pretrained_merged
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
        assert name in macsloth.__all__
