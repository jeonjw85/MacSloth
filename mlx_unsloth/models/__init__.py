"""Model loading and LoRA attachment."""

from mlx_unsloth.export.exporter import (
    push_to_hub_merged,
    save_pretrained_gguf,
    save_pretrained_merged,
)
from mlx_unsloth.models.loader import FastLanguageModel

FastLanguageModel.save_pretrained_merged = staticmethod(save_pretrained_merged)
FastLanguageModel.save_pretrained_gguf = staticmethod(save_pretrained_gguf)
FastLanguageModel.push_to_hub_merged = staticmethod(push_to_hub_merged)

__all__ = ["FastLanguageModel"]
