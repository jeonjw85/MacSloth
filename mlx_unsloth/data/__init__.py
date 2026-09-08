"""Dataset conversion into mlx-lm JSONL."""

from mlx_unsloth.data.dataset import to_mlx_jsonl
from mlx_unsloth.data.errors import (
    EmptyDatasetError,
    MissingAssistantError,
    UnknownRecordFormatError,
    UnknownSpeakerError,
)

__all__ = [
    "EmptyDatasetError",
    "MissingAssistantError",
    "UnknownRecordFormatError",
    "UnknownSpeakerError",
    "to_mlx_jsonl",
]
