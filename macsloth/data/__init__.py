"""Dataset conversion into mlx-lm JSONL."""

from macsloth.data.dataset import to_mlx_jsonl
from macsloth.data.errors import (
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
