"""Typed errors for dataset conversion."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UnknownRecordFormatError(Exception):
    """Raised when a row is not ShareGPT, Alpaca, or OpenAI messages."""

    keys: tuple[str, ...]

    def __str__(self) -> str:
        """Return the error message."""
        return (
            f"record keys {list(self.keys)} are not ShareGPT, Alpaca, "
            "or OpenAI messages"
        )


@dataclass(frozen=True, slots=True)
class UnknownSpeakerError(Exception):
    """Raised when a ShareGPT `from` value is not a known speaker."""

    speaker: str

    def __str__(self) -> str:
        """Return the error message."""
        return f"unknown ShareGPT speaker: {self.speaker}"


@dataclass(frozen=True, slots=True)
class MissingAssistantError(Exception):
    """Raised when a converted record has no assistant turn."""

    def __str__(self) -> str:
        """Return the error message."""
        return "record has no assistant turn"


@dataclass(frozen=True, slots=True)
class EmptyDatasetError(Exception):
    """Raised when conversion would write zero rows."""

    def __str__(self) -> str:
        """Return the error message."""
        return "dataset is empty"


@dataclass(frozen=True, slots=True)
class MissingFieldError(Exception):
    """Raised when a required field is absent."""

    field: str

    def __str__(self) -> str:
        """Return the error message."""
        return f"missing field: {self.field}"


@dataclass(frozen=True, slots=True)
class FieldTypeError(Exception):
    """Raised when a field exists but is the wrong JSON type."""

    field: str
    got: str

    def __str__(self) -> str:
        """Return the error message."""
        return f"field {self.field} has type {self.got}"


@dataclass(frozen=True, slots=True)
class JsonlParseError(Exception):
    """Raised when a source JSONL line is not valid JSON."""

    path: Path
    line: int

    def __str__(self) -> str:
        """Return the error message."""
        return f"invalid JSON in {self.path} at line {self.line}"


@dataclass(frozen=True, slots=True)
class UnknownSplitError(Exception):
    """Raised when `split` is not train/valid/test."""

    split: str

    def __str__(self) -> str:
        """Return the error message."""
        return f"split must be train, valid, or test; got {self.split}"


@dataclass(frozen=True, slots=True)
class DatasetDictError(Exception):
    """Raised when a bare DatasetDict is passed instead of a split."""

    def __str__(self) -> str:
        """Return the error message."""
        return "pass DatasetDict['train'], not the DatasetDict"
