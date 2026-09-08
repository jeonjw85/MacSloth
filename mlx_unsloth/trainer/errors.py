"""Typed errors for LoRA training."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EmptyTrainSetError(Exception):
    """Raised when the data directory has no training rows."""

    path: Path

    def __str__(self) -> str:
        """Return the error message."""
        return f"no training rows in {self.path}"


@dataclass(frozen=True, slots=True)
class InvalidTrainingConfigError(Exception):
    """Raised when a training hyperparameter is out of range."""

    field: str
    value: int | float

    def __str__(self) -> str:
        """Return the error message."""
        return f"invalid {self.field}: {self.value}"


@dataclass(frozen=True, slots=True)
class UnknownOptimizerError(Exception):
    """Raised when the optimizer name is not supported."""

    name: str

    def __str__(self) -> str:
        """Return the error message."""
        return f"unknown optimizer: {self.name}"


@dataclass(frozen=True, slots=True)
class NotADatasetDirError(Exception):
    """Raised when `data` is not a directory of mlx-lm JSONL splits."""

    path: Path

    def __str__(self) -> str:
        """Return the error message."""
        return f"dataset path is not a directory: {self.path}"


@dataclass(frozen=True, slots=True)
class MissingLoraSpecError(Exception):
    """Raised when train runs before get_peft_model attaches LoRA."""

    def __str__(self) -> str:
        """Return the error message."""
        return "get_peft_model is required before train"


@dataclass(frozen=True, slots=True)
class MetalUnavailableError(Exception):
    """Raised when a working-set check needs Metal and it is missing."""

    def __str__(self) -> str:
        """Return the error message."""
        return "Metal is required when min_working_set_bytes is set"


@dataclass(frozen=True, slots=True)
class InsufficientUnifiedMemoryError(Exception):
    """Raised when Metal's recommended working set is below the configured minimum."""

    required_bytes: int
    available_bytes: int

    def __str__(self) -> str:
        """Return the error message."""
        return (
            f"unified memory working set {self.available_bytes} "
            f"< required {self.required_bytes}"
        )
