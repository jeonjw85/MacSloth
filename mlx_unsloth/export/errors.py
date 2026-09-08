"""Typed errors for fuse / GGUF / Hub export."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MissingAdapterError(Exception):
    """Raised when adapter_path lacks mlx-lm adapter files."""

    path: Path

    def __str__(self) -> str:
        """Return the error message."""
        return f"missing adapter file: {self.path}"


@dataclass(frozen=True, slots=True)
class UnsupportedGgufModelError(Exception):
    """Raised when mlx-lm cannot write GGUF for this architecture."""

    model_type: str

    def __str__(self) -> str:
        """Return the error message."""
        return (
            f"GGUF export supports llama, mixtral, and mistral; got {self.model_type}"
        )


@dataclass(frozen=True, slots=True)
class MissingGgufConfigError(Exception):
    """Raised when the GGUF sibling config.json is missing or unreadable."""

    path: Path

    def __str__(self) -> str:
        """Return the error message."""
        return f"missing GGUF sibling config.json: {self.path}"
