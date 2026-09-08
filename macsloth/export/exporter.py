"""Fuse LoRA adapters, optionally write GGUF, optionally upload to the Hub."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.gguf import convert_to_gguf
from mlx_lm.utils import dequantize_model, load, save, upload_to_hub

from macsloth.export.errors import (
    MissingAdapterError,
    MissingGgufConfigError,
    UnsupportedGgufModelError,
)

if TYPE_CHECKING:
    from mlx.nn import Module

GGUF_MODEL_TYPES: Final[frozenset[str]] = frozenset({"llama", "mixtral", "mistral"})
DEFAULT_GGUF_NAME: Final = "ggml-model-f16.gguf"


@dataclass(frozen=True, slots=True)
class _Fused:
    """Fused module plus the config mlx-lm.save needs."""

    model: Module
    config: dict[str, str]
    path: Path


def _gguf_model_type(config: dict[str, str], config_path: Path) -> str:
    if "model_type" not in config:
        raise MissingGgufConfigError(path=config_path)
    model_type = str(config["model_type"])
    if model_type not in GGUF_MODEL_TYPES:
        raise UnsupportedGgufModelError(model_type=model_type)
    return model_type


def _require_adapter(adapter_path: Path) -> None:
    if not adapter_path.is_dir():
        raise MissingAdapterError(path=adapter_path)
    config_file = adapter_path / "adapter_config.json"
    weights_file = adapter_path / "adapters.safetensors"
    if not config_file.is_file():
        raise MissingAdapterError(path=config_file)
    if not weights_file.is_file():
        raise MissingAdapterError(path=weights_file)


def _fuse(
    model_name: str,
    adapter_path: Path,
    save_path: Path,
    dequantize: bool,
) -> _Fused:
    _require_adapter(adapter_path)
    model, tokenizer, config = load(
        model_name,
        adapter_path=str(adapter_path),
        return_config=True,
    )
    fused = [
        (name, module.fuse(dequantize=dequantize))
        for name, module in model.named_modules()
        if hasattr(module, "fuse")
    ]
    if fused:
        model.update_modules(tree_unflatten(fused))
    if dequantize:
        model = dequantize_model(model)
        config.pop("quantization", None)
        config.pop("quantization_config", None)
    save_path.mkdir(parents=True, exist_ok=True)
    save(save_path, model_name, model, tokenizer, config, donate_model=False)
    return _Fused(model=model, config=config, path=save_path)


def save_pretrained_merged(
    model_name: str,
    adapter_path: Path | str,
    save_path: Path | str,
    *,
    dequantize: bool = False,
) -> Path:
    """Fuse adapters into the base MLX weights and save the directory.

    Args:
        model_name: Base MLX repo id or local path.
        adapter_path: Directory with `adapter_config.json` and
            `adapters.safetensors`.
        save_path: Output model directory.
        dequantize: If True, write bf16/fp16 weights instead of quantized.

    Returns:
        `save_path` as a Path.
    """
    dest = Path(save_path)
    fused = _fuse(model_name, Path(adapter_path), dest, dequantize=dequantize)
    return fused.path


def save_pretrained_gguf(
    model_name: str,
    adapter_path: Path | str,
    save_path: Path | str,
    *,
    gguf_name: str = DEFAULT_GGUF_NAME,
) -> Path:
    """Fuse, dequantize, and write a GGUF file under `save_path`.

    mlx-lm only converts llama, mixtral, and mistral in fp16.

    Args:
        model_name: Base MLX repo id or local path.
        adapter_path: Trained adapter directory.
        save_path: Directory that receives fused weights and the GGUF file.
        gguf_name: Filename inside `save_path`.

    Returns:
        Path of the GGUF file.
    """
    dest = Path(save_path)
    fused = _fuse(model_name, Path(adapter_path), dest, dequantize=True)
    _gguf_model_type(fused.config, dest / "config.json")
    weights = dict(tree_flatten(fused.model.parameters()))
    gguf_path = dest / gguf_name
    convert_to_gguf(dest, weights, fused.config, str(gguf_path))
    return gguf_path


def write_ollama_modelfile(gguf_path: Path | str) -> Path:
    """Write an Ollama Modelfile next to a GGUF path.

    Does not invoke the Ollama CLI or HTTP API. The GGUF file need not
    exist. Sibling ``config.json`` must be readable and declare a
    ``model_type`` in ``GGUF_MODEL_TYPES``.

    Args:
        gguf_path: Path of the GGUF file (used for directory and filename).

    Returns:
        Path of the written ``Modelfile``.
    """
    gguf = Path(gguf_path)
    config_path = gguf.parent / "config.json"
    try:
        raw = config_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise MissingGgufConfigError(path=config_path) from exc
    match parsed:
        case dict() as config:
            _gguf_model_type(config, config_path)
        case _:
            raise MissingGgufConfigError(path=config_path)
    modelfile = gguf.parent / "Modelfile"
    modelfile.write_text(f"FROM {gguf.name}\n", encoding="utf-8")
    return modelfile


def push_to_hub_merged(
    model_name: str,
    adapter_path: Path | str,
    save_path: Path | str,
    repo_id: str,
    *,
    dequantize: bool = False,
) -> str:
    """Fuse adapters, save locally, and upload the directory to the Hub.

    Args:
        model_name: Base MLX repo id or local path.
        adapter_path: Trained adapter directory.
        save_path: Local directory to write before upload.
        repo_id: Hugging Face repo id.
        dequantize: If True, upload unquantized weights.

    Returns:
        `repo_id`.
    """
    dest = Path(save_path)
    _fuse(model_name, Path(adapter_path), dest, dequantize=dequantize)
    upload_to_hub(str(dest), repo_id)
    return repo_id
