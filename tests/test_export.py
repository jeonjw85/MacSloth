"""Given/When/Then tests for fuse / GGUF / Hub export."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("mlx_lm")

from macsloth.export import (
    push_to_hub_merged,
    save_pretrained_gguf,
    save_pretrained_merged,
    write_ollama_modelfile,
)
from macsloth.export.errors import (
    MissingAdapterError,
    MissingGgufConfigError,
    UnsupportedGgufModelError,
)

if TYPE_CHECKING:
    from pathlib import Path


class _Fusable:
    def fuse(self, dequantize: bool = False) -> _Fusable:
        self.dequantize = dequantize
        return self


class _Model:
    def named_modules(self) -> list[tuple[str, _Fusable]]:
        return [("q_proj", _Fusable())]

    def update_modules(self, tree: dict[str, _Fusable]) -> None:
        self.tree = tree

    def parameters(self) -> dict[str, tuple[()]]:
        return {}


def _touch_adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapters"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"x")
    return adapter


def _patch_fuse(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> _Model:
    model = _Model()
    tokenizer = SimpleNamespace()
    config = {"model_type": model_type}

    def fake_load(
        name: str,
        adapter_path: str | None = None,
        return_config: bool = False,
    ) -> tuple[_Model, SimpleNamespace, dict[str, str]]:
        assert name == "base-model"
        assert adapter_path is not None
        assert return_config is True
        return model, tokenizer, config

    monkeypatch.setattr("macsloth.export.exporter.load", fake_load)
    monkeypatch.setattr(
        "macsloth.export.exporter.save",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "macsloth.export.exporter.dequantize_model",
        lambda loaded: loaded,
    )
    monkeypatch.setattr(
        "macsloth.export.exporter.tree_unflatten",
        lambda fused: {"q_proj": fused[0][1]} if fused else {},
    )
    monkeypatch.setattr(
        "macsloth.export.exporter.tree_flatten",
        lambda params: [],
    )
    return model


def test_save_pretrained_merged_fuses_and_saves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _touch_adapter(tmp_path)
    _patch_fuse(monkeypatch, "qwen2")

    out = save_pretrained_merged(
        "base-model",
        adapter,
        tmp_path / "fused_model",
    )

    assert out == tmp_path / "fused_model"
    assert out.is_dir()


def test_save_pretrained_gguf_writes_gguf_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _touch_adapter(tmp_path)
    _patch_fuse(monkeypatch, "llama")
    captured: list[str] = []

    def fake_convert(
        dest: Path,
        weights: dict[str, str],
        config: dict[str, str],
        output: str,
    ) -> None:
        captured.append(output)

    monkeypatch.setattr(
        "macsloth.export.exporter.convert_to_gguf",
        fake_convert,
    )

    out = save_pretrained_gguf("base-model", adapter, tmp_path / "fused_model")

    assert out == tmp_path / "fused_model" / "ggml-model-f16.gguf"
    assert captured == [str(out)]


def test_save_pretrained_gguf_missing_model_type_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _touch_adapter(tmp_path)
    model = _Model()
    tokenizer = SimpleNamespace()

    def fake_load(
        name: str,
        adapter_path: str | None = None,
        return_config: bool = False,
    ) -> tuple[_Model, SimpleNamespace, dict[str, str]]:
        del name, adapter_path, return_config
        return model, tokenizer, {}

    monkeypatch.setattr("macsloth.export.exporter.load", fake_load)
    monkeypatch.setattr(
        "macsloth.export.exporter.save",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "macsloth.export.exporter.dequantize_model",
        lambda loaded: loaded,
    )
    monkeypatch.setattr(
        "macsloth.export.exporter.tree_unflatten",
        lambda fused: {},
    )

    with pytest.raises(MissingGgufConfigError):
        save_pretrained_gguf("base-model", adapter, tmp_path / "fused_model")


def test_save_pretrained_gguf_rejects_qwen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _touch_adapter(tmp_path)
    _patch_fuse(monkeypatch, "qwen2")

    with pytest.raises(UnsupportedGgufModelError):
        save_pretrained_gguf("base-model", adapter, tmp_path / "fused_model")


def test_push_to_hub_merged_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _touch_adapter(tmp_path)
    _patch_fuse(monkeypatch, "llama")
    uploaded: list[tuple[str, str]] = []

    def fake_upload(path: str, repo_id: str) -> None:
        uploaded.append((path, repo_id))

    monkeypatch.setattr("macsloth.export.exporter.upload_to_hub", fake_upload)

    repo = push_to_hub_merged(
        "base-model",
        adapter,
        tmp_path / "fused_model",
        "user/fused",
    )

    assert repo == "user/fused"
    assert uploaded == [(str(tmp_path / "fused_model"), "user/fused")]


def test_missing_adapter_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(MissingAdapterError):
        save_pretrained_merged("base-model", tmp_path / "nope", tmp_path / "out")


def test_write_ollama_modelfile_writes_from_gguf_name(tmp_path: Path) -> None:
    dest = tmp_path / "fused"
    dest.mkdir()
    (dest / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    gguf_path = dest / "ggml-model-f16.gguf"

    out = write_ollama_modelfile(gguf_path)

    assert out == dest / "Modelfile"
    assert out.read_bytes() == b"FROM ggml-model-f16.gguf\n"


def test_write_ollama_modelfile_rejects_qwen2(tmp_path: Path) -> None:
    dest = tmp_path / "fused"
    dest.mkdir()
    (dest / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    gguf_path = dest / "ggml-model-f16.gguf"

    with pytest.raises(UnsupportedGgufModelError):
        write_ollama_modelfile(gguf_path)


def test_write_ollama_modelfile_missing_config_raises(tmp_path: Path) -> None:
    dest = tmp_path / "fused"
    dest.mkdir()
    gguf_path = dest / "ggml-model-f16.gguf"
    config_path = dest / "config.json"

    with pytest.raises(MissingGgufConfigError) as exc_info:
        write_ollama_modelfile(gguf_path)

    assert exc_info.value.path == config_path
    assert str(exc_info.value) == f"missing GGUF sibling config.json: {config_path}"


def test_write_ollama_modelfile_empty_object_raises(tmp_path: Path) -> None:
    dest = tmp_path / "fused"
    dest.mkdir()
    (dest / "config.json").write_text("{}", encoding="utf-8")
    gguf_path = dest / "ggml-model-f16.gguf"

    with pytest.raises(MissingGgufConfigError) as exc_info:
        write_ollama_modelfile(gguf_path)

    assert exc_info.value.path == dest / "config.json"


def test_write_ollama_modelfile_json_array_raises(tmp_path: Path) -> None:
    dest = tmp_path / "fused"
    dest.mkdir()
    (dest / "config.json").write_text("[]", encoding="utf-8")
    gguf_path = dest / "ggml-model-f16.gguf"

    with pytest.raises(MissingGgufConfigError):
        write_ollama_modelfile(gguf_path)
