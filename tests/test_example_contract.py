"""Given/When/Then contract tests for the Qwen example script."""

from __future__ import annotations

from pathlib import Path

_QWEN_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "train_qwen_coder.py"


def test_qwen_example_contains_save_pretrained_merged() -> None:
    source = _QWEN_EXAMPLE.read_text(encoding="utf-8")
    assert "save_pretrained_merged" in source


def test_qwen_example_omits_save_pretrained_gguf() -> None:
    source = _QWEN_EXAMPLE.read_text(encoding="utf-8")
    assert "save_pretrained_gguf" not in source


def test_qwen_example_omits_write_ollama_modelfile() -> None:
    source = _QWEN_EXAMPLE.read_text(encoding="utf-8")
    assert "write_ollama_modelfile" not in source


def test_qwen_example_omits_grad_checkpoint_kwarg() -> None:
    source = _QWEN_EXAMPLE.read_text(encoding="utf-8")
    assert "grad_checkpoint=" not in source


def test_qwen_example_omits_min_working_set_bytes() -> None:
    source = _QWEN_EXAMPLE.read_text(encoding="utf-8")
    assert "min_working_set_bytes" not in source
