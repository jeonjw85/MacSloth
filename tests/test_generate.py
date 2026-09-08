"""Given/When/Then tests for FastLanguageModel generate and inference."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_lm")

from macsloth.models.loader import FastLanguageModel


class _EvalModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


def test_for_inference_calls_eval() -> None:
    model = _EvalModel()
    out = FastLanguageModel.for_inference(model)
    assert out is model
    assert model.training is False


def test_generate_forwards_to_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cap:
        prompt: str | None = None
        max_tokens: int | None = None
        verbose: bool | None = None

    captured = _Cap()

    def fake_generate(
        model: SimpleNamespace,
        tokenizer: SimpleNamespace,
        prompt: str,
        max_tokens: int = 256,
        verbose: bool = True,
    ) -> str:
        captured.prompt = prompt
        captured.max_tokens = max_tokens
        captured.verbose = verbose
        return "ok"

    monkeypatch.setattr("macsloth.models.loader.mlx_generate", fake_generate)
    text = FastLanguageModel.generate(
        SimpleNamespace(), SimpleNamespace(), "hi", max_tokens=8
    )
    assert text == "ok"
    assert captured.prompt == "hi"
    assert captured.max_tokens == 8
    assert captured.verbose is False
