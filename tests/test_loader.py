"""Given/When/Then tests for FastLanguageModel load and LoRA attach."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_lm")

from mlx_unsloth.models.loader import (
    FastLanguageModel,
    InvalidLoraRankError,
    LoraTargetError,
    UnsupportedLoadIn4bitError,
    _layer_lora_keys,
)


class _FakeLayer:
    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names

    def named_modules(self) -> list[tuple[str, str]]:
        return [(name, name) for name in self._names]


class _FakeModel:
    def __init__(self, layers: list[_FakeLayer]) -> None:
        self.layers = layers
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True


def test_from_pretrained_loads_named_checkpoint_and_sets_max_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace()
    loaded = SimpleNamespace()

    def fake_load(name: str) -> tuple[SimpleNamespace, SimpleNamespace]:
        assert name == "mlx-community/test-4bit"
        return loaded, tokenizer

    monkeypatch.setattr("mlx_unsloth.models.loader.load", fake_load)

    model, tok = FastLanguageModel.from_pretrained(
        "mlx-community/test-4bit",
        max_seq_length=1024,
    )

    assert model is loaded
    assert tok is tokenizer
    assert tok.model_max_length == 1024


def test_from_pretrained_rejects_load_in_4bit_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_name: str) -> None:
        raise AssertionError

    monkeypatch.setattr("mlx_unsloth.models.loader.load", fail_if_called)

    with pytest.raises(UnsupportedLoadIn4bitError) as exc_info:
        FastLanguageModel.from_pretrained(
            "mlx-community/test-4bit",
            load_in_4bit=False,
        )

    assert str(exc_info.value) == (
        "this wrapper does not load unquantized weights; pass load_in_4bit=True "
        "and use a prequantized mlx-community checkpoint"
    )


def test_layer_lora_keys_match_short_and_full_names() -> None:
    layer = _FakeLayer(("self_attn.q_proj", "self_attn.v_proj", "mlp.up_proj"))
    model = _FakeModel([layer])

    keys = _layer_lora_keys(model, ["q_proj", "self_attn.v_proj"])

    assert keys == ("self_attn.q_proj", "self_attn.v_proj")


def test_layer_lora_keys_raise_when_nothing_matches() -> None:
    model = _FakeModel([_FakeLayer(("mlp.up_proj",))])

    with pytest.raises(LoraTargetError):
        _layer_lora_keys(model, ["q_proj"])


def test_get_peft_model_freezes_and_converts_matched_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _FakeLayer(
        (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        )
    )
    model = _FakeModel([layer, layer])

    class _Capture:
        def __init__(self) -> None:
            self.model: _FakeModel | None = None
            self.num_layers: int | None = None
            self.config: dict[str, int | float | list[str]] | None = None
            self.use_dora: bool | None = None

    captured = _Capture()

    def fake_linear_to_lora_layers(
        converted: _FakeModel,
        num_layers: int,
        config: dict[str, int | float | list[str]],
        use_dora: bool = False,
    ) -> None:
        captured.model = converted
        captured.num_layers = num_layers
        captured.config = config
        captured.use_dora = use_dora

    monkeypatch.setattr(
        "mlx_unsloth.models.loader.linear_to_lora_layers",
        fake_linear_to_lora_layers,
    )

    result = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    assert result is model
    assert model.frozen is True
    assert captured.model is model
    assert captured.num_layers == 2
    assert captured.use_dora is False
    assert captured.config == {
        "rank": 16,
        "scale": 32.0,
        "dropout": 0.05,
        "keys": [
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.v_proj",
        ],
    }
    assert model.lora_spec.rank == 16
    assert model.lora_spec.num_layers == 2
    assert model.lora_spec.keys == (
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    )


def test_get_peft_model_patches_metal_when_dropout_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _FakeLayer(("self_attn.q_proj",))
    model = _FakeModel([layer])
    called: list[bool] = []

    monkeypatch.setattr(
        "mlx_unsloth.models.loader.linear_to_lora_layers",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mlx_unsloth.models.loader._patch_lora_linear_metal",
        lambda: called.append(True),
    )

    FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj"],
    )

    assert called == [True]


def test_get_peft_model_rejects_non_positive_rank() -> None:
    model = _FakeModel([])

    with pytest.raises(InvalidLoraRankError):
        FastLanguageModel.get_peft_model(model, r=0)
