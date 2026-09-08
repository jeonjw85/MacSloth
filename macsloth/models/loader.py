"""Unsloth-shaped load and LoRA attach for MLX models."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from mlx.nn import Module
from mlx_lm import generate as mlx_generate
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

from macsloth.kernels.lora_metal import lora_linear_forward

DEFAULT_LORA_TARGETS: Final[tuple[str, ...]] = (
    "q_proj",
    "v_proj",
)


class _TransformerBlock(Protocol):
    def named_modules(self) -> Sequence[tuple[str, Module]]:
        """Return (name, module) pairs under this block."""


class _LoraHost(Protocol):
    layers: Sequence[_TransformerBlock]

    def freeze(self) -> None:
        """Freeze base weights so only LoRA parameters train."""


@dataclass(frozen=True, slots=True)
class LoraTargetError(Exception):
    """Raised when none of the requested LoRA module names exist on the model."""

    target_modules: tuple[str, ...]

    def __str__(self) -> str:
        """Return the error message."""
        return f"LoRA target_modules matched nothing: {list(self.target_modules)}"


@dataclass(frozen=True, slots=True)
class UnsupportedLoadIn4bitError(Exception):
    """Raised when from_pretrained is asked to load unquantized weights."""

    def __str__(self) -> str:
        """Return the error message."""
        return (
            "this wrapper does not load unquantized weights; pass load_in_4bit=True "
            "and use a prequantized mlx-community checkpoint"
        )


@dataclass(frozen=True, slots=True)
class LoraSpec:
    """LoRA layout written to adapter_config.json for mlx-lm fuse."""

    num_layers: int
    rank: int
    scale: float
    dropout: float
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InvalidLoraRankError(Exception):
    """Raised when LoRA rank is not a positive integer."""

    rank: int

    def __str__(self) -> str:
        """Return the error message."""
        return f"LoRA rank must be >= 1, got {self.rank}"


def _layer_lora_keys(
    model: _LoraHost,
    target_modules: Sequence[str],
) -> tuple[str, ...]:
    targets = set(target_modules)
    matched: set[str] = set()
    for layer in model.layers:
        for name, _module in layer.named_modules():
            short = name.rsplit(".", 1)[-1]
            if name in targets or short in targets:
                matched.add(name)
    if not matched:
        raise LoraTargetError(target_modules=tuple(target_modules))
    return tuple(sorted(matched))


class FastLanguageModel:
    """Load an MLX causal LM and attach LoRA the way Unsloth does."""

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ) -> tuple[Module, TokenizerWrapper]:
        """Load a local or Hugging Face MLX checkpoint.

        Args:
            model_name: Hugging Face repo id or local path.
            max_seq_length: Written onto the tokenizer after load.
            load_in_4bit: Kept for Unsloth call-site compatibility. Must be
                True. MLX checkpoints are already quantized in-repo; this
                flag does not quantize at load time.

        Returns:
            The MLX module and its tokenizer.

        Raises:
            UnsupportedLoadIn4bitError: If `load_in_4bit` is False.
        """
        if load_in_4bit is False:
            raise UnsupportedLoadIn4bitError
        del load_in_4bit
        model, tokenizer = load(model_name)
        tokenizer.model_max_length = max_seq_length
        return model, tokenizer

    @staticmethod
    def for_inference(model: Module) -> Module:
        """Switch the module to eval for generation."""
        model.eval()
        return model

    @staticmethod
    def generate(
        model: Module,
        tokenizer: TokenizerWrapper,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        """Generate text with mlx-lm. Library code does not print."""
        return mlx_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

    @staticmethod
    def get_peft_model(
        model: _LoraHost,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Sequence[str] | None = None,
    ) -> _LoraHost:
        """Freeze the base weights and convert matched linears to LoRA.

        `lora_alpha` is passed to mlx-lm as `scale`. Short names such as
        `q_proj` match layer-local module names (`self_attn.q_proj`).

        Args:
            model: Model returned by `from_pretrained`.
            r: LoRA rank.
            lora_alpha: mlx-lm LoRA scale.
            lora_dropout: Dropout on the LoRA path.
            target_modules: Module name suffixes to convert. Defaults to
                q_proj and v_proj.

        Returns:
            The same model instance with LoRA layers attached.

        Raises:
            InvalidLoraRankError: If `r` is less than 1.
            LoraTargetError: If no module names matched.
        """
        if r < 1:
            raise InvalidLoraRankError(rank=r)

        modules = DEFAULT_LORA_TARGETS if target_modules is None else target_modules
        keys = _layer_lora_keys(model, modules)
        layers = model.layers
        model.freeze()
        linear_to_lora_layers(
            model,
            len(layers),
            {
                "rank": r,
                "scale": float(lora_alpha),
                "dropout": lora_dropout,
                "keys": list(keys),
            },
        )
        model.lora_spec = LoraSpec(
            num_layers=len(layers),
            rank=r,
            scale=float(lora_alpha),
            dropout=lora_dropout,
            keys=keys,
        )
        if lora_dropout == 0.0:
            _patch_lora_linear_metal()
        return model


_LORA_METAL_PATCHED = [False]


def _patch_lora_linear_metal() -> None:
    """Route dropout-free LoRALinear through the custom Metal LoRA delta."""
    if _LORA_METAL_PATCHED[0]:
        return
    original = LoRALinear.__call__

    def patched(self: LoRALinear, x):  # noqa: ANN001, ANN202
        p = getattr(self.dropout, "p", 0.0)
        if p != 0.0:
            return original(self, x)
        y = self.linear(x)
        return lora_linear_forward(x, y, self.lora_a, self.lora_b, float(self.scale))

    LoRALinear.__call__ = patched
    _LORA_METAL_PATCHED[0] = True
