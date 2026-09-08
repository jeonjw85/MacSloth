"""MLX fast kernels and a custom Metal LoRA delta."""

from macsloth.kernels.fast_ops import cross_entropy, rms_norm, rope
from macsloth.kernels.lora_metal import lora_delta, lora_linear_forward

__all__ = [
    "cross_entropy",
    "lora_delta",
    "lora_linear_forward",
    "rms_norm",
    "rope",
]
