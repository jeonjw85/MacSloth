# AGENTS.md

Repo: MacSloth. Python package: `macsloth`.

LoRA/QLoRA fine-tuning on Apple Silicon using MLX. Public API follows Unsloth (`FastLanguageModel`) so call sites stay short.

## Rules

1. Use `mlx` and `mlx-lm` only. No CUDA. No PyTorch training path.
2. Unified memory is the budget. Prefer lazy arrays. Call `mx.eval()` when values must materialize.
3. Keep the Unsloth-shaped API. Do not add parallel load/train entry points without a reason.
4. Export fused weights through `mlx_lm.fuse`, then GGUF / Hugging Face Hub as needed.

## Layout

| Path | Role |
| :--- | :--- |
| `macsloth/models/loader.py` | `FastLanguageModel`: load MLX weights, attach LoRA |
| `macsloth/trainer/lora.py` | Training loop on MLX autograd / optimizers |
| `macsloth/data/dataset.py` | ShareGPT / Alpaca / OpenAI messages -> MLX JSONL |
| `macsloth/export/exporter.py` | Fuse, GGUF, Hub upload |
| `examples/` | Short scripts that start from `FastLanguageModel` |

Files listed above that do not exist yet are planned, not stubs to invent in passing.

## Python

- 3.10+
- Type annotations on public functions. Prefer `X | None`, `list[str]`, `tuple[...]`. Do not use `Any` in public signatures.
- PEP 8. Lint/format with ruff.
- Library code does not print and does not use emojis. Example scripts may print.

## MLX

- Array work in `mlx.core`, not numpy/torch, on the training path.
- Drop arrays you do not keep. Do not leave graphs alive across steps.

## Errors

Fail fast when:

- the machine cannot hold the load
- quantization format is unsupported
- LoRA `target_modules` match nothing on the model

## Changes

- New models: check `mlx-community` quantization layouts before adding them.
- Training loop edits: clear non-retained arrays each step.
- Run `pytest` before you send a change.
