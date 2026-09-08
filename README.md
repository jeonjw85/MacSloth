# mlx-unsloth

LoRA fine-tuning on Apple Silicon with MLX. Not Unsloth; same call-site shape only.

Python 3.10+, macOS, prequantized `mlx-community` checkpoint. No CUDA. No PyTorch train path.

GGUF export is llama / mistral / mixtral only. Qwen: fuse MLX weights, do not call GGUF.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```python
from mlx_unsloth import (
    FastLanguageModel,
    SFTTrainer,
    TrainingConfig,
    save_pretrained_merged,
    to_mlx_jsonl,
)

model, tokenizer = FastLanguageModel.from_pretrained(
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    max_seq_length=512,
)
model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=16)

to_mlx_jsonl(rows, "data")  # Alpaca / ShareGPT / OpenAI messages

SFTTrainer(
    model,
    tokenizer,
    "data",
    TrainingConfig(iters=20, batch_size=4, max_seq_length=512, adapter_path="adapters"),
).train()

save_pretrained_merged(
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "adapters",
    "fused_model",
)
```

See `examples/train_qwen_coder.py`.

## Test

```bash
pytest tests -q
```
