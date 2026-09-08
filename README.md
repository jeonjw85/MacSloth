# MacSloth

LoRA fine-tuning on Apple Silicon with MLX. Python package: `macsloth`.

Load a quantized MLX checkpoint, attach LoRA, train, fuse adapters. Call sites follow the Unsloth shape (`FastLanguageModel`, `SFTTrainer`) so scripts stay short. The training loop is MacSloth's, not Unsloth and not `mlx_lm.tuner.trainer.train`.

## Requirements

- macOS, Apple Silicon
- Python 3.10+
- A prequantized [mlx-community](https://huggingface.co/mlx-community) checkpoint (or a local MLX directory)

No CUDA. No PyTorch training path. `load_in_4bit=False` is rejected; this wrapper does not quantize at load time.

## Install

```bash
pip install -e ".[dev]"
```

## Data

`to_mlx_jsonl` writes mlx-lm chat JSONL (`train.jsonl` / `valid.jsonl` / `test.jsonl`).

Accepted row shapes:

- Alpaca: `instruction`, `output`, optional `input`
- ShareGPT: `conversations` with `from` / `value` (`human`, `gpt`, `system`, …)
- OpenAI: `messages` with `role` / `content`

```python
from macsloth import to_mlx_jsonl

to_mlx_jsonl(rows, "data")              # -> data/train.jsonl
to_mlx_jsonl(rows, "data", split="valid")
to_mlx_jsonl(rows, "out.jsonl")         # explicit file
```

Also accepts a `.json` / `.jsonl` path or a Hugging Face `Dataset` (pass a split, not a `DatasetDict`).

## Train

```python
from macsloth import FastLanguageModel, SFTTrainer, TrainingConfig, to_mlx_jsonl

model, tokenizer = FastLanguageModel.from_pretrained(
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    max_seq_length=512,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,  # passed to mlx-lm as LoRA scale, not alpha/r
    target_modules=["q_proj", "v_proj"],  # default
)

to_mlx_jsonl(rows, "data")

SFTTrainer(
    model,
    tokenizer,
    "data",
    TrainingConfig(
        iters=20,
        batch_size=4,
        max_seq_length=512,
        mask_prompt=True,
        adapter_path="adapters",
        optimizer="adam",  # or adamw
        grad_checkpoint=False,  # True uses less memory, slower
    ),
).train()
```

`train()` writes `adapters/adapter_config.json` and `adapters/adapters.safetensors`.

`grad_accumulation_steps` must stay `1`. Validation is skipped. `min_working_set_bytes` fail-fasts if Metal's recommended working set is too small.

Full example: `examples/train_qwen_coder.py`.

## Export

Fuse LoRA into the base MLX weights:

```python
from macsloth import save_pretrained_merged

save_pretrained_merged(
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "adapters",
    "fused_model",
)
```

GGUF is llama / mistral / mixtral only (mlx-lm fp16 converter). Qwen: fuse MLX weights, do not call GGUF.

```python
from macsloth import save_pretrained_gguf, write_ollama_modelfile

gguf = save_pretrained_gguf("mlx-community/Llama-3.2-1B-Instruct-4bit", "adapters", "fused_model")
write_ollama_modelfile(gguf)  # writes Modelfile next to the GGUF; does not run Ollama
```

Hub upload:

```python
from macsloth import push_to_hub_merged

push_to_hub_merged(base, "adapters", "fused_model", "user/repo")
```

## Generate

```python
from macsloth import FastLanguageModel

FastLanguageModel.for_inference(model)
text = FastLanguageModel.generate(model, tokenizer, prompt, max_tokens=256)
```

## Test

```bash
pytest tests -q
```
