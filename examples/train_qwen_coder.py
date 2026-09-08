"""Load Qwen2.5-Coder, attach LoRA, train, and fuse adapters."""

from mlx_unsloth import (
    FastLanguageModel,
    SFTTrainer,
    TrainingConfig,
    save_pretrained_merged,
    to_mlx_jsonl,
)

MODEL_NAME = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=512,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)

to_mlx_jsonl(
    [
        {
            "instruction": (
                "Write a Python function that returns the nth Fibonacci number."
            ),
            "output": (
                "def fib(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
            ),
        },
        {
            "instruction": "Reverse a string in Python without slicing.",
            "output": (
                "def reverse_string(text):\n"
                "    chars = list(text)\n"
                "    left, right = 0, len(chars) - 1\n"
                "    while left < right:\n"
                "        chars[left], chars[right] = chars[right], chars[left]\n"
                "        left += 1\n"
                "        right -= 1\n"
                "    return ''.join(chars)\n"
            ),
        },
        {
            "instruction": "Count word frequencies in a string.",
            "output": (
                "from collections import Counter\n"
                "\n"
                "def word_counts(text):\n"
                "    return dict(Counter(text.split()))\n"
            ),
        },
        {
            "instruction": "Return True if a string is a palindrome.",
            "output": (
                "def is_palindrome(text):\n"
                "    normalized = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
                "    return normalized == normalized[::-1]\n"
            ),
        },
    ],
    "data",
)

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
    ),
).train()

save_pretrained_merged(MODEL_NAME, "adapters", "fused_model")
