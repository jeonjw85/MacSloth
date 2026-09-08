"""Compare tokens/sec and peak memory against mlx_lm.tuner.trainer.train.

Run: .venv/bin/python scripts/bench_vs_mlx_lm.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, TypedDict

import mlx.core as mx
import mlx.optimizers as optim
from macsloth.trainer.loop import run_sft
from macsloth.trainer.lora import TrainingConfig
from mlx import nn
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import TrainingArgs, default_loss, train

BATCH_SIZE: Final = 4
SEQ_LEN: Final = 128
WARMUP_STEPS: Final = 5
TIMED_STEPS: Final = 20
ITERS: Final = WARMUP_STEPS + TIMED_STEPS
TOKENS_PER_TIMED_STEP: Final = BATCH_SIZE * (SEQ_LEN - 1)
LORA_RANK: Final = 8
VOCAB_SIZE: Final = 256
HIDDEN_SIZE: Final = 256
N_ROWS: Final = BATCH_SIZE
SEED: Final = 0
LEARNING_RATE: Final = 1e-5
WIN_RATIO: Final = 1.10
REPO_ROOT: Final = Path(__file__).resolve().parents[1]
EVIDENCE_JSON: Final = REPO_ROOT / ".omo" / "evidence" / "bench-vs-mlx-lm.json"


@dataclass(frozen=True, slots=True)
class BenchIncompleteError(Exception):
    """Raised when the train callback did not cover warmup + timed steps."""

    iteration: int
    needed: int

    def __str__(self) -> str:
        """Return the error message."""
        return f"bench callback stopped at iter {self.iteration}, need {self.needed}"


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Tokens/sec and peak bytes for one timed train() run."""

    tokens_per_sec: float
    peak_bytes: int


class TrainLossInfo(TypedDict):
    """Fields from mlx-lm train_info that the bench timer reads."""

    iteration: int
    trained_tokens: float


@dataclass(frozen=True, slots=True)
class BenchReport:
    """JSON payload written to .omo/evidence/bench-vs-mlx-lm.json."""

    mlx_lm_tokens_per_sec: float
    ours_tokens_per_sec: float
    mlx_lm_peak_bytes: int
    ours_peak_bytes: int
    tokens_per_sec_ratio: float


class TinyLoraLM(nn.Module):
    """Embedding plus two LoRA linears; logits over a tiny vocab."""

    def __init__(self) -> None:
        """Freeze the base stack, then wrap both linears with LoRA rank 8."""
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)
        self.freeze()
        self.fc1 = LoRALinear.from_base(self.fc1, r=LORA_RANK)
        self.fc2 = LoRALinear.from_base(self.fc2, r=LORA_RANK)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Return logits of shape (batch, seq, vocab)."""
        hidden = self.embed(tokens)
        hidden = self.fc1(hidden)
        return self.fc2(hidden)


class RandomTokenRows:
    """Fixed-length random token rows for CacheDataset."""

    def __init__(self, rows: tuple[tuple[int, ...], ...]) -> None:
        """Store already-sampled rows."""
        self._rows = rows

    def process(self, row: tuple[int, ...]) -> tuple[list[int], int]:
        """Return (tokens, prompt_offset) as mlx-lm datasets do."""
        return (list(row), 0)

    def __getitem__(self, idx: int) -> tuple[int, ...]:
        """Return the raw token row at idx."""
        return self._rows[idx]

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._rows)


class TimedStepsCallback(TrainingCallback):
    """Time ntoks after warmup; mutation is the documented purpose."""

    def __init__(self) -> None:
        """Start with no timed samples."""
        self._t0: float | None = None
        self._tokens_at_start: float = 0.0
        self._tokens_at_end: float = 0.0
        self._elapsed: float = 0.0
        self._peak_bytes: int = 0
        self._last_iteration: int = 0

    def on_train_loss_report(self, train_info: TrainLossInfo) -> None:
        """Reset peak at warmup; stop the clock after timed steps."""
        iteration = int(train_info["iteration"])
        trained = float(train_info["trained_tokens"])
        self._last_iteration = iteration
        if iteration == WARMUP_STEPS:
            mx.reset_peak_memory()
            self._t0 = time.perf_counter()
            self._tokens_at_start = trained
            return
        if iteration == ITERS:
            if self._t0 is None:
                raise BenchIncompleteError(iteration=iteration, needed=WARMUP_STEPS)
            self._elapsed = time.perf_counter() - self._t0
            self._tokens_at_end = trained
            self._peak_bytes = int(mx.get_peak_memory())

    def metrics(self) -> RunMetrics:
        """Return tokens/sec and peak bytes for the timed window."""
        if self._t0 is None or self._elapsed <= 0.0 or self._last_iteration != ITERS:
            raise BenchIncompleteError(
                iteration=self._last_iteration,
                needed=ITERS,
            )
        tokens = self._tokens_at_end - self._tokens_at_start
        return RunMetrics(
            tokens_per_sec=tokens / self._elapsed,
            peak_bytes=self._peak_bytes,
        )


def _token_rows() -> tuple[tuple[int, ...], ...]:
    mx.random.seed(SEED)
    table = mx.random.randint(0, VOCAB_SIZE, (N_ROWS, SEQ_LEN))
    mx.eval(table)
    return tuple(
        tuple(int(table[row, col].item()) for col in range(SEQ_LEN))
        for row in range(N_ROWS)
    )


def _build_model() -> TinyLoraLM:
    mx.random.seed(SEED)
    model = TinyLoraLM()
    mx.eval(model.parameters())
    return model


def run_mlx_lm_train(adapter_file: Path) -> RunMetrics:
    """Run mlx-lm train() for warmup + timed steps on a tiny LoRA module."""
    model = _build_model()
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    dataset = CacheDataset(RandomTokenRows(_token_rows()))
    callback = TimedStepsCallback()
    args = TrainingArgs(
        batch_size=BATCH_SIZE,
        iters=ITERS,
        val_batches=0,
        steps_per_report=WARMUP_STEPS,
        steps_per_eval=ITERS + 1,
        steps_per_save=ITERS + 1,
        max_seq_length=SEQ_LEN,
        adapter_file=str(adapter_file),
        grad_checkpoint=False,
    )
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=dataset,
        val_dataset=None,
        args=args,
        loss=default_loss,
        training_callback=callback,
    )
    return callback.metrics()


def run_ours(adapter_file: Path) -> RunMetrics:
    """Run native run_sft for warmup + timed steps on the same tiny LoRA module."""
    model = _build_model()
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    dataset = CacheDataset(RandomTokenRows(_token_rows()))
    callback = TimedStepsCallback()

    def after_step(iteration: int) -> None:
        if iteration == WARMUP_STEPS:
            mx.clear_cache()
            mx.synchronize()
        if iteration not in (WARMUP_STEPS, ITERS):
            return
        if iteration == ITERS:
            mx.synchronize()
        callback.on_train_loss_report(
            {
                "iteration": iteration,
                "trained_tokens": float(iteration * TOKENS_PER_TIMED_STEP),
            }
        )

    config = TrainingConfig(
        batch_size=BATCH_SIZE,
        iters=ITERS,
        learning_rate=LEARNING_RATE,
        max_seq_length=SEQ_LEN,
        adapter_path=str(adapter_file.parent),
        grad_checkpoint=False,
        val_batches=0,
        steps_per_report=WARMUP_STEPS,
        steps_per_eval=ITERS + 1,
        steps_per_save=ITERS + 1,
        seed=SEED,
        optimizer="adam",
    )
    run_sft(
        model,
        optimizer,
        dataset,
        config,
        after_step=after_step,
    )
    return callback.metrics()


def write_report(report: BenchReport) -> None:
    """Write the bench JSON next to other plan evidence."""
    EVIDENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_JSON.write_text(
        json.dumps(asdict(report), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Run mlx-lm then ours, write JSON, exit 0 only if the win gate holds."""
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Discard one run per path so timed measurements start compiled.
        run_mlx_lm_train(tmp_dir / "prime_mlx.safetensors")
        mlx_metrics = run_mlx_lm_train(tmp_dir / "mlx_adapters.safetensors")
        run_ours(tmp_dir / "prime_ours.safetensors")
        ours_metrics = run_ours(tmp_dir / "ours_adapters.safetensors")
    ratio = ours_metrics.tokens_per_sec / mlx_metrics.tokens_per_sec
    report = BenchReport(
        mlx_lm_tokens_per_sec=mlx_metrics.tokens_per_sec,
        ours_tokens_per_sec=ours_metrics.tokens_per_sec,
        mlx_lm_peak_bytes=mlx_metrics.peak_bytes,
        ours_peak_bytes=ours_metrics.peak_bytes,
        tokens_per_sec_ratio=ratio,
    )
    write_report(report)
    print(json.dumps(asdict(report), indent=2))
    print(f"wrote {EVIDENCE_JSON}")
    gate = ratio >= WIN_RATIO and ours_metrics.peak_bytes <= mlx_metrics.peak_bytes
    if gate:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
