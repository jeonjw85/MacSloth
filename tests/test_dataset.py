"""Given/When/Then tests for ShareGPT / Alpaca / OpenAI -> MLX JSONL."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from datasets import Dataset, DatasetDict
from mlx_unsloth.data import (
    EmptyDatasetError,
    MissingAssistantError,
    UnknownRecordFormatError,
    UnknownSpeakerError,
    to_mlx_jsonl,
)
from mlx_unsloth.data.errors import DatasetDictError

if TYPE_CHECKING:
    from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, list[dict[str, str]]]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_openai_messages_write_train_jsonl(tmp_path: Path) -> None:
    records = [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
    ]

    out = to_mlx_jsonl(records, tmp_path)

    assert out == tmp_path / "train.jsonl"
    assert _read_jsonl(out) == records


def test_sharegpt_conversations_map_speakers(tmp_path: Path) -> None:
    records = [
        {
            "conversations": [
                {"from": "system", "value": "Be brief."},
                {"from": "human", "value": "2+2?"},
                {"from": "gpt", "value": "4"},
            ]
        }
    ]

    out = to_mlx_jsonl(records, tmp_path)

    assert _read_jsonl(out) == [
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
    ]


def test_alpaca_with_input_joins_user_turn(tmp_path: Path) -> None:
    records = [
        {
            "instruction": "Add the numbers.",
            "input": "1 and 2",
            "output": "3",
        }
    ]

    out = to_mlx_jsonl(records, tmp_path)

    assert _read_jsonl(out) == [
        {
            "messages": [
                {"role": "user", "content": "Add the numbers.\n\n1 and 2"},
                {"role": "assistant", "content": "3"},
            ]
        }
    ]


def test_alpaca_without_input_uses_instruction_only(tmp_path: Path) -> None:
    records = [{"instruction": "Say hi.", "output": "Hi"}]

    out = to_mlx_jsonl(records, tmp_path)

    assert _read_jsonl(out) == [
        {
            "messages": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": "Hi"},
            ]
        }
    ]


def test_destination_jsonl_file_keeps_that_path(tmp_path: Path) -> None:
    dest = tmp_path / "custom.jsonl"
    records = [
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        }
    ]

    out = to_mlx_jsonl(records, dest)

    assert out == dest
    assert dest.exists()


def test_split_valid_writes_valid_jsonl(tmp_path: Path) -> None:
    records = [
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        }
    ]

    out = to_mlx_jsonl(records, tmp_path, split="valid")

    assert out == tmp_path / "valid.jsonl"


def test_source_jsonl_file_is_loaded(tmp_path: Path) -> None:
    src = tmp_path / "raw.jsonl"
    src.write_text(
        json.dumps(
            {
                "instruction": "Q",
                "output": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = to_mlx_jsonl(src, tmp_path / "out")

    assert _read_jsonl(out) == [
        {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }
    ]


def test_unknown_format_raises(tmp_path: Path) -> None:
    with pytest.raises(UnknownRecordFormatError):
        to_mlx_jsonl([{"prompt": "x"}], tmp_path)


def test_unknown_sharegpt_speaker_raises(tmp_path: Path) -> None:
    records = [
        {
            "conversations": [
                {"from": "wizard", "value": "hi"},
                {"from": "gpt", "value": "yo"},
            ]
        }
    ]

    with pytest.raises(UnknownSpeakerError):
        to_mlx_jsonl(records, tmp_path)


def test_missing_assistant_raises(tmp_path: Path) -> None:
    records = [{"messages": [{"role": "user", "content": "Hi"}]}]

    with pytest.raises(MissingAssistantError):
        to_mlx_jsonl(records, tmp_path)


def test_empty_source_raises(tmp_path: Path) -> None:
    with pytest.raises(EmptyDatasetError):
        to_mlx_jsonl([], tmp_path)


_OPENAI_ROW: dict[str, list[dict[str, str]]] = {
    "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
}


def test_huggingface_dataset_jsonl_matches_plain_list(tmp_path: Path) -> None:
    list_out = to_mlx_jsonl([_OPENAI_ROW], tmp_path / "list")

    ds_out = to_mlx_jsonl(Dataset.from_list([_OPENAI_ROW]), tmp_path / "ds")

    assert _read_jsonl(ds_out) == _read_jsonl(list_out)


def test_huggingface_datasetdict_train_split_jsonl_matches_plain_list(
    tmp_path: Path,
) -> None:
    ds = Dataset.from_list([_OPENAI_ROW])
    list_out = to_mlx_jsonl([_OPENAI_ROW], tmp_path / "list")

    split_out = to_mlx_jsonl(DatasetDict({"train": ds})["train"], tmp_path / "split")

    assert _read_jsonl(split_out) == _read_jsonl(list_out)


def test_bare_datasetdict_raises_datasetdict_error(tmp_path: Path) -> None:
    ds = Dataset.from_list([_OPENAI_ROW])

    with pytest.raises(DatasetDictError) as exc_info:
        to_mlx_jsonl(DatasetDict({"train": ds}), tmp_path)

    assert str(exc_info.value) == "pass DatasetDict['train'], not the DatasetDict"
