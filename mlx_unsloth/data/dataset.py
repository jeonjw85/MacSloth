"""Convert ShareGPT, Alpaca, and OpenAI messages into mlx-lm JSONL."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, TypeAlias

from datasets import Dataset, DatasetDict

from mlx_unsloth.data.errors import (
    DatasetDictError,
    EmptyDatasetError,
    FieldTypeError,
    JsonlParseError,
    MissingAssistantError,
    MissingFieldError,
    UnknownRecordFormatError,
    UnknownSpeakerError,
    UnknownSplitError,
)

RawRecord: TypeAlias = Mapping[str, str | list[Mapping[str, str]]]


class Role(str, Enum):
    """Chat roles that mlx-lm's ChatDataset accepts."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


SPEAKER_ROLES: Final[dict[str, Role]] = {
    "system": Role.SYSTEM,
    "human": Role.USER,
    "user": Role.USER,
    "gpt": Role.ASSISTANT,
    "assistant": Role.ASSISTANT,
}


class Split(str, Enum):
    """Filenames mlx-lm looks for under a data directory."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn in an mlx-lm chat record."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ChatRecord:
    """Parsed chat example ready to write as JSONL."""

    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        """Reject records that have no assistant completion."""
        if not any(message.role is Role.ASSISTANT for message in self.messages):
            raise MissingAssistantError

    def to_mlx_row(self) -> dict[str, list[dict[str, str]]]:
        """Return the mlx-lm chat JSON object."""
        return {
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in self.messages
            ]
        }


def _require_str(value: object, field: str) -> str:
    match value:
        case str() as text:
            return text
        case _:
            raise FieldTypeError(field=field, got=type(value).__name__)


def _role_for_speaker(speaker: str) -> Role:
    try:
        return SPEAKER_ROLES[speaker.lower()]
    except KeyError:
        raise UnknownSpeakerError(speaker=speaker) from None


def _parse_openai_turn(turn: object) -> ChatMessage:
    match turn:
        case dict() as fields:
            try:
                role_raw = fields["role"]
                content_raw = fields["content"]
            except KeyError as exc:
                raise MissingFieldError(field=str(exc).strip("'")) from None
            try:
                role = Role(_require_str(role_raw, "role"))
            except ValueError:
                raise UnknownSpeakerError(
                    speaker=_require_str(role_raw, "role")
                ) from None
            return ChatMessage(role=role, content=_require_str(content_raw, "content"))
        case _:
            raise FieldTypeError(field="messages", got=type(turn).__name__)


def _parse_sharegpt_turn(turn: object) -> ChatMessage:
    match turn:
        case dict() as fields if "from" in fields:
            try:
                content_raw = fields["value"]
            except KeyError:
                raise MissingFieldError(field="value") from None
            speaker = _require_str(fields["from"], "from")
            return ChatMessage(
                role=_role_for_speaker(speaker),
                content=_require_str(content_raw, "value"),
            )
        case dict() as fields if "role" in fields:
            return _parse_openai_turn(fields)
        case dict() as fields:
            raise UnknownRecordFormatError(keys=tuple(sorted(fields)))
        case _:
            raise FieldTypeError(field="conversations", got=type(turn).__name__)


def _parse_openai(record: Mapping[str, str | list[Mapping[str, str]]]) -> ChatRecord:
    raw_messages = record["messages"]
    match raw_messages:
        case list() as turns:
            return ChatRecord(
                messages=tuple(_parse_openai_turn(turn) for turn in turns)
            )
        case _:
            raise FieldTypeError(field="messages", got=type(raw_messages).__name__)


def _parse_sharegpt(record: Mapping[str, str | list[Mapping[str, str]]]) -> ChatRecord:
    raw_turns = record["conversations"]
    match raw_turns:
        case list() as turns:
            return ChatRecord(
                messages=tuple(_parse_sharegpt_turn(turn) for turn in turns)
            )
        case _:
            raise FieldTypeError(field="conversations", got=type(raw_turns).__name__)


def _parse_alpaca(record: Mapping[str, str | list[Mapping[str, str]]]) -> ChatRecord:
    try:
        instruction = _require_str(record["instruction"], "instruction")
        output = _require_str(record["output"], "output")
    except KeyError as exc:
        raise MissingFieldError(field=str(exc).strip("'")) from None
    extra = _require_str(record.get("input", ""), "input")
    user = instruction if extra == "" else f"{instruction}\n\n{extra}"
    return ChatRecord(
        messages=(
            ChatMessage(role=Role.USER, content=user),
            ChatMessage(role=Role.ASSISTANT, content=output),
        )
    )


def _parse_record(raw: object) -> ChatRecord:
    match raw:
        case dict() as record if "messages" in record:
            return _parse_openai(record)
        case dict() as record if "conversations" in record:
            return _parse_sharegpt(record)
        case dict() as record if "instruction" in record:
            return _parse_alpaca(record)
        case dict() as record:
            raise UnknownRecordFormatError(keys=tuple(sorted(record)))
        case _:
            raise FieldTypeError(field="record", got=type(raw).__name__)


def _load_file(path: Path) -> list[ChatRecord]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        loaded: object = json.loads(text)
        match loaded:
            case list() as rows:
                return [_parse_record(row) for row in rows]
            case _:
                raise FieldTypeError(field="file", got=type(loaded).__name__)
    records: list[ChatRecord] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            continue
        try:
            loaded_line: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JsonlParseError(path=path, line=line_no) from exc
        records.append(_parse_record(loaded_line))
    return records


def _load_records(
    source: Sequence[RawRecord] | Path | str | Dataset,
) -> list[ChatRecord]:
    match source:
        case str() as path:
            return _load_file(Path(path))
        case Path() as path:
            return _load_file(path)
        case DatasetDict():
            raise DatasetDictError
        case Dataset() as dataset:
            return [_parse_record(record) for record in dataset]
        case _:
            return [_parse_record(record) for record in source]


def to_mlx_jsonl(
    source: Sequence[RawRecord] | Path | str | Dataset,
    destination: Path | str,
    split: str = "train",
) -> Path:
    """Convert records to mlx-lm chat JSONL.

    `destination` ending in `.jsonl` is the output file. Otherwise it is a
    directory and the file is `{split}.jsonl`.

    Args:
        source: Rows, a Hugging Face `Dataset`, or a `.json` / `.jsonl` path.
        destination: Output file or directory.
        split: `train`, `valid`, or `test` when writing a directory.

    Returns:
        Path of the written JSONL file.
    """
    records = _load_records(source)
    if not records:
        raise EmptyDatasetError
    try:
        named_split = Split(split)
    except ValueError:
        raise UnknownSplitError(split=split) from None
    dest = Path(destination)
    if dest.suffix == ".jsonl":
        dest.parent.mkdir(parents=True, exist_ok=True)
        path = dest
    else:
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{named_split.value}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_mlx_row(), ensure_ascii=False))
            handle.write("\n")
    return path
