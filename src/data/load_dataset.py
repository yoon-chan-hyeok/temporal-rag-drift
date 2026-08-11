"""Load and validate canonical temporal RAG JSONL datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, resolve_path


@dataclass(frozen=True)
class DatasetRecord:
    """One factual QA example with current and stale retrieved evidence."""

    id: str
    question: str
    gold_answer: str
    current_docs: list[str]
    stale_docs: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    stale_answer: str | None = None

    @classmethod
    def from_mapping(cls, row: dict[str, Any], line_number: int | None = None) -> "DatasetRecord":
        """Validate a raw JSON object and return a canonical record."""
        location = f" at row {line_number}" if line_number is not None else ""
        required = ("id", "question", "gold_answer", "current_docs", "stale_docs")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Missing required keys{location}: {missing}")

        current_docs = row["current_docs"]
        stale_docs = row["stale_docs"]
        if not isinstance(current_docs, list) or not all(isinstance(item, str) for item in current_docs):
            raise ValueError(f"current_docs must be a list of strings{location}")
        if not isinstance(stale_docs, list) or not all(isinstance(item, str) for item in stale_docs):
            raise ValueError(f"stale_docs must be a list of strings{location}")

        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be an object when provided{location}")

        stale_answer = row.get("stale_answer")
        if stale_answer is None:
            stale_answer = metadata.get("stale_answer") or metadata.get("old_answer") or metadata.get("previous_answer")

        return cls(
            id=str(row["id"]),
            question=str(row["question"]),
            gold_answer=str(row["gold_answer"]),
            current_docs=list(current_docs),
            stale_docs=list(stale_docs),
            metadata=dict(metadata),
            stale_answer=str(stale_answer) if stale_answer is not None else None,
        )


def load_dataset(path: str | Path, max_questions: int | None = None, base_dir: str | Path | None = None) -> list[DatasetRecord]:
    """Load canonical JSONL data and optionally truncate it."""
    resolved = resolve_path(path, base_dir=base_dir)
    rows = read_jsonl(resolved)
    records = [DatasetRecord.from_mapping(row, line_number=index) for index, row in enumerate(rows, start=1)]
    if max_questions is not None:
        records = records[: int(max_questions)]
    return records


def records_by_id(records: list[DatasetRecord]) -> dict[str, DatasetRecord]:
    """Return a mapping from record id to record, rejecting duplicate ids."""
    mapping: dict[str, DatasetRecord] = {}
    for record in records:
        if record.id in mapping:
            raise ValueError(f"Duplicate dataset id: {record.id}")
        mapping[record.id] = record
    return mapping
