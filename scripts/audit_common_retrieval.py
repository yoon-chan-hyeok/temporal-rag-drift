"""Audit answer/evidence/qrel coverage for common-retriever cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit retrieval support coverage.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--per-question-output", required=True)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalized(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'")


def text_present(value: str, rows: list[dict[str, Any]]) -> bool:
    needle = normalized(value)
    if not needle:
        return False
    return any(needle in normalized(row.get("text")) for row in rows)


def title_present(value: str, rows: list[dict[str, Any]]) -> bool:
    needle = normalized(value)
    if not needle:
        return False
    return any(needle == normalized(row.get("title")) for row in rows)


def rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def audit(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_question: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") or {}
        retrieval = metadata.get("retrieval") or {}
        source = str(metadata.get("source") or "")
        is_clark = source == "clark_news"
        old_rows = retrieval.get("stale") or []
        new_rows = retrieval.get("current" if is_clark else "fresh") or []
        mixed_rows = retrieval.get("mixed") or []
        has_mixed = bool(mixed_rows)

        old_answer = str(
            record.get("stale_answer")
            or metadata.get("old_answer")
            or ""
        )
        new_answer = str(
            record.get("gold_answer")
            or metadata.get("current_answer")
            or ""
        )
        title = str(metadata.get("document_title") or "")
        old_evidence = str(metadata.get("old_evidence") or "")
        new_evidence = str(metadata.get("current_evidence") or "")
        official_hits = metadata.get("official_support_hits") or {}
        old_official = official_hits.get("stale") or {}
        new_official = official_hits.get("current") or {}

        row = {
            "id": record.get("id"),
            "source": source,
            "change_label": metadata.get("change_label") or metadata.get("answer_status"),
            "question": record.get("question"),
            "old_answer": old_answer,
            "new_answer": new_answer,
            "old_answer_in_old_topk": text_present(old_answer, old_rows),
            "new_answer_in_new_topk": text_present(new_answer, new_rows),
            "old_answer_in_mixed_topk": text_present(old_answer, mixed_rows) if has_mixed else None,
            "new_answer_in_mixed_topk": text_present(new_answer, mixed_rows) if has_mixed else None,
            "target_title_in_old_topk": title_present(title, old_rows),
            "target_title_in_new_topk": title_present(title, new_rows),
            "target_title_in_mixed_topk": title_present(title, mixed_rows) if has_mixed else None,
            "old_evidence_in_old_topk": text_present(old_evidence, old_rows),
            "new_evidence_in_new_topk": text_present(new_evidence, new_rows),
            "old_evidence_in_mixed_topk": text_present(old_evidence, mixed_rows) if has_mixed else None,
            "new_evidence_in_mixed_topk": text_present(new_evidence, mixed_rows) if has_mixed else None,
            "official_old_support_mapped": int(old_official.get("support_url_count", 0)) > 0,
            "official_new_support_mapped": int(new_official.get("support_url_count", 0)) > 0,
            "official_old_hit_at_k": bool(old_official.get("hit_at_k", False)),
            "official_new_hit_at_k": bool(new_official.get("hit_at_k", False)),
            "official_old_best_rank": old_official.get("best_rank"),
            "official_new_best_rank": new_official.get("best_rank"),
        }
        per_question.append(row)

    boolean_fields = [
        "old_answer_in_old_topk",
        "new_answer_in_new_topk",
        "old_answer_in_mixed_topk",
        "new_answer_in_mixed_topk",
        "target_title_in_old_topk",
        "target_title_in_new_topk",
        "target_title_in_mixed_topk",
        "old_evidence_in_old_topk",
        "new_evidence_in_new_topk",
        "old_evidence_in_mixed_topk",
        "new_evidence_in_mixed_topk",
    ]
    summary: dict[str, Any] = {
        "records": len(per_question),
        "source_counts": {},
    }
    for row in per_question:
        source = str(row["source"])
        summary["source_counts"][source] = summary["source_counts"].get(source, 0) + 1
    for field in boolean_fields:
        summary[f"{field}_rate"] = rate(
            [bool(row[field]) for row in per_question if row[field] is not None]
        )

    old_mapped = [row for row in per_question if row["official_old_support_mapped"]]
    new_mapped = [row for row in per_question if row["official_new_support_mapped"]]
    summary["official_old_support_mapped"] = len(old_mapped)
    summary["official_new_support_mapped"] = len(new_mapped)
    summary["official_old_hit_at_k_rate"] = rate(
        [bool(row["official_old_hit_at_k"]) for row in old_mapped]
    )
    summary["official_new_hit_at_k_rate"] = rate(
        [bool(row["official_new_hit_at_k"]) for row in new_mapped]
    )
    summary["official_both_hit_at_k_rate"] = rate(
        [
            bool(row["official_old_hit_at_k"] and row["official_new_hit_at_k"])
            for row in per_question
            if row["official_old_support_mapped"] and row["official_new_support_mapped"]
        ]
    )
    return per_question, summary


def main() -> None:
    args = parse_args()
    records = read_jsonl(resolve(args.input))
    per_question, summary = audit(records)
    summary_output = resolve(args.summary_output)
    per_question_output = resolve(args.per_question_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    per_question_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(per_question[0]) if per_question else []
    with per_question_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_question)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(summary_output)
    print(per_question_output)


if __name__ == "__main__":
    main()
