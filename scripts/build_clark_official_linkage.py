"""Reconstruct official CLARK question-fact-article provenance.

CLARK questions are deterministically generated from the human-validated rows in
``property_to_results.csv``. This script replays those templates and writes one
qrel row per row in ``questions.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import write_jsonl
from src.utils.text import compact_whitespace, normalize_text, stable_text_hash


PROPERTY_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "employer": {
        "subject_q": ("Who is the employer of {subject}?", "{object}"),
        "binary_q": ("Is {subject} an employee of {object}?", "yes"),
    },
    "chief executive officer": {
        "subject_q": ("Who is the CEO of {subject}?", "{object}"),
        "object_q": ("What company is {object} the CEO of?", "{subject}"),
        "binary_q": ("Is {object} the CEO of {subject}?", "yes"),
    },
    "chairperson": {
        "subject_q": ("Who is the chairperson of {subject}?", "{object}"),
        "object_q": ("What organization is {object} the chairperson of?", "{subject}"),
        "binary_q": ("Is {object} the chairperson of {subject}?", "yes"),
    },
    "head of state": {
        "subject_q": ("Who is the head of state of {subject}?", "{object}"),
        "object_q": ("Where is {object} the head of state of?", "{subject}"),
        "binary_q": ("Is {object} the head of state of {subject}?", "yes"),
    },
    "position held": {
        "subject_q": ("What government position does {subject} hold?", "{object}"),
        "binary_q": ("Does {subject} hold government position {object}?", "yes"),
    },
    "member of sports team": {
        "subject_q": ("What sports team is {subject} a member of?", "{object}"),
        "binary_q": ("Is {subject} a member of {object}?", "yes"),
    },
    "unmarried partner": {
        "subject_q": ("Who is the unmarried partner of {subject}?", "{object}"),
        "object_q": ("Who is the unmarried partner of {object}?", "{subject}"),
        "binary_q": ("Is {object} the unmarried partner of {subject}?", "yes"),
    },
    "residence": {
        "subject_q": ("Where does {subject} reside?", "{object}"),
        "binary_q": ("Does {subject} reside in {object}?", "yes"),
    },
    "headquarters location": {
        "subject_q": ("Where is the headquarters location of {subject}?", "{object}"),
        "binary_q": ("Is the headquarters location of {subject} in {object}?", "yes"),
    },
    "member of political party": {
        "subject_q": ("What political party is {subject} a member of?", "{object}"),
        "binary_q": ("Is {subject} a member of {object}?", "yes"),
    },
}


@dataclass(frozen=True)
class FactRow:
    row_index: int
    fact_id: str
    subject: str
    property_label: str
    object_label: str
    start_ts: pd.Timestamp | None
    end_ts: pd.Timestamp | None
    selected_link: str
    source_date: pd.Timestamp | None
    span: str
    annotator: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official CLARK question-to-article qrels.")
    parser.add_argument(
        "--property-results",
        default="data/external/clark/property_to_results.csv",
    )
    parser.add_argument(
        "--questions-csv",
        default="data/external/clark/questions.csv",
    )
    parser.add_argument(
        "--external-sources",
        default="data/external/clark/external_sources.json",
    )
    parser.add_argument(
        "--output",
        default="data/external/clark/question_article_qrels.jsonl",
    )
    parser.add_argument(
        "--audit-output",
        default="data/external/clark/official_linkage_audit.json",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return compact_whitespace(str(value))


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    text = clean(value)
    if not text:
        return None
    timestamp = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp


def timestamp_text(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None else None


def interval_overlap(
    left_start: pd.Timestamp | None,
    left_end: pd.Timestamp | None,
    right_start: pd.Timestamp | None,
    right_end: pd.Timestamp | None,
) -> bool:
    if left_end is not None and right_start is not None and left_end <= right_start:
        return False
    if right_end is not None and left_start is not None and right_end <= left_start:
        return False
    return True


def interval_match_score(
    question_start: pd.Timestamp | None,
    question_end: pd.Timestamp | None,
    fact_start: pd.Timestamp | None,
    fact_end: pd.Timestamp | None,
) -> int:
    score = 0
    if question_start == fact_start:
        score += 2
    if question_end == fact_end:
        score += 2
    if interval_overlap(question_start, question_end, fact_start, fact_end):
        score += 1
    return score


def load_facts(path: Path) -> list[FactRow]:
    frame = pd.read_csv(path)
    facts: list[FactRow] = []
    for row_index, row in frame.iterrows():
        subject = clean(row.get("subjectLabel"))
        property_label = clean(row.get("propertyLabel"))
        object_label = clean(row.get("objectLabel"))
        selected_link = clean(row.get("selected_link"))
        fact_id = f"clark_fact_{stable_text_hash(f'{row_index}|{subject}|{property_label}|{object_label}|{selected_link}')}"
        facts.append(
            FactRow(
                row_index=int(row_index),
                fact_id=fact_id,
                subject=subject,
                property_label=property_label,
                object_label=object_label,
                start_ts=parse_timestamp(row.get("startDate")),
                end_ts=parse_timestamp(row.get("endDate")),
                selected_link=selected_link,
                source_date=parse_timestamp(row.get("source_date")),
                span=clean(row.get("span")),
                annotator=clean(row.get("annotator")),
            )
        )
    return facts


def generated_entries(fact: FactRow) -> list[tuple[str, str, str]]:
    if not fact.object_label:
        if fact.property_label == "chairperson":
            return [
                (
                    "subject_q",
                    f"Who is the chairperson of {fact.subject}?",
                    "no one",
                ),
                (
                    "absence_q",
                    f"Is there a chairperson of {fact.subject}?",
                    "no",
                ),
            ]
        if fact.property_label == "unmarried partner":
            return [
                (
                    "subject_q",
                    f"Who is the unmarried partner of {fact.subject}?",
                    "no one",
                ),
                (
                    "absence_q",
                    f"Does {fact.subject} have a partner?",
                    "no",
                ),
            ]
    templates = PROPERTY_TEMPLATES.get(fact.property_label, {})
    entries: list[tuple[str, str, str]] = []
    for template_name, (question_template, answer_template) in templates.items():
        question = question_template.format(
            subject=fact.subject,
            object=fact.object_label,
        )
        answer = answer_template.format(
            subject=fact.subject,
            object=fact.object_label,
        )
        entries.append((template_name, compact_whitespace(question), compact_whitespace(answer)))
    return entries


def fact_payload(
    fact: FactRow,
    external_sources: dict[str, Any],
    *,
    support_kind: str,
) -> dict[str, Any]:
    versions = external_sources.get(fact.selected_link, {})
    archive_versions: list[dict[str, str]] = []
    if isinstance(versions, dict):
        for timestamp_key, payload in versions.items():
            if not isinstance(payload, dict):
                continue
            archive_versions.append(
                {
                    "timestamp": clean(payload.get("source_timestamp")) or clean(timestamp_key),
                    "archive_url": clean(payload.get("archive_url")) or fact.selected_link,
                }
            )
    archive_versions.sort(key=lambda row: row["timestamp"])
    return {
        "fact_id": fact.fact_id,
        "fact_row_index": fact.row_index,
        "subject": fact.subject,
        "property": fact.property_label,
        "object": fact.object_label,
        "valid_start": timestamp_text(fact.start_ts),
        "valid_end": timestamp_text(fact.end_ts),
        "source_url": fact.selected_link,
        "source_date": timestamp_text(fact.source_date),
        "evidence_span": fact.span,
        "annotator": fact.annotator,
        "support_kind": support_kind,
        "archive_versions": archive_versions,
    }


def build_linkage(
    property_results: Path,
    questions_csv: Path,
    external_sources_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts = load_facts(property_results)
    external_sources = json.loads(external_sources_path.read_text(encoding="utf-8"))
    questions = pd.read_csv(questions_csv)

    direct_index: dict[tuple[str, str], list[tuple[str, FactRow]]] = defaultdict(list)
    binary_target_index: dict[str, list[FactRow]] = defaultdict(list)
    absence_target_index: dict[str, list[FactRow]] = defaultdict(list)
    subject_property_index: dict[tuple[str, str], list[FactRow]] = defaultdict(list)

    for fact in facts:
        subject_property_index[(fact.subject, fact.property_label)].append(fact)
        for template_name, question, answer in generated_entries(fact):
            question_key = normalize_text(question)
            answer_key = normalize_text(answer)
            direct_index[(question_key, answer_key)].append((template_name, fact))
            if template_name == "binary_q":
                binary_target_index[question_key].append(fact)
            elif template_name == "absence_q":
                absence_target_index[question_key].append(fact)

    qrels: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    relation_status: dict[str, Counter[str]] = defaultdict(Counter)

    for row_index, row in questions.iterrows():
        question = clean(row.get("Question"))
        answer = clean(row.get("Answer"))
        question_type = clean(row.get("Question type"))
        start_ts = parse_timestamp(row.get("Start timestamp"))
        end_ts = parse_timestamp(row.get("End timestamp"))
        known_start_ts = parse_timestamp(row.get("Known start timestamp"))
        known_end_ts = parse_timestamp(row.get("Known end timestamp"))
        question_key = normalize_text(question)
        answer_key = normalize_text(answer)

        support_facts: list[tuple[FactRow, str]] = []
        direct_matches = direct_index.get((question_key, answer_key), [])
        if direct_matches:
            scores = [
                interval_match_score(start_ts, end_ts, fact.start_ts, fact.end_ts)
                for _, fact in direct_matches
            ]
            best_score = max(scores)
            support_facts = [
                (fact, "direct")
                for (_, fact), score in zip(direct_matches, scores)
                if score == best_score
            ]
            status = "direct"
        elif answer_key == normalize_text("no") and binary_target_index.get(question_key):
            competing: dict[str, FactRow] = {}
            target_facts = binary_target_index[question_key]
            for target in target_facts:
                peers = subject_property_index[(target.subject, target.property_label)]
                for peer in peers:
                    if peer.fact_id == target.fact_id or peer.object_label == target.object_label:
                        continue
                    if interval_overlap(start_ts, end_ts, peer.start_ts, peer.end_ts):
                        competing[peer.fact_id] = peer
            support_facts = [
                (fact, "competing_current_fact")
                for fact in sorted(
                    competing.values(),
                    key=lambda item: (
                        item.source_date or pd.Timestamp.min.tz_localize("UTC"),
                        item.fact_id,
                    ),
                )
            ]
            status = "binary_no_competing" if support_facts else "binary_no_unmapped"
        elif answer_key == normalize_text("yes") and absence_target_index.get(question_key):
            competing = {}
            for target in absence_target_index[question_key]:
                peers = subject_property_index[(target.subject, target.property_label)]
                for peer in peers:
                    if not peer.object_label:
                        continue
                    if interval_overlap(start_ts, end_ts, peer.start_ts, peer.end_ts):
                        competing[peer.fact_id] = peer
            support_facts = [
                (fact, "competing_present_fact")
                for fact in sorted(
                    competing.values(),
                    key=lambda item: (
                        item.source_date or pd.Timestamp.min.tz_localize("UTC"),
                        item.fact_id,
                    ),
                )
            ]
            status = "absence_yes_competing" if support_facts else "absence_yes_unmapped"
        else:
            status = "unmapped"

        support_payloads = [
            fact_payload(fact, external_sources, support_kind=support_kind)
            for fact, support_kind in support_facts
        ]
        qrel_id = f"clark_qrel_{stable_text_hash(f'{row_index}|{question}|{answer}|{start_ts}|{end_ts}')}"
        qrel = {
            "qrel_id": qrel_id,
            "question_row_index": int(row_index),
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "start_timestamp": timestamp_text(start_ts),
            "end_timestamp": timestamp_text(end_ts),
            "known_start_timestamp": timestamp_text(known_start_ts),
            "known_end_timestamp": timestamp_text(known_end_ts),
            "mapping_status": status,
            "support_facts": support_payloads,
            "support_urls": sorted(
                {
                    payload["source_url"]
                    for payload in support_payloads
                    if payload.get("source_url")
                }
            ),
        }
        qrels.append(qrel)
        status_counts[status] += 1
        relation_status[question_type][status] += 1

    unique_fact_links = {fact.selected_link for fact in facts if fact.selected_link}
    source_links = set(external_sources)
    support_with_archives = sum(
        bool(fact.get("archive_versions"))
        for qrel in qrels
        for fact in qrel["support_facts"]
    )
    support_total = sum(len(qrel["support_facts"]) for qrel in qrels)
    audit = {
        "property_results": str(property_results),
        "questions_csv": str(questions_csv),
        "external_sources": str(external_sources_path),
        "input_sha256": {
            "property_results": file_sha256(property_results),
            "questions_csv": file_sha256(questions_csv),
            "external_sources": file_sha256(external_sources_path),
        },
        "fact_rows": len(facts),
        "question_rows": len(qrels),
        "unique_fact_source_urls": len(unique_fact_links),
        "fact_source_urls_in_external_sources": len(unique_fact_links & source_links),
        "fact_source_url_coverage": (
            len(unique_fact_links & source_links) / len(unique_fact_links)
            if unique_fact_links
            else 0.0
        ),
        "mapping_status_counts": dict(sorted(status_counts.items())),
        "mapped_question_rows": sum(
            count
            for status, count in status_counts.items()
            if status not in {"unmapped", "binary_no_unmapped", "absence_yes_unmapped"}
        ),
        "question_mapping_coverage": (
            sum(
                count
                for status, count in status_counts.items()
                if status not in {"unmapped", "binary_no_unmapped", "absence_yes_unmapped"}
            )
            / len(qrels)
            if qrels
            else 0.0
        ),
        "support_fact_references": support_total,
        "support_fact_references_with_archive_versions": support_with_archives,
        "support_archive_coverage": (
            support_with_archives / support_total if support_total else 0.0
        ),
        "relation_status": {
            relation: dict(sorted(counts.items()))
            for relation, counts in sorted(relation_status.items())
        },
    }
    return qrels, audit


def main() -> None:
    args = parse_args()
    output = resolve(args.output)
    audit_output = resolve(args.audit_output)
    qrels, audit = build_linkage(
        resolve(args.property_results),
        resolve(args.questions_csv),
        resolve(args.external_sources),
    )
    write_jsonl(qrels, output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(output)
    print(audit_output)


if __name__ == "__main__":
    main()
