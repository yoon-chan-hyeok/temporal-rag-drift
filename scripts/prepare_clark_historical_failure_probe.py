"""Build a blinded P1-P5 diagnostic cohort from completed CLARK experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import write_json


CHANGED_PREDICTIONS = (
    ROOT
    / "outputs"
    / "runs"
    / "clark_changed_primary_luna"
    / "analysis_changed_only"
    / "locked_changed_predictions.csv"
)
FOUR_STATE_PREDICTIONS = (
    ROOT
    / "outputs"
    / "runs"
    / "clark_sequential_luna"
    / "analysis_natural_disjoint_primary"
    / "locked_4state_predictions.csv"
)
SOURCE_DATASETS = (
    ROOT / "data" / "processed" / "clark_changed_primary" / "calibration_changed.jsonl",
    ROOT / "data" / "processed" / "clark_changed_primary" / "validation_changed.jsonl",
    ROOT / "data" / "processed" / "clark_changed_primary" / "locked_changed.jsonl",
    ROOT / "data" / "processed" / "clark_sequential_ml" / "calibration_changed_raw.jsonl",
    ROOT / "data" / "processed" / "clark_sequential_ml" / "locked_changed_raw.jsonl",
    ROOT / "data" / "processed" / "clark_sequential_ml" / "calibration_stable_raw.jsonl",
    ROOT / "data" / "processed" / "clark_sequential_ml" / "locked_stable_raw.jsonl",
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "clark_historical_failure_probe"
CONDITIONS = (
    "p1_natural",
    "p2_support_presence",
    "p3_support_first",
    "p4_evidence_only",
    "p5_fact_card",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--new-degradation", type=int, default=22)
    parser.add_argument("--persistent", type=int, default=18)
    parser.add_argument("--adaptive-control", type=int, default=22)
    parser.add_argument("--normal-control", type=int, default=22)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def record_richness(record: dict[str, Any]) -> tuple[int, int, int]:
    metadata = record.get("metadata") or {}
    linkage = metadata.get("official_linkage") or {}
    current_linkage = linkage.get("current") or {}
    facts = current_linkage.get("support_facts") or []
    details = metadata.get("retrieval_details") or {}
    current_details = details.get("current_docs") or []
    return (len(facts), len(current_details), len(record.get("current_docs") or []))


def load_source_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in SOURCE_DATASETS:
        for record in read_jsonl(path):
            question_id = str(record.get("id", ""))
            if not question_id:
                continue
            previous = records.get(question_id)
            if previous is None or record_richness(record) > record_richness(previous):
                copied = dict(record)
                copied["_source_dataset"] = str(path.relative_to(ROOT))
                records[question_id] = copied
    return records


def metadata_for(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def current_linkage(record: dict[str, Any]) -> dict[str, Any]:
    linkage = metadata_for(record).get("official_linkage") or {}
    value = linkage.get("current") if isinstance(linkage, dict) else None
    return value if isinstance(value, dict) else {}


def relation(record: dict[str, Any]) -> str:
    return str(current_linkage(record).get("question_type") or "unknown").strip().lower()


def transition(record: dict[str, Any]) -> str:
    metadata = metadata_for(record)
    time_x = str(metadata.get("time_x") or "")[:10]
    time_y = str(metadata.get("time_y") or "")[:10]
    return f"{time_x}_to_{time_y}"


def binary_style(record: dict[str, Any]) -> str:
    answer = str(record.get("gold_answer", "")).strip().lower()
    return "binary" if answer in {"yes", "no"} else "open"


def probe_eligible(record: dict[str, Any]) -> bool:
    return bool(record.get("current_docs")) and bool(support_facts(record))


def stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    return (transition(record), relation(record), binary_style(record))


def select_rows(
    rows: list[dict[str, str]],
    state: str,
    limit: int,
    records: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("outcome_state") == state
        and row.get("question_id") in records
        and probe_eligible(records[str(row.get("question_id"))])
    ]
    selected.sort(key=lambda row: str(row.get("question_id")))
    return selected[:limit]


def select_matched_controls(
    candidates: list[dict[str, str]],
    targets: list[dict[str, str]],
    limit: int,
    records: dict[str, dict[str, Any]],
    excluded: set[str],
) -> list[dict[str, str]]:
    pool = [
        row
        for row in candidates
        if row.get("question_id") not in excluded
        and str(row.get("question_id")) in records
        and probe_eligible(records[str(row.get("question_id"))])
    ]
    pool.sort(key=lambda row: str(row.get("question_id")))
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pool:
        record = records.get(str(row.get("question_id")))
        if record is not None:
            buckets[stratum(record)].append(row)

    desired = Counter(
        stratum(records[str(row["question_id"])])
        for row in targets
        if str(row.get("question_id")) in records
    )
    chosen: list[dict[str, str]] = []
    for key, count in desired.most_common():
        take = min(count, len(buckets.get(key, [])), limit - len(chosen))
        chosen.extend(buckets.get(key, [])[:take])
        buckets[key] = buckets.get(key, [])[take:]
        if len(chosen) >= limit:
            break

    chosen_ids = {str(row.get("question_id")) for row in chosen}
    if len(chosen) < limit:
        target_rel_binary = Counter((key[1], key[2]) for key in desired.elements())
        for key, _ in target_rel_binary.most_common():
            for row in pool:
                question_id = str(row.get("question_id"))
                if question_id in chosen_ids:
                    continue
                record = records[question_id]
                if (relation(record), binary_style(record)) == key:
                    chosen.append(row)
                    chosen_ids.add(question_id)
                    if len(chosen) >= limit:
                        return chosen

    for row in pool:
        question_id = str(row.get("question_id"))
        if question_id not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(question_id)
            if len(chosen) >= limit:
                break
    return chosen


def normalized_url(value: str) -> str:
    parsed = urlparse(value.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def support_hits(record: dict[str, Any]) -> dict[str, Any]:
    hits = metadata_for(record).get("official_support_hits") or {}
    value = hits.get("current") if isinstance(hits, dict) else None
    return value if isinstance(value, dict) else {}


def support_facts(record: dict[str, Any]) -> list[dict[str, Any]]:
    facts = current_linkage(record).get("support_facts") or []
    return [fact for fact in facts if isinstance(fact, dict) and fact.get("evidence_span")]


def rewrite_rank(document: str, rank: int) -> str:
    header = f"[retrieved rank {rank}]"
    if re.match(r"^\[retrieved rank \d+\]", document):
        return re.sub(r"^\[retrieved rank \d+\]", header, document, count=1)
    return f"{header}\n{document}"


def rerank_documents(documents: list[str]) -> list[str]:
    return [rewrite_rank(document, index) for index, document in enumerate(documents, start=1)]


def evidence_document(fact: dict[str, Any], rank: int = 1) -> str:
    source_url = str(fact.get("source_url") or "")
    host = urlparse(source_url).netloc.removeprefix("www.") or "source"
    return "\n".join(
        [
            f"[retrieved rank {rank}]",
            f"Timestamp: {fact.get('source_date') or fact.get('valid_start') or 'unknown'}",
            f"Title: Evidence from {host}",
            f"Source URL: {source_url}",
            f"Document ID: {fact.get('fact_id') or 'official-evidence'}",
            f"Chunk: {fact.get('fact_row_index') or 0}",
            str(fact.get("evidence_span") or "").strip(),
        ]
    )


def find_support_index(record: dict[str, Any], documents: list[str]) -> int | None:
    hit = support_hits(record)
    best_rank = hit.get("best_rank")
    if best_rank is not None:
        try:
            index = int(best_rank) - 1
            if 0 <= index < len(documents):
                return index
        except (TypeError, ValueError):
            pass
    urls = {
        normalized_url(str(url))
        for url in current_linkage(record).get("support_urls") or []
        if str(url).strip()
    }
    for index, document in enumerate(documents):
        normalized_document = document.lower().replace("www.", "")
        if any(url and url.lower() in normalized_document for url in urls):
            return index
    return None


def build_fact_card(record: dict[str, Any], facts: list[dict[str, Any]]) -> str:
    linkage = current_linkage(record)
    lines = [
        f"Evaluation date: {str(metadata_for(record).get('time_y') or '')[:10]}",
        f"Question subject: {facts[0].get('subject') if facts else 'unknown'}",
        f"Question relation: {linkage.get('question_type') or 'unknown'}",
        f"Verified answer to the question: {record.get('gold_answer')}",
    ]
    for fact in facts:
        lines.extend(
            [
                (
                    "Current fact: "
                    f"{fact.get('subject')} | {fact.get('property')} | {fact.get('object')}"
                ),
                f"Valid from: {fact.get('valid_start') or 'unknown'}",
                f"Source date: {fact.get('source_date') or 'unknown'}",
                f"Source URL: {fact.get('source_url') or ''}",
            ]
        )
    return "\n".join(lines)


def build_condition_docs(record: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    natural = rerank_documents([str(doc) for doc in record.get("current_docs") or []])
    facts = support_facts(record)
    if not natural:
        raise ValueError(f"No current documents for {record.get('id')}")
    if not facts:
        raise ValueError(f"No official current support facts for {record.get('id')}")

    natural_support_index = find_support_index(record, natural)
    injected = natural_support_index is None
    if injected:
        presence = list(natural[:9])
        presence.append(evidence_document(facts[0], rank=10))
        presence_support_index = len(presence) - 1
    else:
        presence = list(natural)
        presence_support_index = int(natural_support_index)
    presence = rerank_documents(presence)

    support_doc = presence[presence_support_index]
    support_first = [support_doc] + [
        document for index, document in enumerate(presence) if index != presence_support_index
    ]
    support_first = rerank_documents(support_first)
    evidence_only = [evidence_document(fact, rank=index) for index, fact in enumerate(facts, start=1)]
    fact_card = [build_fact_card(record, facts)]

    condition_docs = {
        "p1_natural": natural,
        "p2_support_presence": presence,
        "p3_support_first": support_first,
        "p4_evidence_only": evidence_only,
        "p5_fact_card": fact_card,
    }
    intervention = {
        "natural_support_hit": natural_support_index is not None,
        "natural_support_rank": (
            int(natural_support_index) + 1 if natural_support_index is not None else None
        ),
        "p2_support_injected": injected,
        "p2_changed_from_p1": presence != natural,
        "p3_support_moved_to_rank1": presence_support_index != 0,
        "official_support_fact_count": len(facts),
    }
    return condition_docs, intervention


def cohort_entry(
    row: dict[str, str],
    cohort_group: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    condition_docs, intervention = build_condition_docs(record)
    metadata = dict(metadata_for(record))
    time_y = metadata.get("time_y") or metadata.get("current_snapshot")
    metadata.update(
        {
            "condition_docs": condition_docs,
            "condition_time": {condition: time_y for condition in CONDITIONS},
            "probe_condition_blinded": True,
            "historical_cohort_group": cohort_group,
            "historical_outcome_state": row.get("outcome_state"),
            "historical_accuracy_stale": float(row.get("accuracy_stale") or 0.0),
            "historical_accuracy_current": float(row.get("accuracy_current") or 0.0),
            "historical_accuracy_drop": float(row.get("accuracy_drop") or 0.0),
            "historical_source_dataset": record.get("_source_dataset"),
            "probe_intervention": intervention,
        }
    )
    return {
        "id": record["id"],
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "stale_answer": record.get("stale_answer", ""),
        "gold_old": record.get("gold_old", [record.get("stale_answer", "")]),
        "gold_new": record.get("gold_new", [record["gold_answer"]]),
        "current_docs": condition_docs["p1_natural"],
        "stale_docs": record.get("stale_docs") or [],
        "metadata": metadata,
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_audit(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "question_id",
        "cohort_group",
        "historical_outcome_state",
        "question",
        "time_x",
        "time_y",
        "old_answer",
        "current_answer",
        "question_type",
        "answer_style",
        "natural_support_hit",
        "natural_support_rank",
        "p2_support_injected",
        "p3_support_moved_to_rank1",
        "official_evidence",
        "source_url",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            metadata = metadata_for(row)
            facts = support_facts(row)
            fact = facts[0] if facts else {}
            intervention = metadata.get("probe_intervention") or {}
            writer.writerow(
                {
                    "question_id": row["id"],
                    "cohort_group": metadata.get("historical_cohort_group"),
                    "historical_outcome_state": metadata.get("historical_outcome_state"),
                    "question": row["question"],
                    "time_x": metadata.get("time_x"),
                    "time_y": metadata.get("time_y"),
                    "old_answer": row.get("stale_answer"),
                    "current_answer": row.get("gold_answer"),
                    "question_type": relation(row),
                    "answer_style": binary_style(row),
                    "natural_support_hit": intervention.get("natural_support_hit"),
                    "natural_support_rank": intervention.get("natural_support_rank"),
                    "p2_support_injected": intervention.get("p2_support_injected"),
                    "p3_support_moved_to_rank1": intervention.get("p3_support_moved_to_rank1"),
                    "official_evidence": fact.get("evidence_span"),
                    "source_url": fact.get("source_url"),
                }
            )


def validate_cohort(rows: list[dict[str, Any]]) -> None:
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Probe cohort contains duplicate question IDs")
    forbidden = tuple(condition.lower() for condition in CONDITIONS)
    for row in rows:
        metadata = metadata_for(row)
        condition_docs = metadata.get("condition_docs") or {}
        condition_time = metadata.get("condition_time") or {}
        if tuple(condition_docs) != CONDITIONS:
            raise ValueError(f"Condition order mismatch for {row.get('id')}")
        if set(condition_time) != set(CONDITIONS) or len(set(condition_time.values())) != 1:
            raise ValueError(f"Condition time mismatch for {row.get('id')}")
        for documents in condition_docs.values():
            payload = "\n".join(str(value) for value in documents).lower()
            if any(marker in payload for marker in forbidden):
                raise ValueError(f"Condition label leaked into context for {row.get('id')}")
        intervention = metadata.get("probe_intervention") or {}
        if not intervention.get("p2_changed_from_p1"):
            if condition_docs["p1_natural"] != condition_docs["p2_support_presence"]:
                raise ValueError(f"Unchanged P2 differs from P1 for {row.get('id')}")
        if str(row.get("gold_answer", "")).lower() not in condition_docs["p5_fact_card"][0].lower():
            raise ValueError(f"P5 fact card omits current answer for {row.get('id')}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_source_records()
    changed_rows = read_csv(CHANGED_PREDICTIONS)
    state_rows = read_csv(FOUR_STATE_PREDICTIONS)

    new_degradation = select_rows(
        changed_rows, "new_degradation", args.new_degradation, records
    )
    persistent = select_rows(
        state_rows, "persistent_failure", args.persistent, records
    )
    excluded = {str(row["question_id"]) for row in new_degradation + persistent}
    adaptive_candidates = [
        row
        for row in changed_rows
        if row.get("outcome_state") == "recovery_or_adaptive_success"
    ]
    adaptive = select_matched_controls(
        adaptive_candidates,
        new_degradation,
        args.adaptive_control,
        records,
        excluded,
    )
    excluded.update(str(row["question_id"]) for row in adaptive)
    normal_candidates = [row for row in state_rows if row.get("outcome_state") == "normal"]
    normal = select_matched_controls(
        normal_candidates,
        persistent,
        args.normal_control,
        records,
        excluded,
    )

    selections = (
        [(row, "new_degradation") for row in new_degradation]
        + [(row, "persistent_failure") for row in persistent]
        + [(row, "adaptive_control") for row in adaptive]
        + [(row, "normal_control") for row in normal]
    )
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen: set[str] = set()
    for prediction, group in selections:
        question_id = str(prediction.get("question_id"))
        if question_id in seen:
            continue
        record = records.get(question_id)
        if record is None:
            failures.append({"question_id": question_id, "reason": "source_record_missing"})
            continue
        try:
            output_rows.append(cohort_entry(prediction, group, record))
            seen.add(question_id)
        except ValueError as exc:
            failures.append({"question_id": question_id, "reason": str(exc)})

    validate_cohort(output_rows)
    dataset_path = output_dir / "clark_historical_failure_probe.jsonl"
    write_jsonl(output_rows, dataset_path)
    write_audit(output_rows, output_dir / "cohort_audit.csv")
    group_counts = Counter(
        str(metadata_for(row).get("historical_cohort_group")) for row in output_rows
    )
    support_hits_count = sum(
        bool((metadata_for(row).get("probe_intervention") or {}).get("natural_support_hit"))
        for row in output_rows
    )
    manifest = {
        "schema_version": 1,
        "design": "blinded_clark_historical_failure_mechanism_probe",
        "dataset": str(dataset_path.relative_to(ROOT)),
        "conditions": list(CONDITIONS),
        "question_count": len(output_rows),
        "response_count_at_16_samples": len(output_rows) * len(CONDITIONS) * 16,
        "group_counts": dict(group_counts),
        "natural_current_support_hit_count": support_hits_count,
        "natural_current_support_miss_count": len(output_rows) - support_hits_count,
        "condition_labels_exposed_to_model": False,
        "failures": failures,
        "sources": {
            "changed_predictions": str(CHANGED_PREDICTIONS.relative_to(ROOT)),
            "four_state_predictions": str(FOUR_STATE_PREDICTIONS.relative_to(ROOT)),
            "record_datasets": [str(path.relative_to(ROOT)) for path in SOURCE_DATASETS],
        },
    }
    write_json(manifest, output_dir / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
