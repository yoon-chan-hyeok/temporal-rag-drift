"""Build a detector-linked P1-P5 cohort from the frozen CLARK future test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_clark_historical_failure_probe import (
    CONDITIONS,
    binary_style,
    build_condition_docs,
    metadata_for,
    read_jsonl,
    relation,
    support_facts,
)
from src.utils.io import write_json


DEFAULT_DATASET = (
    ROOT
    / "data"
    / "processed"
    / "clark_t0_temporal_transfer"
    / "future_t1_t4_all_changed.jsonl"
)
DEFAULT_PREDICTIONS = (
    ROOT
    / "outputs"
    / "runs"
    / "clark_t0_temporal_transfer_luna"
    / "core4_only_multimodel_transfer"
    / "frozen_future_predictions.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "clark_detector_linked_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--detector", default="additive_gam")
    parser.add_argument(
        "--matched-tn",
        type=int,
        default=None,
        help="TN controls; default equals the number of false positives.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    metadata = metadata_for(record)
    return (
        str(metadata.get("transition_id") or ""),
        relation(record),
        binary_style(record),
    )


def match_tn_controls(
    false_positives: list[dict[str, str]],
    true_negatives: list[dict[str, str]],
    records: dict[str, dict[str, Any]],
    risk_column: str,
    limit: int,
) -> list[dict[str, str]]:
    available = {str(row["question_id"]): row for row in true_negatives}
    selected: list[dict[str, str]] = []
    for target in sorted(false_positives, key=lambda row: str(row["question_id"])):
        if len(selected) >= limit or not available:
            break
        target_record = records[str(target["question_id"])]
        target_stratum = stratum(target_record)
        target_risk = float(target[risk_column])

        def distance(candidate: dict[str, str]) -> tuple[float, str]:
            candidate_id = str(candidate["question_id"])
            candidate_stratum = stratum(records[candidate_id])
            penalty = 0.0
            if candidate_stratum[0] != target_stratum[0]:
                penalty += 10.0
            if candidate_stratum[1] != target_stratum[1]:
                penalty += 3.0
            if candidate_stratum[2] != target_stratum[2]:
                penalty += 1.0
            penalty += abs(float(candidate[risk_column]) - target_risk)
            return penalty, candidate_id

        chosen = min(available.values(), key=distance)
        selected.append(chosen)
        available.pop(str(chosen["question_id"]))
    if len(selected) < limit:
        remainder = sorted(
            available.values(),
            key=lambda row: (-float(row[risk_column]), str(row["question_id"])),
        )
        selected.extend(remainder[: limit - len(selected)])
    return selected


def cohort_name(target: bool, alarm: bool) -> str:
    if target and alarm:
        return "detector_true_positive"
    if target and not alarm:
        return "detector_false_negative"
    if not target and alarm:
        return "detector_false_positive"
    return "detector_true_negative_control"


def build_row(
    record: dict[str, Any],
    prediction: dict[str, str],
    detector: str,
) -> dict[str, Any]:
    condition_docs, intervention = build_condition_docs(record)
    metadata = dict(metadata_for(record))
    time_y = metadata.get("time_y") or metadata.get("current_snapshot")
    target = as_bool(prediction["target_new_degradation"])
    alarm = as_bool(prediction[f"alarm__{detector}"])
    metadata.update(
        {
            "condition_docs": condition_docs,
            "condition_time": {condition: time_y for condition in CONDITIONS},
            "probe_condition_blinded": True,
            "probe_design": "detector_linked_locked_future",
            "detector_model": detector,
            "detector_confusion_group": cohort_name(target, alarm),
            "detector_risk": float(prediction[f"risk__{detector}"]),
            "detector_alarm": alarm,
            "target_new_degradation": target,
            "historical_outcome_state": prediction.get("outcome_state"),
            "historical_accuracy_stale": float(prediction["accuracy_stale"]),
            "historical_accuracy_current": float(prediction["accuracy_current"]),
            "historical_accuracy_drop": float(prediction["accuracy_drop"]),
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


def validate(rows: list[dict[str, Any]]) -> None:
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Detector-linked probe contains duplicate event IDs")
    forbidden = tuple(condition.lower() for condition in CONDITIONS)
    for row in rows:
        metadata = metadata_for(row)
        docs = metadata.get("condition_docs") or {}
        times = metadata.get("condition_time") or {}
        if tuple(docs) != CONDITIONS:
            raise ValueError(f"Condition order mismatch: {row['id']}")
        if set(times) != set(CONDITIONS) or len(set(times.values())) != 1:
            raise ValueError(f"Condition time mismatch: {row['id']}")
        for documents in docs.values():
            payload = "\n".join(str(value) for value in documents).lower()
            if any(marker in payload for marker in forbidden):
                raise ValueError(f"Condition name leaked into context: {row['id']}")
        if str(row["gold_answer"]).lower() not in docs["p5_fact_card"][0].lower():
            raise ValueError(f"P5 fact card omits current answer: {row['id']}")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_audit(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "question_id",
        "confusion_group",
        "transition_id",
        "question",
        "time_x",
        "time_y",
        "old_answer",
        "current_answer",
        "historical_accuracy_stale",
        "historical_accuracy_current",
        "historical_accuracy_drop",
        "detector_risk",
        "detector_alarm",
        "target_new_degradation",
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
            intervention = metadata.get("probe_intervention") or {}
            facts = support_facts(row)
            fact = facts[0] if facts else {}
            writer.writerow(
                {
                    "question_id": row["id"],
                    "confusion_group": metadata.get("detector_confusion_group"),
                    "transition_id": metadata.get("transition_id"),
                    "question": row["question"],
                    "time_x": metadata.get("time_x"),
                    "time_y": metadata.get("time_y"),
                    "old_answer": row.get("stale_answer"),
                    "current_answer": row.get("gold_answer"),
                    "historical_accuracy_stale": metadata.get("historical_accuracy_stale"),
                    "historical_accuracy_current": metadata.get("historical_accuracy_current"),
                    "historical_accuracy_drop": metadata.get("historical_accuracy_drop"),
                    "detector_risk": metadata.get("detector_risk"),
                    "detector_alarm": metadata.get("detector_alarm"),
                    "target_new_degradation": metadata.get("target_new_degradation"),
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


def main() -> None:
    args = parse_args()
    records = {str(row["id"]): row for row in read_jsonl(args.dataset.resolve())}
    predictions = read_csv(args.predictions.resolve())
    risk_column = f"risk__{args.detector}"
    alarm_column = f"alarm__{args.detector}"
    required = {"question_id", "target_new_degradation", risk_column, alarm_column}
    if not predictions or not required.issubset(predictions[0]):
        raise ValueError(f"Prediction columns missing: {sorted(required)}")
    missing = [row["question_id"] for row in predictions if row["question_id"] not in records]
    if missing:
        raise ValueError(f"Prediction records missing from dataset: {missing[:5]}")

    positives = [row for row in predictions if as_bool(row["target_new_degradation"])]
    false_positives = [
        row
        for row in predictions
        if not as_bool(row["target_new_degradation"]) and as_bool(row[alarm_column])
    ]
    true_negatives = [
        row
        for row in predictions
        if not as_bool(row["target_new_degradation"]) and not as_bool(row[alarm_column])
    ]
    matched_tn_count = args.matched_tn if args.matched_tn is not None else len(false_positives)
    matched_tn = match_tn_controls(
        false_positives,
        true_negatives,
        records,
        risk_column,
        matched_tn_count,
    )
    selected = positives + false_positives + matched_tn
    selected.sort(key=lambda row: str(row["question_id"]))
    output_rows = [
        build_row(records[str(row["question_id"])], row, args.detector)
        for row in selected
    ]
    validate(output_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "clark_detector_linked_probe.jsonl"
    write_jsonl(output_rows, dataset_path)
    write_audit(output_rows, args.output_dir / "cohort_audit.csv")
    counts = Counter(
        str(metadata_for(row).get("detector_confusion_group")) for row in output_rows
    )
    manifest = {
        "schema_version": 1,
        "design": "locked_future_detector_linked_p1_p5_probe",
        "detector": args.detector,
        "source_future_events": len(predictions),
        "source_future_new_degradation": len(positives),
        "probe_event_count": len(output_rows),
        "probe_group_counts": dict(counts),
        "conditions": list(CONDITIONS),
        "samples_per_condition_default": 16,
        "requests_at_16_samples": len(output_rows) * len(CONDITIONS) * 16,
        "condition_labels_exposed_to_model": False,
        "selection": "all positives + all false positives + risk/stratum-matched true negatives",
        "performance_scope": "screening performance is evaluated on all 341 frozen future events",
        "diagnostic_scope": "probe localization is evaluated on the selected 144-event diagnostic cohort",
        "dataset": str(dataset_path.relative_to(ROOT)),
        "predictions": str(args.predictions.resolve().relative_to(ROOT)),
    }
    write_json(manifest, args.output_dir / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
