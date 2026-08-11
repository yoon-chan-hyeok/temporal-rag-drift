"""Prepare a T0-calibrated CLARK temporal-transfer cohort with response reuse."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl


SOURCE = ROOT / "data" / "processed" / "clark_changed_primary" / "transitions"
OUTPUT = ROOT / "data" / "processed" / "clark_t0_temporal_transfer"
RUN_ROOT = ROOT / "outputs" / "runs" / "clark_t0_temporal_transfer_luna"
OLD_RUN_ROOT = ROOT / "outputs" / "runs" / "clark_changed_primary_luna"
EXPECTED_SAMPLES_PER_EVENT = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def question(row: dict[str, Any]) -> str:
    return str(row.get("question") or "").strip()


def complete_response_map() -> tuple[dict[tuple[str, str, int], dict[str, Any]], set[str]]:
    keyed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for run_name in ("validation_changed", "locked_changed"):
        path = OLD_RUN_ROOT / run_name / "samples" / "responses.jsonl"
        for row in read_jsonl(path):
            if row.get("error") or not str(row.get("answer") or "").strip():
                continue
            key = (
                str(row.get("question_id")),
                str(row.get("condition")),
                int(row.get("sample_idx", -1)),
            )
            keyed.setdefault(key, row)
    counts = Counter(key[0] for key in keyed)
    complete_ids = {
        question_id
        for question_id, count in counts.items()
        if count == EXPECTED_SAMPLES_PER_EVENT
    }
    return keyed, complete_ids


def annotate(
    row: dict[str, Any],
    *,
    transition_id: str,
    detector_split: str,
) -> dict[str, Any]:
    output = dict(row)
    metadata = dict(output.get("metadata") or {})
    metadata.update(
        {
            "temporal_partition": "locked_future",
            "transition_id": transition_id,
            "detector_split": detector_split,
            "calibration_question_disjoint": True,
            "future_repeat_allowed": True,
            "analysis_unit": "question_by_db_update_event",
        }
    )
    output["metadata"] = metadata
    return output


def prepare() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    response_map, reusable_ids = complete_response_map()

    calibration = read_jsonl(SOURCE / "t0_changed.jsonl")
    calibration_questions = {question(row) for row in calibration}
    future: list[dict[str, Any]] = []
    per_transition: dict[str, dict[str, int]] = {}
    for index in range(1, 5):
        transition_id = f"t{index}"
        candidates = [
            row
            for row in read_jsonl(SOURCE / f"{transition_id}_changed.jsonl")
            if question(row) not in calibration_questions
        ]
        existing_n = sum(str(row["id"]) in reusable_ids for row in candidates)
        per_transition[transition_id] = {
            "events": len(candidates),
            "existing_exploratory": existing_n,
            "new_confirmatory": len(candidates) - existing_n,
        }
        for row in candidates:
            record_id = str(row["id"])
            split = (
                "future_existing_exploratory"
                if record_id in reusable_ids
                else "future_new_confirmatory"
            )
            future.append(
                annotate(row, transition_id=transition_id, detector_split=split)
            )

    future_ids = {str(row["id"]) for row in future}
    existing = [
        row
        for row in future
        if (row.get("metadata") or {}).get("detector_split")
        == "future_existing_exploratory"
    ]
    confirmatory = [
        row
        for row in future
        if (row.get("metadata") or {}).get("detector_split")
        == "future_new_confirmatory"
    ]
    if len(calibration) != 167 or len(future) != 341:
        raise ValueError(
            f"Unexpected CLARK cohort size: calibration={len(calibration)}, future={len(future)}"
        )
    if len(existing) != 146 or len(confirmatory) != 195:
        raise ValueError(
            f"Unexpected reuse split: existing={len(existing)}, confirmatory={len(confirmatory)}"
        )

    paths = {
        "calibration": OUTPUT / "calibration_t0_changed.jsonl",
        "future_all": OUTPUT / "future_t1_t4_all_changed.jsonl",
        "future_existing": OUTPUT / "future_existing_exploratory.jsonl",
        "future_confirmatory": OUTPUT / "future_new_confirmatory.jsonl",
    }
    write_jsonl(calibration, paths["calibration"])
    write_jsonl(future, paths["future_all"])
    write_jsonl(existing, paths["future_existing"])
    write_jsonl(confirmatory, paths["future_confirmatory"])

    response_path = RUN_ROOT / "future_all_changed" / "samples" / "responses.jsonl"
    current = {
        (
            str(row.get("question_id")),
            str(row.get("condition")),
            int(row.get("sample_idx", -1)),
        ): row
        for row in read_jsonl(response_path)
        if str(row.get("question_id")) in future_ids
    }
    for key, row in response_map.items():
        if key[0] in future_ids and key[0] in reusable_ids:
            current.setdefault(key, row)
    seeded = [current[key] for key in sorted(current)]
    response_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(seeded, response_path)

    seeded_counts = Counter(str(row.get("question_id")) for row in seeded)
    incomplete_seeded = {
        record_id: count
        for record_id, count in seeded_counts.items()
        if count not in (EXPECTED_SAMPLES_PER_EVENT,)
    }
    if incomplete_seeded:
        raise ValueError(f"Incomplete seeded response groups: {incomplete_seeded}")

    manifest = {
        "schema_version": 1,
        "design": "T0-only calibration; frozen detector; T1-T4 temporal transfer",
        "analysis_unit": "question_by_db_update_event",
        "detector_family": "quadratic_logistic fixed from prior exploratory study",
        "calibration": {
            "transition": "2021-12-22_to_2022-08-31",
            "events": len(calibration),
            "future_question_disjoint": True,
        },
        "future": {
            "events": len(future),
            "unique_questions": len({question(row) for row in future}),
            "existing_exploratory": len(existing),
            "new_confirmatory": len(confirmatory),
            "per_transition": per_transition,
        },
        "response_reuse": {
            "seeded_events": sum(count == EXPECTED_SAMPLES_PER_EVENT for count in seeded_counts.values()),
            "seeded_responses": len(seeded),
            "expected_new_requests": len(confirmatory) * EXPECTED_SAMPLES_PER_EVENT,
        },
        "files": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "confirmatory_ids": sorted(str(row["id"]) for row in confirmatory),
    }
    manifest_path = OUTPUT / "cohort_lock_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(manifest_path)
    return manifest


if __name__ == "__main__":
    prepare()
