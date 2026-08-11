"""Prepare expanded CLARK cohorts for changed-only detector evaluation.

The six official CLARK checkpoints define five adjacent transitions. Questions
are assigned to the first transition in which they appear, which makes the
calibration, validation, and locked partitions globally question-disjoint.
Changed questions are the detector population. Stable questions are retained
only as the null reference and false-alarm control.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_clark_temporal_cohort as clark
from src.utils.io import read_jsonl, write_jsonl


PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
BUILDER = ROOT / "scripts" / "build_clark_temporal_cohort.py"
AUDITOR = ROOT / "scripts" / "audit_common_retrieval.py"
QUESTIONS = ROOT / "data" / "external" / "clark" / "questions.csv"
EVIDENCE = ROOT / "data" / "external" / "clark" / "external_sources.json"
TIMESTAMP_MAP = ROOT / "data" / "external" / "clark" / "timestamp_to_questions.json"
URL_TIMESTAMP_MAP = (
    ROOT / "data" / "external" / "clark" / "url_timestamp_to_archive_url.jsonl"
)
QRELS = ROOT / "data" / "external" / "clark" / "question_article_qrels.jsonl"
COMMON_INDEX = ROOT / "outputs" / "indexes" / "clark_news_common.sqlite"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "clark_changed_primary"
DEFAULT_AUDIT = ROOT / "outputs" / "retrieval_audits" / "clark_changed_primary"

CHECKPOINTS = (
    "2021-12-22T00:00:00Z",
    "2022-08-31T00:00:00Z",
    "2023-01-29T00:00:00Z",
    "2023-07-31T00:00:00Z",
    "2023-11-21T00:00:00Z",
    "2024-04-19T00:00:00Z",
)

TRANSITIONS = tuple(
    {
        "id": f"t{index}",
        "time_x": CHECKPOINTS[index],
        "time_y": CHECKPOINTS[index + 1],
        "role": "calibration" if index == 0 else "validation" if index == 1 else "locked",
    }
    for index in range(len(CHECKPOINTS) - 1)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument(
        "--dense-devices",
        default="cuda:0,cuda:1",
        help="Comma-separated devices used by parallel transition builders.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Retrieve and materialize only changed cohorts.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Reuse transition retrieval files and rebuild merged partitions/audits.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def run(command: list[str], log_path: Path | None = None) -> None:
    print("[run]", subprocess.list2cmdline(command), flush=True)
    if log_path is None:
        result = subprocess.run(command, cwd=ROOT, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    if result.returncode != 0:
        suffix = f" See {log_path}." if log_path else ""
        raise RuntimeError(f"Command failed with exit code {result.returncode}.{suffix}")


def candidate_sets() -> list[dict[str, Any]]:
    histories = clark.load_question_spans(QUESTIONS)
    timestamp_map = clark.load_timestamp_map(TIMESTAMP_MAP)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for transition in TRANSITIONS:
        changed, stable = candidate_questions(
            histories,
            timestamp_map,
            transition["time_x"],
            transition["time_y"],
        )
        all_candidates = changed | stable
        selected_changed = changed - seen
        selected_stable = stable - seen
        output.append(
            {
                **transition,
                "changed": changed,
                "stable": stable,
                "selected_changed": selected_changed,
                "selected_stable": selected_stable,
                "excluded_prior_question": all_candidates & seen,
            }
        )
        seen.update(all_candidates)
    return output


def candidate_questions(
    histories: dict[str, list[clark.QuestionSpan]],
    timestamp_map: dict[str, list[str]],
    time_x_text: str,
    time_y_text: str,
) -> tuple[set[str], set[str]]:
    time_x = clark.parse_timestamp(time_x_text)
    time_y = clark.parse_timestamp(time_y_text)
    if time_x is None or time_y is None:
        raise ValueError("Invalid CLARK checkpoint timestamp.")
    allowed = clark.allowed_questions_for_checkpoint(
        timestamp_map,
        time_x_text,
        time_y_text,
        "intersection",
    )
    changed: set[str] = set()
    stable: set[str] = set()
    for question, spans in histories.items():
        if allowed is not None and question not in allowed:
            continue
        old = clark.active_answer(spans, time_x, mode="known")
        new = clark.active_answer(spans, time_y, mode="known")
        if old is None or new is None:
            continue
        destination = (
            changed if not clark.normalized_equal(old.answer, new.answer) else stable
        )
        destination.add(question)
    return changed, stable


def write_plan(
    output_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    changed_only: bool,
) -> dict[str, Any]:
    roles: dict[str, dict[str, int]] = {}
    transitions: list[dict[str, Any]] = []
    for item in candidates:
        role = str(item["role"])
        roles.setdefault(role, {"changed": 0, "stable": 0, "total": 0})
        changed_n = len(item["selected_changed"])
        stable_n = len(item["selected_stable"])
        roles[role]["changed"] += changed_n
        roles[role]["stable"] += stable_n
        roles[role]["total"] += changed_n + stable_n
        transitions.append(
            {
                "id": item["id"],
                "role": role,
                "time_x": item["time_x"],
                "time_y": item["time_y"],
                "changed_candidates": len(item["changed"]),
                "stable_candidates": len(item["stable"]),
                "selected_changed": changed_n,
                "selected_stable": stable_n,
                "excluded_seen_in_prior_transition": len(item["excluded_prior_question"]),
            }
        )
    manifest = {
        "schema_version": 2,
        "design": (
            "changed-only pilot without stable generation"
            if changed_only
            else "changed-only detector; stable null reference/control"
        ),
        "selection": "official checkpoint intersection, known time",
        "question_partition": "first eligible adjacent transition; globally question-disjoint",
        "transitions": transitions,
        "roles": roles,
        "active_roles": {
            role: {"changed": values["changed"]}
            for role, values in roles.items()
        }
        if changed_only
        else roles,
        "primary_endpoint": "changed new_degradation vs recovery_or_adaptive_success",
        "normalization": (
            "calibration/development changed empirical CDF"
            if changed_only
            else "calibration/development stable empirical CDF"
        ),
        "stable_uses": []
        if changed_only
        else [
            "empirical-CDF normalization",
            "development p90 risk threshold",
            "locked false-alarm control",
        ],
        "changed_only": changed_only,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "candidate_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(path)
    return manifest


def transition_paths(output_dir: Path, transition_id: str) -> dict[str, Path]:
    transition_dir = output_dir / "transitions"
    return {
        "all": transition_dir / f"{transition_id}_all.jsonl",
        "changed": transition_dir / f"{transition_id}_changed.jsonl",
        "stable": transition_dir / f"{transition_id}_stable.jsonl",
    }


def build_transition(
    item: dict[str, Any],
    output_dir: Path,
    log_dir: Path,
    *,
    device: str,
    top_k: int,
    candidate_k: int,
    changed_only: bool,
) -> None:
    paths = transition_paths(output_dir, str(item["id"]))
    paths["all"].parent.mkdir(parents=True, exist_ok=True)
    command = [
            str(PYTHON),
            str(BUILDER),
            "--questions-csv",
            str(QUESTIONS),
            "--evidence-file",
            str(EVIDENCE),
            "--timestamp-map",
            str(TIMESTAMP_MAP),
            "--url-timestamp-map",
            str(URL_TIMESTAMP_MAP),
            "--official-linkage-file",
            str(QRELS),
            "--time-x",
            str(item["time_x"]),
            "--time-y",
            str(item["time_y"]),
            "--selection",
            "intersection",
            "--time-mode",
            "known",
            "--output",
            str(paths["all"]),
            "--output-changed",
            str(paths["changed"]),
            "--output-stable",
            str(paths["stable"]),
            "--retrieval-method",
            "common_hybrid",
            "--common-index",
            str(COMMON_INDEX),
            "--top-k",
            str(top_k),
            "--candidate-k",
            str(candidate_k),
            "--dense-model",
            "BAAI/bge-large-en-v1.5",
            "--dense-device",
            device,
            "--dense-local-files-only",
            "--article-dedup",
        ]
    if changed_only:
        command.extend(["--change-label", "changed"])
    run(command, log_dir / f"build_{item['id']}.log")


def build_device_group(
    items: list[dict[str, Any]],
    output_dir: Path,
    log_dir: Path,
    *,
    device: str,
    top_k: int,
    candidate_k: int,
    changed_only: bool,
) -> None:
    """Build one fixed transition queue per GPU to prevent device overlap."""
    for item in items:
        build_transition(
            item,
            output_dir,
            log_dir,
            device=device,
            top_k=top_k,
            candidate_k=candidate_k,
            changed_only=changed_only,
        )


def annotate_record(record: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    metadata = dict(output.get("metadata") or {})
    metadata.update(
        {
            "temporal_partition": item["role"],
            "transition_id": item["id"],
            "global_question_disjoint": True,
            "detector_population": (
                "primary_changed" if metadata.get("change_label") == "changed" else "stable_null"
            ),
        }
    )
    output["metadata"] = metadata
    return output


def merge_partitions(
    output_dir: Path,
    audit_dir: Path,
    candidates: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    top_k: int,
    candidate_k: int,
    changed_only: bool,
) -> None:
    active_labels = ("changed",) if changed_only else ("changed", "stable")
    merged: dict[tuple[str, str], list[dict[str, Any]]] = {
        (role, label): []
        for role in ("calibration", "validation", "locked")
        for label in active_labels
    }
    selected_questions: dict[str, str] = {}
    for item in candidates:
        paths = transition_paths(output_dir, str(item["id"]))
        for label in active_labels:
            allowed = item[f"selected_{label}"]
            records = [
                annotate_record(row, item)
                for row in read_jsonl(paths[label])
                if str(row.get("question") or "").strip() in allowed
            ]
            for row in records:
                question = str(row.get("question") or "").strip()
                if question in selected_questions:
                    raise ValueError(
                        f"Question overlap between {selected_questions[question]} and {item['id']}: {question}"
                    )
                selected_questions[question] = str(item["id"])
            merged[(str(item["role"]), label)].extend(records)

    audit_dir.mkdir(parents=True, exist_ok=True)
    realized: dict[str, dict[str, int]] = {}
    for role in ("calibration", "validation", "locked"):
        realized[role] = {}
        for label in active_labels:
            path = output_dir / f"{role}_{label}.jsonl"
            rows = merged[(role, label)]
            write_jsonl(rows, path)
            realized[role][label] = len(rows)
            run(
                [
                    str(PYTHON),
                    str(AUDITOR),
                    "--input",
                    str(path),
                    "--summary-output",
                    str(audit_dir / f"{role}_{label}_summary.json"),
                    "--per-question-output",
                    str(audit_dir / f"{role}_{label}_per_question.csv"),
                ],
                audit_dir / "logs" / f"audit_{role}_{label}.log",
            )

    expected = {
        role: {
            label: int(manifest["roles"][role][label])
            for label in active_labels
        }
        for role in ("calibration", "validation", "locked")
    }
    if realized != expected:
        raise ValueError(f"Retrieved partition counts differ from candidate plan: {realized} != {expected}")

    final_manifest = {
        **manifest,
        "retrieval": {
            "method": "common_hybrid",
            "index": str(COMMON_INDEX),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "dense_model": "BAAI/bge-large-en-v1.5",
            "article_dedup": True,
            "timestamp_cutoff": True,
        },
        "realized": realized,
        "unique_questions": len(selected_questions),
        "changed_only": changed_only,
    }
    (audit_dir / "cohort_manifest.json").write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    audit_dir = resolve(args.audit_dir)
    candidates = candidate_sets()
    manifest = write_plan(
        output_dir,
        candidates,
        changed_only=bool(args.changed_only),
    )
    if args.plan_only:
        return

    required = (PYTHON, QUESTIONS, EVIDENCE, TIMESTAMP_MAP, URL_TIMESTAMP_MAP, QRELS, COMMON_INDEX)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CLARK resources:\n" + "\n".join(map(str, missing)))

    if not args.finalize_only:
        devices = [value.strip() for value in args.dense_devices.split(",") if value.strip()]
        if not devices:
            raise ValueError("--dense-devices must contain at least one device.")
        workers = max(1, min(int(args.workers), len(devices), len(candidates)))
        device_groups = [candidates[index::workers] for index in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    build_device_group,
                    group,
                    output_dir,
                    audit_dir / "logs",
                    device=devices[index],
                    top_k=int(args.top_k),
                    candidate_k=int(args.candidate_k),
                    changed_only=bool(args.changed_only),
                )
                for index, group in enumerate(device_groups)
            ]
            for future in futures:
                future.result()
    merge_partitions(
        output_dir,
        audit_dir,
        candidates,
        manifest,
        top_k=int(args.top_k),
        candidate_k=int(args.candidate_k),
        changed_only=bool(args.changed_only),
    )
    print(audit_dir / "cohort_manifest.json")


if __name__ == "__main__":
    main()
