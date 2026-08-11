"""Run the locked T0-only CLARK temporal-transfer experiment with Luna."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_clark_changed_primary_luna import (
    CACHED_INPUT_PRICE_PER_M,
    INPUT_PRICE_PER_M,
    MODEL_NAME,
    OUTPUT_PRICE_PER_M,
    experiment_config,
)
from src.pipeline.sample_responses_async_openai import build_jobs
from src.utils.io import load_yaml, read_jsonl, write_json, write_yaml


PYTHON = Path(sys.executable)
PREPARE = ROOT / "scripts" / "prepare_clark_t0_temporal_transfer.py"
FREEZE = ROOT / "scripts" / "freeze_clark_t0_transfer_detector.py"
SAMPLER = ROOT / "scripts" / "sample_openai_async.py"
EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
DISTANCES = ROOT / "scripts" / "analyze_distribution_shift_runs.py"
EVALUATE = ROOT / "scripts" / "evaluate_clark_t0_temporal_transfer.py"
DATASET = ROOT / "data" / "processed" / "clark_t0_temporal_transfer" / "future_t1_t4_all_changed.jsonl"
COHORT_LOCK = ROOT / "data" / "processed" / "clark_t0_temporal_transfer" / "cohort_lock_manifest.json"
RUN_ROOT = ROOT / "outputs" / "runs" / "clark_t0_temporal_transfer_luna"
RUN_DIR = RUN_ROOT / "future_all_changed"
DETECTOR_LOCK = RUN_ROOT / "detector_t0" / "detector_lock.json"
CONFIG_PATH = ROOT / "configs" / "clark_t0_temporal_transfer_luna" / "future_all_changed.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "cost", "sample", "metrics", "evaluate"),
        default="cost",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--expected-output-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests-per-minute", type=int, default=450)
    parser.add_argument("--tokens-per-minute", type=int, default=180000)
    parser.add_argument("--metrics-gpu", default="1")
    parser.add_argument("--confirm-api-cost", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, log_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    if log_path is None:
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    if result.returncode:
        suffix = f" See {log_path}." if log_path else ""
        raise RuntimeError(f"Command failed with exit code {result.returncode}.{suffix}")


def prepare() -> None:
    run([str(PYTHON), str(PREPARE)])
    run([str(PYTHON), str(FREEZE)])


def write_config(args: argparse.Namespace) -> dict:
    if not DATASET.exists() or not DETECTOR_LOCK.exists():
        raise FileNotFoundError("Run --stage prepare before cost or sampling.")
    config = experiment_config("t0_temporal_transfer_future", DATASET, args)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(config, CONFIG_PATH)
    return config


def completed_keys() -> set[tuple[str, str, int]]:
    return {
        (str(row.get("question_id")), str(row.get("condition")), int(row.get("sample_idx", -1)))
        for row in read_jsonl(RUN_DIR / "samples" / "responses.jsonl")
        if str(row.get("answer") or "").strip() and not row.get("error")
    }


def validate_seeded_contexts(config: dict) -> None:
    jobs, _ = build_jobs(config, set())
    expected = {(job.record.id, job.condition): job.context_hash for job in jobs}
    mismatches = []
    for row in read_jsonl(RUN_DIR / "samples" / "responses.jsonl"):
        key = (str(row.get("question_id")), str(row.get("condition")))
        if key in expected and str(row.get("context_hash")) != expected[key]:
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"Seeded response context mismatch for {len(set(mismatches))} events")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cost_plan(config: dict, args: argparse.Namespace) -> dict:
    completed = completed_keys()
    jobs, total = build_jobs(config, completed)
    pending_input = sum(job.estimated_input_tokens for job in jobs)
    first_prompt_input = 0
    seen_prompts: set[tuple[str, str]] = set()
    for job in jobs:
        key = (job.record.id, job.condition)
        if key not in seen_prompts:
            seen_prompts.add(key)
            first_prompt_input += job.estimated_input_tokens
    cached_input = pending_input - first_prompt_input
    expected_output = len(jobs) * args.expected_output_tokens
    uncached = (
        pending_input / 1_000_000 * INPUT_PRICE_PER_M
        + expected_output / 1_000_000 * OUTPUT_PRICE_PER_M
    )
    cached = (
        first_prompt_input / 1_000_000 * INPUT_PRICE_PER_M
        + cached_input / 1_000_000 * CACHED_INPUT_PRICE_PER_M
        + expected_output / 1_000_000 * OUTPUT_PRICE_PER_M
    )
    plan = {
        "model": MODEL_NAME,
        "future_events": 341,
        "reused_events": len(completed) // 32,
        "new_confirmatory_events": len(jobs) // 32,
        "total_requests": total,
        "reused_requests": len(completed),
        "pending_requests": len(jobs),
        "pending_input_tokens": pending_input,
        "first_prompt_input_tokens": first_prompt_input,
        "repeat_cached_input_tokens": cached_input,
        "expected_output_tokens": expected_output,
        "estimated_uncached_cost_usd": round(uncached, 4),
        "repeat_prompt_cache_scenario_usd": round(cached, 4),
        "sampling_gate": "--confirm-api-cost",
        "locks": {
            "cohort_sha256": file_sha256(COHORT_LOCK),
            "detector_lock_sha256": file_sha256(DETECTOR_LOCK),
            "config_sha256": file_sha256(CONFIG_PATH),
        },
    }
    write_json(plan, RUN_ROOT / "cost_plan.json")
    locked_cost_path = RUN_ROOT / "cost_plan_confirmatory_locked.json"
    if len(jobs) and not locked_cost_path.exists():
        write_json(plan, locked_cost_path)
    experiment_lock_path = RUN_ROOT / "confirmatory_experiment_lock.json"
    previous_lock = (
        json.loads(experiment_lock_path.read_text(encoding="utf-8"))
        if experiment_lock_path.exists()
        else {}
    )
    if previous_lock.get("paid_sampling_started"):
        changed = {
            key: (previous_lock.get(key), value)
            for key, value in plan["locks"].items()
            if previous_lock.get(key) != value
        }
        if changed:
            raise RuntimeError(f"Confirmatory lock changed after sampling started: {changed}")
    write_json(
        {
            "schema_version": 1,
            **plan["locks"],
            "paid_sampling_started": bool(previous_lock.get("paid_sampling_started", False)),
            **(
                {"pending_requests_at_start": previous_lock["pending_requests_at_start"]}
                if "pending_requests_at_start" in previous_lock
                else {}
            ),
        },
        experiment_lock_path,
    )
    return plan


def metrics(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.metrics_gpu
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_dir = RUN_ROOT / "launcher_logs"
    metric_outputs = (
        RUN_DIR / "metrics" / "sample_level_metrics.csv",
        RUN_DIR / "metrics" / "question_level_metrics.csv",
    )
    if not all(path.exists() and path.stat().st_size for path in metric_outputs):
        run(
            [str(PYTHON), str(EXPERIMENT), "--skip_sampling", "--run_dir", str(RUN_DIR)],
            log_path=log_dir / "metrics_future_all.log",
            env=env,
        )
    distance = RUN_DIR / "distribution_shift" / "per_question_distribution_shift.csv"
    if not distance.exists() or not distance.stat().st_size:
        run(
            [
                str(PYTHON),
                str(DISTANCES),
                "--run-dir",
                str(RUN_DIR),
                "--label",
                "clark_t0_transfer_future_all",
                "--output-dir",
                str(RUN_DIR / "distribution_shift"),
            ],
            log_path=log_dir / "distances_future_all.log",
            env=env,
        )


def main() -> None:
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"}:
        prepare()
        if args.stage == "prepare":
            return
    config = write_config(args)
    validate_seeded_contexts(config)
    plan = cost_plan(config, args)
    print(json.dumps(plan, indent=2))
    if args.stage == "cost":
        return
    if args.stage in {"sample", "all"}:
        if not args.confirm_api_cost:
            raise RuntimeError("Review cost_plan.json and rerun with --confirm-api-cost.")
        if plan["pending_requests"]:
            lock_path = RUN_ROOT / "confirmatory_experiment_lock.json"
            sampling_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            sampling_lock["paid_sampling_started"] = True
            sampling_lock["pending_requests_at_start"] = plan["pending_requests"]
            write_json(sampling_lock, lock_path)
            run(
                [
                    str(PYTHON),
                    str(SAMPLER),
                    "--config",
                    str(CONFIG_PATH),
                    "--run-dir",
                    str(RUN_DIR),
                    "--concurrency",
                    str(args.concurrency),
                    "--requests-per-minute",
                    str(args.requests_per_minute),
                    "--tokens-per-minute",
                    str(args.tokens_per_minute),
                ],
                log_path=RUN_ROOT / "launcher_logs" / "sample_confirmatory.log",
            )
    if args.stage in {"metrics", "all"}:
        metrics(args)
    if args.stage in {"evaluate", "all"}:
        run([str(PYTHON), str(EVALUATE)])


if __name__ == "__main__":
    main()
