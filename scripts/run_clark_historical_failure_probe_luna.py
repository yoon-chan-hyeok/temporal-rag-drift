"""Prepare, cost, sample, score, and analyze the CLARK failure probe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.sample_responses_async_openai import build_jobs
from src.utils.io import load_yaml, write_json, write_yaml


PYTHON = Path(sys.executable)
PREPARE = ROOT / "scripts" / "prepare_clark_historical_failure_probe.py"
SAMPLER = ROOT / "scripts" / "sample_openai_async.py"
EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
DISTANCES = ROOT / "scripts" / "analyze_distribution_shift_runs.py"
ANALYZER = ROOT / "scripts" / "analyze_clark_historical_failure_probe.py"
BASE_CONFIG = ROOT / "configs" / "actual" / "clark_t0_future_all_changed.yaml"
DATASET = (
    ROOT
    / "data"
    / "processed"
    / "clark_historical_failure_probe"
    / "clark_historical_failure_probe.jsonl"
)
CONFIG = ROOT / "configs" / "clark_historical_failure_probe_luna.yaml"
DEFAULT_RUN_DIR = ROOT / "outputs" / "runs" / "clark_historical_failure_probe_luna"
MODEL_NAME = "gpt-5.6-luna"
INPUT_PRICE_PER_M = 0.20
OUTPUT_PRICE_PER_M = 1.20
CONDITIONS = (
    "p1_natural",
    "p2_support_presence",
    "p3_support_first",
    "p4_evidence_only",
    "p5_fact_card",
)

SYSTEM_PROMPT = """
You are answering a time-conditioned question.
Use only the provided retrieved context and treat the stated date as the
evaluation time. Do not use outside or parametric knowledge.

Prefer the newest directly relevant evidence available at or before that date.
If older and newer passages conflict, answer from the newest relevant passage.
Ignore passages that do not concern the entity and relation in the question.

If the context is insufficient, answer exactly:
The retrieved context is insufficient.

Return only the shortest natural-language final answer. For a yes/no question,
return only Yes or No. Do not explain, cite documents, or restate the question.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "cost", "sample", "metrics", "analyze"),
        default="all",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--expected-output-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests-per-minute", type=int, default=450)
    parser.add_argument("--tokens-per-minute", type=int, default=180000)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--confirm-api-cost", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def run_checked(
    command: list[str],
    *,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
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
    if result.returncode != 0:
        suffix = f" See {log_path}." if log_path else ""
        raise RuntimeError(f"Command failed with exit code {result.returncode}.{suffix}")


def prepare() -> None:
    run_checked([str(PYTHON), str(PREPARE)])


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": "openai_compatible",
        "model_name": MODEL_NAME,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 96,
        "n_samples": args.samples,
        "n_samples_list": sorted(
            value for value in {4, 8, args.samples} if value <= args.samples
        ),
        "system_prompt_override": SYSTEM_PROMPT,
        "use_condition_time_prefix": True,
        "condition_time_prefix_template": (
            "As of {time}, using only evidence available up to that date, "
            "answer the following question: {question}"
        ),
        "use_chat_template": True,
        "request_timeout": 180,
        "max_retries": 6,
        "retry_min_seconds": 2,
        "retry_max_seconds": 60,
        "api_key_env": "OPENAI_API_KEY",
        "async_concurrency": args.concurrency,
        "requests_per_minute": args.requests_per_minute,
        "tokens_per_minute": args.tokens_per_minute,
        "use_max_completion_tokens": True,
        "omit_seed": True,
        "reasoning_effort": "none",
    }


def write_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(BASE_CONFIG)
    config["experiment_name"] = "clark_historical_failure_probe_luna"
    config["dataset"]["path"] = relative(DATASET)
    config["dataset"]["max_questions"] = args.max_questions
    config["conditions"] = list(CONDITIONS)
    config["generation"] = generation_config(args)
    config["embedding"]["device"] = "cuda:0"
    config["clustering"]["nli_device"] = "cuda:0"
    config["metrics"]["nli_device"] = "cuda:0"
    config["metrics"]["target_by_condition"] = {
        condition: "current" for condition in CONDITIONS
    }
    config["drift"]["enabled"] = False
    config["stats"]["bootstrap_rounds"] = 2000
    config["stats"]["permutation_rounds"] = 2000
    config["stats"]["comparisons"] = [
        ["p1_natural", condition] for condition in CONDITIONS[1:]
    ]
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(config, CONFIG)
    return config


def cost_plan(config: dict[str, Any], expected_output_tokens: int) -> dict[str, Any]:
    jobs, total = build_jobs(config, set())
    forbidden_markers = tuple(value.lower() for value in CONDITIONS)
    leaked = [
        (job.record.id, job.condition)
        for job in jobs
        if any(
            marker in f"{job.question}\n{job.context}".lower()
            for marker in forbidden_markers
        )
    ]
    if leaked:
        raise ValueError(f"Probe condition label leaked into model input: {leaked[:5]}")
    questions_by_record: dict[str, set[str]] = {}
    for job in jobs:
        questions_by_record.setdefault(job.record.id, set()).add(job.question)
    inconsistent_questions = [
        question_id
        for question_id, rendered in questions_by_record.items()
        if len(rendered) != 1
    ]
    if inconsistent_questions:
        raise ValueError(
            "Conditions rendered different question/time prompts for: "
            f"{inconsistent_questions[:5]}"
        )
    input_tokens = sum(job.estimated_input_tokens for job in jobs)
    expected_output_total = len(jobs) * expected_output_tokens
    max_output_total = sum(
        job.estimated_tokens - job.estimated_input_tokens for job in jobs
    )
    return {
        "model": MODEL_NAME,
        "questions": len({job.record.id for job in jobs}),
        "conditions": len(CONDITIONS),
        "samples_per_condition": config["generation"]["n_samples"],
        "requests": len(jobs),
        "total_request_slots": total,
        "estimated_input_tokens": input_tokens,
        "expected_output_tokens": expected_output_total,
        "estimated_cost_usd": round(
            input_tokens / 1_000_000 * INPUT_PRICE_PER_M
            + expected_output_total / 1_000_000 * OUTPUT_PRICE_PER_M,
            4,
        ),
        "batch_reference_cost_usd": round(
            (
                input_tokens / 1_000_000 * INPUT_PRICE_PER_M
                + expected_output_total / 1_000_000 * OUTPUT_PRICE_PER_M
            )
            * 0.5,
            4,
        ),
        "max_output_ceiling_cost_usd": round(
            input_tokens / 1_000_000 * INPUT_PRICE_PER_M
            + max_output_total / 1_000_000 * OUTPUT_PRICE_PER_M,
            4,
        ),
        "paid_sampling_requires": "--confirm-api-cost",
        "blinding_validation": {
            "condition_labels_in_model_input": False,
            "identical_question_and_time_across_conditions": True,
            "system_prompt_identical_across_conditions": True,
        },
    }


def sample(args: argparse.Namespace, run_dir: Path, log_dir: Path) -> None:
    run_checked(
        [
            str(PYTHON),
            str(SAMPLER),
            "--config",
            str(CONFIG),
            "--run-dir",
            str(run_dir),
            "--concurrency",
            str(args.concurrency),
            "--requests-per-minute",
            str(args.requests_per_minute),
            "--tokens-per-minute",
            str(args.tokens_per_minute),
        ],
        log_path=log_dir / "sampling.log",
    )


def metrics(args: argparse.Namespace, run_dir: Path, log_dir: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run_checked(
        [str(PYTHON), str(EXPERIMENT), "--skip_sampling", "--run_dir", str(run_dir)],
        log_path=log_dir / "metrics.log",
        env=env,
    )
    distance_command = [
        str(PYTHON),
        str(DISTANCES),
        "--run-dir",
        relative(run_dir),
        "--label",
        "clark_historical_failure_probe_luna",
        "--output-dir",
        relative(run_dir / "distribution_shift"),
    ]
    for condition in CONDITIONS[1:]:
        distance_command.extend(["--comparison", condition, "p1_natural"])
    run_checked(
        distance_command,
        log_path=log_dir / "distribution_shift.log",
        env=env,
    )


def analyze(run_dir: Path, log_dir: Path) -> None:
    run_checked(
        [str(PYTHON), str(ANALYZER), "--run-dir", str(run_dir)],
        log_path=log_dir / "analysis.log",
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    log_dir = run_dir / "launcher_logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "prepare", "cost"}:
        prepare()
    if not DATASET.exists():
        raise FileNotFoundError(f"Prepared dataset missing: {DATASET}")
    config = write_config(args)
    plan = cost_plan(config, args.expected_output_tokens)
    write_json(plan, run_dir / "cost_plan.json")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.stage in {"prepare", "cost"} or args.dry_run:
        return
    if args.stage in {"all", "sample"}:
        if not args.confirm_api_cost:
            raise RuntimeError(
                "Paid Luna sampling is blocked. Review cost_plan.json and rerun "
                "with --confirm-api-cost."
            )
        sample(args, run_dir, log_dir)
    if args.stage in {"all", "metrics"}:
        metrics(args, run_dir, log_dir)
    if args.stage in {"all", "analyze"}:
        analyze(run_dir, log_dir)
    if args.stage == "all":
        print(run_dir / "probe_analysis" / "report_ko.md")


if __name__ == "__main__":
    main()
