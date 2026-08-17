"""Prepare, cost, sample, score, and analyze the detector-linked CLARK probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_clark_historical_failure_probe_luna import (
    CONDITIONS,
    SYSTEM_PROMPT,
    cost_plan,
    run_checked,
)
from src.utils.io import load_yaml, write_json, write_yaml


PYTHON = Path(sys.executable)
PREPARE = ROOT / "scripts" / "prepare_clark_detector_linked_probe.py"
SAMPLER = ROOT / "scripts" / "sample_openai_async.py"
EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
ANALYZER = ROOT / "scripts" / "analyze_clark_detector_linked_probe.py"
BASE_CONFIG = ROOT / "configs" / "actual" / "clark_detector_linked_probe_luna.yaml"
DATASET = (
    ROOT
    / "data"
    / "processed"
    / "clark_detector_linked_probe"
    / "clark_detector_linked_probe.jsonl"
)
DEFAULT_RUN_DIR = ROOT / "outputs" / "runs" / "clark_detector_linked_probe_luna"
MODEL_NAME = "gpt-5.6-luna"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "cost", "sample", "metrics", "analyze"),
        default="all",
    )
    parser.add_argument("--samples", type=int, default=16)
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


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": "openai_compatible",
        "model_name": MODEL_NAME,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 96,
        "n_samples": args.samples,
        "n_samples_list": sorted(value for value in {4, 8, args.samples} if value <= args.samples),
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


def prepare() -> None:
    run_checked([str(PYTHON), str(PREPARE)])


def write_config(args: argparse.Namespace, run_dir: Path) -> tuple[dict[str, Any], Path]:
    config = load_yaml(BASE_CONFIG)
    config["experiment_name"] = "clark_detector_linked_probe_luna"
    config["dataset"]["path"] = relative(DATASET)
    config["dataset"]["max_questions"] = None
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
    config_path = run_dir / "config.yaml"
    write_yaml(config, config_path)
    return config, config_path


def sample(
    args: argparse.Namespace,
    run_dir: Path,
    log_dir: Path,
    config_path: Path,
) -> None:
    run_checked(
        [
            str(PYTHON),
            str(SAMPLER),
            "--config",
            str(config_path),
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
    config, config_path = write_config(args, run_dir)
    plan = cost_plan(config, args.expected_output_tokens)
    plan["screening_evaluation_events"] = 341
    plan["diagnostic_probe_events"] = 144
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
        sample(args, run_dir, log_dir, config_path)
    if args.stage in {"all", "metrics"}:
        metrics(args, run_dir, log_dir)
    if args.stage in {"all", "analyze"}:
        analyze(run_dir, log_dir)
    if args.stage == "all":
        print(run_dir / "linked_probe_analysis" / "report_ko.md")


if __name__ == "__main__":
    main()
