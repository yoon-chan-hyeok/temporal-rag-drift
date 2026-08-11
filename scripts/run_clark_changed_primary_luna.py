"""Run expanded changed-only CLARK calibration, validation, and locked test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.sample_responses_async_openai import build_jobs
from src.utils.io import load_yaml, write_json, write_yaml


PYTHON = Path(sys.executable)
PREPARE = ROOT / "scripts" / "prepare_clark_changed_primary.py"
ASYNC_SAMPLER = ROOT / "scripts" / "sample_openai_async.py"
EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
DISTANCES = ROOT / "scripts" / "analyze_distribution_shift_runs.py"
ANALYZER = ROOT / "scripts" / "analyze_clark_changed_primary.py"
CHANGED_ONLY_ANALYZER = ROOT / "scripts" / "analyze_clark_changed_only.py"
BASE_CONFIG = ROOT / "configs" / "actual" / "clark_t0_future_all_changed.yaml"
DATA_DIR = ROOT / "data" / "processed" / "clark_changed_primary"
CONFIG_DIR = ROOT / "configs" / "clark_changed_primary_luna"
DEFAULT_RUN_ROOT = ROOT / "outputs" / "runs" / "clark_changed_primary_luna"

MODEL_NAME = "gpt-5.6-luna"
INPUT_PRICE_PER_M = 0.20
CACHED_INPUT_PRICE_PER_M = 0.02
OUTPUT_PRICE_PER_M = 1.20

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

SPECS = {
    f"{split}_{label}": {
        "dataset": DATA_DIR / f"{split}_{label}.jsonl",
        "gpu_group": index % 2,
    }
    for index, (split, label) in enumerate(
        (split, label)
        for split in ("calibration", "validation", "locked")
        for label in ("changed", "stable")
    )
}


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
    parser.add_argument("--gpu-0", default="0")
    parser.add_argument("--gpu-1", default="1")
    parser.add_argument(
        "--serial-metrics",
        action="store_true",
        help="Compute unfinished metric runs sequentially on one GPU.",
    )
    parser.add_argument(
        "--metrics-serial-gpu",
        default="1",
        help="Physical GPU used with --serial-metrics (default: 1).",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Generate and analyze only changed cohorts; omit stable API calls.",
    )
    parser.add_argument(
        "--confirm-api-cost",
        action="store_true",
        help="Required before paid Luna sampling starts.",
    )
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


def active_specs(changed_only: bool) -> dict[str, dict[str, Any]]:
    if not changed_only:
        return SPECS
    selected = {
        name: spec
        for name, spec in SPECS.items()
        if name.endswith("_changed")
    }
    # Balance 167 questions on GPU 0 against 46+100 on GPU 1.
    return {
        name: {
            **spec,
            "gpu_group": 0 if name == "calibration_changed" else 1,
        }
        for name, spec in selected.items()
    }


def datasets_ready(specs: dict[str, dict[str, Any]]) -> bool:
    return all(spec["dataset"].exists() for spec in specs.values())


def prepare(changed_only: bool) -> None:
    command = [str(PYTHON), str(PREPARE)]
    if changed_only:
        command.append("--changed-only")
    run_checked(command)


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


def experiment_config(name: str, dataset: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(BASE_CONFIG)
    config["experiment_name"] = f"clark_changed_primary_luna_{name}"
    config["dataset"]["path"] = relative(dataset)
    config["dataset"]["max_questions"] = args.max_questions
    config["generation"] = generation_config(args)
    config["embedding"]["device"] = "cuda:0"
    config["clustering"]["nli_device"] = "cuda:0"
    config["metrics"]["nli_device"] = "cuda:0"
    config["stats"]["bootstrap_rounds"] = 2000
    config["stats"]["permutation_rounds"] = 2000
    return config


def write_configs(
    args: argparse.Namespace,
    specs: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output: dict[str, Path] = {}
    for name, spec in specs.items():
        path = CONFIG_DIR / f"{name}.yaml"
        write_yaml(experiment_config(name, spec["dataset"], args), path)
        output[name] = path
    return output


def build_cost_plan(
    config_paths: dict[str, Path],
    specs: dict[str, dict[str, Any]],
    expected_output_tokens: int,
) -> dict[str, Any]:
    requests = 0
    input_tokens = 0
    max_output_tokens = 0
    first_prompt_tokens = 0
    by_run: dict[str, dict[str, int]] = {}
    for name, config_path in config_paths.items():
        config = load_yaml(config_path)
        jobs, _ = build_jobs(config, set())
        run_input = sum(job.estimated_input_tokens for job in jobs)
        by_run[name] = {"requests": len(jobs), "estimated_input_tokens": run_input}
        requests += len(jobs)
        input_tokens += run_input
        max_output_tokens += sum(job.estimated_tokens - job.estimated_input_tokens for job in jobs)
        first_prompt: set[tuple[str, str]] = set()
        for job in jobs:
            key = (job.record.id, job.condition)
            if key not in first_prompt:
                first_prompt.add(key)
                first_prompt_tokens += job.estimated_input_tokens
    expected_output_total = requests * expected_output_tokens
    uncached_cost = input_tokens / 1_000_000 * INPUT_PRICE_PER_M + expected_output_total / 1_000_000 * OUTPUT_PRICE_PER_M
    cached_input_tokens = max(0, input_tokens - first_prompt_tokens)
    cache_cost = first_prompt_tokens / 1_000_000 * INPUT_PRICE_PER_M + cached_input_tokens / 1_000_000 * CACHED_INPUT_PRICE_PER_M + expected_output_total / 1_000_000 * OUTPUT_PRICE_PER_M
    ceiling_cost = input_tokens / 1_000_000 * INPUT_PRICE_PER_M + max_output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    return {
        "model": MODEL_NAME,
        "samples_per_condition": int(next(iter(load_yaml(path)["generation"]["n_samples"] for path in config_paths.values()))),
        "questions": sum(len(read_dataset(spec["dataset"])) for spec in specs.values()),
        "requests": requests,
        "estimated_input_tokens": input_tokens,
        "expected_output_tokens": expected_output_total,
        "by_run": by_run,
        "api_cost_usd": round(uncached_cost, 4),
        "batch_reference_cost_usd": round(uncached_cost * 0.5, 4),
        "repeat_prompt_cache_scenario_usd": round(cache_cost, 4),
        "max_output_ceiling_cost_usd": round(ceiling_cost, 4),
        "paid_sampling_requires": "--confirm-api-cost",
    }


def read_dataset(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line for line in handle if line.strip()]


def sample(
    config_paths: dict[str, Path],
    specs: dict[str, dict[str, Any]],
    run_root: Path,
    args: argparse.Namespace,
    log_dir: Path,
) -> None:
    for name in specs:
        print(f"[sample/Luna] {name}", flush=True)
        run_checked(
            [
                str(PYTHON),
                str(ASYNC_SAMPLER),
                "--config",
                str(config_paths[name]),
                "--run-dir",
                str(run_root / name),
                "--concurrency",
                str(args.concurrency),
                "--requests-per-minute",
                str(args.requests_per_minute),
                "--tokens-per-minute",
                str(args.tokens_per_minute),
            ],
            log_path=log_dir / f"sample_{name}.log",
        )


def metrics_group(
    group: int,
    gpu: str,
    specs: dict[str, dict[str, Any]],
    run_root: Path,
    log_dir: Path,
) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for name, spec in specs.items():
        if int(spec["gpu_group"]) != group:
            continue
        run_dir = run_root / name
        metric_outputs = (
            run_dir / "metrics" / "sample_level_metrics.csv",
            run_dir / "metrics" / "question_level_metrics.csv",
        )
        if all(path.exists() and path.stat().st_size > 0 for path in metric_outputs):
            print(f"[metrics/skip-complete] {name}", flush=True)
        else:
            print(f"[metrics/GPU {gpu}] {name}", flush=True)
            run_checked(
                [
                    str(PYTHON),
                    str(EXPERIMENT),
                    "--skip_sampling",
                    "--run_dir",
                    str(run_dir),
                ],
                log_path=log_dir / f"metrics_{name}.log",
                env=env,
            )

        distance_output = (
            run_dir / "distribution_shift" / "per_question_distribution_shift.csv"
        )
        if distance_output.exists() and distance_output.stat().st_size > 0:
            print(f"[distances/skip-complete] {name}", flush=True)
        else:
            print(f"[distances/GPU {gpu}] {name}", flush=True)
            run_checked(
                [
                    str(PYTHON),
                    str(DISTANCES),
                    "--run-dir",
                    str(run_dir),
                    "--label",
                    name,
                    "--output-dir",
                    str(run_dir / "distribution_shift"),
                ],
                log_path=log_dir / f"distances_{name}.log",
                env=env,
            )


def metrics(
    specs: dict[str, dict[str, Any]],
    run_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
) -> None:
    if args.serial_metrics:
        serial_specs = {
            name: {**spec, "gpu_group": 0}
            for name, spec in specs.items()
        }
        metrics_group(
            0,
            args.metrics_serial_gpu,
            serial_specs,
            run_root,
            log_dir,
        )
        return

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(metrics_group, 0, args.gpu_0, specs, run_root, log_dir),
            executor.submit(metrics_group, 1, args.gpu_1, specs, run_root, log_dir),
        ]
        for future in futures:
            future.result()


def analyze(
    specs: dict[str, dict[str, Any]],
    run_root: Path,
    log_dir: Path,
    *,
    changed_only: bool,
) -> None:
    command = [
        str(PYTHON),
        str(CHANGED_ONLY_ANALYZER if changed_only else ANALYZER),
    ]
    for split in ("calibration", "validation", "locked"):
        labels = ("changed",) if changed_only else ("changed", "stable")
        for label in labels:
            command.extend(
                [f"--{split}-{label}-run", relative(run_root / f"{split}_{label}")]
            )
    output_dir = run_root / (
        "analysis_changed_only" if changed_only else "analysis_changed_primary"
    )
    command.extend(
        [
            "--output-dir",
            relative(output_dir),
            "--experiment-label",
            (
                "clark_all_checkpoints_changed_only_luna"
                if changed_only
                else "clark_all_checkpoints_changed_only_stable_null_luna"
            ),
            "--drop-threshold",
            "0.10",
        ]
    )
    if not changed_only:
        command.extend(["--stable-risk-quantile", "0.90"])
    analysis_name = "analysis_changed_only" if changed_only else "analysis_changed_primary"
    run_checked(command, log_path=log_dir / f"{analysis_name}.log")
    manifest = {
        "schema_version": 2,
        "model": MODEL_NAME,
        "design": "changed-only detector; stable null calibration/control",
        "stable_generation_omitted": changed_only,
        "calibration_transition": "2021-12-22_to_2022-08-31",
        "validation_transition": "2022-08-31_to_2023-01-29",
        "locked_transitions": [
            "2023-01-29_to_2023-07-31",
            "2023-07-31_to_2023-11-21",
            "2023-11-21_to_2024-04-19",
        ],
        "runs": {name: relative(run_root / name) for name in specs},
        "primary_analysis": relative(output_dir),
    }
    write_json(manifest, run_root / "run_manifest.json")


def main() -> None:
    args = parse_args()
    specs = active_specs(args.changed_only)
    run_root = args.run_root.resolve()
    log_dir = run_root / "launcher_logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "prepare" or (
        args.stage == "all" and not datasets_ready(specs)
    ):
        prepare(args.changed_only)
    if not datasets_ready(specs):
        raise FileNotFoundError(
            "Prepared CLARK partitions are missing. Run with --stage prepare first."
        )

    config_paths = write_configs(args, specs)
    plan = build_cost_plan(config_paths, specs, args.expected_output_tokens)
    plan["changed_only"] = args.changed_only
    write_json(plan, run_root / "cost_plan.json")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.stage in {"prepare", "cost"} or args.dry_run:
        return
    if args.stage in {"all", "sample"}:
        if not args.confirm_api_cost:
            raise RuntimeError(
                "Paid Luna sampling is blocked. Review cost_plan.json and rerun with --confirm-api-cost."
            )
        sample(config_paths, specs, run_root, args, log_dir)
    if args.stage in {"all", "metrics"}:
        metrics(specs, run_root, log_dir, args)
    if args.stage in {"all", "analyze"}:
        analyze(specs, run_root, log_dir, changed_only=args.changed_only)
    if args.stage == "all":
        output_name = "analysis_changed_only" if args.changed_only else "analysis_changed_primary"
        print(run_root / output_name / "report_ko.md")


if __name__ == "__main__":
    main()
