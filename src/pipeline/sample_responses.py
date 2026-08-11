"""Sampling pipeline for current/stale/mixed DB interventions."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data.build_contexts import build_context, parse_context_config
from src.data.load_dataset import DatasetRecord, load_dataset
from src.generation.base import GenerationConfig, build_generator
from src.utils.io import (
    append_jsonl,
    copy_config_to_run,
    init_run_subdirs,
    load_yaml,
    project_root,
    read_jsonl,
    resolve_path,
    setup_logging,
    timestamped_run_dir,
)
from src.utils.seed import derive_seed, set_seed
from src.utils.text import stable_text_hash

LOGGER = logging.getLogger(__name__)


def run_sampling(config_path: str | Path, run_dir: str | Path | None = None) -> Path:
    """Sample model responses and checkpoint them to ``responses.jsonl``."""
    root = project_root()
    config = load_yaml(resolve_path(config_path, base_dir=root))
    seed = int(config.get("seed", 42))
    set_seed(seed)

    if run_dir is None:
        outputs_root = root / "outputs" / "runs"
        run_dir_path = timestamped_run_dir(
            outputs_root, str(config.get("experiment_name", "clark_temporal"))
        )
    else:
        run_dir_path = resolve_path(run_dir, base_dir=root)
        run_dir_path.mkdir(parents=True, exist_ok=True)

    subdirs = init_run_subdirs(run_dir_path)
    setup_logging(subdirs["logs"] / "sampling.log")
    copy_config_to_run(config, run_dir_path)

    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset config must be a mapping")
    records = load_dataset(
        dataset_cfg.get("path", "data/processed/temporal_canonical.jsonl"),
        max_questions=dataset_cfg.get("max_questions"),
        base_dir=root,
    )
    conditions = [str(condition) for condition in config.get("conditions", ["current_only", "stale_only", "mixed"])]
    generation_cfg_raw = config.get("generation", {})
    if not isinstance(generation_cfg_raw, dict):
        raise ValueError("generation config must be a mapping")
    generation_cfg = GenerationConfig.from_mapping(generation_cfg_raw)
    n_samples = _max_requested_samples(generation_cfg_raw, generation_cfg.n_samples)
    context_cfg = parse_context_config(config.get("context") if isinstance(config.get("context"), dict) else None)

    responses_path = subdirs["samples"] / "responses.jsonl"
    existing = read_jsonl(responses_path)
    completed = {
        (str(row.get("question_id")), str(row.get("condition")), int(row.get("sample_idx", -1)))
        for row in existing
        if "answer" in row and row.get("error") in (None, "")
    }
    LOGGER.info("Loaded %d records; %d samples already checkpointed", len(records), len(completed))

    generator = build_generator(generation_cfg)
    total = len(records) * len(conditions) * n_samples
    with tqdm(total=total, desc="sampling") as progress:
        progress.update(len(completed))
        for record in records:
            for condition in conditions:
                context_seed = derive_seed(seed, record.id, condition, "context")
                context = build_context(record, condition, context_cfg, seed=context_seed)
                context_hash = stable_text_hash(context)
                prompt_question = _prompt_question(record, condition, generation_cfg)
                for sample_idx in range(n_samples):
                    key = (record.id, condition, sample_idx)
                    if key in completed:
                        continue
                    sample_seed = derive_seed(seed, record.id, condition, sample_idx)
                    try:
                        answer = generator.generate(prompt_question, context, sample_seed=sample_seed)
                        row = _response_row(
                            record_id=record.id,
                            question=record.question,
                            gold_answer=record.gold_answer,
                            condition=condition,
                            sample_idx=sample_idx,
                            answer=answer,
                            context_hash=context_hash,
                            model_name=generation_cfg.model_name,
                            backend=generation_cfg.backend,
                            sample_seed=sample_seed,
                        )
                    except Exception as exc:
                        LOGGER.exception("Generation failed for %s/%s/%d", record.id, condition, sample_idx)
                        row = _response_row(
                            record_id=record.id,
                            question=record.question,
                            gold_answer=record.gold_answer,
                            condition=condition,
                            sample_idx=sample_idx,
                            answer="",
                            context_hash=context_hash,
                            model_name=generation_cfg.model_name,
                            backend=generation_cfg.backend,
                            sample_seed=sample_seed,
                            error=repr(exc),
                        )
                    append_jsonl([row], responses_path)
                    completed.add(key)
                    progress.update(1)

    LOGGER.info("Sampling complete: %s", responses_path)
    return run_dir_path


def _max_requested_samples(generation_cfg: dict[str, Any], default: int) -> int:
    """Use max(n_samples, max(n_samples_list)) so stability curves can reuse checkpoints."""
    n_samples_list = generation_cfg.get("n_samples_list") or []
    if isinstance(n_samples_list, list) and n_samples_list:
        return max(int(default), max(int(value) for value in n_samples_list))
    return int(default)


def _prompt_question(record: DatasetRecord, condition: str, generation_cfg: GenerationConfig) -> str:
    """Return the question text to place in the prompt for one condition."""
    extra = generation_cfg.extra or {}
    if not bool(extra.get("use_condition_time_prefix", False)):
        return record.question

    metadata = record.metadata or {}
    condition_times = metadata.get("condition_time")
    condition_time = (
        condition_times.get(condition)
        if isinstance(condition_times, dict)
        else None
    )
    time_x = (
        metadata.get("time_x")
        or metadata.get("old_snapshot")
        or metadata.get("old_month")
    )
    time_y = (
        metadata.get("time_y")
        or metadata.get("new_snapshot")
        or metadata.get("current_month")
    )
    if condition_time is not None:
        time_value = condition_time
    elif condition == "stale_only":
        time_value = time_x
    elif condition in {"current_only", "mixed"}:
        time_value = time_y
    else:
        time_value = None
    if time_value is None:
        return record.question

    rendered_time = _render_time_label(str(time_value))
    template = str(
        extra.get(
            "condition_time_prefix_template",
            "As of {time}, answer the following question: {question}",
        )
    )
    return template.format(time=rendered_time, question=record.question, condition=condition)


def _render_time_label(value: str) -> str:
    """Convert an ISO-like timestamp into a short human-readable date label."""
    text = value.strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    return text


def _response_row(
    record_id: str,
    question: str,
    gold_answer: str,
    condition: str,
    sample_idx: int,
    answer: str,
    context_hash: str,
    model_name: str,
    backend: str,
    sample_seed: int | None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a checkpoint row."""
    row: dict[str, Any] = {
        "question_id": record_id,
        "question": question,
        "gold_answer": gold_answer,
        "condition": condition,
        "sample_idx": sample_idx,
        "answer": answer,
        "context_hash": context_hash,
        "model_name": model_name,
        "backend": backend,
        "sample_seed": sample_seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if error:
        row["error"] = error
    return row
