"""Build normalized shift and uncertainty scores for CLARK run outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import load_run_config, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
SHIFT_COLUMNS = ("swd", "mmd_rbf", "energy", "cluster_js", "centroid_gap")
UNCERTAINTY_COLUMNS = ("delta_entropy", "delta_volume")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference.astype(float))
    left = np.searchsorted(ordered, values.astype(float), side="left")
    right = np.searchsorted(ordered, values.astype(float), side="right")
    return (left + right) / (2.0 * len(ordered))


def load_run_frame(run_dir: Path, split: str, variant: str) -> pd.DataFrame:
    metrics_path = run_dir / "metrics" / "question_level_metrics.csv"
    distances_path = (
        run_dir / "distribution_shift" / "per_question_distribution_shift.csv"
    )
    if not metrics_path.exists() or not distances_path.exists():
        raise FileNotFoundError(
            f"Missing metrics for {split}/{variant}: {metrics_path}, {distances_path}"
        )

    metrics = pd.read_csv(metrics_path)
    if "embedding_model_role" in metrics:
        metrics = metrics[metrics["embedding_model_role"] == "primary"].copy()
    accuracy_column = (
        "target_correct_rate" if "target_correct_rate" in metrics else "accuracy"
    )
    keep = [
        "question_id",
        "condition",
        accuracy_column,
        "semantic_entropy",
        "semantic_volume",
    ]
    wide = metrics[keep].pivot_table(
        index="question_id", columns="condition", aggfunc="first"
    )
    wide.columns = [f"{metric}__{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    required = [
        f"{accuracy_column}__stale_only",
        f"{accuracy_column}__current_only",
        "semantic_entropy__stale_only",
        "semantic_entropy__current_only",
        "semantic_volume__stale_only",
        "semantic_volume__current_only",
    ]
    missing = [column for column in required if column not in wide]
    if missing:
        raise ValueError(f"{split}/{variant} metrics miss {missing}")

    wide["accuracy_stale"] = wide[f"{accuracy_column}__stale_only"]
    wide["accuracy_current"] = wide[f"{accuracy_column}__current_only"]
    wide["accuracy_drop"] = wide["accuracy_stale"] - wide["accuracy_current"]
    wide["delta_entropy"] = (
        wide["semantic_entropy__current_only"]
        - wide["semantic_entropy__stale_only"]
    )
    wide["delta_volume"] = (
        wide["semantic_volume__current_only"]
        - wide["semantic_volume__stale_only"]
    )

    distances = pd.read_csv(distances_path)
    distances = distances[
        distances["comparison"] == "current_only__vs__stale_only"
    ].copy()
    frame = distances.merge(wide, on="question_id", how="inner")

    config = load_run_config(run_dir)
    dataset_rows = read_jsonl(resolve(config["dataset"]["path"]))
    metadata: list[dict[str, Any]] = []
    for row in dataset_rows:
        meta = row.get("metadata") or {}
        metadata.append(
            {
                "question_id": str(row["id"]),
                "natural_question_id": str(
                    meta.get("natural_record_id") or row["id"]
                ),
                "question": str(row.get("question", "")),
                "change_label": str(meta.get("change_label", "")),
                "old_answer": str(
                    meta.get("old_answer", row.get("stale_answer", ""))
                ),
                "current_answer": str(
                    meta.get("current_answer", row.get("gold_answer", ""))
                ),
                "time_x": str(meta.get("time_x", "")),
                "time_y": str(meta.get("time_y", "")),
            }
        )
    frame = frame.merge(pd.DataFrame(metadata), on="question_id", how="left")
    frame["split"] = split
    frame["variant"] = variant
    return frame


def stable_references(stable: pd.DataFrame) -> dict[str, list[float]]:
    return {
        column: sorted(stable[column].astype(float).tolist())
        for column in (*SHIFT_COLUMNS, *UNCERTAINTY_COLUMNS)
    }


def add_scores(
    frame: pd.DataFrame, references: dict[str, list[float]]
) -> pd.DataFrame:
    output = frame.copy()
    for column in (*SHIFT_COLUMNS, *UNCERTAINTY_COLUMNS):
        output[f"percentile__{column}"] = empirical_percentile(
            output[column].to_numpy(dtype=float),
            np.asarray(references[column], dtype=float),
        )
    output["shift_score"] = output[
        [f"percentile__{column}" for column in SHIFT_COLUMNS]
    ].mean(axis=1)
    output["uncertainty_score"] = output[
        [f"percentile__{column}" for column in UNCERTAINTY_COLUMNS]
    ].mean(axis=1)
    output["low_shift_low_uncertainty"] = (
        (1.0 - output["shift_score"]) * (1.0 - output["uncertainty_score"])
    )
    output["high_shift_low_uncertainty"] = (
        output["shift_score"] * (1.0 - output["uncertainty_score"])
    )
    output["low_shift_high_uncertainty"] = (
        (1.0 - output["shift_score"]) * output["uncertainty_score"]
    )
    output["high_shift_high_uncertainty"] = (
        output["shift_score"] * output["uncertainty_score"]
    )
    return output
