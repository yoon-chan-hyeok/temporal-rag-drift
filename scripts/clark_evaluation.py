"""Outcome labels and evaluation utilities for CLARK temporal transfer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def load_runs(helper: Any, paths: list[str], *, split: str) -> pd.DataFrame:
    frames = [
        helper.load_run_frame(resolve(path), split, "natural") for path in paths
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["question_id"], keep="first"
    ).reset_index(drop=True)
    if combined["change_label"].isin(["changed", "stable"]).sum() != len(combined):
        counts = combined["change_label"].value_counts(dropna=False).to_dict()
        raise ValueError(f"{split} has missing/invalid change_label values: {counts}")
    return combined


def assign_outcome(
    frame: pd.DataFrame,
    *,
    drop_threshold: float,
    failure_threshold: float,
    success_threshold: float,
) -> pd.Series:
    old_accuracy = frame["accuracy_stale"].astype(float)
    new_accuracy = frame["accuracy_current"].astype(float)
    drop = old_accuracy - new_accuracy
    gain = new_accuracy - old_accuracy
    persistent = (old_accuracy < failure_threshold) & (
        new_accuracy < failure_threshold
    )
    degradation = (~persistent) & (drop >= drop_threshold)
    adaptive = (
        ~persistent
        & ~degradation
        & (
            (gain >= drop_threshold)
            | ((frame["change_label"] == "changed") & (new_accuracy >= success_threshold))
        )
    )
    outcome = pd.Series("normal", index=frame.index, dtype="object")
    outcome.loc[adaptive] = "recovery_or_adaptive_success"
    outcome.loc[degradation] = "new_degradation"
    outcome.loc[persistent] = "persistent_failure"
    return outcome


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities.astype(float), [1.0])))
    best: tuple[float, float, float] = (-1.0, -1.0, 0.5)
    for threshold in candidates:
        predictions = probabilities >= threshold
        precision, _, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        candidate = (float(f1), float(precision), float(threshold))
        if candidate > best:
            best = candidate
    return best[2]


def threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    prevalence = float(labels.mean()) if len(labels) else math.nan
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "risk_lift": (
            float(precision / prevalence)
            if prevalence and not math.isnan(prevalence)
            else math.nan
        ),
    }


def binary_scores(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return math.nan, math.nan
    return (
        float(roc_auc_score(labels, probabilities)),
        float(average_precision_score(labels, probabilities)),
    )


def bootstrap_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    rounds: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        key: []
        for key in (
            "auroc",
            "auprc",
            "precision",
            "recall",
            "f1",
            "risk_lift",
            "auprc_gain_over_prevalence",
        )
    }
    for _ in range(rounds):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        sampled_probabilities = probabilities[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        auroc, auprc = binary_scores(sampled_labels, sampled_probabilities)
        result = threshold_metrics(sampled_labels, sampled_probabilities, threshold)
        samples["auroc"].append(auroc)
        samples["auprc"].append(auprc)
        for metric in ("precision", "recall", "f1", "risk_lift"):
            value = float(result[metric])
            if math.isfinite(value):
                samples[metric].append(value)
        samples["auprc_gain_over_prevalence"].append(
            auprc - float(sampled_labels.mean())
        )
    output: dict[str, float] = {}
    for metric, values in samples.items():
        output[f"{metric}_ci_low"] = (
            float(np.quantile(values, 0.025)) if values else math.nan
        )
        output[f"{metric}_ci_high"] = (
            float(np.quantile(values, 0.975)) if values else math.nan
        )
    output["bootstrap_valid_rounds"] = float(len(samples["auprc"]))
    return output
