"""Drift magnitude and label assignment."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "delta_semantic_volume": 0.4,
    "delta_semantic_entropy": 0.4,
    "centroid_shift": 0.2,
}


def quality_score(row: pd.Series) -> float:
    """Compute Q = Accuracy - .5*ContradictionRate - .5*StaleAnswerRate."""
    return float(row["accuracy"] - 0.5 * row["contradiction_rate"] - 0.5 * row["stale_answer_rate"])


def zscore(values: pd.Series) -> pd.Series:
    """Population z-score with zero variance guarded to zero."""
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    std = float(numeric.std(ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - float(numeric.mean())) / std


def assign_drift_labels(
    question_metrics: pd.DataFrame,
    tau_shift: float = 1.0,
    tau_q: float = 0.05,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Attach shift magnitude, quality deltas, and drift labels."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    df = question_metrics.copy()
    df["quality_score"] = df.apply(quality_score, axis=1)

    baseline_condition = choose_baseline_condition(df)
    baselines = df[df["condition"] == baseline_condition].set_index("question_id")
    if baselines.empty:
        raise ValueError("baseline rows are required for drift labeling")
    stale_condition = "stale_only" if (df["condition"] == "stale_only").any() else "stale_100"
    mixed_condition = "mixed" if (df["condition"] == "mixed").any() else "stale_50"
    stale_rows = df[df["condition"] == stale_condition].set_index("question_id")
    mixed_rows = df[df["condition"] == mixed_condition].set_index("question_id")

    baseline_cols = [
        "semantic_entropy",
        "semantic_volume",
        "quality_score",
        "target_correct_rate",
        "current_answer_rate",
        "stale_answer_rate",
        "n_clusters",
    ]
    for column in baseline_cols:
        if column in baselines.columns:
            df[f"{column}_current"] = df["question_id"].map(baselines[column])
    for column in ("target_correct_rate", "current_answer_rate", "stale_answer_rate", "n_clusters"):
        if column in stale_rows.columns:
            df[f"{column}_stale_db"] = df["question_id"].map(stale_rows[column])
        if column in mixed_rows.columns:
            df[f"{column}_mixed_db"] = df["question_id"].map(mixed_rows[column])

    df["delta_semantic_entropy"] = df["semantic_entropy"] - df["semantic_entropy_current"]
    df["delta_semantic_volume"] = df["semantic_volume"] - df["semantic_volume_current"]
    df["delta_quality"] = df["quality_score"] - df["quality_score_current"]
    if "n_clusters_current" in df.columns:
        df["delta_n_clusters_vs_current"] = df["n_clusters"] - df["n_clusters_current"]
    df["abs_delta_semantic_entropy"] = df["delta_semantic_entropy"].abs()
    df["abs_delta_semantic_volume"] = df["delta_semantic_volume"].abs()
    df["abs_delta_n_clusters"] = df["delta_n_clusters_vs_current"].abs() if "delta_n_clusters_vs_current" in df.columns else 0.0

    df["z_delta_semantic_entropy"] = zscore(df["abs_delta_semantic_entropy"])
    df["z_delta_semantic_volume"] = zscore(df["abs_delta_semantic_volume"])
    df["z_centroid_shift"] = zscore(df["centroid_shift"])
    df["z_delta_n_clusters"] = zscore(df["abs_delta_n_clusters"])
    df["shift_magnitude"] = (
        float(weights.get("delta_semantic_volume", 0.4)) * df["z_delta_semantic_volume"]
        + float(weights.get("delta_semantic_entropy", 0.4)) * df["z_delta_semantic_entropy"]
        + float(weights.get("centroid_shift", 0.2)) * df["z_centroid_shift"]
        + float(weights.get("delta_n_clusters", 0.0)) * df["z_delta_n_clusters"]
    )
    _attach_temporal_drift_metrics(df, tau_shift=tau_shift, tau_q=tau_q)
    df["drift_label"] = [
        label_drift(condition, sm, dq, baseline_condition=baseline_condition, tau_shift=tau_shift, tau_q=tau_q)
        for condition, sm, dq in zip(df["condition"], df["shift_magnitude"], df["delta_quality"], strict=False)
    ]
    df["benchmark_drift_label"] = [
        label_benchmark_drift(condition, drop, shift, baseline_condition=baseline_condition, tau_shift=tau_shift, tau_q=tau_q)
        for condition, drop, shift in zip(
            df["condition"],
            df["current_performance_drop"],
            df["shift_magnitude"],
            strict=False,
        )
    ]
    return df


def choose_baseline_condition(df: pd.DataFrame) -> str:
    """Return the condition to use as the no-stale baseline."""
    conditions = set(df["condition"].astype(str))
    if "current_only" in conditions:
        return "current_only"
    if "stale_0" in conditions:
        return "stale_0"
    return sorted(conditions)[0]


def _attach_temporal_drift_metrics(df: pd.DataFrame, tau_shift: float, tau_q: float) -> None:
    """Attach DB-freshness-specific drift metrics in-place."""
    target_current = numeric_column(df, "target_correct_rate_current")
    target_stale = numeric_column(df, "target_correct_rate_stale_db")
    target_mixed = numeric_column(df, "target_correct_rate_mixed_db")
    current_answer_current = numeric_column(df, "current_answer_rate_current")
    stale_answer_current = numeric_column(df, "stale_answer_rate_current")

    df["benchmark_delta_current_minus_stale"] = target_current - target_stale
    df["benchmark_delta_stale_minus_current"] = target_stale - target_current
    df["current_performance_drop"] = np.maximum(0.0, -df["benchmark_delta_current_minus_stale"])
    df["current_performance_gain"] = np.maximum(0.0, df["benchmark_delta_current_minus_stale"])
    df["mixed_delta_current_minus_stale"] = target_mixed - target_stale
    df["mixed_delta_current_minus_current"] = target_mixed - target_current

    df["condition_current_pull"] = df["current_answer_rate"] - df["stale_answer_rate"]
    df["condition_stale_pull"] = df["stale_answer_rate"] - df["current_answer_rate"]
    df["answer_flip_magnitude_vs_current"] = (
        (numeric_column(df, "current_answer_rate") - current_answer_current).abs()
        + (numeric_column(df, "stale_answer_rate") - stale_answer_current).abs()
    )
    df["mixed_conflict_rate"] = np.where(
        df["condition"] == "mixed",
        np.minimum(
            numeric_column(df, "current_answer_rate"),
            numeric_column(df, "stale_answer_rate"),
        ),
        0.0,
    )
    df["temporal_answer_flip"] = (
        (df["condition"] == "stale_only")
        & (df["stale_answer_rate"] > df["current_answer_rate"] + tau_q)
    ).astype(int)
    df["mixed_conflict"] = (df["mixed_conflict_rate"] > tau_q).astype(int)
    df["semantic_shift_increased"] = (df["shift_magnitude"] >= tau_shift).astype(int)
    df["benchmark_semantic_drift"] = (
        (df["current_performance_drop"] > tau_q) & (df["semantic_shift_increased"] == 1)
    ).astype(int)


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column or a zero series when missing."""
    if column not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def label_drift(
    condition: str,
    shift_magnitude: float,
    delta_quality: float,
    baseline_condition: str = "current_only",
    tau_shift: float = 1.0,
    tau_q: float = 0.05,
) -> str:
    """Classify a condition as baseline, inert, adaptive, harmful, or benign."""
    if condition == baseline_condition:
        return "baseline"
    if shift_magnitude < tau_shift and abs(delta_quality) < tau_q:
        return "inert_change"
    if shift_magnitude >= tau_shift and delta_quality >= tau_q:
        return "adaptive_shift"
    if shift_magnitude >= tau_shift and delta_quality <= -tau_q:
        return "harmful_drift"
    return "benign_shift"


def label_benchmark_drift(
    condition: str,
    current_performance_drop: float,
    shift_magnitude: float,
    baseline_condition: str = "current_only",
    tau_shift: float = 1.0,
    tau_q: float = 0.05,
) -> str:
    """Label drift using time-conditioned benchmark regression plus semantic shift."""
    if condition == baseline_condition:
        return "baseline"
    if current_performance_drop > tau_q and shift_magnitude >= tau_shift:
        return "benchmark_regression_with_semantic_shift"
    if current_performance_drop > tau_q:
        return "benchmark_regression_without_semantic_shift"
    if shift_magnitude >= tau_shift:
        return "semantic_shift_without_benchmark_regression"
    return "no_benchmark_drift"
