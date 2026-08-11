"""Statistical testing pipeline for condition comparisons."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.stats.bootstrap import bootstrap_paired_mean_diff_ci
from src.stats.permutation import cohens_dz, paired_permutation_test
from src.utils.io import ensure_dir, load_run_config, project_root, resolve_path, setup_logging

LOGGER = logging.getLogger(__name__)

COMPARISONS = [
    ("current_only", "stale_only"),
    ("current_only", "mixed"),
    ("stale_only", "mixed"),
]

METRICS = [
    "semantic_entropy",
    "semantic_volume",
    "centroid_shift",
    "n_clusters",
    "delta_n_clusters",
    "accuracy",
    "target_correct_rate",
    "current_answer_rate",
    "stale_answer_rate",
    "missing_rate",
    "harmful_other_rate",
    "ambiguous_rate",
    "contradiction_rate",
    "current_alignment",
    "quality_score",
    "shift_magnitude",
    "benchmark_delta_current_minus_stale",
    "current_performance_drop",
    "mixed_delta_current_minus_current",
    "condition_current_pull",
    "condition_stale_pull",
    "answer_flip_magnitude_vs_current",
    "mixed_conflict_rate",
    "semantic_shift_increased",
    "benchmark_semantic_drift",
]


def run_stats(run_dir: str | Path) -> Path:
    """Run paired permutation tests and bootstrap CIs for a completed run."""
    root = project_root()
    run_dir_path = resolve_path(run_dir, base_dir=root)
    config = load_run_config(run_dir_path)
    stats_dir = ensure_dir(run_dir_path / "stats")
    setup_logging(run_dir_path / "logs" / "stats.log")

    metrics_path = run_dir_path / "metrics" / "drift_labels.csv"
    if not metrics_path.exists():
        metrics_path = run_dir_path / "metrics" / "question_level_metrics.csv"
    df = pd.read_csv(metrics_path)
    if "embedding_model_role" in df.columns:
        df = df[df["embedding_model_role"] == "primary"].copy()

    stats_cfg = config.get("stats", {}) if isinstance(config.get("stats"), dict) else {}
    rounds_perm = int(stats_cfg.get("permutation_rounds", 5000))
    rounds_boot = int(stats_cfg.get("bootstrap_rounds", 2000))
    alpha = float(stats_cfg.get("alpha", 0.05))
    use_wilcoxon = bool(stats_cfg.get("wilcoxon", True))
    seed = int(config.get("seed", 42))
    comparisons = _configured_comparisons(config, df)

    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        pivot = df.pivot(index="question_id", columns="condition", values=metric)
        for left, right in comparisons:
            if left not in pivot.columns or right not in pivot.columns:
                continue
            paired = pivot[[left, right]].dropna()
            if paired.empty:
                continue
            x = paired[left].to_numpy(dtype=float)
            y = paired[right].to_numpy(dtype=float)
            mean_diff, ci_low, ci_high = bootstrap_paired_mean_diff_ci(
                x,
                y,
                rounds=rounds_boot,
                alpha=alpha,
                seed=seed,
            )
            p_value = paired_permutation_test(x, y, rounds=rounds_perm, seed=seed)
            row = {
                "metric": metric,
                "comparison": f"{left}_vs_{right}",
                "baseline_condition": left,
                "comparison_condition": right,
                "n_pairs": int(len(paired)),
                "mean_baseline": float(np.mean(x)),
                "mean_comparison": float(np.mean(y)),
                "mean_diff_comparison_minus_baseline": mean_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "permutation_p_value": p_value,
                "effect_size_cohens_dz": cohens_dz(x, y),
            }
            if use_wilcoxon:
                row["wilcoxon_p_value"] = _wilcoxon_p_value(x, y)
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(stats_dir / "stats_results.csv", index=False)
    LOGGER.info("Stats complete: %s", stats_dir / "stats_results.csv")
    return run_dir_path


def _configured_comparisons(config: dict[str, Any], df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return configured paired comparisons, falling back to sensible defaults."""
    stats_cfg = config.get("stats", {}) if isinstance(config.get("stats"), dict) else {}
    raw = stats_cfg.get("comparisons")
    if isinstance(raw, list) and raw:
        comparisons: list[tuple[str, str]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                comparisons.append((str(item[0]), str(item[1])))
        if comparisons:
            return comparisons

    conditions = set(df["condition"].astype(str))
    if all(condition in conditions for condition in ("current_only", "stale_only", "mixed")):
        return COMPARISONS
    if "stale_0" in conditions:
        ordered = sorted(
            [condition for condition in conditions if condition.startswith("stale_")],
            key=lambda value: int(value.split("_", 1)[1]) if value.split("_", 1)[1].isdigit() else 10_000,
        )
        return [("stale_0", condition) for condition in ordered if condition != "stale_0"]
    ordered = sorted(conditions)
    return [(ordered[0], condition) for condition in ordered[1:]]


def _wilcoxon_p_value(x: np.ndarray, y: np.ndarray) -> float:
    """Return Wilcoxon signed-rank p-value when SciPy can compute it."""
    try:
        from scipy.stats import wilcoxon

        if np.allclose(x, y):
            return 1.0
        return float(wilcoxon(y - x, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")
