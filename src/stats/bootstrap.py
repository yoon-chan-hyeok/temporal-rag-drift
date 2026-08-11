"""Bootstrap confidence intervals for paired metric differences."""

from __future__ import annotations

import numpy as np


def bootstrap_paired_mean_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    rounds: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Return mean(y-x), lower CI, upper CI with paired bootstrap resampling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired bootstrap inputs must have the same shape")
    if x.size == 0:
        return 0.0, 0.0, 0.0
    diff = y - x
    rng = np.random.default_rng(seed)
    boot = np.empty(rounds, dtype=float)
    for index in range(rounds):
        sample_idx = rng.integers(0, diff.size, size=diff.size)
        boot[index] = float(np.mean(diff[sample_idx]))
    mean_diff = float(np.mean(diff))
    lower = float(np.quantile(boot, alpha / 2))
    upper = float(np.quantile(boot, 1 - alpha / 2))
    return mean_diff, lower, upper
