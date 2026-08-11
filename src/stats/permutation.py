"""Paired permutation tests."""

from __future__ import annotations

import numpy as np


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    rounds: int = 5000,
    seed: int | None = None,
) -> float:
    """Two-sided paired permutation test using random sign flips."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired permutation inputs must have the same shape")
    if x.size == 0:
        return 1.0
    diff = y - x
    observed = abs(float(np.mean(diff)))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(rounds):
        signs = rng.choice(np.array([-1.0, 1.0]), size=diff.size)
        statistic = abs(float(np.mean(diff * signs)))
        if statistic >= observed:
            extreme += 1
    return float((extreme + 1) / (rounds + 1))


def cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    """Paired standardized effect size: mean(diff) / std(diff)."""
    diff = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
    if diff.size <= 1:
        return 0.0
    std = float(np.std(diff, ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(np.mean(diff) / std)
