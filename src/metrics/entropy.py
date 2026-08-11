"""Semantic entropy over response clusters."""

from __future__ import annotations

import numpy as np


def semantic_entropy(labels: np.ndarray) -> float:
    """Compute H = -sum_k p_k log(p_k) from cluster labels."""
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    return max(0.0, entropy)
