"""Centroid-based embedding shift metrics."""

from __future__ import annotations

import numpy as np

from src.embedding.embed import normalize_embeddings


def centroid(embeddings: np.ndarray) -> np.ndarray:
    """Return the mean vector over normalized embeddings."""
    if embeddings.shape[0] == 0:
        return np.zeros((embeddings.shape[1] if embeddings.ndim == 2 else 0,), dtype=float)
    return normalize_embeddings(embeddings).mean(axis=0)


def centroid_shift(embeddings: np.ndarray, baseline_embeddings: np.ndarray) -> float:
    """Compute Euclidean distance between condition and baseline centroids."""
    if embeddings.shape[0] == 0 or baseline_embeddings.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(centroid(embeddings) - centroid(baseline_embeddings)))
