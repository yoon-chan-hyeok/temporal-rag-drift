"""Semantic volume from response embedding geometry."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from src.embedding.embed import normalize_embeddings


def semantic_volume(embeddings: np.ndarray, pca_dim: int = 10, epsilon: float = 1e-10) -> float:
    """Compute log det(V_tilde^T V_tilde + epsilon I) after normalization and PCA."""
    if embeddings.shape[0] <= 1 or embeddings.shape[1] == 0:
        return 0.0
    normalized = normalize_embeddings(embeddings)
    dim = min(int(pca_dim), normalized.shape[0] - 1, normalized.shape[1])
    if dim <= 0:
        return 0.0
    transformed = PCA(n_components=dim, random_state=0).fit_transform(normalized)
    gram = transformed.T @ transformed
    jitter = float(epsilon)
    for _ in range(12):
        matrix = gram + jitter * np.eye(gram.shape[0])
        sign, logdet = np.linalg.slogdet(matrix)
        if sign > 0 and np.isfinite(logdet):
            return float(logdet)
        jitter *= 10.0
    diagonal = np.diag(gram) + jitter
    diagonal = np.where(diagonal <= 0, jitter, diagonal)
    return float(np.sum(np.log(diagonal)))
