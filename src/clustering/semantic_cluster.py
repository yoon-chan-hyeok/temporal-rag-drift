"""Semantic clustering for sampled response embeddings."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.embedding.embed import normalize_embeddings
from src.metrics.nli import NLIConfig, NLIEntailmentScorer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusteringConfig:
    """Semantic clustering settings."""

    method: str = "hdbscan"
    cosine_threshold: float = 0.82
    min_cluster_size: int = 2
    nli_model: str = "microsoft/deberta-large-mnli"
    nli_batch_size: int = 16
    nli_device: str | int | None = None
    entailment_threshold: float = 0.5
    contradiction_threshold: float = 0.5
    equivalence_rule: str = "bidirectional"
    lexical_f1_threshold: float = 0.92
    numeric_mismatch_splits: bool = True
    local_files_only: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "ClusteringConfig":
        """Parse clustering config from a mapping."""
        return cls(
            method=str(data.get("method", "hdbscan")),
            cosine_threshold=float(data.get("cosine_threshold", 0.82)),
            min_cluster_size=int(data.get("min_cluster_size", 2)),
            nli_model=str(data.get("nli_model", "microsoft/deberta-large-mnli")),
            nli_batch_size=int(data.get("nli_batch_size", data.get("batch_size", 16))),
            nli_device=data.get("nli_device", data.get("device")),
            entailment_threshold=float(data.get("entailment_threshold", 0.5)),
            contradiction_threshold=float(data.get("contradiction_threshold", 0.5)),
            equivalence_rule=str(data.get("equivalence_rule", "bidirectional")),
            lexical_f1_threshold=float(data.get("lexical_f1_threshold", 0.92)),
            numeric_mismatch_splits=bool(data.get("numeric_mismatch_splits", True)),
            local_files_only=bool(data.get("local_files_only", False)),
        )

    def nli_config(self) -> NLIConfig:
        """Return the matching NLI scorer configuration."""
        return NLIConfig(
            model_name=self.nli_model,
            batch_size=self.nli_batch_size,
            device=self.nli_device,
            entailment_threshold=self.entailment_threshold,
            contradiction_threshold=self.contradiction_threshold,
            lexical_f1_threshold=self.lexical_f1_threshold,
            local_files_only=self.local_files_only,
        )


def cluster_embeddings(embeddings: np.ndarray, config: ClusteringConfig) -> np.ndarray:
    """Cluster embeddings and return integer labels."""
    if embeddings.shape[0] == 0:
        return np.asarray([], dtype=int)
    if embeddings.shape[0] == 1:
        return np.asarray([0], dtype=int)
    method = config.method.lower()
    if method == "hdbscan":
        try:
            return _hdbscan_labels(embeddings, config)
        except Exception as exc:
            LOGGER.warning("HDBSCAN failed; falling back to agglomerative clustering: %s", exc)
            return _agglomerative_labels(embeddings, config.cosine_threshold)
    if method == "agglomerative":
        return _agglomerative_labels(embeddings, config.cosine_threshold)
    if method == "nli":
        raise ValueError("NLI clustering requires answer texts; call cluster_texts_by_nli instead")
    raise ValueError(f"Unsupported cluster method: {config.method}")


def cluster_texts_by_nli(
    answers: list[str],
    question: str,
    config: ClusteringConfig,
    scorer: NLIEntailmentScorer,
) -> np.ndarray:
    """Cluster answers with question-conditioned semantic equivalence."""
    if not answers:
        return np.asarray([], dtype=int)
    if len(answers) == 1:
        return np.asarray([0], dtype=int)

    parent = list(range(len(answers)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(answers)):
        for right in range(left + 1, len(answers)):
            if config.numeric_mismatch_splits and numeric_mismatch(answers[left], answers[right]):
                continue
            if scorer.equivalent(question, answers[left], answers[right], rule=config.equivalence_rule):
                union(left, right)

    labels = np.asarray([find(index) for index in range(len(answers))], dtype=int)
    return compact_labels(labels)


def extract_numbers(text: str) -> set[str]:
    """Extract normalized numeric strings from text."""
    values: set[str] = set()
    for raw in re.findall(r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?(?:%|st|nd|rd|th)?", text.lower()):
        cleaned = re.sub(r"(st|nd|rd|th)$", "", raw.rstrip("%"))
        cleaned = cleaned.replace(",", "")
        if cleaned:
            values.add(cleaned)
    return values


def numeric_mismatch(left: str, right: str) -> bool:
    """Return whether both texts contain numbers but disagree numerically."""
    left_numbers = extract_numbers(left)
    right_numbers = extract_numbers(right)
    return bool(left_numbers and right_numbers and (left_numbers - right_numbers) and (right_numbers - left_numbers))


def _hdbscan_labels(embeddings: np.ndarray, config: ClusteringConfig) -> np.ndarray:
    """Run HDBSCAN on normalized embeddings."""
    import hdbscan

    normalized = normalize_embeddings(embeddings)
    min_cluster_size = max(2, min(config.min_cluster_size, embeddings.shape[0]))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = np.asarray(clusterer.fit_predict(normalized), dtype=int)
    return relabel_noise_as_singletons(labels)


def _agglomerative_labels(embeddings: np.ndarray, cosine_threshold: float) -> np.ndarray:
    """Cluster with average-linkage agglomerative clustering and cosine threshold."""
    normalized = normalize_embeddings(embeddings)
    distance_threshold = max(0.0, min(2.0, 1.0 - cosine_threshold))
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    labels = np.asarray(model.fit_predict(normalized), dtype=int)
    return compact_labels(labels)


def relabel_noise_as_singletons(labels: np.ndarray) -> np.ndarray:
    """Turn HDBSCAN noise labels into singleton clusters for entropy accounting."""
    labels = labels.copy()
    next_label = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
    for index, label in enumerate(labels):
        if label == -1:
            labels[index] = next_label
            next_label += 1
    return compact_labels(labels)


def compact_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster labels to contiguous integers."""
    mapping: dict[int, int] = {}
    compact: list[int] = []
    for label in labels.tolist():
        if int(label) not in mapping:
            mapping[int(label)] = len(mapping)
        compact.append(mapping[int(label)])
    return np.asarray(compact, dtype=int)
