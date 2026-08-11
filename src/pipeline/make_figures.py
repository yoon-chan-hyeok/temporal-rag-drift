"""Figure generation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.clustering.semantic_cluster import ClusteringConfig, cluster_embeddings, cluster_texts_by_nli
from src.embedding.embed import EmbeddingConfig, TextEmbedder
from src.metrics.entropy import semantic_entropy
from src.metrics.nli import NLIEntailmentScorer
from src.metrics.volume import semantic_volume
from src.utils.io import ensure_dir, load_run_config, project_root, read_jsonl, resolve_path, setup_logging

LOGGER = logging.getLogger(__name__)


def run_make_figures(run_dir: str | Path) -> Path:
    """Create all requested figures for a completed run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = project_root()
    run_dir_path = resolve_path(run_dir, base_dir=root)
    config = load_run_config(run_dir_path)
    figures_dir = ensure_dir(run_dir_path / "figures")
    setup_logging(run_dir_path / "logs" / "figures.log")

    question_df = pd.read_csv(run_dir_path / "metrics" / "question_level_metrics.csv")
    question_df = question_df[question_df["embedding_model_role"] == "primary"].copy()
    drift_path = run_dir_path / "metrics" / "drift_labels.csv"
    drift_df = pd.read_csv(drift_path) if drift_path.exists() else None
    sample_df = pd.read_csv(run_dir_path / "metrics" / "sample_level_metrics.csv")
    responses = [row for row in read_jsonl(run_dir_path / "samples" / "responses.jsonl") if row.get("answer")]

    fig_cfg = config.get("figures", {}) if isinstance(config.get("figures"), dict) else {}
    dpi = int(fig_cfg.get("dpi", 180))

    _figure_condition_bars(question_df, figures_dir / "figure1_condition_metric_bars.png", dpi=dpi)
    if drift_df is not None and "shift_magnitude" in drift_df.columns:
        _figure_shift_violin(drift_df, figures_dir / "figure2_shift_magnitude_distribution.png", dpi=dpi)
    _figure_stability(config, responses, sample_df, figures_dir / "figure3_sample_stability.png", dpi=dpi)
    _figure_projection(config, responses, figures_dir / "figure4_response_embedding_projection.png", dpi=dpi)
    LOGGER.info("Figures complete: %s", figures_dir)
    return run_dir_path


def _figure_condition_bars(question_df: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("semantic_entropy", "SemEnt"),
        ("n_clusters", "Clusters"),
        ("semantic_volume", "SemVol"),
        ("accuracy", "Accuracy"),
        ("stale_answer_rate", "StaleRate"),
    ]
    preferred_order = ["current_only", "stale_only", "mixed"]
    conditions = [condition for condition in preferred_order if condition in set(question_df["condition"].astype(str))]
    if not conditions:
        conditions = sorted(question_df["condition"].astype(str).unique().tolist())
    colors = {"current_only": "#2f6f73", "stale_only": "#b5533c", "mixed": "#6a5acd"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=False):
        summary = question_df.groupby("condition")[metric].agg(["mean", "sem"]).reindex(conditions)
        axis.bar(
            summary.index.astype(str),
            summary["mean"],
            yerr=summary["sem"].fillna(0.0),
            color=[colors.get(condition, "#666666") for condition in summary.index.astype(str)],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _figure_shift_violin(drift_df: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    data = []
    labels = []
    for condition in ["stale_only", "mixed"]:
        values = drift_df.loc[drift_df["condition"] == condition, "shift_magnitude"].dropna().to_numpy(dtype=float)
        if len(values):
            data.append(values)
            labels.append(f"current vs {condition.replace('_only', '')}")
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    if data:
        axis.violinplot(data, showmeans=True, showextrema=True)
        axis.boxplot(data, widths=0.16, patch_artist=True, boxprops={"facecolor": "#ffffff", "alpha": 0.7})
        axis.set_xticks(range(1, len(labels) + 1), labels=labels, rotation=15)
    axis.set_ylabel("Shift magnitude")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _figure_stability(config: dict[str, Any], responses: list[dict[str, Any]], sample_df: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    generation_cfg = config.get("generation", {}) if isinstance(config.get("generation"), dict) else {}
    n_values = generation_cfg.get("n_samples_list") or [8, 16, int(generation_cfg.get("n_samples", 16))]
    n_values = sorted({int(value) for value in n_values if int(value) > 0})
    max_sample_idx = max(int(row.get("sample_idx", 0)) for row in responses) if responses else 0
    n_values = [n for n in n_values if n <= max_sample_idx + 1]
    if not n_values:
        n_values = [max_sample_idx + 1]

    embed_cfg = EmbeddingConfig.from_mapping(config.get("embedding", {}) if isinstance(config.get("embedding"), dict) else {})
    cluster_cfg = ClusteringConfig.from_mapping(config.get("clustering", {}) if isinstance(config.get("clustering"), dict) else {})
    cluster_scorer = NLIEntailmentScorer(cluster_cfg.nli_config()) if cluster_cfg.method.lower() == "nli" else None
    embedder = TextEmbedder(
        embed_cfg.primary_model,
        batch_size=embed_cfg.batch_size,
        normalize=embed_cfg.normalize,
        allow_hashing_fallback=embed_cfg.allow_hashing_fallback,
        device=embed_cfg.device,
        local_files_only=embed_cfg.local_files_only,
    )
    response_df = pd.DataFrame(responses).reset_index().rename(columns={"index": "response_index"})
    embeddings = embedder.encode(response_df["answer"].astype(str).tolist())
    rows = []
    for n in n_values:
        entropies = []
        volumes = []
        accuracies = []
        for (question_id, condition), group in response_df.groupby(["question_id", "condition"], sort=False):
            subset = group.sort_values("sample_idx").head(n)
            indices = subset["response_index"].to_numpy(dtype=int)
            if cluster_cfg.method.lower() == "nli":
                if cluster_scorer is None:
                    raise RuntimeError("NLI clustering was requested but no NLI scorer is available")
                labels = cluster_texts_by_nli(
                    subset["answer"].astype(str).tolist(),
                    str(subset["question"].iloc[0]),
                    cluster_cfg,
                    cluster_scorer,
                )
            else:
                labels = cluster_embeddings(embeddings[indices], cluster_cfg)
            entropies.append(semantic_entropy(labels))
            volumes.append(semantic_volume(embeddings[indices], pca_dim=embed_cfg.pca_dim))
            sample_subset = sample_df[
                (sample_df["question_id"] == question_id)
                & (sample_df["condition"] == condition)
                & (sample_df["sample_idx"].isin(subset["sample_idx"]))
            ]
            if not sample_subset.empty:
                accuracies.append(float(sample_subset["correct"].mean()))
        rows.append(
            {
                "n": n,
                "semantic_entropy": float(np.mean(entropies)) if entropies else 0.0,
                "semantic_volume": float(np.mean(volumes)) if volumes else 0.0,
                "accuracy_variance": float(np.var(accuracies)) if accuracies else 0.0,
            }
        )
    stability = pd.DataFrame(rows)
    stability.to_csv(path.with_suffix(".csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ["semantic_entropy", "semantic_volume", "accuracy_variance"],
        ["SemEnt", "SemVol", "Accuracy variance"],
        strict=False,
    ):
        axis.plot(stability["n"], stability[metric], marker="o", color="#2f6f73")
        axis.set_title(title)
        axis.set_xlabel("n samples")
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _figure_projection(config: dict[str, Any], responses: list[dict[str, Any]], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    if not responses:
        return
    fig_cfg = config.get("figures", {}) if isinstance(config.get("figures"), dict) else {}
    max_points = int(fig_cfg.get("max_projection_points", 3000))
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    if len(responses) > max_points:
        indices = sorted(rng.choice(np.arange(len(responses)), size=max_points, replace=False).tolist())
        responses = [responses[index] for index in indices]

    embed_cfg = EmbeddingConfig.from_mapping(config.get("embedding", {}) if isinstance(config.get("embedding"), dict) else {})
    embedder = TextEmbedder(
        embed_cfg.primary_model,
        batch_size=embed_cfg.batch_size,
        normalize=embed_cfg.normalize,
        allow_hashing_fallback=embed_cfg.allow_hashing_fallback,
        device=embed_cfg.device,
        local_files_only=embed_cfg.local_files_only,
    )
    embeddings = embedder.encode([str(row["answer"]) for row in responses])
    projection = _project_2d(embeddings, seed=seed)
    colors = {"current_only": "#2f6f73", "stale_only": "#b5533c", "mixed": "#6a5acd"}
    preferred_order = ["current_only", "stale_only", "mixed"]
    present = {str(row["condition"]) for row in responses}
    ordered_conditions = [condition for condition in preferred_order if condition in present] + [
        condition for condition in sorted(present) if condition not in preferred_order
    ]

    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for condition in ordered_conditions:
        mask = np.array([row["condition"] == condition for row in responses])
        if np.any(mask):
            axis.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=14,
                alpha=0.55,
                color=colors.get(condition, "#666666"),
                label=condition,
                linewidths=0,
            )
    axis.set_title("2D response embedding projection")
    axis.set_xlabel("Projection 1")
    axis.set_ylabel("Projection 2")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _project_2d(embeddings: np.ndarray, seed: int) -> np.ndarray:
    """Use UMAP when available, else PCA."""
    if embeddings.shape[0] <= 2:
        padded = np.zeros((embeddings.shape[0], 2), dtype=float)
        if embeddings.shape[1] > 0:
            padded[:, : min(2, embeddings.shape[1])] = embeddings[:, : min(2, embeddings.shape[1])]
        return padded
    try:
        import umap

        return umap.UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(embeddings)
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)
