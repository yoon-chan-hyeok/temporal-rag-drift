"""Compute answer-distribution distances for completed CLARK run directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering.semantic_cluster import ClusteringConfig, cluster_embeddings
from src.embedding.embed import EmbeddingConfig, TextEmbedder, normalize_embeddings
from src.utils.io import ensure_dir, load_run_config, project_root, read_jsonl, setup_logging, write_csv

ROOT = project_root()


def sliced_wasserstein_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    num_projections: int = 128,
    seed: int = 42,
) -> float:
    """Approximate Wasserstein distance with random 1D projections."""
    if left.size == 0 or right.size == 0:
        return 0.0
    dim = left.shape[1]
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(num_projections, dim))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True) + 1e-12
    distances: list[float] = []
    for vector in projections:
        left_proj = np.sort(left @ vector)
        right_proj = np.sort(right @ vector)
        quantiles = np.linspace(0.0, 1.0, max(len(left_proj), len(right_proj)))
        left_interp = np.interp(quantiles, np.linspace(0.0, 1.0, len(left_proj)), left_proj)
        right_interp = np.interp(quantiles, np.linspace(0.0, 1.0, len(right_proj)), right_proj)
        distances.append(float(np.mean(np.abs(left_interp - right_interp))))
    return float(np.mean(distances))


def pairwise_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distances."""
    delta = left[:, None, :] - right[None, :, :]
    return np.linalg.norm(delta, axis=2)


def multivariate_energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Multivariate energy distance via pairwise Euclidean distances."""
    if left.size == 0 or right.size == 0:
        return 0.0
    cross = pairwise_euclidean(left, right).mean()
    within_left = pairwise_euclidean(left, left).mean()
    within_right = pairwise_euclidean(right, right).mean()
    return float(max(0.0, 2.0 * cross - within_left - within_right))


def rbf_mmd(left: np.ndarray, right: np.ndarray) -> float:
    """RBF-kernel MMD with a median-distance bandwidth heuristic."""
    if left.size == 0 or right.size == 0:
        return 0.0
    combined = np.vstack([left, right])
    dists = pairwise_euclidean(combined, combined)
    upper = dists[np.triu_indices_from(dists, k=1)]
    sigma = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    gamma = 1.0 / max(2.0 * sigma * sigma, 1e-12)

    def kernel(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
        sq = np.sum((mat_a[:, None, :] - mat_b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * sq)

    k_xx = kernel(left, left)
    k_yy = kernel(right, right)
    k_xy = kernel(left, right)
    m = left.shape[0]
    n = right.shape[0]
    term_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1)) if m > 1 else 0.0
    term_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1)) if n > 1 else 0.0
    term_xy = k_xy.mean()
    return float(max(0.0, term_xx + term_yy - 2.0 * term_xy))


def js_divergence(left_probs: np.ndarray, right_probs: np.ndarray) -> float:
    """Jensen-Shannon divergence between discrete distributions."""
    left = np.asarray(left_probs, dtype=float)
    right = np.asarray(right_probs, dtype=float)
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    mean = 0.5 * (left + right)

    def kl_div(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], 1e-12, None))))

    return 0.5 * kl_div(left, mean) + 0.5 * kl_div(right, mean)


def cluster_js(left: np.ndarray, right: np.ndarray, cluster_cfg: ClusteringConfig) -> tuple[float, int]:
    """Cluster pooled embeddings and compute JS divergence over cluster proportions."""
    pooled = np.vstack([left, right])
    labels = cluster_embeddings(pooled, cluster_cfg)
    n_left = left.shape[0]
    left_labels = labels[:n_left]
    right_labels = labels[n_left:]
    unique = sorted(set(labels.tolist()))
    left_counts = np.asarray([(left_labels == label).sum() for label in unique], dtype=float)
    right_counts = np.asarray([(right_labels == label).sum() for label in unique], dtype=float)
    return float(js_divergence(left_counts, right_counts)), int(len(unique))


def bootstrap_ci(values: np.ndarray, *, rounds: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap a mean confidence interval."""
    if values.size == 0:
        return (0.0, 0.0)
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(rounds, dtype=float)
    for index in range(rounds):
        sample = rng.choice(values, size=values.size, replace=True)
        means[index] = sample.mean()
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def natural_comparisons(conditions: list[str]) -> list[tuple[str, str]]:
    """Return available pairwise comparisons for temporal run conditions."""
    pairs = [("current_only", "mixed"), ("current_only", "stale_only"), ("mixed", "stale_only")]
    return [pair for pair in pairs if pair[0] in conditions and pair[1] in conditions]


def load_responses(run_dirs: list[str]) -> pd.DataFrame:
    """Load and concatenate raw responses from multiple shard runs."""
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        path = ROOT / run_dir / "samples" / "responses.jsonl"
        for row in read_jsonl(path):
            item = dict(row)
            item["run_dir"] = run_dir
            rows.append(item)
    if not rows:
        raise ValueError(f"No responses found for run dirs: {run_dirs}")
    return pd.DataFrame(rows)


def experiment_conditions(config: dict[str, object]) -> list[str]:
    """Return condition names from a run config."""
    raw = config.get("conditions", [])
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute temporal answer-distribution distances.")
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        required=True,
        help="Run directory relative to repo root. Pass once per shard run.",
    )
    parser.add_argument(
        "--label",
        default="custom",
        help="Short experiment label written into the output CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/runs/distribution_shift_custom",
        help="Output directory for CSV summaries, relative to repo root.",
    )
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=1000,
        help="Bootstrap rounds for summary confidence intervals.",
    )
    parser.add_argument(
        "--comparison",
        nargs=2,
        action="append",
        metavar=("CONDITION_A", "CONDITION_B"),
        help=(
            "Explicit condition pair. Repeat for multiple pairs. When omitted, "
            "the standard current/stale/mixed comparisons are used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(ROOT / args.output_dir)
    setup_logging(output_dir / "distribution_shift.log")

    config = load_run_config(ROOT / args.run_dirs[0])
    responses = load_responses(args.run_dirs)
    conditions = experiment_conditions(config)
    comparisons = (
        [(str(left), str(right)) for left, right in args.comparison]
        if args.comparison
        else natural_comparisons(conditions)
    )
    unknown = sorted(
        {
            condition
            for pair in comparisons
            for condition in pair
            if condition not in conditions
        }
    )
    if unknown:
        raise ValueError(f"Unknown comparison conditions: {unknown}; available: {conditions}")
    if not comparisons:
        raise ValueError(f"No temporal comparisons available for conditions: {conditions}")

    embedding_cfg = EmbeddingConfig.from_mapping(
        config.get("embedding", {}) if isinstance(config.get("embedding"), dict) else {}
    )
    clustering_cfg_raw = config.get("clustering", {}) if isinstance(config.get("clustering"), dict) else {}
    base_cluster_cfg = ClusteringConfig.from_mapping(clustering_cfg_raw)
    js_cluster_cfg = ClusteringConfig(
        method="agglomerative",
        cosine_threshold=base_cluster_cfg.cosine_threshold,
        min_cluster_size=base_cluster_cfg.min_cluster_size,
        nli_model=base_cluster_cfg.nli_model,
        nli_batch_size=base_cluster_cfg.nli_batch_size,
        nli_device=base_cluster_cfg.nli_device,
        entailment_threshold=base_cluster_cfg.entailment_threshold,
        contradiction_threshold=base_cluster_cfg.contradiction_threshold,
        equivalence_rule=base_cluster_cfg.equivalence_rule,
        lexical_f1_threshold=base_cluster_cfg.lexical_f1_threshold,
        numeric_mismatch_splits=base_cluster_cfg.numeric_mismatch_splits,
        local_files_only=base_cluster_cfg.local_files_only,
    )

    embedder = TextEmbedder(
        model_name=embedding_cfg.primary_model,
        batch_size=embedding_cfg.batch_size,
        normalize=embedding_cfg.normalize,
        allow_hashing_fallback=embedding_cfg.allow_hashing_fallback,
        device=embedding_cfg.device,
        local_files_only=embedding_cfg.local_files_only,
    )
    answers = responses["answer"].astype(str).tolist()
    embeddings = embedder.encode(answers)
    embeddings = normalize_embeddings(np.asarray(embeddings, dtype=np.float32))
    responses = responses.reset_index().rename(columns={"index": "response_index"})

    grouped: dict[tuple[str, str], np.ndarray] = {}
    for (question_id, condition), group in responses.groupby(["question_id", "condition"], sort=False):
        grouped[(str(question_id), str(condition))] = group["response_index"].to_numpy(dtype=int)

    per_question_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for condition_a, condition_b in comparisons:
        rows_for_comparison: list[dict[str, object]] = []
        question_ids = sorted(
            {
                qid
                for (qid, condition) in grouped
                if condition == condition_a or condition == condition_b
            }
        )
        for question_id in question_ids:
            indices_a = grouped.get((question_id, condition_a))
            indices_b = grouped.get((question_id, condition_b))
            if indices_a is None or indices_b is None:
                continue
            embed_a = embeddings[indices_a]
            embed_b = embeddings[indices_b]
            row = {
                "experiment_label": args.label,
                "question_id": question_id,
                "comparison": f"{condition_a}__vs__{condition_b}",
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_a": int(embed_a.shape[0]),
                "n_b": int(embed_b.shape[0]),
                "swd": sliced_wasserstein_distance(embed_a, embed_b),
                "mmd_rbf": rbf_mmd(embed_a, embed_b),
                "energy": multivariate_energy_distance(embed_a, embed_b),
                "centroid_gap": float(np.linalg.norm(embed_a.mean(axis=0) - embed_b.mean(axis=0))),
            }
            cluster_js_value, union_clusters = cluster_js(embed_a, embed_b, js_cluster_cfg)
            row["cluster_js"] = cluster_js_value
            row["union_clusters"] = union_clusters
            rows_for_comparison.append(row)
            per_question_rows.append(row)

        if not rows_for_comparison:
            continue
        frame = pd.DataFrame(rows_for_comparison)
        summary: dict[str, object] = {
            "experiment_label": args.label,
            "comparison": f"{condition_a}__vs__{condition_b}",
            "condition_a": condition_a,
            "condition_b": condition_b,
            "n_questions": int(len(frame)),
        }
        for metric in ("swd", "mmd_rbf", "energy", "cluster_js", "centroid_gap", "union_clusters"):
            values = frame[metric].to_numpy(dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_std"] = float(values.std(ddof=0))
            if metric != "union_clusters":
                ci_low, ci_high = bootstrap_ci(values, rounds=args.bootstrap_rounds)
                summary[f"{metric}_ci_low"] = ci_low
                summary[f"{metric}_ci_high"] = ci_high
        summary_rows.append(summary)

    write_csv(per_question_rows, output_dir / "per_question_distribution_shift.csv")
    write_csv(summary_rows, output_dir / "summary_distribution_shift.csv")
    print(output_dir)


if __name__ == "__main__":
    main()
