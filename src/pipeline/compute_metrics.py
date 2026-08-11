"""Metric computation pipeline over sampled responses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.clustering.semantic_cluster import ClusteringConfig, cluster_embeddings, cluster_texts_by_nli
from src.data.load_dataset import load_dataset, records_by_id
from src.embedding.embed import EmbeddingConfig, TextEmbedder
from src.metrics.centroid import centroid_shift
from src.metrics.drift import assign_drift_labels
from src.metrics.entropy import semantic_entropy
from src.metrics.quality import (
    QualityConfig,
    classify_answer,
    contradiction_to_reference,
    current_alignment,
    matches_any_reference,
    pairwise_contradiction_rate,
    score_accuracy,
    stale_references,
)
from src.metrics.nli import NLIEntailmentScorer
from src.metrics.volume import semantic_volume
from src.utils.io import (
    ensure_dir,
    load_run_config,
    project_root,
    read_jsonl,
    resolve_path,
    setup_logging,
)

LOGGER = logging.getLogger(__name__)


def run_compute_metrics(run_dir: str | Path) -> Path:
    """Compute sample, question, condition, and drift metrics for a run."""
    root = project_root()
    run_dir_path = resolve_path(run_dir, base_dir=root)
    config = load_run_config(run_dir_path)
    metrics_dir = ensure_dir(run_dir_path / "metrics")
    report_dir = ensure_dir(run_dir_path / "report")
    setup_logging(run_dir_path / "logs" / "metrics.log")

    responses = [row for row in read_jsonl(run_dir_path / "samples" / "responses.jsonl") if row.get("answer")]
    if not responses:
        raise ValueError(f"No usable responses found under {run_dir_path / 'samples'}")

    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset config must be a mapping")
    records = load_dataset(
        dataset_cfg.get("path", "data/processed/temporal_canonical.jsonl"),
        max_questions=dataset_cfg.get("max_questions"),
        base_dir=root,
    )
    record_map = records_by_id(records)
    embed_cfg = EmbeddingConfig.from_mapping(config.get("embedding", {}) if isinstance(config.get("embedding"), dict) else {})
    cluster_cfg = ClusteringConfig.from_mapping(config.get("clustering", {}) if isinstance(config.get("clustering"), dict) else {})
    quality_cfg = QualityConfig.from_mapping(config.get("metrics", {}) if isinstance(config.get("metrics"), dict) else {})
    cluster_scorer = (
        NLIEntailmentScorer(cluster_cfg.nli_config()) if cluster_cfg.method.lower() == "nli" else None
    )
    quality_scorer = None
    if quality_cfg.uses_nli():
        if cluster_scorer is not None and cluster_cfg.nli_config() == quality_cfg.nli_config():
            quality_scorer = cluster_scorer
        else:
            quality_scorer = NLIEntailmentScorer(quality_cfg.nli_config())

    sample_rows = _compute_sample_metrics(responses, record_map, quality_cfg, quality_scorer)
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(metrics_dir / "sample_level_metrics.csv", index=False)

    question_rows: list[dict[str, Any]] = []
    embedding_specs = [("primary", embed_cfg.primary_model)]
    if embed_cfg.use_secondary_for_sensitivity and embed_cfg.secondary_model:
        embedding_specs.append(("secondary", embed_cfg.secondary_model))

    answers = [str(row["answer"]) for row in responses]
    response_df = pd.DataFrame(responses).reset_index().rename(columns={"index": "response_index"})

    for role, model_name in embedding_specs:
        embedder = TextEmbedder(
            model_name=model_name,
            batch_size=embed_cfg.batch_size,
            normalize=embed_cfg.normalize,
            allow_hashing_fallback=embed_cfg.allow_hashing_fallback,
            device=embed_cfg.device,
            local_files_only=embed_cfg.local_files_only,
        )
        embeddings = embedder.encode(answers)
        question_rows.extend(
            _compute_question_metrics_for_embeddings(
                response_df=response_df,
                sample_df=sample_df,
                embeddings=embeddings,
                role=role,
                model_name=model_name,
                cluster_cfg=cluster_cfg,
                cluster_scorer=cluster_scorer,
                pca_dim=embed_cfg.pca_dim,
            )
        )

    question_df = pd.DataFrame(question_rows)
    question_df = _add_cluster_deltas(question_df)
    question_df.to_csv(metrics_dir / "question_level_metrics.csv", index=False)

    primary_df = question_df[question_df["embedding_model_role"] == "primary"].copy()
    condition_summary = _condition_summary(primary_df)
    condition_summary.to_csv(metrics_dir / "condition_level_summary.csv", index=False)

    drift_cfg = config.get("drift", {}) if isinstance(config.get("drift"), dict) else {}
    drift_enabled = bool(drift_cfg.get("enabled", True))
    drift_df: pd.DataFrame | None = None
    if drift_enabled:
        drift_df = assign_drift_labels(
            primary_df,
            tau_shift=float(drift_cfg.get("tau_shift", 1.0)),
            tau_q=float(drift_cfg.get("tau_q", 0.05)),
            weights=drift_cfg.get("weights") if isinstance(drift_cfg.get("weights"), dict) else None,
        )
        drift_df.to_csv(metrics_dir / "drift_labels.csv", index=False)
    _write_summary(
        report_dir / "summary.md",
        config,
        responses,
        sample_df,
        primary_df,
        drift_df,
        drift_enabled=drift_enabled,
    )
    LOGGER.info("Metrics complete: %s", metrics_dir)
    return run_dir_path


def _compute_sample_metrics(
    responses: list[dict[str, Any]],
    record_map: dict[str, Any],
    quality_cfg: QualityConfig,
    quality_scorer: NLIEntailmentScorer | None,
) -> list[dict[str, Any]]:
    """Compute per-response quality flags and fuzzy scores."""
    rows: list[dict[str, Any]] = []
    for response_index, row in enumerate(tqdm(responses, desc="sample metrics")):
        question_id = str(row["question_id"])
        record = record_map[question_id]
        answer = str(row["answer"])
        condition = str(row["condition"])
        classification = classify_answer(answer, record, condition, quality_cfg, scorer=quality_scorer)
        target_accuracy = score_accuracy(
            answer,
            str(classification["target_answer"]),
            threshold=quality_cfg.accuracy_f1_threshold,
        )
        current_accuracy = score_accuracy(answer, record.gold_answer, threshold=quality_cfg.accuracy_f1_threshold)
        stale_refs = stale_references(record)
        stale_match, stale_best_f1 = matches_any_reference(answer, stale_refs, threshold=quality_cfg.stale_f1_threshold)
        rows.append(
            {
                "response_index": response_index,
                "question_id": question_id,
                "condition": condition,
                "sample_idx": int(row["sample_idx"]),
                "quality_label": classification["quality_label"],
                "target_reference": classification["target_reference"],
                "target_answer": classification["target_answer"],
                "correct": classification["target_correct"],
                "target_correct": classification["target_correct"],
                "target_exact_match": target_accuracy["exact_match"],
                "target_substring_match": target_accuracy["substring_match"],
                "target_token_f1": target_accuracy["token_f1"],
                "target_rouge_l": target_accuracy["rouge_l"],
                "current_answer_match": classification["current_answer_match"],
                "current_exact_match": current_accuracy["exact_match"],
                "current_substring_match": current_accuracy["substring_match"],
                "current_token_f1": current_accuracy["token_f1"],
                "current_rouge_l": current_accuracy["rouge_l"],
                "stale_answer_match": classification["stale_answer_match"],
                "stale_answer_best_f1": max(float(classification["stale_answer_best_f1"]), stale_best_f1),
                "fuzzy_stale_answer_match": stale_match,
                "missing": classification["is_missing"],
                "harmful_other": classification["harmful_other"],
                "ambiguous": classification["ambiguous"],
                "contradiction_to_gold": contradiction_to_reference(answer, record.gold_answer),
                "current_alignment": current_alignment(answer, record),
            }
        )
    return rows


def _compute_question_metrics_for_embeddings(
    response_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    embeddings: np.ndarray,
    role: str,
    model_name: str,
    cluster_cfg: ClusteringConfig,
    cluster_scorer: NLIEntailmentScorer | None,
    pca_dim: int,
) -> list[dict[str, Any]]:
    """Compute distribution and aggregated quality metrics for each question-condition group."""
    rows: list[dict[str, Any]] = []
    grouped_indices: dict[tuple[str, str], np.ndarray] = {}
    for (question_id, condition), group in response_df.groupby(["question_id", "condition"], sort=False):
        grouped_indices[(str(question_id), str(condition))] = group["response_index"].to_numpy(dtype=int)

    for (question_id, condition), indices in tqdm(grouped_indices.items(), desc=f"question metrics/{role}"):
        condition_embeddings = embeddings[indices]
        current_indices = grouped_indices.get((question_id, _baseline_condition(grouped_indices, question_id)))
        baseline_embeddings = embeddings[current_indices] if current_indices is not None else condition_embeddings
        group_df = response_df.loc[indices].sort_values("response_index")
        answers = group_df["answer"].astype(str).tolist()
        question = str(group_df["question"].iloc[0]) if not group_df.empty else ""
        if cluster_cfg.method.lower() == "nli":
            if cluster_scorer is None:
                raise RuntimeError("NLI clustering was requested but no NLI scorer is available")
            labels = cluster_texts_by_nli(answers, question, cluster_cfg, cluster_scorer)
        else:
            labels = cluster_embeddings(condition_embeddings, cluster_cfg)
        sample_group = sample_df[
            (sample_df["question_id"] == question_id) & (sample_df["condition"] == condition)
        ].copy()
        pairwise_rate = pairwise_contradiction_rate(answers)
        gold_rate = float(sample_group["contradiction_to_gold"].mean()) if not sample_group.empty else 0.0
        contradiction_rate = max(pairwise_rate, gold_rate)
        label_counts = sample_group["quality_label"].value_counts().to_dict() if not sample_group.empty else {}
        rows.append(
            {
                "question_id": question_id,
                "condition": condition,
                "embedding_model_role": role,
                "embedding_model": model_name,
                "n_samples": int(len(indices)),
                "n_clusters": int(len(set(labels.tolist()))),
                "semantic_entropy": semantic_entropy(labels),
                "semantic_volume": semantic_volume(condition_embeddings, pca_dim=pca_dim),
                "centroid_shift": centroid_shift(condition_embeddings, baseline_embeddings),
                "accuracy": float(sample_group["correct"].mean()) if not sample_group.empty else 0.0,
                "target_correct_rate": float(sample_group["target_correct"].mean()) if not sample_group.empty else 0.0,
                "current_answer_rate": float(sample_group["current_answer_match"].mean()) if not sample_group.empty else 0.0,
                "stale_answer_rate": float(sample_group["stale_answer_match"].mean()) if not sample_group.empty else 0.0,
                "missing_rate": float(sample_group["missing"].mean()) if not sample_group.empty else 0.0,
                "harmful_other_rate": float(sample_group["harmful_other"].mean()) if not sample_group.empty else 0.0,
                "ambiguous_rate": float(sample_group["ambiguous"].mean()) if not sample_group.empty else 0.0,
                "perfect_current_rate": float(label_counts.get("perfect_current", 0) / max(len(sample_group), 1)),
                "perfect_stale_rate": float(label_counts.get("perfect_stale", 0) / max(len(sample_group), 1)),
                "mean_target_token_f1": float(sample_group["target_token_f1"].mean()) if not sample_group.empty else 0.0,
                "mean_target_rouge_l": float(sample_group["target_rouge_l"].mean()) if not sample_group.empty else 0.0,
                "contradiction_rate": contradiction_rate,
                "pairwise_contradiction_rate": pairwise_rate,
                "gold_contradiction_rate": gold_rate,
                "current_alignment": float(sample_group["current_alignment"].mean()) if not sample_group.empty else 0.0,
            }
        )
    return rows


def _add_cluster_deltas(question_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-question cluster count deltas relative to the baseline condition."""
    if question_df.empty or "n_clusters" not in question_df.columns:
        return question_df
    output = question_df.copy()
    baseline = _baseline_condition_from_frame(output)
    baselines = output[output["condition"] == baseline][
        ["question_id", "embedding_model_role", "n_clusters"]
    ].rename(columns={"n_clusters": "baseline_n_clusters"})
    output = output.merge(baselines, on=["question_id", "embedding_model_role"], how="left")
    output["baseline_n_clusters"] = output["baseline_n_clusters"].fillna(output["n_clusters"]).astype(int)
    output["delta_n_clusters"] = output["n_clusters"].astype(int) - output["baseline_n_clusters"].astype(int)
    return output


def _baseline_condition(grouped_indices: dict[tuple[str, str], np.ndarray], question_id: str) -> str:
    """Return the preferred baseline condition for a question."""
    if (question_id, "current_only") in grouped_indices:
        return "current_only"
    if (question_id, "stale_0") in grouped_indices:
        return "stale_0"
    candidates = [condition for qid, condition in grouped_indices if qid == question_id]
    return sorted(candidates)[0] if candidates else ""


def _baseline_condition_from_frame(df: pd.DataFrame) -> str:
    """Return the preferred baseline condition in a metric dataframe."""
    conditions = set(df["condition"].astype(str))
    if "current_only" in conditions:
        return "current_only"
    if "stale_0" in conditions:
        return "stale_0"
    return sorted(conditions)[0]


def _condition_summary(question_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize primary question-level metrics by condition."""
    metrics = [
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
    ]
    rows: list[dict[str, Any]] = []
    for condition, group in question_df.groupby("condition", sort=False):
        row: dict[str, Any] = {"condition": condition, "n_questions": int(group["question_id"].nunique())}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_se"] = float(row[f"{metric}_std"] / np.sqrt(max(len(values), 1)))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_summary(
    path: Path,
    config: dict[str, Any],
    responses: list[dict[str, Any]],
    sample_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    drift_df: pd.DataFrame | None,
    drift_enabled: bool = True,
) -> None:
    """Write a compact Markdown summary report."""
    label_counts = drift_df["drift_label"].value_counts().to_dict() if drift_enabled and drift_df is not None and "drift_label" in drift_df.columns else {}
    condition_means = primary_df.groupby("condition")[
        [
            "semantic_entropy",
            "semantic_volume",
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
        ]
    ].mean()
    drift_columns = [
        "benchmark_delta_current_minus_stale",
        "current_performance_drop",
        "condition_current_pull",
        "condition_stale_pull",
        "answer_flip_magnitude_vs_current",
        "mixed_conflict_rate",
        "semantic_shift_increased",
        "benchmark_semantic_drift",
    ]
    available_drift_columns = (
        [column for column in drift_columns if column in drift_df.columns]
        if drift_enabled and drift_df is not None
        else []
    )
    drift_means = pd.DataFrame()
    if drift_enabled and drift_df is not None and available_drift_columns:
        drift_means = drift_df.groupby("condition")[available_drift_columns].mean()
    benchmark_label_counts = (
        drift_df["benchmark_drift_label"].value_counts().to_dict()
        if drift_enabled and drift_df is not None and "benchmark_drift_label" in drift_df.columns
        else {}
    )
    experiment_name = str(config.get("experiment_name", "")).strip() or "Temporal RAG Drift Summary"
    lines = [
        f"# {experiment_name} Summary",
        "",
        f"- Responses: {len(responses)}",
        f"- Questions: {primary_df['question_id'].nunique()}",
        f"- Conditions: {', '.join(map(str, primary_df['condition'].unique()))}",
        f"- Sample-level metric rows: {len(sample_df)}",
        f"- Drift labels: {label_counts}" if drift_enabled else "- Drift labels: disabled",
        f"- Benchmark drift labels: {benchmark_label_counts}" if drift_enabled else "- Benchmark drift labels: disabled",
        "",
        "## Condition Means",
        "",
        condition_means.to_markdown(),
        "",
        "## Benchmark Drift Means",
        "",
        drift_means.to_markdown() if not drift_means.empty else "_No benchmark drift columns available._",
        "",
        "## Output Files",
        "",
        "- samples/responses.jsonl",
        "- metrics/sample_level_metrics.csv",
        "- metrics/question_level_metrics.csv",
        "- metrics/condition_level_summary.csv",
    ]
    if drift_enabled:
        lines.append("- metrics/drift_labels.csv")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
