"""Evaluate the frozen CLARK T0 detector on exploratory and confirmatory future events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_clark_changed_only import plot_locked
from scripts.analyze_clark_changed_primary import FEATURES, add_outcomes, load_module, load_one, primary_subset
from scripts.clark_evaluation import binary_scores, threshold_metrics
from src.utils.io import read_jsonl


DEFAULT_RUN = ROOT / "outputs" / "runs" / "clark_t0_temporal_transfer_luna" / "future_all_changed"
DEFAULT_DETECTOR = ROOT / "outputs" / "runs" / "clark_t0_temporal_transfer_luna" / "detector_t0" / "frozen_detector_t0.joblib"
DEFAULT_OUTPUT = ROOT / "outputs" / "runs" / "clark_t0_temporal_transfer_luna" / "analysis"
DEFAULT_DATASET = ROOT / "data" / "processed" / "clark_t0_temporal_transfer" / "future_t1_t4_all_changed.jsonl"


def cohort_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = frame["target_new_degradation"].to_numpy(dtype=int)
    probabilities = frame["risk_probability"].to_numpy(dtype=float)
    auroc, auprc = binary_scores(labels, probabilities)
    operating = threshold_metrics(labels, probabilities, threshold)
    return {
        "n": len(frame),
        "unique_questions": frame["question"].nunique(),
        "new_degradation_n": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": auroc,
        "auprc": auprc,
        **operating,
    }


def clustered_bootstrap(
    frame: pd.DataFrame,
    threshold: float,
    *,
    rounds: int,
    seed: int,
) -> dict[str, float]:
    groups = {name: part.index.to_numpy() for name, part in frame.groupby("question")}
    names = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(rounds):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([groups[str(name)] for name in sampled])
        part = frame.loc[indices]
        labels = part["target_new_degradation"].to_numpy(dtype=int)
        probabilities = part["risk_probability"].to_numpy(dtype=float)
        if len(np.unique(labels)) < 2:
            continue
        auroc, auprc = binary_scores(labels, probabilities)
        operating = threshold_metrics(labels, probabilities, threshold)
        rows.append({"auroc": auroc, "auprc": auprc, **operating})
    output: dict[str, float] = {"cluster_bootstrap_valid_rounds": float(len(rows))}
    for metric in ("auroc", "auprc", "precision", "recall", "f1", "risk_lift"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        output[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
        output[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    helper = load_module(
        "clark_t0_transfer_score_helper",
        ROOT / "scripts" / "clark_score_features.py",
    )
    artifact = joblib.load(args.detector)
    raw = load_one(helper, str(args.future_run), "future_t1_t4", "changed")
    outcome_args = SimpleNamespace(
        drop_threshold=0.10,
        failure_threshold=0.50,
        success_threshold=0.50,
    )
    scored = add_outcomes(
        helper.add_scores(raw, artifact["references"]), outcome_args
    )
    split_map = {
        str(row["id"]): str((row.get("metadata") or {}).get("detector_split"))
        for row in read_jsonl(args.dataset)
    }
    scored["detector_split"] = scored["question_id"].map(split_map)
    primary = primary_subset(scored)
    estimator = artifact["estimator"]
    threshold = float(artifact["threshold"])
    primary["risk_probability"] = estimator.predict_proba(
        primary[FEATURES].to_numpy(dtype=float)
    )[:, 1]
    primary["alarm"] = primary["risk_probability"] >= threshold

    cohorts = {
        "future_all": primary,
        "future_existing_exploratory": primary[
            primary["detector_split"] == "future_existing_exploratory"
        ],
        "future_new_confirmatory": primary[
            primary["detector_split"] == "future_new_confirmatory"
        ],
    }
    summaries: list[dict[str, Any]] = []
    for index, (name, frame) in enumerate(cohorts.items()):
        summary = {"cohort": name, **cohort_metrics(frame, threshold)}
        summary.update(
            clustered_bootstrap(
                frame,
                threshold,
                rounds=args.bootstrap_rounds,
                seed=args.seed + 1000 * index,
            )
        )
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(
        args.output_dir / "cohort_summary.csv", index=False, encoding="utf-8-sig"
    )

    transition_rows: list[dict[str, Any]] = []
    for cohort_name, cohort in cohorts.items():
        for time_y, transition in cohort.groupby("time_y", sort=True):
            transition_rows.append(
                {
                    "cohort": cohort_name,
                    "time_y": time_y,
                    **cohort_metrics(transition, threshold),
                }
            )
            if cohort_name == "future_new_confirmatory":
                start = pd.Timestamp(transition["time_x"].iloc[0]).strftime("%Y-%m-%d")
                end = pd.Timestamp(time_y).strftime("%Y-%m-%d")
                f1 = transition_rows[-1]["f1"]
                plot_locked(
                    transition,
                    args.output_dir
                    / f"confirmatory_scatter_{start.replace('-', '')}_{end.replace('-', '')}.png",
                    estimator,
                    threshold,
                    title=f"Confirmatory {start} to {end} | F1 = {f1:.3f}",
                )
    pd.DataFrame(transition_rows).to_csv(
        args.output_dir / "transition_summary.csv", index=False, encoding="utf-8-sig"
    )
    primary.to_csv(
        args.output_dir / "future_predictions.csv", index=False, encoding="utf-8-sig"
    )
    confirmatory = summary_frame[
        summary_frame["cohort"] == "future_new_confirmatory"
    ].iloc[0]
    report = [
        "# CLARK T0-only temporal transfer",
        "",
        "- Detector: frozen quadratic logistic calibrated on T0 only",
        f"- Frozen threshold: `{threshold:.6f}`",
        "- Confirmatory responses were not available during detector fitting",
        "- Inference uses question-clustered bootstrap confidence intervals",
        "",
        f"- Confirmatory primary n: {int(confirmatory['n'])}",
        f"- AUROC / AUPRC: {confirmatory['auroc']:.3f} / {confirmatory['auprc']:.3f}",
        f"- Precision / recall / F1: {confirmatory['precision']:.3f} / {confirmatory['recall']:.3f} / {confirmatory['f1']:.3f}",
    ]
    (args.output_dir / "report_ko.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(args.output_dir / "report_ko.md")


if __name__ == "__main__":
    main()
