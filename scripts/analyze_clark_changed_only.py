"""Calibrate and lock a CLARK detector using changed questions only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_clark_changed_primary import (
    FEATURES,
    PRIMARY_STATES,
    add_outcomes,
    ensure_binary,
    json_safe,
    load_module,
    load_one,
    primary_subset,
    select_validation_model,
)
from scripts.clark_evaluation import (
    best_f1_threshold,
    binary_scores,
    bootstrap_binary_metrics,
    threshold_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-changed-run", required=True)
    parser.add_argument("--validation-changed-run", required=True)
    parser.add_argument("--locked-changed-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-label", default="clark_changed_only")
    parser.add_argument("--drop-threshold", type=float, default=0.10)
    parser.add_argument("--failure-threshold", type=float, default=0.50)
    parser.add_argument("--success-threshold", type=float, default=0.50)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--simplicity-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def transition_metrics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for time_y, part in frame.groupby("time_y", sort=True):
        labels = part["target_new_degradation"].to_numpy(dtype=int)
        probability = part["risk_probability"].to_numpy(dtype=float)
        auroc, auprc = binary_scores(labels, probability)
        result = threshold_metrics(labels, probability, threshold)
        rows.append(
            {
                "time_y": str(time_y),
                "n": len(part),
                "new_degradation_n": int(labels.sum()),
                "prevalence": float(labels.mean()),
                "auroc": auroc,
                "auprc": auprc,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
        )
    return pd.DataFrame(rows)


def plot_locked(
    frame: pd.DataFrame,
    output_path: Path,
    estimator: Any,
    threshold: float,
    *,
    title: str = "Locked CLARK: frozen detector alarms",
) -> None:
    colors = frame["outcome_state"].map(
        {
            "new_degradation": "#d9473f",
            "recovery_or_adaptive_success": "#16846b",
        }
    )
    fig, axis = plt.subplots(figsize=(7.2, 6.0))

    grid_axis = np.linspace(0.0, 1.0, 301)
    grid_shift, grid_uncertainty = np.meshgrid(grid_axis, grid_axis)
    grid = np.column_stack([grid_shift.ravel(), grid_uncertainty.ravel()])
    grid_probability = estimator.predict_proba(grid)[:, 1].reshape(grid_shift.shape)
    axis.contourf(
        grid_shift,
        grid_uncertainty,
        grid_probability,
        levels=[threshold, 1.0],
        colors=["#f2b38d"],
        alpha=0.22,
    )
    axis.contour(
        grid_shift,
        grid_uncertainty,
        grid_probability,
        levels=[threshold],
        colors=["#17212b"],
        linewidths=1.45,
    )
    axis.scatter(
        frame["shift_score"],
        frame["uncertainty_score"],
        c=colors,
        s=np.where(frame["alarm"], 72, 42),
        alpha=0.84,
        edgecolors=np.where(frame["alarm"], "#17212b", "white"),
        linewidths=np.where(frame["alarm"], 1.1, 0.45),
    )
    axis.axvline(0.5, color="#c7ced6", linewidth=1)
    axis.axhline(0.5, color="#c7ced6", linewidth=1)
    for x, y, label in (
        (0.03, 0.04, "LL"),
        (0.03, 0.96, "LH"),
        (0.97, 0.04, "HL"),
        (0.97, 0.96, "HH"),
    ):
        axis.text(
            x,
            y,
            label,
            ha="left" if x < 0.5 else "right",
            va="bottom" if y < 0.5 else "top",
            color="#7d8792",
            fontsize=9,
            fontweight="bold",
        )
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Shift score (changed-development CDF)")
    axis.set_ylabel("Uncertainty score (changed-development CDF)")
    axis.set_title(title, fontweight="bold")
    axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="#d9473f",
                markeredgecolor="white", markersize=7, label="New degradation",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="#16846b",
                markeredgecolor="white", markersize=7, label="Adaptive success",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="white",
                markeredgecolor="#17212b", markeredgewidth=1.4, markersize=8,
                label="Detector alarm",
            ),
            Patch(
                facecolor="#f2b38d", edgecolor="#17212b", alpha=0.35,
                label=f"Alarm region (p >= {threshold:.3f})",
            ),
        ],
        loc="lower right",
        frameon=True,
        fontsize=8,
    )
    axis.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    helper = load_module(
        "clark_changed_only_score_helper",
        ROOT / "scripts" / "clark_score_features.py",
    )
    ml = load_module(
        "clark_changed_only_ml_helper",
        ROOT / "scripts" / "clark_detector_models.py",
    )

    calibration_raw = load_one(
        helper, args.calibration_changed_run, "calibration", "changed"
    )
    validation_raw = load_one(
        helper, args.validation_changed_run, "validation", "changed"
    )
    locked_raw = load_one(helper, args.locked_changed_run, "locked", "changed")

    # No stable samples are generated in this pilot. T0 changed questions define
    # the development scale; no locked observation enters the reference.
    calibration_reference = helper.stable_references(calibration_raw)
    calibration = add_outcomes(
        helper.add_scores(calibration_raw, calibration_reference), args
    )
    validation = add_outcomes(
        helper.add_scores(validation_raw, calibration_reference), args
    )
    calibration_primary = primary_subset(calibration)
    validation_primary = primary_subset(validation)
    ensure_binary(calibration_primary, "calibration", args.min_class_count)
    ensure_binary(validation_primary, "validation", args.min_class_count)

    specs = ml.model_specs(args.seed)
    model_rows: list[dict[str, Any]] = []
    for model_index, (model_name, (estimator, grid)) in enumerate(specs.items()):
        x_calibration = calibration_primary[FEATURES].to_numpy(dtype=float)
        y_calibration = calibration_primary["target_new_degradation"].to_numpy(dtype=int)
        oof, parameter_frequency = ml.repeated_nested_oof(
            x_calibration,
            y_calibration,
            estimator=estimator,
            grid=grid,
            repeats=args.outer_repeats,
            seed=args.seed + 100 * model_index,
        )
        threshold = best_f1_threshold(y_calibration, oof)
        search = ml.fit_tuned(
            estimator,
            grid,
            x_calibration,
            y_calibration,
            seed=args.seed + 1000 + model_index,
        )
        validation_probability = search.predict_proba(
            validation_primary[FEATURES].to_numpy(dtype=float)
        )[:, 1]
        y_validation = validation_primary["target_new_degradation"].to_numpy(dtype=int)
        calibration_auroc, calibration_auprc = binary_scores(y_calibration, oof)
        validation_auroc, validation_auprc = binary_scores(
            y_validation, validation_probability
        )
        validation_result = threshold_metrics(
            y_validation, validation_probability, threshold
        )
        model_rows.append(
            {
                "model": model_name,
                "calibration_n": len(y_calibration),
                "calibration_positive": int(y_calibration.sum()),
                "calibration_oof_auroc": calibration_auroc,
                "calibration_oof_auprc": calibration_auprc,
                "calibration_oof_f1_threshold": threshold,
                "validation_n": len(y_validation),
                "validation_positive": int(y_validation.sum()),
                "validation_auroc": validation_auroc,
                "validation_auprc": validation_auprc,
                "validation_precision": validation_result["precision"],
                "validation_recall": validation_result["recall"],
                "validation_f1": validation_result["f1"],
                "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
                "nested_parameter_frequency": json.dumps(
                    parameter_frequency, sort_keys=True, default=str
                ),
            }
        )
    model_table = pd.DataFrame(model_rows)
    selected_model = select_validation_model(model_table, args.simplicity_margin)
    model_table["selected"] = model_table["model"] == selected_model
    model_table.to_csv(
        output_dir / "development_model_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    development_raw = pd.concat(
        [calibration_raw, validation_raw], ignore_index=True
    ).drop_duplicates("question_id")
    development_reference = helper.stable_references(development_raw)
    development = add_outcomes(
        helper.add_scores(development_raw, development_reference), args
    )
    locked = add_outcomes(helper.add_scores(locked_raw, development_reference), args)
    development_primary = primary_subset(development)
    locked_primary = primary_subset(locked)
    ensure_binary(development_primary, "development", args.min_class_count)
    ensure_binary(locked_primary, "locked", 1)

    estimator, grid = specs[selected_model]
    x_development = development_primary[FEATURES].to_numpy(dtype=float)
    y_development = development_primary["target_new_degradation"].to_numpy(dtype=int)
    development_oof, parameter_frequency = ml.repeated_nested_oof(
        x_development,
        y_development,
        estimator=estimator,
        grid=grid,
        repeats=args.outer_repeats,
        seed=args.seed + 8000,
    )
    risk_threshold = best_f1_threshold(y_development, development_oof)
    final_search = ml.fit_tuned(
        estimator,
        grid,
        x_development,
        y_development,
        seed=args.seed + 9000,
    )
    locked_primary["risk_probability"] = final_search.predict_proba(
        locked_primary[FEATURES].to_numpy(dtype=float)
    )[:, 1]
    locked_primary["alarm"] = locked_primary["risk_probability"] >= risk_threshold
    y_locked = locked_primary["target_new_degradation"].to_numpy(dtype=int)
    probability = locked_primary["risk_probability"].to_numpy(dtype=float)
    auroc, auprc = binary_scores(y_locked, probability)
    operating = threshold_metrics(y_locked, probability, risk_threshold)
    confidence = bootstrap_binary_metrics(
        y_locked,
        probability,
        risk_threshold,
        rounds=args.bootstrap_rounds,
        seed=args.seed + 12000,
    )
    summary = {
        "experiment": args.experiment_label,
        "population": "changed only",
        "normalization": "calibration+validation changed empirical CDF",
        "selected_model": selected_model,
        "selected_hyperparameters": json_safe(final_search.best_params_),
        "threshold_source": "development changed repeated-nested-OOF max F1",
        "threshold": risk_threshold,
        "locked_n": len(y_locked),
        "locked_positive": int(y_locked.sum()),
        "locked_prevalence": float(y_locked.mean()),
        "locked_auroc": auroc,
        "locked_auprc": auprc,
        **{f"locked_{key}": value for key, value in operating.items()},
        **confidence,
        "stable_control_available": False,
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / "primary_summary.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "primary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    columns = [
        "question_id",
        "question",
        "time_x",
        "time_y",
        "old_answer",
        "current_answer",
        "accuracy_stale",
        "accuracy_current",
        "accuracy_drop",
        "shift_score",
        "uncertainty_score",
        "outcome_state",
        "target_new_degradation",
        "risk_probability",
        "alarm",
    ]
    locked_primary[columns].to_csv(
        output_dir / "locked_changed_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transition_metrics(locked_primary, risk_threshold).to_csv(
        output_dir / "locked_transition_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    artifact = {
        "schema_version": 1,
        "experiment": args.experiment_label,
        "features": FEATURES,
        "normalization": "development changed empirical CDF",
        "references": development_reference,
        "selected_model": selected_model,
        "selected_hyperparameters": json_safe(final_search.best_params_),
        "threshold": risk_threshold,
        "development_parameter_frequency": json_safe(parameter_frequency),
        "locked_labels_used_for_selection": False,
        "stable_control_available": False,
    }
    (output_dir / "frozen_detector.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    joblib.dump(
        {**artifact, "estimator": final_search.best_estimator_},
        output_dir / "frozen_detector.joblib",
    )
    plot_locked(
        locked_primary,
        output_dir / "locked_changed_scatter.png",
        final_search.best_estimator_,
        risk_threshold,
    )
    for time_y, transition in locked_primary.groupby("time_y", sort=True):
        time_x = str(transition["time_x"].iloc[0])
        start_date = pd.Timestamp(time_x).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(time_y).strftime("%Y-%m-%d")
        labels = transition["target_new_degradation"].to_numpy(dtype=int)
        probabilities = transition["risk_probability"].to_numpy(dtype=float)
        result = threshold_metrics(labels, probabilities, risk_threshold)
        plot_locked(
            transition,
            output_dir
            / (
                "locked_changed_scatter_"
                f"{start_date.replace('-', '')}_{end_date.replace('-', '')}.png"
            ),
            final_search.best_estimator_,
            risk_threshold,
            title=f"CLARK {start_date} to {end_date} | F1 = {result['f1']:.3f}",
        )
    report = [
        f"# {args.experiment_label}",
        "",
        "- Population: changed questions only",
        "- Endpoint: new degradation vs adaptive success",
        "- Normalization: development changed empirical CDF",
        "- Stable responses were not generated; stable FPR is unavailable",
        "- Locked labels were not used for model or threshold selection",
        "",
        f"- Selected model: `{selected_model}`",
        f"- Locked n / positives: {len(y_locked)} / {int(y_locked.sum())}",
        f"- AUROC / AUPRC: {auroc:.3f} / {auprc:.3f}",
        f"- Precision / recall / F1: {operating['precision']:.3f} / {operating['recall']:.3f} / {operating['f1']:.3f}",
    ]
    (output_dir / "report_ko.md").write_text("\n".join(report), encoding="utf-8")
    print(output_dir / "report_ko.md")


if __name__ == "__main__":
    main()
