"""Fit a changed-only CLARK detector and evaluate stable questions as controls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.clark_evaluation import (
    assign_outcome,
    best_f1_threshold,
    binary_scores,
    bootstrap_binary_metrics,
    load_runs,
    threshold_metrics,
)


FEATURES = ["shift_score", "uncertainty_score"]
PRIMARY_STATES = ("new_degradation", "recovery_or_adaptive_success")
MODEL_COMPLEXITY = {
    "l2_logistic": 0,
    "quadratic_logistic": 1,
    "additive_gam": 2,
    "gaussian_process": 3,
    "rbf_svm": 4,
}
BASELINE_SCORES = (
    "shift_score",
    "uncertainty_score",
    "low_shift_high_uncertainty",
    "high_shift_high_uncertainty",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("calibration", "validation", "locked"):
        parser.add_argument(f"--{split}-changed-run", required=True)
        parser.add_argument(f"--{split}-stable-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-label", default="clark_changed_primary")
    parser.add_argument("--drop-threshold", type=float, default=0.10)
    parser.add_argument("--failure-threshold", type=float, default=0.50)
    parser.add_argument("--success-threshold", type=float, default=0.50)
    parser.add_argument("--stable-risk-quantile", type=float, default=0.90)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--simplicity-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_one(helper: Any, path: str, split: str, expected_label: str) -> pd.DataFrame:
    frame = load_runs(helper, [path], split=split)
    observed = set(frame["change_label"].astype(str))
    if observed != {expected_label}:
        raise ValueError(f"{split}/{expected_label} contains labels {sorted(observed)}")
    return frame


def add_outcomes(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    output = frame.copy()
    output["outcome_state"] = assign_outcome(
        output,
        drop_threshold=args.drop_threshold,
        failure_threshold=args.failure_threshold,
        success_threshold=args.success_threshold,
    )
    return output


def primary_subset(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[frame["outcome_state"].isin(PRIMARY_STATES)].copy()
    output["target_new_degradation"] = (
        output["outcome_state"] == "new_degradation"
    ).astype(int)
    return output.reset_index(drop=True)


def ensure_binary(frame: pd.DataFrame, split: str, minimum: int) -> None:
    counts = frame["target_new_degradation"].value_counts().to_dict()
    if min(int(counts.get(0, 0)), int(counts.get(1, 0))) < minimum:
        raise ValueError(
            f"{split} changed-only primary endpoint has insufficient classes: {counts}; "
            f"minimum={minimum}"
        )


def select_validation_model(rows: pd.DataFrame, margin: float) -> str:
    valid = rows.dropna(subset=["validation_f1", "validation_auprc"]).copy()
    if valid.empty:
        raise ValueError("No model produced valid validation F1/AUPRC values.")
    best = float(valid["validation_f1"].max())
    candidates = valid[valid["validation_f1"] >= best - margin].copy()
    candidates["complexity"] = candidates["model"].map(MODEL_COMPLEXITY)
    candidates = candidates.sort_values(
        ["complexity", "validation_f1", "validation_auprc", "validation_auroc"],
        ascending=[True, False, False, False],
    )
    return str(candidates.iloc[0]["model"])


def quantile_threshold(values: np.ndarray, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("--stable-risk-quantile must be between 0 and 1.")
    return float(np.quantile(values.astype(float), quantile, method="higher"))


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auroc, auprc = binary_scores(labels, scores)
    return {"auroc": auroc, "auprc": auprc}


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def state_counts(split: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    counts = frame["outcome_state"].value_counts()
    return [
        {
            "split": split,
            "change_label": str(frame["change_label"].iloc[0]),
            "state": state,
            "count": int(count),
            "rate": float(count / len(frame)),
        }
        for state, count in counts.items()
    ]


def plot_locked(
    changed: pd.DataFrame,
    stable: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharex=True, sharey=True)
    changed_colors = changed["outcome_state"].map(
        {
            "new_degradation": "#d9473f",
            "recovery_or_adaptive_success": "#16846b",
            "persistent_failure": "#8c6bb1",
            "normal": "#9aa6b2",
        }
    )
    axes[0].scatter(
        changed["shift_score"],
        changed["uncertainty_score"],
        c=changed_colors,
        s=42,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.45,
    )
    stable_colors = np.where(stable["alarm"], "#d9473f", "#9aa6b2")
    axes[1].scatter(
        stable["shift_score"],
        stable["uncertainty_score"],
        c=stable_colors,
        s=42,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.45,
    )
    for axis, title in zip(
        axes,
        ("Locked changed: primary population", "Locked stable: null control"),
    ):
        axis.axvline(0.5, color="#c7ced6", linewidth=1)
        axis.axhline(0.5, color="#c7ced6", linewidth=1)
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Shift percentile score")
        axis.set_title(title, fontweight="bold")
        axis.grid(alpha=0.16)
    axes[0].set_ylabel("Uncertainty percentile score")
    fig.suptitle("CLARK changed-only detector with stable null control", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def transition_metrics(
    changed_primary: pd.DataFrame,
    stable: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    transitions = sorted(
        set(changed_primary["time_y"].astype(str)) | set(stable["time_y"].astype(str))
    )
    for time_y in transitions:
        changed_part = changed_primary[changed_primary["time_y"].astype(str) == time_y]
        stable_part = stable[stable["time_y"].astype(str) == time_y]
        labels = changed_part["target_new_degradation"].to_numpy(dtype=int)
        probabilities = changed_part["risk_probability"].to_numpy(dtype=float)
        auroc, auprc = binary_scores(labels, probabilities)
        threshold_result = threshold_metrics(labels, probabilities, threshold)
        stable_no_degradation = stable_part["outcome_state"] != "new_degradation"
        rows.append(
            {
                "time_y": time_y,
                "changed_primary_n": len(changed_part),
                "new_degradation_n": int(labels.sum()),
                "prevalence": float(labels.mean()) if len(labels) else math.nan,
                "auroc": auroc,
                "auprc": auprc,
                "precision": threshold_result["precision"],
                "recall": threshold_result["recall"],
                "f1": threshold_result["f1"],
                "stable_n": len(stable_part),
                "stable_alarm_rate": (
                    float(stable_part["alarm"].mean()) if len(stable_part) else math.nan
                ),
                "stable_no_new_degradation_fpr": (
                    float(stable_part.loc[stable_no_degradation, "alarm"].mean())
                    if stable_no_degradation.any()
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    helper = load_module(
        "clark_score_helper",
        ROOT / "scripts" / "clark_score_features.py",
    )
    ml = load_module(
        "clark_ml_helper",
        ROOT / "scripts" / "clark_detector_models.py",
    )

    raw: dict[tuple[str, str], pd.DataFrame] = {}
    for split in ("calibration", "validation", "locked"):
        for label in ("changed", "stable"):
            raw[(split, label)] = load_one(
                helper,
                getattr(args, f"{split}_{label}_run"),
                split,
                label,
            )

    # Model-family selection uses T0 stable CDF and T0/T1 changed labels only.
    calibration_reference = helper.stable_references(raw[("calibration", "stable")])
    calibration_changed = add_outcomes(
        helper.add_scores(raw[("calibration", "changed")], calibration_reference),
        args,
    )
    validation_changed = add_outcomes(
        helper.add_scores(raw[("validation", "changed")], calibration_reference),
        args,
    )
    calibration_primary = primary_subset(calibration_changed)
    validation_primary = primary_subset(validation_changed)
    ensure_binary(calibration_primary, "calibration", args.min_class_count)
    ensure_binary(validation_primary, "validation", args.min_class_count)

    model_rows: list[dict[str, Any]] = []
    specs = ml.model_specs(args.seed)
    for model_index, (model_name, (estimator, grid)) in enumerate(specs.items()):
        x_cal = calibration_primary[FEATURES].to_numpy(dtype=float)
        y_cal = calibration_primary["target_new_degradation"].to_numpy(dtype=int)
        oof, parameter_counts = ml.repeated_nested_oof(
            x_cal,
            y_cal,
            estimator=estimator,
            grid=grid,
            repeats=args.outer_repeats,
            seed=args.seed + 100 * model_index,
        )
        search = ml.fit_tuned(estimator, grid, x_cal, y_cal, seed=args.seed + 1000 + model_index)
        validation_probability = search.predict_proba(
            validation_primary[FEATURES].to_numpy(dtype=float)
        )[:, 1]
        y_validation = validation_primary["target_new_degradation"].to_numpy(dtype=int)
        calibration_auroc, calibration_auprc = binary_scores(y_cal, oof)
        calibration_threshold = best_f1_threshold(y_cal, oof)
        validation_at_calibration_threshold = threshold_metrics(
            y_validation,
            validation_probability,
            calibration_threshold,
        )
        validation_auroc, validation_auprc = binary_scores(y_validation, validation_probability)
        model_rows.append(
            {
                "model": model_name,
                "calibration_n": len(y_cal),
                "calibration_positive": int(y_cal.sum()),
                "calibration_oof_auroc": calibration_auroc,
                "calibration_oof_auprc": calibration_auprc,
                "calibration_oof_f1_threshold": calibration_threshold,
                "validation_n": len(y_validation),
                "validation_positive": int(y_validation.sum()),
                "validation_auroc": validation_auroc,
                "validation_auprc": validation_auprc,
                "validation_precision": validation_at_calibration_threshold["precision"],
                "validation_recall": validation_at_calibration_threshold["recall"],
                "validation_f1": validation_at_calibration_threshold["f1"],
                "calibration_best_params": json.dumps(
                    search.best_params_, sort_keys=True, default=str
                ),
                "nested_parameter_frequency": json.dumps(
                    parameter_counts, sort_keys=True, default=str
                ),
            }
        )
    model_table = pd.DataFrame(model_rows)
    selected_model = select_validation_model(model_table, args.simplicity_margin)
    model_table["selected"] = model_table["model"] == selected_model
    model_table.to_csv(output_dir / "development_model_selection.csv", index=False, encoding="utf-8-sig")

    # After selection, combine all development data, refit, and freeze before T2-T4.
    development_stable_raw = pd.concat(
        [raw[("calibration", "stable")], raw[("validation", "stable")]],
        ignore_index=True,
    )
    final_reference = helper.stable_references(development_stable_raw)
    development_changed = add_outcomes(
        helper.add_scores(
            pd.concat(
                [raw[("calibration", "changed")], raw[("validation", "changed")]],
                ignore_index=True,
            ),
            final_reference,
        ),
        args,
    )
    development_stable = add_outcomes(helper.add_scores(development_stable_raw, final_reference), args)
    locked_changed = add_outcomes(helper.add_scores(raw[("locked", "changed")], final_reference), args)
    locked_stable = add_outcomes(helper.add_scores(raw[("locked", "stable")], final_reference), args)
    development_primary = primary_subset(development_changed)
    locked_primary = primary_subset(locked_changed)
    ensure_binary(development_primary, "development refit", args.min_class_count)
    ensure_binary(locked_primary, "locked", 1)

    estimator, grid = specs[selected_model]
    x_development = development_primary[FEATURES].to_numpy(dtype=float)
    y_development = development_primary["target_new_degradation"].to_numpy(dtype=int)
    development_oof, development_parameter_frequency = ml.repeated_nested_oof(
        x_development,
        y_development,
        estimator=estimator,
        grid=grid,
        repeats=args.outer_repeats,
        seed=args.seed + 8000,
    )
    f1_risk_threshold = best_f1_threshold(y_development, development_oof)
    final_search = ml.fit_tuned(
        estimator,
        grid,
        x_development,
        y_development,
        seed=args.seed + 9000,
    )
    stable_development_probability = final_search.predict_proba(
        development_stable[FEATURES].to_numpy(dtype=float)
    )[:, 1]
    stable_p90_threshold = quantile_threshold(
        stable_development_probability,
        args.stable_risk_quantile,
    )

    locked_changed["risk_probability"] = final_search.predict_proba(
        locked_changed[FEATURES].to_numpy(dtype=float)
    )[:, 1]
    locked_changed["alarm_f1"] = (
        locked_changed["risk_probability"] >= f1_risk_threshold
    )
    locked_changed["alarm_stable_p90"] = (
        locked_changed["risk_probability"] >= stable_p90_threshold
    )
    locked_changed["alarm"] = locked_changed["alarm_f1"]
    locked_stable["risk_probability"] = final_search.predict_proba(
        locked_stable[FEATURES].to_numpy(dtype=float)
    )[:, 1]
    locked_stable["alarm_f1"] = (
        locked_stable["risk_probability"] >= f1_risk_threshold
    )
    locked_stable["alarm_stable_p90"] = (
        locked_stable["risk_probability"] >= stable_p90_threshold
    )
    locked_stable["alarm"] = locked_stable["alarm_f1"]
    locked_primary = locked_changed[
        locked_changed["outcome_state"].isin(PRIMARY_STATES)
    ].copy()
    locked_primary["target_new_degradation"] = (
        locked_primary["outcome_state"] == "new_degradation"
    ).astype(int)

    locked_labels = locked_primary["target_new_degradation"].to_numpy(dtype=int)
    locked_probability = locked_primary["risk_probability"].to_numpy(dtype=float)
    primary_auroc, primary_auprc = binary_scores(locked_labels, locked_probability)
    primary_threshold = threshold_metrics(
        locked_labels,
        locked_probability,
        f1_risk_threshold,
    )
    stable_p90_result = threshold_metrics(
        locked_labels,
        locked_probability,
        stable_p90_threshold,
    )
    confidence = bootstrap_binary_metrics(
        locked_labels,
        locked_probability,
        f1_risk_threshold,
        rounds=args.bootstrap_rounds,
        seed=args.seed + 12000,
    )

    stable_no_new_degradation = locked_stable["outcome_state"] != "new_degradation"
    stable_fpr = float(
        locked_stable.loc[stable_no_new_degradation, "alarm_f1"].mean()
    )
    stable_p90_fpr = float(
        locked_stable.loc[
            stable_no_new_degradation,
            "alarm_stable_p90",
        ].mean()
    )
    primary_summary = {
        "experiment": args.experiment_label,
        "selected_model": selected_model,
        "selected_hyperparameters": json_safe(final_search.best_params_),
        "model_selection": "validation F1 at calibration-OOF threshold",
        "risk_threshold_source": "development changed repeated-nested-OOF max F1",
        "risk_threshold": f1_risk_threshold,
        "stable_p90_threshold": stable_p90_threshold,
        "locked_changed_total": len(locked_changed),
        "locked_primary_n": len(locked_primary),
        "locked_primary_positive": int(locked_labels.sum()),
        "locked_primary_prevalence": float(locked_labels.mean()),
        "locked_primary_auroc": primary_auroc,
        "locked_primary_auprc": primary_auprc,
        **{f"locked_{key}": value for key, value in primary_threshold.items()},
        **{
            f"locked_stable_p90_{key}": value
            for key, value in stable_p90_result.items()
        },
        **confidence,
        "locked_stable_n": len(locked_stable),
        "locked_stable_alarm_rate": float(locked_stable["alarm_f1"].mean()),
        "locked_stable_p90_alarm_rate": float(
            locked_stable["alarm_stable_p90"].mean()
        ),
        "locked_stable_no_new_degradation_n": int(stable_no_new_degradation.sum()),
        "locked_stable_false_positive_rate": stable_fpr,
        "locked_stable_p90_false_positive_rate": stable_p90_fpr,
        "locked_stable_new_degradation_rate": float(
            (locked_stable["outcome_state"] == "new_degradation").mean()
        ),
    }
    (output_dir / "primary_summary.json").write_text(
        json.dumps(primary_summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    pd.DataFrame([primary_summary]).to_csv(
        output_dir / "primary_summary.csv", index=False, encoding="utf-8-sig"
    )

    baseline_rows: list[dict[str, Any]] = []
    for score_name in BASELINE_SCORES:
        development_primary_values = development_primary[score_name].to_numpy(dtype=float)
        threshold = best_f1_threshold(y_development, development_primary_values)
        stable_threshold = quantile_threshold(
            development_stable[score_name].to_numpy(dtype=float),
            args.stable_risk_quantile,
        )
        locked_values = locked_primary[score_name].to_numpy(dtype=float)
        auroc, auprc = binary_scores(locked_labels, locked_values)
        result = threshold_metrics(locked_labels, locked_values, threshold)
        stable_result = threshold_metrics(locked_labels, locked_values, stable_threshold)
        baseline_rows.append(
            {
                "score": score_name,
                "threshold": threshold,
                "locked_auroc": auroc,
                "locked_auprc": auprc,
                **{f"locked_{key}": value for key, value in result.items()},
                **{
                    f"locked_stable_p90_{key}": value
                    for key, value in stable_result.items()
                },
                "locked_stable_alarm_rate": float(
                    (locked_stable[score_name] >= threshold).mean()
                ),
                "locked_stable_p90_alarm_rate": float(
                    (locked_stable[score_name] >= stable_threshold).mean()
                ),
            }
        )
    pd.DataFrame(baseline_rows).to_csv(
        output_dir / "locked_baseline_scores.csv", index=False, encoding="utf-8-sig"
    )

    prediction_columns = [
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
        "risk_probability",
        "alarm_f1",
        "alarm_stable_p90",
        "alarm",
    ]
    locked_changed[prediction_columns].to_csv(
        output_dir / "locked_changed_predictions.csv", index=False, encoding="utf-8-sig"
    )
    locked_stable[prediction_columns].to_csv(
        output_dir / "locked_stable_control.csv", index=False, encoding="utf-8-sig"
    )
    transition_metrics(locked_primary, locked_stable, f1_risk_threshold).to_csv(
        output_dir / "locked_transition_metrics.csv", index=False, encoding="utf-8-sig"
    )
    counts = []
    for split, label, frame in (
        ("calibration", "changed", calibration_changed),
        ("validation", "changed", validation_changed),
        ("development", "changed", development_changed),
        ("locked", "changed", locked_changed),
        ("locked", "stable", locked_stable),
    ):
        counts.extend(state_counts(split, frame.assign(change_label=label)))
    pd.DataFrame(counts).to_csv(output_dir / "outcome_state_counts.csv", index=False, encoding="utf-8-sig")

    artifact = {
        "schema_version": 1,
        "experiment": args.experiment_label,
        "features": FEATURES,
        "selected_model": selected_model,
        "selected_hyperparameters": json_safe(final_search.best_params_),
        "risk_threshold_source": "development changed repeated-nested-OOF max F1",
        "risk_threshold": f1_risk_threshold,
        "stable_p90_threshold": stable_p90_threshold,
        "development_nested_parameter_frequency": json_safe(
            development_parameter_frequency
        ),
        "stable_risk_quantile": args.stable_risk_quantile,
        "drop_threshold": args.drop_threshold,
        "failure_threshold": args.failure_threshold,
        "success_threshold": args.success_threshold,
        "stable_references": final_reference,
        "locked_labels_used_for_selection": False,
        "primary_population": "changed only",
        "primary_endpoint": "new_degradation vs recovery_or_adaptive_success",
    }
    (output_dir / "frozen_detector.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    joblib.dump(
        {**artifact, "estimator": final_search.best_estimator_},
        output_dir / "frozen_detector.joblib",
    )
    plot_locked(locked_changed, locked_stable, output_dir / "locked_changed_stable_scatter.png")

    report_lines = [
        f"# {args.experiment_label}",
        "",
        "## Primary design",
        "",
        "- Detector fitting and model selection: changed questions only",
        "- Stable questions: CDF normalization, p90 threshold, and false-alarm control only",
        "- Primary endpoint: new degradation vs recovery/adaptive success",
        "- Persistent failures and ambiguous normal changed cases are excluded from the primary endpoint",
        "- Locked labels were not used for model or threshold selection",
        "",
        "## Locked result",
        "",
        f"- Selected model: `{selected_model}`",
        f"- Changed primary: n={len(locked_primary)}, degradation={int(locked_labels.sum())}",
        f"- AUROC: {primary_auroc:.3f}",
        f"- AUPRC: {primary_auprc:.3f} (prevalence {float(locked_labels.mean()):.3f})",
        f"- Primary threshold: development changed OOF max-F1 ({f1_risk_threshold:.4f})",
        f"- Precision / recall / F1: {primary_threshold['precision']:.3f} / {primary_threshold['recall']:.3f} / {primary_threshold['f1']:.3f}",
        f"- Stable-control alarm rate at F1 threshold: {float(locked_stable['alarm_f1'].mean()):.3f}",
        f"- Stable no-new-degradation FPR at F1 threshold: {stable_fpr:.3f}",
        f"- Conservative stable-p90 precision / recall / F1: {stable_p90_result['precision']:.3f} / {stable_p90_result['recall']:.3f} / {stable_p90_result['f1']:.3f}",
        f"- Stable no-new-degradation FPR at stable p90: {stable_p90_fpr:.3f}",
        "",
        "The detector consumes only shift and uncertainty at inference time. Gold answers are used only for retrospective evaluation.",
    ]
    (output_dir / "report_ko.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(output_dir / "report_ko.md")


if __name__ == "__main__":
    main()
