"""Fit Core4 detectors on CLARK T0 and evaluate frozen future updates.

The script consumes completed CLARK run directories. It never reads question
text or article text: Energy, cluster JS, entropy, volume and timestamp-valid
accuracy are reconstructed from the metric CSV files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.clark_evaluation import best_f1_threshold, threshold_metrics


RAW_FEATURES = ("energy", "cluster_js", "delta_entropy", "delta_volume")
REPRESENTATIONS = ("ecdf", "rank_gaussian", "robust_z")
DISPLAY_NAMES = {
    "l2_logistic": "L2 Logistic",
    "elastic_net": "Elastic Net",
    "quadratic_logistic": "Quadratic Logistic",
    "additive_gam": "Additive GAM",
    "rbf_svm": "RBF-SVM",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}
TUNED_MODELS = (
    "l2_logistic",
    "elastic_net",
    "quadratic_logistic",
    "additive_gam",
    "rbf_svm",
    "extra_trees",
    "hist_gradient_boosting",
)
FIXED_MODELS = ("xgboost", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--future-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*TUNED_MODELS, *FIXED_MODELS),
        default=[*TUNED_MODELS, *FIXED_MODELS],
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
    )
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def load_run(run_dir: Path, split: str) -> pd.DataFrame:
    metrics_path = run_dir / "metrics" / "question_level_metrics.csv"
    shift_path = run_dir / "distribution_shift" / "per_question_distribution_shift.csv"
    if not metrics_path.exists() or not shift_path.exists():
        raise FileNotFoundError(f"Missing CLARK metrics under {run_dir}")

    metrics = pd.read_csv(metrics_path)
    if "embedding_model_role" in metrics:
        metrics = metrics[metrics["embedding_model_role"] == "primary"].copy()
    accuracy = "target_correct_rate" if "target_correct_rate" in metrics else "accuracy"
    wide = metrics[
        ["question_id", "condition", accuracy, "semantic_entropy", "semantic_volume"]
    ].pivot_table(index="question_id", columns="condition", aggfunc="first")
    wide.columns = [f"{metric}__{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()

    required = [
        f"{accuracy}__stale_only",
        f"{accuracy}__current_only",
        "semantic_entropy__stale_only",
        "semantic_entropy__current_only",
        "semantic_volume__stale_only",
        "semantic_volume__current_only",
    ]
    missing = [column for column in required if column not in wide]
    if missing:
        raise ValueError(f"{run_dir} misses {missing}")

    wide["accuracy_stale"] = wide[f"{accuracy}__stale_only"].astype(float)
    wide["accuracy_current"] = wide[f"{accuracy}__current_only"].astype(float)
    wide["accuracy_drop"] = wide["accuracy_stale"] - wide["accuracy_current"]
    wide["delta_entropy"] = (
        wide["semantic_entropy__current_only"]
        - wide["semantic_entropy__stale_only"]
    )
    wide["delta_volume"] = (
        wide["semantic_volume__current_only"]
        - wide["semantic_volume__stale_only"]
    )

    shift = pd.read_csv(shift_path)
    shift = shift[shift["comparison"] == "current_only__vs__stale_only"].copy()
    frame = shift.merge(wide, on="question_id", how="inner")
    frame["persistent_failure"] = (
        (frame["accuracy_stale"] < 0.50) & (frame["accuracy_current"] < 0.50)
    )
    frame["target_new_degradation"] = (
        ~frame["persistent_failure"] & (frame["accuracy_drop"] >= 0.10)
    ).astype(int)
    frame["question_group"] = frame["question_id"].astype(str).str.rsplit("_", n=1).str[-1]
    frame["transition"] = (
        frame["question_id"].astype(str).str.extract(r"clark_(\d{8})_(\d{8})")[1]
    )
    frame["split"] = split
    if frame[list(RAW_FEATURES)].isna().any().any():
        raise ValueError(f"{run_dir} contains missing Core4 values")
    return frame


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    ranks = np.searchsorted(ordered, np.asarray(values, dtype=float), side="right")
    return np.clip((ranks - 0.5) / len(ordered), 1e-4, 1.0 - 1e-4)


def robust_z(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    median = float(np.median(reference))
    q1, q3 = np.quantile(reference, [0.25, 0.75])
    scale = float((q3 - q1) / 1.349)
    if not math.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(reference))
    if not math.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return np.clip((np.asarray(values, dtype=float) - median) / scale, -8.0, 8.0)


def add_representations(
    calibration: pd.DataFrame, future: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = calibration.copy()
    future = future.copy()
    references = {
        feature: calibration[feature].to_numpy(dtype=float) for feature in RAW_FEATURES
    }
    for frame in (calibration, future):
        for feature, reference in references.items():
            cdf = empirical_cdf(reference, frame[feature].to_numpy(dtype=float))
            frame[f"ecdf__{feature}"] = cdf
            frame[f"rank_gaussian__{feature}"] = norm.ppf(cdf)
            frame[f"robust_z__{feature}"] = robust_z(
                reference, frame[feature].to_numpy(dtype=float)
            )
    return calibration, future


def feature_columns(representation: str) -> list[str]:
    return [f"{representation}__{feature}" for feature in RAW_FEATURES]


def safe_splits(labels: np.ndarray, requested: int) -> int:
    counts = np.bincount(labels.astype(int), minlength=2)
    return max(2, min(requested, int(counts.min())))


def tuned_spec(name: str, seed: int) -> tuple[Any, dict[str, list[Any]]]:
    logistic = lambda **kwargs: LogisticRegression(
        class_weight="balanced", max_iter=7000, random_state=seed, **kwargs
    )
    if name == "l2_logistic":
        return Pipeline([("scale", StandardScaler()), ("model", logistic())]), {
            "model__C": [0.1, 1.0, 10.0]
        }
    if name == "elastic_net":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", logistic(penalty="elasticnet", solver="saga")),
            ]
        ), {"model__C": [0.3, 1.0, 3.0], "model__l1_ratio": [0.25, 0.75]}
    if name == "quadratic_logistic":
        return Pipeline(
            [
                ("quadratic", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", logistic(solver="liblinear")),
            ]
        ), {"model__C": [0.01, 0.1, 1.0, 10.0]}
    if name == "additive_gam":
        return Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(
                        degree=2, include_bias=False, extrapolation="linear"
                    ),
                ),
                ("scale", StandardScaler()),
                ("model", logistic(solver="liblinear")),
            ]
        ), {"spline__n_knots": [3, 4], "model__C": [0.1, 1.0, 10.0]}
    if name == "rbf_svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ), {"model__C": [0.5, 2.0, 8.0], "model__gamma": ["scale", 0.5, 2.0]}
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ), {
            "max_depth": [2, 4, None],
            "min_samples_leaf": [2, 5],
            "max_features": ["sqrt", 1.0],
        }
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=180, random_state=seed
        ), {
            "learning_rate": [0.04, 0.10],
            "max_leaf_nodes": [3, 7, 15],
            "l2_regularization": [0.0, 1.0],
            "min_samples_leaf": [5, 10],
        }
    raise ValueError(name)


def fit_search(
    estimator: Any,
    grid: dict[str, list[Any]],
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> GridSearchCV:
    cv = StratifiedKFold(
        n_splits=safe_splits(y, 4), shuffle=True, random_state=seed
    )
    search = GridSearchCV(
        clone(estimator), grid, scoring="average_precision", cv=cv, n_jobs=-1
    )
    search.fit(x, y)
    return search


def tuned_oof(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, Counter[str], Any]:
    estimator, grid = tuned_spec(name, seed)
    cv = RepeatedStratifiedKFold(
        n_splits=safe_splits(y, 5), n_repeats=repeats, random_state=seed
    )
    predictions: list[list[float]] = [[] for _ in range(len(y))]
    counts: Counter[str] = Counter()
    for fold, (train, test) in enumerate(cv.split(x, y)):
        search = fit_search(estimator, grid, x[train], y[train], seed + fold + 1)
        probability = search.predict_proba(x[test])[:, 1]
        for index, value in zip(test, probability):
            predictions[int(index)].append(float(value))
        counts[json.dumps(search.best_params_, sort_keys=True, default=str)] += 1
    final = fit_search(estimator, grid, x, y, seed + 10000).best_estimator_
    return np.asarray([np.mean(values) for values in predictions]), counts, final


def fixed_model(name: str, seed: int, y: np.ndarray) -> Any:
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("Install xgboost to reproduce the XGBoost row") from error
        positive = max(1, int(y.sum()))
        negative = max(1, int(len(y) - positive))
        return XGBClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5.0,
            scale_pos_weight=negative / positive,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=2,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        alpha=0.01,
                        learning_rate_init=0.003,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=40,
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(name)


def fixed_oof(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, Any]:
    cv = RepeatedStratifiedKFold(
        n_splits=safe_splits(y, 5), n_repeats=repeats, random_state=seed
    )
    predictions: list[list[float]] = [[] for _ in range(len(y))]
    for fold, (train, test) in enumerate(cv.split(x, y)):
        model = fixed_model(name, seed + fold + 1, y[train])
        if name == "mlp":
            model.fit(
                x[train],
                y[train],
                model__sample_weight=compute_sample_weight("balanced", y[train]),
            )
        else:
            model.fit(x[train], y[train])
        probability = model.predict_proba(x[test])[:, 1]
        for index, value in zip(test, probability):
            predictions[int(index)].append(float(value))
    final = fixed_model(name, seed + 10000, y)
    if name == "mlp":
        final.fit(x, y, model__sample_weight=compute_sample_weight("balanced", y))
    else:
        final.fit(x, y)
    return np.asarray([np.mean(values) for values in predictions]), final


def score(
    labels: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    result = threshold_metrics(labels, probability, threshold)
    result.update(
        {
            "n": len(labels),
            "positive": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "auroc": float(roc_auc_score(labels, probability)),
            "auprc": float(average_precision_score(labels, probability)),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration, future = add_representations(
        load_run(args.calibration_run.resolve(), "t0"),
        load_run(args.future_run.resolve(), "future"),
    )
    y_t0 = calibration["target_new_degradation"].to_numpy(dtype=int)
    y_future = future["target_new_degradation"].to_numpy(dtype=int)
    selection_rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], Any] = {}

    for model_index, name in enumerate(args.models):
        for representation in args.representations:
            columns = feature_columns(representation)
            x_t0 = calibration[columns].to_numpy(dtype=float)
            seed = args.seed + model_index * 1000
            if name in TUNED_MODELS:
                oof, parameters, model = tuned_oof(
                    name, x_t0, y_t0, args.outer_repeats, seed
                )
                parameter_summary = dict(parameters)
            else:
                oof, model = fixed_oof(name, x_t0, y_t0, args.outer_repeats, seed)
                parameter_summary = {"configuration": "fixed"}
            threshold = best_f1_threshold(y_t0, oof)
            metrics = score(y_t0, oof, threshold)
            fitted[(name, representation)] = model
            selection_rows.append(
                {
                    "model": name,
                    "display_name": DISPLAY_NAMES[name],
                    "representation": representation,
                    **{f"t0_{key}": value for key, value in metrics.items()},
                    "parameters": json.dumps(parameter_summary, sort_keys=True),
                }
            )

    candidates = pd.DataFrame(selection_rows)
    selected = (
        candidates.sort_values(
            ["model", "t0_f1", "t0_auprc", "t0_auroc"],
            ascending=[True, False, False, False],
        )
        .groupby("model", as_index=False)
        .first()
    )
    selected = selected.sort_values(
        ["t0_f1", "t0_auprc", "t0_auroc"], ascending=False
    ).reset_index(drop=True)
    selected["selected_global_t0"] = False
    selected.loc[0, "selected_global_t0"] = True
    selected.to_csv(
        args.output_dir / "t0_selected_per_model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    future_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        columns = feature_columns(str(row.representation))
        model = fitted[(str(row.model), str(row.representation))]
        probability = model.predict_proba(future[columns].to_numpy(dtype=float))[:, 1]
        threshold = float(row.t0_threshold)
        future_rows.append(
            {
                "model": row.model,
                "display_name": row.display_name,
                "representation": row.representation,
                **score(y_future, probability, threshold),
            }
        )
        for transition, part in future.assign(risk=probability).groupby(
            "transition", sort=True
        ):
            labels = part["target_new_degradation"].to_numpy(dtype=int)
            transition_rows.append(
                {
                    "model": row.model,
                    "display_name": row.display_name,
                    "representation": row.representation,
                    "transition": transition,
                    **score(labels, part["risk"].to_numpy(dtype=float), threshold),
                }
            )
    pd.DataFrame(future_rows).to_csv(
        args.output_dir / "frozen_future_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(transition_rows).to_csv(
        args.output_dir / "frozen_transition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
