"""Small model family used to calibrate the CLARK risk detector."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVC


def model_specs(seed: int) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "l2_logistic": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "quadratic_logistic": (
            Pipeline(
                [
                    ("quadratic", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "additive_gam": (
            Pipeline(
                [
                    (
                        "spline",
                        SplineTransformer(
                            degree=2,
                            include_bias=False,
                            extrapolation="linear",
                        ),
                    ),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"spline__n_knots": [3, 4], "model__C": [0.1, 1.0, 10.0]},
        ),
        "gaussian_process": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        GaussianProcessClassifier(
                            kernel=ConstantKernel(
                                1.0, constant_value_bounds="fixed"
                            )
                            * RBF(1.0, length_scale_bounds="fixed"),
                            optimizer=None,
                            max_iter_predict=200,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "model__kernel": [
                    ConstantKernel(1.0, constant_value_bounds="fixed")
                    * RBF(value, length_scale_bounds="fixed")
                    for value in (0.3, 0.7, 1.5)
                ]
            },
        ),
        "rbf_svm": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            probability=True,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.1, 0.5, 1.0, 2.0],
            },
        ),
    }


def safe_splits(labels: np.ndarray, requested: int) -> int:
    counts = np.bincount(labels.astype(int), minlength=2)
    return max(2, min(requested, int(counts.min())))


def fit_tuned(
    estimator: Pipeline,
    grid: dict[str, list[Any]],
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> GridSearchCV:
    inner = StratifiedKFold(
        n_splits=safe_splits(y, 4), shuffle=True, random_state=seed
    )
    search = GridSearchCV(
        clone(estimator),
        grid,
        scoring="average_precision",
        cv=inner,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    return search


def repeated_nested_oof(
    x: np.ndarray,
    y: np.ndarray,
    *,
    estimator: Pipeline,
    grid: dict[str, list[Any]],
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, Counter[str]]:
    outer = RepeatedStratifiedKFold(
        n_splits=safe_splits(y, 5), n_repeats=repeats, random_state=seed
    )
    predictions: list[list[float]] = [[] for _ in range(len(y))]
    parameter_counts: Counter[str] = Counter()
    for fold, (train_index, test_index) in enumerate(outer.split(x, y)):
        search = fit_tuned(
            estimator,
            grid,
            x[train_index],
            y[train_index],
            seed=seed + fold + 1,
        )
        probability = search.predict_proba(x[test_index])[:, 1]
        for index, value in zip(test_index, probability):
            predictions[int(index)].append(float(value))
        parameter_counts[
            json.dumps(search.best_params_, sort_keys=True, default=str)
        ] += 1
    if any(not values for values in predictions):
        raise RuntimeError("Some questions did not receive out-of-fold predictions.")
    return (
        np.asarray([float(np.mean(values)) for values in predictions]),
        parameter_counts,
    )
