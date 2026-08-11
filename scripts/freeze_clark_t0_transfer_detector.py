"""Freeze a quadratic shift-uncertainty detector using CLARK T0 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_clark_changed_primary import FEATURES, add_outcomes, load_module, load_one, primary_subset
from scripts.clark_evaluation import best_f1_threshold, binary_scores, threshold_metrics


DEFAULT_CALIBRATION = ROOT / "outputs" / "runs" / "clark_changed_primary_luna" / "calibration_changed"
DEFAULT_OUTPUT = ROOT / "outputs" / "runs" / "clark_t0_temporal_transfer_luna" / "detector_t0"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    helper = load_module(
        "clark_t0_score_helper", ROOT / "scripts" / "clark_score_features.py"
    )
    ml = load_module(
        "clark_t0_ml_helper", ROOT / "scripts" / "clark_detector_models.py"
    )
    outcome_args = SimpleNamespace(
        drop_threshold=0.10,
        failure_threshold=0.50,
        success_threshold=0.50,
    )
    raw = load_one(helper, str(args.calibration_run), "calibration_t0", "changed")
    references = helper.stable_references(raw)
    calibration = primary_subset(add_outcomes(helper.add_scores(raw, references), outcome_args))
    x = calibration[FEATURES].to_numpy(dtype=float)
    y = calibration["target_new_degradation"].to_numpy(dtype=int)
    estimator, grid = ml.model_specs(args.seed)["quadratic_logistic"]
    oof, parameter_frequency = ml.repeated_nested_oof(
        x,
        y,
        estimator=estimator,
        grid=grid,
        repeats=args.outer_repeats,
        seed=args.seed,
    )
    threshold = best_f1_threshold(y, oof)
    search = ml.fit_tuned(estimator, grid, x, y, seed=args.seed + 1000)
    auroc, auprc = binary_scores(y, oof)
    operating = threshold_metrics(y, oof, threshold)
    artifact = {
        "schema_version": 1,
        "design": "T0-only frozen temporal-transfer detector",
        "features": FEATURES,
        "normalization": "T0 changed empirical CDF",
        "references": references,
        "selected_model": "quadratic_logistic",
        "family_selection": "fixed before confirmatory sampling from prior exploratory CLARK study",
        "selected_hyperparameters": search.best_params_,
        "threshold": float(threshold),
        "calibration_n": len(y),
        "calibration_positive": int(y.sum()),
        "calibration_oof_auroc": auroc,
        "calibration_oof_auprc": auprc,
        "calibration_oof_operating": operating,
        "parameter_frequency": parameter_frequency,
        "future_data_used_for_fit_or_threshold": False,
    }
    json_path = args.output_dir / "frozen_detector_t0.json"
    json_path.write_text(json.dumps(artifact, indent=2, default=float), encoding="utf-8")
    joblib_path = args.output_dir / "frozen_detector_t0.joblib"
    joblib.dump({**artifact, "estimator": search.best_estimator_}, joblib_path)
    lock = {
        "detector_json": str(json_path.relative_to(ROOT)),
        "detector_json_sha256": file_sha256(json_path),
        "detector_joblib": str(joblib_path.relative_to(ROOT)),
        "detector_joblib_sha256": file_sha256(joblib_path),
    }
    (args.output_dir / "detector_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print(json.dumps({**artifact, **lock}, indent=2, default=float))


if __name__ == "__main__":
    main()
