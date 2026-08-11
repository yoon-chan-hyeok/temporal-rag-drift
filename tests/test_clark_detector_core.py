from __future__ import annotations

import unittest

import pandas as pd

from scripts.clark_detector_models import model_specs
from scripts.clark_score_features import add_scores, stable_references


class ClarkDetectorCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = pd.DataFrame(
            {
                "swd": [0.1, 0.2, 0.3],
                "mmd_rbf": [0.1, 0.2, 0.3],
                "energy": [0.1, 0.2, 0.3],
                "cluster_js": [0.1, 0.2, 0.3],
                "centroid_gap": [0.1, 0.2, 0.3],
                "delta_entropy": [-0.1, 0.0, 0.1],
                "delta_volume": [-1.0, 0.0, 1.0],
            }
        )

    def test_percentile_axes_are_bounded(self) -> None:
        references = stable_references(self.reference)
        scored = add_scores(self.reference, references)

        self.assertTrue(scored["shift_score"].between(0.0, 1.0).all())
        self.assertTrue(scored["uncertainty_score"].between(0.0, 1.0).all())
        self.assertLess(scored.loc[0, "shift_score"], scored.loc[2, "shift_score"])

    def test_calibration_family_contains_frozen_model(self) -> None:
        specs = model_specs(seed=42)

        self.assertIn("quadratic_logistic", specs)
        estimator, grid = specs["quadratic_logistic"]
        self.assertIn("quadratic", estimator.named_steps)
        self.assertIn("model__C", grid)


if __name__ == "__main__":
    unittest.main()
