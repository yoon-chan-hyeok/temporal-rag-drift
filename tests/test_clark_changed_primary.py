from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.analyze_clark_changed_primary import (
    primary_subset,
    quantile_threshold,
    select_validation_model,
)
from scripts.prepare_clark_changed_primary import TRANSITIONS
from scripts.run_clark_changed_primary_luna import active_specs


class ClarkChangedPrimaryDesignTest(unittest.TestCase):
    def test_primary_endpoint_excludes_persistent_and_normal(self) -> None:
        frame = pd.DataFrame(
            {
                "question_id": ["a", "b", "c", "d"],
                "outcome_state": [
                    "new_degradation",
                    "recovery_or_adaptive_success",
                    "persistent_failure",
                    "normal",
                ],
            }
        )

        selected = primary_subset(frame)

        self.assertEqual(selected["question_id"].tolist(), ["a", "b"])
        self.assertEqual(selected["target_new_degradation"].tolist(), [1, 0])

    def test_stable_quantile_threshold_is_frozen_from_reference(self) -> None:
        threshold = quantile_threshold(np.asarray([0.1, 0.2, 0.3, 0.8, 0.9]), 0.80)

        self.assertEqual(threshold, 0.9)

    def test_temporal_roles_are_calibration_validation_then_locked(self) -> None:
        self.assertEqual(
            [transition["role"] for transition in TRANSITIONS],
            ["calibration", "validation", "locked", "locked", "locked"],
        )

    def test_model_selection_prioritizes_frozen_validation_f1(self) -> None:
        candidates = pd.DataFrame(
            {
                "model": ["l2_logistic", "additive_gam"],
                "validation_f1": [0.70, 0.60],
                "validation_auprc": [0.45, 0.90],
                "validation_auroc": [0.70, 0.90],
            }
        )

        selected = select_validation_model(candidates, margin=0.02)

        self.assertEqual(selected, "l2_logistic")

    def test_changed_only_runner_omits_stable_and_balances_gpu_groups(self) -> None:
        specs = active_specs(changed_only=True)

        self.assertEqual(
            set(specs),
            {"calibration_changed", "validation_changed", "locked_changed"},
        )
        self.assertEqual(specs["calibration_changed"]["gpu_group"], 0)
        self.assertEqual(specs["validation_changed"]["gpu_group"], 1)
        self.assertEqual(specs["locked_changed"]["gpu_group"], 1)
        self.assertTrue(
            all(
                transition["time_x"] < transition["time_y"]
                for transition in TRANSITIONS
            )
        )


if __name__ == "__main__":
    unittest.main()
