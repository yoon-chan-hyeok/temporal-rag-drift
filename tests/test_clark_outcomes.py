from __future__ import annotations

import unittest

import pandas as pd

from scripts.clark_evaluation import assign_outcome


class ClarkOutcomeTest(unittest.TestCase):
    def test_priority_and_four_outcomes(self) -> None:
        frame = pd.DataFrame(
            {
                "accuracy_stale": [0.40, 0.80, 0.20, 0.80, 0.80],
                "accuracy_current": [0.20, 0.60, 0.60, 0.80, 0.80],
                "change_label": [
                    "changed",
                    "stable",
                    "changed",
                    "changed",
                    "stable",
                ],
            }
        )

        states = assign_outcome(
            frame,
            drop_threshold=0.10,
            failure_threshold=0.50,
            success_threshold=0.50,
        )

        self.assertEqual(
            states.tolist(),
            [
                "persistent_failure",
                "new_degradation",
                "recovery_or_adaptive_success",
                "recovery_or_adaptive_success",
                "normal",
            ],
        )

    def test_persistent_failure_precedes_small_recovery(self) -> None:
        frame = pd.DataFrame(
            {
                "accuracy_stale": [0.00],
                "accuracy_current": [0.20],
                "change_label": ["changed"],
            }
        )

        state = assign_outcome(
            frame,
            drop_threshold=0.10,
            failure_threshold=0.50,
            success_threshold=0.50,
        ).iloc[0]

        self.assertEqual(state, "persistent_failure")


if __name__ == "__main__":
    unittest.main()
