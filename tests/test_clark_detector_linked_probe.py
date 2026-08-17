from __future__ import annotations

import unittest

import pandas as pd

from scripts.analyze_clark_detector_linked_probe import add_recovery_labels
from scripts.prepare_clark_historical_failure_probe import CONDITIONS


class DetectorLinkedProbeTest(unittest.TestCase):
    def make_row(self, accuracies: tuple[float, float, float, float]) -> dict[str, object]:
        row: dict[str, object] = {
            "historical_accuracy_stale": 1.0,
            "accuracy__p1_natural": 0.5,
            "p2_support_injected": True,
            "p3_support_moved_to_rank1": True,
        }
        for condition, accuracy in zip(CONDITIONS[1:], accuracies, strict=True):
            row[f"accuracy__{condition}"] = accuracy
        return row

    def test_recovery_uses_pre_update_accuracy_target(self) -> None:
        result = add_recovery_labels(
            pd.DataFrame([self.make_row((0.9375, 1.0, 1.0, 1.0))]),
            degradation_delta=0.10,
            recovery_gain=0.10,
            recovery_tolerance=0.0625,
        )
        self.assertTrue(bool(result.loc[0, "degradation_reproduced"]))
        self.assertEqual(
            result.loc[0, "earliest_recovery_stage"], "p2_support_presence"
        )
        self.assertEqual(
            result.loc[0, "mechanism_candidate"], "retrieval_coverage_failure"
        )

    def test_p5_is_used_only_after_earlier_stages_fail(self) -> None:
        result = add_recovery_labels(
            pd.DataFrame([self.make_row((0.50, 0.55, 0.60, 1.0))]),
            degradation_delta=0.10,
            recovery_gain=0.10,
            recovery_tolerance=0.0625,
        )
        self.assertEqual(result.loc[0, "earliest_recovery_stage"], "p5_fact_card")
        self.assertEqual(
            result.loc[0, "mechanism_candidate"],
            "evidence_utilization_or_answer_realization_failure",
        )


if __name__ == "__main__":
    unittest.main()
