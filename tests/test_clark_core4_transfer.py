from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.evaluate_clark_core4_transfer import (
    RAW_FEATURES,
    add_representations,
    empirical_cdf,
    robust_z,
)


class Core4TransferTest(unittest.TestCase):
    def test_future_normalization_uses_t0_reference(self) -> None:
        calibration = pd.DataFrame(
            {feature: [0.0, 1.0, 2.0, 3.0] for feature in RAW_FEATURES}
        )
        future = pd.DataFrame(
            {feature: [4.0, -1.0] for feature in RAW_FEATURES}
        )
        _, transformed = add_representations(calibration, future)
        for feature in RAW_FEATURES:
            self.assertGreater(transformed.loc[0, f"robust_z__{feature}"], 0.0)
            self.assertLess(transformed.loc[1, f"robust_z__{feature}"], 0.0)
            self.assertGreater(transformed.loc[0, f"ecdf__{feature}"], 0.80)

    def test_ecdf_and_robust_z_are_finite(self) -> None:
        reference = np.asarray([1.0, 1.0, 1.0])
        values = np.asarray([1.0, 2.0])
        self.assertTrue(np.isfinite(empirical_cdf(reference, values)).all())
        self.assertTrue(np.isfinite(robust_z(reference, values)).all())


if __name__ == "__main__":
    unittest.main()
