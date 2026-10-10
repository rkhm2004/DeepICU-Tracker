import unittest

import numpy as np
import pandas as pd

from src.clinical_validation import (
    _binary_metrics,
    _kaplan_meier,
    _phase_type_survival,
    _safe_auc,
)


class TestClinicalValidationHelpers(unittest.TestCase):
    def test_kaplan_meier_accounts_for_right_censoring(self):
        # One event at hour 1, one censored at hour 2, one event at hour 3.
        survival = _kaplan_meier(
            times=[1.0, 2.0, 3.0],
            events=[1, 0, 1],
            grid=[0.0, 1.0, 2.0, 3.0],
        )
        np.testing.assert_allclose(survival, [1.0, 2 / 3, 2 / 3, 0.0])

    def test_phase_type_survival_matches_exponential_case(self):
        result = _phase_type_survival(0, np.array([[-0.2]]), 5.0)
        self.assertAlmostEqual(result, float(np.exp(-1.0)), places=7)

    def test_critical_state_has_zero_survival(self):
        self.assertEqual(_phase_type_survival(3, np.array([[-0.2]]), 0.0), 0.0)

    def test_auc_is_undefined_for_single_class(self):
        self.assertIsNone(_safe_auc([0, 0, 0], [0.1, 0.2, 0.3]))

    def test_binary_metrics_report_events_and_brier(self):
        metrics = _binary_metrics([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.6])
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["events"], 2)
        self.assertEqual(metrics["non_events"], 2)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertIn("brier_score", metrics)


if __name__ == "__main__":
    unittest.main()
