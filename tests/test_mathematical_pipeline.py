import unittest

import numpy as np
import pandas as pd

from src.data_pipeline.mimic_parser import (
    assign_stay_splits,
    build_observation_index,
    causal_impute_features,
)
from src.evaluation import _first_critical_remaining_hours
from src.markov_mgf.ctmc_estimator import estimate_generator, transition_matrix
from src.markov_mgf.mgf_calculator import phase_type_mgf, phase_type_moments



class TestCausalPreprocessing(unittest.TestCase):
    def test_hour_index_never_extends_past_outtime(self):
        stays = pd.DataFrame({
            "subject_id": [1, 2, 3],
            "stay_id": [11, 22, 33],
            "intime": pd.to_datetime([
                "2025-01-01 00:00:00",
                "2025-01-01 00:00:00",
                "2025-01-01 00:00:00",
            ]),
            "outtime": pd.to_datetime([
                "2025-01-01 02:30:00",
                "2025-01-01 02:00:00",
                "2025-01-04 12:00:00",
            ]),
        })
        index = build_observation_index(stays, [11, 22, 33], window_hours=48)
        counts = index.groupby("stay_id")["hour"].count().to_dict()
        self.assertEqual(counts, {11: 3, 22: 2, 33: 48})
        duration = stays.set_index("stay_id")
        for row in index.itertuples(index=False):
            hours = (
                duration.loc[row.stay_id, "outtime"]
                - duration.loc[row.stay_id, "intime"]
            ).total_seconds() / 3600.0
            self.assertLess(row.hour, hours)

    def test_missing_earlier_value_does_not_use_future_measurement(self):
        frame = pd.DataFrame({
            "stay_id": [1, 1, 2, 2],
            "hour": [0, 1, 0, 1],
            "heart_rate": [70.0, 80.0, np.nan, 180.0],
            "sbp": [110.0, 120.0, np.nan, 150.0],
            "wbc": [7.0, 8.0, np.nan, 18.0],
        })
        train_mask = frame["stay_id"] == 1
        imputed, medians = causal_impute_features(frame, train_mask)
        # The missing first hour of stay 2 uses training medians, not hour-1's
        # future measurement (180 bpm / 150 mmHg / WBC 18).
        self.assertEqual(imputed.loc[2, "heart_rate"], medians["heart_rate"])
        self.assertNotEqual(imputed.loc[2, "heart_rate"], 180.0)
        self.assertEqual(imputed.loc[2, "heart_rate_was_missing"], 1)
        self.assertTrue(np.isfinite(imputed[["heart_rate", "sbp", "wbc"]]).all().all())

class TestCTMC(unittest.TestCase):
    def test_patient_boundaries_are_not_crossed(self):
        df = pd.DataFrame(
            {
                "stay_id": [1, 1, 2, 2],
                "hour": [0, 1, 0, 1],
                "state": [0, 1, 1, 2],
            }
        )

        Q, counts, exposure = estimate_generator(df)

        self.assertEqual(counts[0, 1], 1)
        self.assertEqual(counts[1, 2], 1)
        self.assertEqual(counts[1, 0], 0)

        self.assertTrue(np.allclose(Q.sum(axis=1), 0.0))
        for i in range(3):
            for j in range(4):
                if i != j:
                    self.assertGreaterEqual(Q[i, j], 0.0)
        self.assertTrue(np.allclose(Q[3], 0.0))
        self.assertEqual(exposure[0], 1.0)
        self.assertEqual(exposure[1], 1.0)

    def test_invalid_time_step_is_rejected(self):
        df = pd.DataFrame({
            "stay_id": [1, 1],
            "hour": [0, 1],
            "state": [0, 1],
        })
        with self.assertRaises(ValueError):
            estimate_generator(df, dt_hours=0)

    def test_duplicate_stay_hour_is_rejected(self):
        df = pd.DataFrame({
            "stay_id": [1, 1],
            "hour": [0, 0],
            "state": [0, 1],
        })
        with self.assertRaises(ValueError):
            estimate_generator(df)

    def test_transition_matrix_is_stochastic(self):
        Q = np.array(
            [
                [-0.15, 0.10, 0.00, 0.05],
                [0.00, -0.30, 0.20, 0.10],
                [0.00, 0.00, -0.50, 0.50],
                [0.00, 0.00, 0.00, 0.00],
            ]
        )
        P = transition_matrix(Q, 2.0)

        self.assertTrue(np.all(P >= -1e-12))
        self.assertTrue(np.allclose(P.sum(axis=1), 1.0, atol=1e-10))
        with self.assertRaises(ValueError):
            transition_matrix(Q, -1.0)


class TestPhaseType(unittest.TestCase):
    def setUp(self):
        self.Q = np.array(
            [
                [-0.15, 0.10, 0.00, 0.05],
                [0.00, -0.30, 0.20, 0.10],
                [0.00, 0.00, -0.50, 0.50],
                [0.00, 0.00, 0.00, 0.00],
            ]
        )
        self.T = self.Q[:3, :3]

    def test_mgf_at_zero(self):
        for state in range(3):
            self.assertAlmostEqual(
                phase_type_mgf(0.0, state, self.T),
                1.0,
                places=8,
            )

    def test_moments_are_positive(self):
        means, variances, stds = phase_type_moments(self.T)
        self.assertTrue(np.all(means > 0))
        self.assertTrue(np.all(variances >= 0))
        self.assertTrue(np.all(stds >= 0))

    def test_nonabsorbing_transient_generator_is_rejected(self):
        from src.markov_mgf.mgf_calculator import validate_transient_generator
        invalid = np.array([[-1.0, 0.0], [0.0, 0.0]])
        with self.assertRaises(ValueError):
            validate_transient_generator(invalid)

    def test_negative_off_diagonal_rate_is_rejected(self):
        from src.markov_mgf.mgf_calculator import validate_transient_generator
        invalid = np.array([[-1.0, -0.2], [0.0, -1.0]])
        with self.assertRaises(ValueError):
            validate_transient_generator(invalid)

    def test_mgf_rejects_values_outside_its_convergence_domain(self):
        with self.assertRaises(ValueError):
            phase_type_mgf(1e6, 0, self.T)



class TestDataSplitting(unittest.TestCase):
    def test_first_critical_remaining_time_excludes_post_critical(self):
        df = pd.DataFrame({"hour": [0, 1, 2, 3], "state": [0, 3, 2, 1]})
        remaining = _first_critical_remaining_hours(df)

        self.assertEqual(remaining.iloc[0], 1.0)
        self.assertEqual(remaining.iloc[1], 0.0)
        self.assertTrue(np.isnan(remaining.iloc[2]))
        self.assertTrue(np.isnan(remaining.iloc[3]))


    def test_empirical_transition_matrix_includes_self_transitions_and_stops_at_critical(self):
        from src.evaluation import _empirical_one_step_transition_matrix

        df = pd.DataFrame(
            {
                "stay_id": [1, 1, 1, 1, 2, 2],
                "hour": [0, 1, 2, 3, 0, 1],
                "state": [0, 0, 3, 1, 1, 2],
            }
        )
        probabilities, counts = _empirical_one_step_transition_matrix(df)

        self.assertEqual(counts[0, 0], 1)
        self.assertEqual(counts[0, 3], 1)
        self.assertEqual(counts[1, 2], 1)
        self.assertEqual(counts[3].sum(), 0)
        self.assertTrue(np.allclose(probabilities[0], [0.5, 0.0, 0.0, 0.5]))

    def test_stay_split_is_reproducible_and_exclusive(self):
        stay_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        first = assign_stay_splits(stay_ids, seed=42)
        second = assign_stay_splits(stay_ids, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(stay_ids))
        self.assertEqual(set(first.values()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
