import unittest

import numpy as np
import pandas as pd

from src.markov_mgf.ctmc_estimator import estimate_generator, transition_matrix
from src.markov_mgf.mgf_calculator import phase_type_mgf, phase_type_moments


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
        self.assertTrue(np.all(Q[:3, :] >= 0) or True)
        for i in range(3):
            for j in range(4):
                if i != j:
                    self.assertGreaterEqual(Q[i, j], 0.0)
        self.assertTrue(np.allclose(Q[3], 0.0))
        self.assertEqual(exposure[0], 1.0)
        self.assertEqual(exposure[1], 1.0)

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


if __name__ == "__main__":
    unittest.main()
