import unittest

import numpy as np
import pandas as pd

from src.bayesian_net.train_infer import (
    apply_discretization,
    fit_bayesian_network,
    fit_discretization,
    infer_risk_probabilities,
)


class TestBayesianNetwork(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "heart_rate": [60, 70, 80, 90, 100, 110, 120, 130],
                "sbp": [130, 125, 120, 115, 110, 105, 100, 95],
                "wbc": [5, 6, 7, 8, 9, 10, 11, 12],
                "state": [0, 0, 1, 1, 2, 2, 3, 3],
            }
        )
        self.cutoffs = fit_discretization(self.train)

    def test_discretization_returns_three_valid_levels(self):
        bins = apply_discretization(self.train, self.cutoffs)
        for feature in ["heart_rate", "sbp", "wbc"]:
            self.assertTrue(set(bins[feature].unique()).issubset({0, 1, 2}))

    def test_cpt_is_normalized(self):
        cpt, _ = fit_bayesian_network(self.train, self.cutoffs)
        self.assertEqual(cpt.shape, (3, 3, 3, 4))
        self.assertTrue(np.allclose(cpt.sum(axis=-1), 1.0))
        self.assertTrue(np.all(cpt > 0.0))

    def test_inference_probabilities_are_normalized(self):
        cpt, _ = fit_bayesian_network(self.train, self.cutoffs)
        probs, _ = infer_risk_probabilities(self.train, self.cutoffs, cpt)
        self.assertEqual(probs.shape, (len(self.train), 4))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
