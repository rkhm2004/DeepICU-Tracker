import unittest

import numpy as np
import pandas as pd

from src.bayesian_net.train_infer import (
    apply_discretization,
    compute_classification_metrics,
    fit_bayesian_network,
    fit_discretization,
    infer_risk_probabilities,
    merge_bn_inputs,
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

    def test_invalid_laplace_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_bayesian_network(self.train, self.cutoffs, alpha=0.0)

    def test_nonfinite_inference_feature_is_rejected(self):
        cpt, _ = fit_bayesian_network(self.train, self.cutoffs)
        broken = self.train.copy()
        broken.loc[0, "heart_rate"] = np.inf
        with self.assertRaises(ValueError):
            infer_risk_probabilities(broken, self.cutoffs, cpt)

    def test_classification_metrics_include_all_states(self):
        y_true = np.array([0, 1, 2, 3, 3])
        y_pred = np.array([0, 1, 1, 2, 3])
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (4, 4))
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertIn("Critical", metrics["per_class_recall"])
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertGreaterEqual(metrics["balanced_accuracy"], 0.0)
        self.assertIn("VAE-derived", metrics["metric_target"])

    def test_input_alignment_uses_keys_not_row_order(self):
        latent = pd.DataFrame({
            "subject_id": [1, 1, 2, 2],
            "stay_id": [11, 11, 22, 22],
            "hour": [0, 1, 0, 1],
            "split": ["train"] * 4,
            "state": [0, 1, 2, 3],
            "state_name": ["Low Risk", "Medium Risk", "High Risk", "Critical"],
        })
        index = latent[["subject_id", "stay_id", "hour", "split"]].copy()
        index["heart_rate"] = [60.0, 70.0, 80.0, 90.0]
        index["sbp"] = [130.0, 120.0, 110.0, 100.0]
        index["wbc"] = [5.0, 6.0, 7.0, 8.0]
        merged = merge_bn_inputs(latent, index.iloc[::-1].reset_index(drop=True))
        self.assertEqual(merged["heart_rate"].tolist(), [60.0, 70.0, 80.0, 90.0])

    def test_input_alignment_rejects_duplicate_keys(self):
        latent = pd.DataFrame({
            "subject_id": [1, 1],
            "stay_id": [11, 11],
            "hour": [0, 0],
            "split": ["train", "train"],
            "state": [0, 0],
            "state_name": ["Low Risk", "Low Risk"],
        })
        index = pd.DataFrame({
            "subject_id": [1],
            "stay_id": [11],
            "hour": [0],
            "split": ["train"],
            "heart_rate": [70.0],
            "sbp": [120.0],
            "wbc": [8.0],
        })
        with self.assertRaises(ValueError):
            merge_bn_inputs(latent, index)


if __name__ == "__main__":
    unittest.main()
