"""
Test the metrics
"""

#!/usr/bin/env python
import os
import sys
import unittest

import numpy as np

from dysts.metrics import (
    coefficient_of_variation,
    compute_metrics,
    estimate_kl_divergence,
    mae,
    mape,
    marre,
    mse,
    ope,
    pearson,
    r2_score,
    rmsle,
    smape,
    spearman,
)

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(WORKING_DIR, "dysts"))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 1.9, 3.2, 4.1, 5.1])
        self.y_train = np.array([0.9, 1.8, 2.9, 3.8, 4.9])

    # # TODO: check the ground truth value for this test
    # def test_mase(self):
    #     result = mase(self.y_true, self.y_pred, self.y_train)
    #     self.assertAlmostEqual(result, 0.10434, places=2)

    def test_mse(self):
        result = mse(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.01599, places=2)

    def test_mae(self):
        result = mae(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.1199, places=2)

    def test_coefficient_of_variation(self):
        result = coefficient_of_variation(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 3.26598632, places=2)

    def test_marre(self):
        result = marre(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 2.99999, places=2)

    def test_ope(self):
        result = ope(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.099999, places=2)

    def test_rmsle(self):
        result = rmsle(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.04934, places=2)

    def test_r2_score(self):
        result = r2_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.992, places=2)

    def test_mape(self):
        result = mape(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 5.2333, places=2)

    def test_smape(self):
        result = smape(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 5.110 / 2, places=2)

    def test_spearman(self):
        result = spearman(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.9999, places=2)  # type: ignore

    def test_pearson(self):
        result = pearson(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.9999, places=2)  # type: ignore


class TestEstimateKLDivergence(unittest.TestCase):
    def setUp(self):
        # Set up some example orbits
        self.true_orbit = np.random.randn(100, 2)
        self.generated_orbit = np.random.randn(100, 2)

    def test_kl_divergence_shape(self):
        # Test if the function works with different shapes
        true_orbit_1d = np.random.randn(100)
        generated_orbit_1d = np.random.randn(100)
        kl_div = estimate_kl_divergence(true_orbit_1d, generated_orbit_1d)
        self.assertIsInstance(kl_div, float)

    def test_kl_divergence_value(self):
        # Test if the function returns a float value
        kl_div = estimate_kl_divergence(self.true_orbit, self.generated_orbit)
        self.assertIsInstance(kl_div, float)

    def test_kl_divergence_same_orbit(self):
        # Test if the KL divergence is close to zero for the same orbits
        kl_div = estimate_kl_divergence(self.true_orbit, self.true_orbit)
        self.assertAlmostEqual(kl_div, 0, places=1)

    def test_kl_divergence_different_orbits(self):
        # Test if the KL divergence is positive for different orbits
        kl_div = estimate_kl_divergence(self.true_orbit, self.generated_orbit)
        self.assertGreater(kl_div, 0)

    def test_kl_divergence_sigma_scale(self):
        # Test if the function works with a specified sigma_scale
        kl_div = estimate_kl_divergence(
            self.true_orbit, self.generated_orbit, sigma_scale=0.5
        )
        self.assertIsInstance(kl_div, float)

    def test_kl_divergence_auto_sigma_scale(self):
        # Test if the function works with none sigma_scale
        kl_div = estimate_kl_divergence(
            self.true_orbit, self.generated_orbit, sigma_scale=None
        )
        self.assertIsInstance(kl_div, float)

    def test_compute_metrics_batched_kl_divergence(self):
        # Test if the function works with a batched dimension
        y_true = np.random.randn(10, 100, 2)
        y_pred = np.random.randn(10, 100, 2)
        metrics = compute_metrics(
            y_true, y_pred, include=["kl_divergence"], batch_axis=0
        )
        self.assertEqual(set(metrics.keys()), set(["kl_divergence"]))


class TestComputeMetrics(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.y_true = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_pred = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])

    def test_compute_metrics_include(self):
        # Test including only specific metrics
        include = ["mse", "mae"]
        metrics = compute_metrics(self.y_true, self.y_pred, include=include)
        self.assertEqual(set(metrics.keys()), set(include))

    def test_compute_metrics_invalid_include(self):
        # Test with invalid metric name
        with self.assertRaises(AssertionError):
            compute_metrics(self.y_true, self.y_pred, include=["invalid_metric"])

    def test_compute_metrics_shape_mismatch(self):
        # Test with non-broadcastable shapes
        y_pred_wrong = np.array([[1, 2], [3, 4]])
        with self.assertRaises(AssertionError):
            compute_metrics(self.y_true, y_pred_wrong)

    def test_compute_metrics_batched(self):
        # Test with a batched dimension
        y_true = np.random.randn(10, 1000, 3)
        y_pred = np.random.randn(10, 1000, 3)
        include = [
            "mse",
            "mae",
            "smape",
            "r2_score",
            "hellinger_distance",
        ]
        metrics = compute_metrics(y_true, y_pred, include=include, batch_axis=0)
        avg_metrics = {key: 0.0 for key in metrics.keys()}
        for i in range(y_true.shape[0]):
            yt = y_true[i]
            yp = y_pred[i]
            submetrics = compute_metrics(yt, yp, include=include)
            for metric_name, metric_value in submetrics.items():
                avg_metrics[metric_name] += metric_value / y_true.shape[0]

        for metric_name, metric_value in metrics.items():
            self.assertAlmostEqual(metric_value, metrics[metric_name], places=2)


if __name__ == "__main__":
    unittest.main()
