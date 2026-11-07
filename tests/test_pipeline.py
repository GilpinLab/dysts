"""
Tests for the add-system pipeline
> python -m pytest tests/test_pipeline.py
"""

import unittest

import numpy as np

from dysts.flows import Lorenz, Rossler
from dysts.pipeline import (
    align_period,
    dominant_period_in_samples,
    flag_suspicious,
    make_system,
    points_per_oscillation,
)


class TestPipeline(unittest.TestCase):
    def test_points_per_oscillation_sine(self):
        """A pure sinusoid with a known period is recovered to within a few percent."""
        t = np.arange(20000)
        for period in (37.0, 100.0, 250.0):
            x = np.sin(2 * np.pi * t / period) + 0.05 * np.random.randn(len(t))
            est = points_per_oscillation(x)
            self.assertLess(abs(est / period - 1), 0.05, f"{est} vs {period}")

    def test_reference_is_self_consistent(self):
        """Aligning a database system to itself leaves its period unchanged."""
        ref = Rossler()
        eq = Rossler()
        period0 = eq.period
        align_period(eq, reference=ref, max_iters=2)
        self.assertLess(abs(eq.period / period0 - 1), 0.1)

    def test_alignment_recovers_rescaled_period(self):
        """A deliberately wrong period is corrected by the alignment step."""
        ref = Rossler()
        target = dominant_period_in_samples(ref)
        lorenz = Lorenz()
        eq = make_system(
            Lorenz,
            parameters=lorenz.params,
            initial_conditions=lorenz.ic,
            dt=lorenz.dt,
            period=lorenz.period * 4,
        )
        align_period(eq, reference=ref)
        achieved = dominant_period_in_samples(eq)
        self.assertLess(abs(achieved / target - 1), 0.25)

    def test_flags(self):
        entry = {
            "embedding_dimension": 3,
            "lyapunov_spectrum_estimated": [0.9, 0.0, -14.6],
            "maximum_lyapunov_estimated": 0.85,
            "kaplan_yorke_dimension": 2.06,
            "correlation_dimension": 2.0,
        }
        self.assertEqual(flag_suspicious(entry), [])
        bad = dict(entry, maximum_lyapunov_estimated=0.1, kaplan_yorke_dimension=3.5)
        flags = flag_suspicious(bad)
        self.assertTrue(any("disagree" in f for f in flags))
        self.assertTrue(any("Kaplan-Yorke" in f for f in flags))


if __name__ == "__main__":
    unittest.main()
