"""
Tests for the add-system pipeline
> python -m pytest tests/test_pipeline.py
"""

import unittest

import numpy as np

from dysts.analysis import (
    find_lyapunov_exponents,
    kaplan_yorke_dimension,
    lyapunov_spectrum,
    max_lyapunov_exponent,
)
from dysts.flows import Lorenz, Rossler, SprottB
from dysts.pipeline import (
    align_period,
    dominant_period_in_samples,
    flag_suspicious,
    lyapunov_spectrum_variational,
    make_system,
    points_per_oscillation,
)
from dysts.utils import jac_fd


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

    def test_lyapunov_spectrum_lorenz(self):
        """The variational spectrum recovers the known Lorenz exponents and their sum."""
        res = lyapunov_spectrum(Lorenz(), n_runs=2, n_periods=150, full_output=True)
        lyap, se = res["spectrum"], res["standard_error"]
        self.assertEqual(res["jacobian"], "analytic")
        self.assertAlmostEqual(lyap[0], 0.906, delta=max(4 * se[0], 0.04))
        self.assertLess(abs(lyap[1]), 0.02)
        self.assertAlmostEqual(lyap.sum(), -(10 + 1 + 8 / 3), delta=0.01)
        self.assertEqual(len(res["max_exponent_by_run"]), 2)

    def test_lyapunov_spectrum_fallbacks(self):
        """Finite-difference Jacobians and the clone method agree with the analytic case."""
        eq = SprottB()  # published largest exponent 0.210 (Sprott 1994)
        for kw in ({"jacobian": "finite_difference"}, {"method": "clones"}):
            res = lyapunov_spectrum(eq, n_runs=1, n_periods=200, full_output=True, **kw)
            self.assertAlmostEqual(res["spectrum"][0], 0.210, delta=0.03, msg=str(kw))
            self.assertAlmostEqual(res["spectrum"].sum(), -1.0, delta=0.01, msg=str(kw))

    def test_jac_fd(self):
        """Central differences match the analytic Jacobian to high accuracy."""
        eq = Lorenz()
        x = np.array([3.0, -40.0, 25.0])
        ja = np.array(eq.jac(x, 0.0))
        jf = jac_fd(lambda y: np.array(eq.rhs(y, 0.0)), x)
        self.assertTrue(np.allclose(ja, jf, rtol=1e-7, atol=1e-6))

    def test_deprecated_lyapunov_functions(self):
        """Retired estimators warn and return the variational result."""
        eq = Lorenz()
        with self.assertWarns(DeprecationWarning):
            lyap = find_lyapunov_exponents(eq, 100 * 500, pts_per_period=500)
        self.assertAlmostEqual(lyap[0], 0.906, delta=0.1)
        with self.assertWarns(DeprecationWarning):
            lyap = lyapunov_spectrum_variational(eq, eq.ic, 50 * eq.period, eq.period / 5)
        self.assertAlmostEqual(lyap.sum(), -(10 + 1 + 8 / 3), delta=0.05)
        with self.assertWarns(DeprecationWarning):
            self.assertGreater(max_lyapunov_exponent(eq, max_walltime=60), 0.5)

    def test_kaplan_yorke_all_negative(self):
        """A fixed point (all exponents negative) has dimension zero."""
        self.assertEqual(kaplan_yorke_dimension([-0.1, -0.5, -2.0]), 0.0)
        self.assertAlmostEqual(
            kaplan_yorke_dimension([0.9, 0.0, -14.6]), 2 + 0.9 / 14.6
        )

    def test_flags(self):
        entry = {
            "embedding_dimension": 3,
            "lyapunov_spectrum_estimated": [0.9, 0.0, -14.6],
            "maximum_lyapunov_estimated": 0.9,
            "kaplan_yorke_dimension": 2.06,
            "correlation_dimension": 2.0,
        }
        self.assertEqual(flag_suspicious(entry), [])
        bad = dict(
            entry,
            lyapunov_spectrum_estimated=[0.9, -0.5, -14.6],
            kaplan_yorke_dimension=3.5,
        )
        flags = flag_suspicious(bad)
        self.assertTrue(any("near-zero" in f for f in flags))
        self.assertTrue(any("Kaplan-Yorke" in f for f in flags))
        self.assertTrue(
            any("not positive" in f for f in flag_suspicious(
                dict(entry, lyapunov_spectrum_estimated=[0.0, -0.1, -1.0])
            ))
        )
        # conservative systems legitimately have Kaplan-Yorke dimension equal to d
        conservative = dict(entry, hamiltonian=True, kaplan_yorke_dimension=3.0)
        self.assertFalse(
            any("Kaplan-Yorke" in f for f in flag_suspicious(conservative))
        )


class TestDeprecatedAliases(unittest.TestCase):
    def test_aizawa_alias(self):
        """Aizawa still resolves to Langford, warns, and is not listed twice."""
        from dysts import flows
        from dysts.systems import get_attractor_list

        with self.assertWarns(DeprecationWarning):
            from dysts.flows import Aizawa
        self.assertIs(Aizawa, flows.Langford)
        self.assertEqual(Aizawa().name, "Langford")
        names = get_attractor_list()
        self.assertIn("Langford", names)
        self.assertNotIn("Aizawa", names)
        self.assertFalse(hasattr(flows, "NotASystem"))


if __name__ == "__main__":
    unittest.main()
