"""
Test the models and regularizer
> python test_generation.py
"""

#!/usr/bin/env python
import os
import sys
import unittest

import numpy as np

from dysts.systems import get_attractor_list
from dysts.utils import (
    cartesian_to_polar,
    make_surrogate,
    polar_to_cartesian,
    signif,
    standardize_ts,
)

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(WORKING_DIR, "dysts"))


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.r = 5
        self.th = np.pi / 4
        self.x = 3.5355339059
        self.y = 3.5355339059
        self.x_arr = np.array([1, 2, 3])
        self.y_arr = np.array([4, 5, 6])
        self.d = {"a": 1}
        self.a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_polar_to_cartesian(self):
        x, y = polar_to_cartesian(self.r, self.th)
        self.assertAlmostEqual(x, self.x)
        self.assertAlmostEqual(y, self.y)

    def test_cartesian_to_polar(self):
        r, th = cartesian_to_polar(self.x, self.y)
        self.assertAlmostEqual(r, self.r)
        self.assertAlmostEqual(th, self.th)

    def test_signif(self):
        res = signif(1.2345678, figs=4)
        self.assertEqual(res, 1.235)

    def test_standardize_ts(self):
        res = standardize_ts(self.a)
        expected = np.array(
            [
                [-1.22474487, -1.22474487, -1.22474487],
                [0.0, 0.0, 0.0],
                [1.22474487, 1.22474487, 1.22474487],
            ]
        )
        self.assertTrue(np.allclose(res, expected))

    # def test_integrate_dyn(self):
    #     def f(y, t):
    #         return -y
    #     ic = [1]
    #     tvals = np.linspace(0, 1, 10)
    #     print((integrate_dyn(f, ic, tvals), np.exp(-tvals)), flush=True)
    #     self.assertTrue(np.allclose(integrate_dyn(f, ic, tvals), np.exp(-tvals)))

    # def test_freq_from_fft(self):
    #     sig = np.sin(2*np.pi*5*np.linspace(0, 1, 100))
    #     self.assertAlmostEqual(freq_from_fft(sig, fs=100), 5)

    # def test_resample_timepoints(self):
    #     def model(y, t):
    #         return -y
    #     ic = [1]
    #     tpts = np.linspace(0, 1, 10)
    #     self.assertTrue(np.array_equal(resample_timepoints(model, ic, tpts, pts_per_period=100), np.linspace(0, 10, 1000)))

    def test_make_surrogate(self):
        data = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
        surr_data = make_surrogate(data, "rp")
        self.assertAlmostEqual(len(surr_data), len(data))

    def test_attractor_list(self):
        """
        Test fetching list of attractors
        """
        assert len(get_attractor_list()) > 130


if __name__ == "__main__":
    unittest.main()
