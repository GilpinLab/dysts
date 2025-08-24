"""
Test the models and regularizer
> python test_generation.py
"""

#!/usr/bin/env python
import os
import random
import sys
import unittest

import numpy as np

import dysts.flows as dfl
import dysts.maps as dmp
from dysts.base import DynMap, DynSys, DynSysDelay
from dysts.flows import Lorenz
from dysts.systems import get_attractor_list

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(WORKING_DIR, "tests", "test_data")

sys.path.insert(1, os.path.join(WORKING_DIR, "dysts"))

NUM_TEST_SYSTEMS = 10


class TestModels(unittest.TestCase):
    """
    Tests integration
    """

    def test_trajectory(self):
        """
        Test generating a trajectory
        """
        model = Lorenz()
        sol = model.make_trajectory(100)
        assert sol is not None, "Generated trajectory is None"
        assert sol.shape == (100, 3), "Generated time series has the wrong shape"  # type: ignore

    def test_trajectory_noise(self):
        """
        Test generating a trajectory with stochasticity
        """
        # check if sdeint is installed
        try:
            import sdeint  # type: ignore
        except ImportError:
            self.skipTest("sdeint is not installed")

        model = Lorenz()
        sol = model.make_trajectory(100, noise=0.01)
        assert sol is not None, "Generated trajectory is None"
        assert sol.shape == (100, 3), "Generated time series has the wrong shape"  # type: ignore

    def test_random_continuous_systems(self):
        continuous_systems = get_attractor_list(sys_class="continuous_no_delay")
        random_systems = random.sample(
            continuous_systems, min(NUM_TEST_SYSTEMS, len(continuous_systems))
        )

        for system_name in random_systems:
            with self.subTest(system=system_name):
                print(f"Testing {system_name}")
                system = getattr(dfl, system_name)()
                self.assertIsInstance(system, DynSys)

                sol = system.make_trajectory(256, return_times=True)
                self.assertIsInstance(sol, tuple)
                self.assertEqual(len(sol), 2)
                self.assertIsInstance(sol[0], np.ndarray)
                self.assertIsInstance(sol[1], np.ndarray)
                self.assertEqual(sol[0].shape[0], 256)
                self.assertEqual(sol[1].shape[0], 256)

    def test_random_delay_systems(self):
        delay_systems = get_attractor_list(sys_class="delay")
        random_systems = random.sample(
            delay_systems, min(NUM_TEST_SYSTEMS, len(delay_systems))
        )

        for system_name in random_systems:
            with self.subTest(system=system_name):
                print(f"Testing {system_name}")
                system = getattr(dfl, system_name)()
                self.assertIsInstance(system, DynSysDelay)

                sol = system.make_trajectory(256, return_times=True)
                self.assertIsInstance(sol, tuple)
                self.assertEqual(len(sol), 2)
                self.assertIsInstance(sol[0], np.ndarray)
                self.assertIsInstance(sol[1], np.ndarray)
                self.assertEqual(sol[0].shape[0], 256)
                self.assertEqual(sol[1].shape[0], 256)

    def test_random_discrete_maps(self):
        discrete_maps = get_attractor_list(sys_class="discrete")
        random_systems = random.sample(
            discrete_maps, min(NUM_TEST_SYSTEMS, len(discrete_maps))
        )

        for system_name in random_systems:
            with self.subTest(system=system_name):
                print(f"Testing {system_name}")
                system = getattr(dmp, system_name)()
                self.assertIsInstance(system, DynMap)

                sol = system.make_trajectory(256)
                self.assertIsInstance(sol, np.ndarray)
                self.assertEqual(sol.shape[0], 256)


class TestJacobian(unittest.TestCase):
    """Perform a grad check to ensure that the Jacobian is implemented correctly"""

    def test_all_jacobians(self):
        eps = 1e-8

        equation_names = get_attractor_list()
        for name in equation_names:
            eq = getattr(dfl, name)()

            # skip if no analytic Jacobian implemented
            if not eq.has_jacobian():
                continue

            jac_analytic = eq.jac(eq.ic, 0)

            d = len(eq.ic)
            jac_fd = np.zeros((d, d))
            for i in range(d):
                ei = np.zeros(d)
                ei[i] = 1
                jac_fd[:, i] = (
                    np.array(eq.rhs(eq.ic + eps * ei, 0))
                    - np.array(eq.rhs(eq.ic - eps * ei, 0))
                ) / (2 * eps)

            self.assertTrue(
                np.allclose(jac_analytic, jac_fd, atol=1e-5),
                f"Jacobian for {name} is incorrect",
            )


if __name__ == "__main__":
    unittest.main()
