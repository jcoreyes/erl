import unittest

import numpy as np

from rlkit.misc import hyperparameter as hp
from rlkit.testing.testing_utils import is_binomial_trial_likely, are_dict_lists_equal


class TestHyperparameters(unittest.TestCase):
    def test_log_float_param(self):
        param = hp.LogFloatParam("variable", 1e-5, 1e-1)
        n = 10000
        num_success = 0
        threshold = 1e-3
        for _ in range(n):
            if param.generate() > threshold:
                num_success += 1
        p = 0.5
        self.assertTrue(is_binomial_trial_likely(n, p, num_success))

    def test_linear_float_param(self):
        param = hp.LinearFloatParam("variable", -10, 10)
        n = 10000
        num_success = 0
        threshold = 0
        for _ in range(n):
            if param.generate() > threshold:
                num_success += 1
        p = 0.5
        self.assertTrue(is_binomial_trial_likely(n, p, num_success))


class TestHyperparameterSweeper(unittest.TestCase):
    def test_sweep_hyperparameters(self):
        sweeper = hp.RandomHyperparameterSweeper([
            hp.LinearFloatParam("v1", -10, 10),
            hp.LogFloatParam("v2", 1e-5, 1e-1),
        ])
        n = 100
        num_successes = np.zeros((2, 2))
        threshold_v1 = 0
        threshold_v2 = 1e-3

        def update_success(v1, v2):
            success_v1 = int(v1 > threshold_v1)
            success_v2 = int(v2 > threshold_v2)
            num_successes[success_v1, success_v2] += 1

        sweeper.sweep_hyperparameters(update_success, n)
        p = 0.25
        for i in range(2):
            for j in range(2):
                self.assertTrue(
                    is_binomial_trial_likely(n, p, num_successes[i, j]))


class TestDeterministicHyperparameterSweeper(unittest.TestCase):
    def test_deterministic_sweeper_basic(self):
        sweeper = hp.DeterministicHyperparameterSweeper(
            {
                "a": [1, 2],
                "b": [-1, -2],
            },
            default_parameters={
                "c": 3,
            },
        )
        dicts = list(sweeper.iterate_hyperparameters())
        expected_dicts = [
            {
                "a": 1,
                "b": -1,
                "c": 3,
            },
            {
                "a": 2,
                "b": -1,
                "c": 3,
            },
            {
                "a": 1,
                "b": -2,
                "c": 3,
            },
            {
                "a": 2,
                "b": -2,
                "c": 3,
            },
        ]
        self.assertTrue(are_dict_lists_equal(dicts, expected_dicts),
                        "Expected: {0}\nActual: {1}".format(str(expected_dicts),
                                                            str(dicts)))

    def test_deterministic_sweeper_basic_many_types(self):
        sweeper = hp.DeterministicHyperparameterSweeper(
            {
                "a": [1, 2],
                "b": [False, True],
            },
            default_parameters={
                "c": 'a',
            },
        )
        dicts = list(sweeper.iterate_hyperparameters())
        expected_dicts = [
            {
                "a": 1,
                "b": False,
                "c": 'a',
            },
            {
                "a": 2,
                "b": False,
                "c": 'a',
            },
            {
                "a": 1,
                "b": True,
                "c": 'a',
            },
            {
                "a": 2,
                "b": True,
                "c": 'a',
            },
        ]
        self.assertTrue(are_dict_lists_equal(dicts, expected_dicts),
                        "Expected: {0}\nActual: {1}".format(str(expected_dicts),
                                                            str(dicts)))


if __name__ == '__main__':
    unittest.main()
