import unittest

import numpy as np
from railrl.testing.tf_test_case import TFTestCase

from railrl.exploration_strategies.onehot_sampler import OneHotSampler
from railrl.testing.testing_utils import is_binomial_trial_likely
from railrl.utils.testing import StubPolicy


class TestOneHotSampler(TFTestCase):
    def test_deterministic_onehot_sampled_correct(self):
        policy = StubPolicy(np.array([1.0, 0., 0.]))
        sampler = OneHotSampler()
        action = sampler.get_action(None, None, policy)

        self.assertNpArraysEqual(np.array([1, 0, 0]), action)

    def test_uniform_onehot_sampled_correct(self):
        n = 500
        prob_a = 0.5
        prob_b = 1 - prob_a
        policy = StubPolicy(np.array([prob_a, prob_b]))
        sampler = OneHotSampler()
        num_a, num_b = 0, 0
        a_vector = np.array([1, 0])
        b_vector = np.array([0, 1])
        for _ in range(n):
            action = sampler.get_action(None, None, policy)
            if (action == a_vector).all():
                num_a += 1
            if (action == b_vector).all():
                num_b += 1

        self.assertTrue(is_binomial_trial_likely(n, prob_a, num_a))
        self.assertTrue(is_binomial_trial_likely(n, prob_b, num_b))

    def test_nonuniform_onehot_sampled_correct(self):
        n = 500
        prob_a = 0.2
        prob_b = 1 - prob_a
        policy = StubPolicy(np.array([prob_a, prob_b]))
        sampler = OneHotSampler()
        num_a, num_b = 0, 0
        a_vector = np.array([1, 0])
        b_vector = np.array([0, 1])
        for _ in range(n):
            action = sampler.get_action(None, None, policy)
            if (action == a_vector).all():
                num_a += 1
            if (action == b_vector).all():
                num_b += 1

        self.assertTrue(is_binomial_trial_likely(n, prob_a, num_a))
        self.assertTrue(is_binomial_trial_likely(n, prob_b, num_b))


if __name__ == '__main__':
    unittest.main()