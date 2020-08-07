import unittest
import torch
import numpy as np
from torch.distributions import Normal
from rlkit.testing.np_test_case import NPTestCase
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.distributions import TanhNormal

class TestNormalDistribution(NPTestCase):
    """
    Mostly just to test that Normal works as expected
    """
    def test_log_prob_type(self):
        normal = Normal(0, 1)
        x = ptu.from_numpy(np.array([0]))
        log_prob = normal.log_prob(x)
        self.assertIsInstance(log_prob, torch.autograd.Variable)

    def test_log_prob_value(self):
        normal = Normal(0, 1)
        x = ptu.from_numpy(np.array([0]))
        log_prob = normal.log_prob(x)

        log_prob_np = ptu.get_numpy(log_prob)
        log_prob_expected = np.log(np.array([1 / np.sqrt(2 * np.pi)]))
        self.assertNpArraysEqual(
            log_prob_expected,
            log_prob_np,
        )

    def test_log_prob_value_two(self):
        normal = Normal(0, 1)
        x = ptu.from_numpy(np.array([1]))
        log_prob = normal.log_prob(x)

        log_prob_np = ptu.get_numpy(log_prob)
        log_prob_expected = np.log(np.array([1 / np.sqrt(2 * np.pi)])) - 0.5
        self.assertNpArraysEqual(
            log_prob_expected,
            log_prob_np,
        )

    def test_log_prob_gradient(self):
        """
        d/d mu log f_X(x) = - 2 (mu - x)
        d/d sigma log f_X(x) = 1/sigma^3 - 1/sigma
        :return:
        """
        mean_var = ptu.from_numpy(np.array([0]))
        std_var = ptu.from_numpy(np.array([0.25]))
        normal = Normal(mean_var, std_var)
        x = ptu.from_numpy(np.array([1]))
        log_prob = normal.log_prob(x)

        gradient = ptu.from_numpy(np.array([1]))

        log_prob.backward(gradient)

        self.assertNpArraysEqual(
            ptu.get_numpy(mean_var.grad),
            np.array([16]),
        )
        self.assertNpArraysEqual(
            ptu.get_numpy(std_var.grad),
            np.array([4**3 - 4]),
        )

class TestTanhNormalDistribution(NPTestCase):
    def test_log_prob_type(self):
        tanh_normal = TanhNormal(0, 1)
        x = ptu.from_numpy(np.array([0]))
        log_prob = tanh_normal.log_prob(x)
        self.assertIsInstance(log_prob, torch.autograd.Variable)

    def test_log_prob_value(self):
        tanh_normal = TanhNormal(0, 1)
        z = np.array([1])
        x_np = np.tanh(z)
        x = ptu.from_numpy(x_np)
        log_prob = tanh_normal.log_prob(x)

        log_prob_np = ptu.get_numpy(log_prob)
        log_prob_expected = (
            np.log(np.array([1 / np.sqrt(2 * np.pi)])) - 0.5  # from Normal
            - np.log(1 - x_np**2)
        )
        self.assertNpArraysEqual(
            log_prob_expected,
            log_prob_np,
        )

    def test_log_prob_value_give_pre_tanh_value(self):
        tanh_normal = TanhNormal(0, 1)
        z_np = np.array([1])
        x_np = np.tanh(z_np)
        z = ptu.from_numpy(z_np)
        x = ptu.from_numpy(x_np)
        log_prob = tanh_normal.log_prob(x, pre_tanh_value=z)

        log_prob_np = ptu.get_numpy(log_prob)
        log_prob_expected = (
            np.log(np.array([1 / np.sqrt(2 * np.pi)])) - 0.5  # from Normal
            - np.log(1 - x_np**2)
        )
        self.assertNpArraysEqual(
            log_prob_expected,
            log_prob_np,
        )

    def test_log_prob_gradient(self):
        """
        Same thing. Tanh term drops out since tanh has no params
        d/d mu log f_X(x) = - 2 (mu - x)
        d/d sigma log f_X(x) = 1/sigma^3 - 1/sigma
        :return:
        """
        mean_var = ptu.from_numpy(np.array([0]), requires_grad=True)
        std_var = ptu.from_numpy(np.array([0.25]), requires_grad=True)
        tanh_normal = TanhNormal(mean_var, std_var)
        z = ptu.from_numpy(np.array([1]))
        x = torch.tanh(z)
        log_prob = tanh_normal.log_prob(x, pre_tanh_value=z)

        gradient = ptu.from_numpy(np.array([1]))

        log_prob.backward(gradient)

        self.assertNpArraysEqual(
            ptu.get_numpy(mean_var.grad),
            np.array([16]),
        )
        self.assertNpArraysEqual(
            ptu.get_numpy(std_var.grad),
            np.array([4**3 - 4]),
        )

if __name__ == '__main__':
    unittest.main()
