import unittest

import numpy as np

from rlkit.testing.np_test_case import NPTestCase
from rlkit.torch import pytorch_util as ptu
from rlkit.torch import modules


class TestHuberLoss(NPTestCase):
    def test_huber_loss_delta_1(self):
        criterion = modules.HuberLoss(1)

        x = np.array([
            [-0.1],
        ])
        x_hat = np.array([
            [0],
        ])
        expected_loss = np.array([
            0.5 * 0.1 * 0.1,
        ])

        x_var = ptu.Variable(ptu.from_numpy(x).float())
        x_hat_var = ptu.Variable(ptu.from_numpy(x_hat).float())
        result_var = criterion(x_var, x_hat_var)
        result = ptu.get_numpy(result_var)
        self.assertNpAlmostEqual(expected_loss, result)

        x = np.array([
            [4],
        ])
        x_hat = np.array([
            [6],
        ])
        expected_loss = np.array([
            2 - 0.5,
        ])

        x_var = ptu.Variable(ptu.from_numpy(x).float())
        x_hat_var = ptu.Variable(ptu.from_numpy(x_hat).float())
        result_var = criterion(x_var, x_hat_var)
        result = ptu.get_numpy(result_var)
        self.assertNpAlmostEqual(expected_loss, result)

    def test_huber_loss_delta_3(self):
        criterion = modules.HuberLoss(3)

        x = np.array([
            [0],
        ])
        x_hat = np.array([
            [5],
        ])
        expected_loss = np.array([
            3 * (5 - 3/2),
        ])

        x_var = ptu.Variable(ptu.from_numpy(x).float())
        x_hat_var = ptu.Variable(ptu.from_numpy(x_hat).float())
        result_var = criterion(x_var, x_hat_var)
        result = ptu.get_numpy(result_var)
        self.assertNpAlmostEqual(expected_loss, result)

        x = np.array([
            [4],
        ])
        x_hat = np.array([
            [6],
        ])
        expected_loss = np.array([
            0.5 * 2 * 2,
        ])

        x_var = ptu.Variable(ptu.from_numpy(x).float())
        x_hat_var = ptu.Variable(ptu.from_numpy(x_hat).float())
        result_var = criterion(x_var, x_hat_var)
        result = ptu.get_numpy(result_var)
        self.assertNpAlmostEqual(expected_loss, result)


class TestBatchSquareVector(NPTestCase):
    def test_batch_square_diagonal_module(self):
        x = np.array([
            [2, 7],
        ])
        diag_vals = np.array([
            [2, 1],
        ])
        expected = np.array([
            [57]  # 2^2 * 2 + 7^2 * 1 = 8 + 49 = 57
        ])

        x_var = ptu.Variable(ptu.from_numpy(x).float())
        diag_var = ptu.Variable(ptu.from_numpy(diag_vals).float())
        net = modules.BatchSquareDiagonal(2)
        result_var = net(vector=x_var, diag_values=diag_var)
        result = ptu.get_numpy(result_var)

        self.assertNpAlmostEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
