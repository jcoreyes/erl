import unittest

import numpy as np

from rlkit.testing.np_test_case import NPTestCase
from rlkit.torch import pytorch_util as ptu


class TestBatchDiag(NPTestCase):
    def test_batch_diag_one(self):
        """
        Check y = x^T diag(d) x
        """
        diags = np.array([
            [2, 1],
        ])
        expected = np.array([
            [
                [2, 0],
                [0, 1],
            ]
        ])

        diag_var = ptu.from_numpy(diags).float()
        result_var = ptu.batch_diag(diag_var)
        result = ptu.get_numpy(result_var)
        self.assertNpAlmostEqual(expected, result)

    def test_batch_diag_multiple(self):
        """
        Check y = x^T diag(d) x
        """
        diags = np.array([
            [2, 1],
            [3, 4],
        ])
        expected = np.array([
            [
                [2, 0],
                [0, 1],
            ],
            [
                [3, 0],
                [0, 4],
            ]
        ])

        diag_var = ptu.from_numpy(diags).float()
        result_var = ptu.batch_diag(diag_var)
        result = ptu.get_numpy(result_var)
        self.assertNpAlmostEqual(expected, result)


class TestBatchSquareVector(NPTestCase):
    def test_values_correct_single_diag(self):
        """
        Check y = x^T diag(d) x
        """
        x = np.array([
            [2, 7],
        ])
        M = np.array([
            [
                [2, 0],
                [0, 1],
            ]
        ])
        expected = np.array([
            [57]  # 2^2 * 2 + 7^2 * 1 = 8 + 49 = 57
        ])

        x_var = ptu.from_numpy(x).float()
        M_var = ptu.from_numpy(M).float()
        result_var = ptu.batch_square_vector(vector=x_var, M=M_var)
        result = ptu.get_numpy(result_var)

        self.assertNpAlmostEqual(expected, result)

    def test_values_correct_batches_diag(self):
        """
        Check y = x^T diag(d) x
        batch-wise
        """
        x = np.array([
            [1, 1],
            [2, 1],
        ])
        M = np.array([
            [
                [3, 0],
                [0, -1],
            ],
            [
                [1, 0],
                [0, 1],
            ]
        ])

        expected = np.array([
            [2],  # 1^2 * 3 + 1^1 * (-1) = 2
            [5],  # 2^2 * 1 + 1^1 * (1) = 5
        ])
        x_var = ptu.from_numpy(x).float()
        M_var = ptu.from_numpy(M).float()
        result_var = ptu.batch_square_vector(vector=x_var, M=M_var)
        result = ptu.get_numpy(result_var)

        self.assertNpAlmostEqual(expected, result)

    def test_values_correct_single_full_matrix(self):
        """
        Check y = x^T diag(d) x
        """
        x = np.array([
            [2, 7],
        ])
        M = np.array([
            [
                [2, -1],
                [2, 1],
            ]
        ])
        expected = np.array([
            [71]  # 2^2 * 2 + 7^2 * 1 + 2*7*(2-1) = 8 + 49 + 14 = 71
        ])

        x_var = ptu.from_numpy(x).float()
        M_var = ptu.from_numpy(M).float()
        result_var = ptu.batch_square_vector(vector=x_var, M=M_var)
        result = ptu.get_numpy(result_var)

        self.assertNpAlmostEqual(expected, result)

    def test_values_correct_batch_full_matrix(self):
        """
        Check y = x^T diag(d) x
        """
        x = np.array([
            [2, 7],
            [2, 7],
        ])
        M = np.array([
            [
                [2, -1],
                [2, 1],
            ],
            [
                [0, -.1],
                [.2, .1],
            ]
        ])
        expected = np.array([
            [71],  # 2^2 * 2 + 7^2 * 1 + 2*7*(2-1) = 8 + 49 + 14 = 71
            [6.3],  # .2^2 * 0 + .7^2 * 1 + .2*.7*(2-1) = 0 + 4.9 + 1.4 = 6.3
        ])

        x_var = ptu.from_numpy(x).float()
        M_var = ptu.from_numpy(M).float()
        result_var = ptu.batch_square_vector(vector=x_var, M=M_var)
        result = ptu.get_numpy(result_var)

        self.assertNpAlmostEqual(expected, result)


if __name__ == '__main__':
    unittest.main()