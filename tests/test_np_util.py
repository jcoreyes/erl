import unittest

import numpy as np

from rlkit.misc import np_util
from rlkit.testing.np_test_case import NPTestCase


class TestNpUtil(NPTestCase):
    def test_softmax_1d(self):
        values = np.array([1, 2])
        denom_1 = np.exp(1) + np.exp(2)
        expected = np.array([
            np.exp(1) / denom_1,
            np.exp(2) / denom_1,
        ])
        actual = np_util.softmax(values)
        self.assertNpAlmostEqual(actual, expected)

    def test_softmax_2d(self):
        values = np.array([
            [
                1, 2,
            ],
            [
                2, 3,
            ],
        ])
        denom_1 = np.exp(1) + np.exp(2)
        denom_2 = np.exp(2) + np.exp(3)
        expected = np.array([
            [
                np.exp(1) / denom_1,
                np.exp(2) / denom_1,
            ],
            [
                np.exp(2) / denom_2,
                np.exp(3) / denom_2,
            ],
        ])
        actual = np_util.softmax(values, axis=1)
        self.assertNpAlmostEqual(actual, expected)

    def test_softmax_3d(self):
        values = np.arange(8).reshape(2, 2, 2)
        # Pairs: 0-2, 1-3, 4-6, 5-7
        denom_02 = np.exp(0) + np.exp(2)
        denom_13 = np.exp(1) + np.exp(3)
        denom_46 = np.exp(4) + np.exp(6)
        denom_57 = np.exp(5) + np.exp(7)
        expected1 = np.array([
            [
                np.exp(0) / denom_02,
                np.exp(1) / denom_13,
            ],
            [
                np.exp(2) / denom_02,
                np.exp(3) / denom_13,
            ],
        ])
        expected2 = np.array([
            [
                np.exp(4) / denom_46,
                np.exp(5) / denom_57,
            ],
            [
                np.exp(6) / denom_46,
                np.exp(7) / denom_57,
                ],
        ])
        expected = np.array([expected1, expected2])
        actual = np_util.softmax(values, axis=1)
        self.assertNpAlmostEqual(actual, expected)

    def test_subsequences(self):
        M = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])
        start_indices = [0, 1, 2]
        length = 2
        subsequences = np_util.subsequences(M, start_indices, length)
        expected = np.array([
            [
                [0, 1],
                [2, 3],
            ],
            [
                [2, 3],
                [4, 5],
            ],
            [
                [4, 5],
                [6, 7],
            ],
        ])
        self.assertNpEqual(subsequences, expected)

    def test_subsequences_out_of_order(self):
        M = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])
        start_indices = [1, 1, 0]
        length = 2
        subsequences = np_util.subsequences(M, start_indices, length)
        expected = np.array([
            [
                [2, 3],
                [4, 5],
            ],
            [
                [2, 3],
                [4, 5],
            ],
            [
                [0, 1],
                [2, 3],
            ],
        ])
        self.assertNpEqual(subsequences, expected)

    def test_subsequences_start_offset(self):
        M = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])
        start_indices = [0, 1]
        length = 2
        subsequences = np_util.subsequences(M, start_indices, length,
                                            start_offset=1)
        expected = np.array([
            [
                [2, 3],
                [4, 5],
            ],
            [
                [4, 5],
                [6, 7],
            ],
        ])
        self.assertNpEqual(subsequences, expected)

    def test_assign_subsequence_all(self):
        M = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])
        start_indices = [0, 2]
        length = 2
        new_values = np.array([
            [
                [-1, -2],
                [-3, -4],
            ],
            [
                [-5, -6],
                [-7, -8],
            ],
        ])
        np_util.assign_subsequences(M, new_values, start_indices, length)
        expected = np.array([
            [-1, -2],
            [-3, -4],
            [-5, -6],
            [-7, -8],
        ])
        self.assertNpEqual(M, expected)

    def test_assign_subsequence_gap(self):
        M = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
        ])
        start_indices = [0, 3]
        length = 2
        new_values = np.array([
            [
                [-1, -2],
                [-3, -4],
            ],
            [
                [-5, -6],
                [-7, -8],
            ],
        ])
        np_util.assign_subsequences(M, new_values, start_indices, length)
        expected = np.array([
            [-1, -2],
            [-3, -4],
            [4, 5],
            [-5, -6],
            [-7, -8],
        ])
        self.assertNpEqual(M, expected)

    def test_assign_subsequence_overlap(self):
        M = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])
        start_indices = [0, 1]
        length = 2
        new_values = np.array([
            [
                [-1, -2],
                [-3, -4],
            ],
            [
                [-5, -6],
                [-7, -8],
            ],
        ])
        np_util.assign_subsequences(M, new_values, start_indices, length)
        expected = np.array([
            [-1, -2],
            [-5, -6],
            [-7, -8],
            [6, 7],
        ])
        self.assertNpEqual(M, expected)

    def test_assign_subsequence_overlap_order_flipped(self):
        M = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])
        start_indices = [1, 0]
        length = 2
        new_values = np.array([
            [
                [-5, -6],
                [-7, -8],
            ],
            [
                [-1, -2],
                [-3, -4],
            ],
        ])
        np_util.assign_subsequences(M, new_values, start_indices, length,
                                    start_offset=0)
        expected = np.array([
            [-1, -2],
            [-3, -4],
            [-7, -8],
            [6, 7],
        ])
        self.assertNpEqual(M, expected)

    def test_assign_subsequence_overlap_offset(self):
        M = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])
        start_indices = [0, 1]
        length = 2
        new_values = np.array([
            [
                [-1, -2],
                [-3, -4],
            ],
            [
                [-5, -6],
                [-7, -8],
            ],
        ])
        np_util.assign_subsequences(M, new_values, start_indices, length,
                                    start_offset=1)
        expected = np.array([
            [0, 1],
            [-1, -2],
            [-5, -6],
            [-7, -8],
        ])
        self.assertNpEqual(M, expected)

    def test_assign_subsequence_completely_overwrite(self):
        M = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])
        start_indices = [0, 0]
        length = 2
        new_values = np.array([
            [
                [-1, -2],
                [-3, -4],
            ],
            [
                [-5, -6],
                [-7, -8],
            ],
        ])
        np_util.assign_subsequences(M, new_values, start_indices, length)
        expected = np.array([
            [-5, -6],
            [-7, -8],
            [4, 5],
            [6, 7],
        ])
        self.assertNpEqual(M, expected)

    def test_assign_subsequence_mismatch_lengths_throws_error(self):
        M = np.zeros((4, 2))
        start_indices = [0, 1]
        length = 2
        # new_values first dimension doesn't match start_indices
        new_values = np.zeros((3, 2, 2))
        with self.assertRaises(AssertionError):
            np_util.assign_subsequences(M, new_values, start_indices, length,
                                        start_offset=1)

        new_values = np.zeros((2, 2, 2))
        # now new_values is right, but length is wrong
        length = 1
        with self.assertRaises(AssertionError):
            np_util.assign_subsequences(M, new_values, start_indices, length,
                                        start_offset=1)

        length = 2
        # now length is right, but new_values and M dimensions don't match
        M = np.zeros((4, 3))
        with self.assertRaises(AssertionError):
            np_util.assign_subsequences(M, new_values, start_indices, length,
                                        start_offset=1)

        M = np.zeros((4, 2))
        # now M is right, but the index will be out of bounds
        start_indices = [0, 3]
        with self.assertRaises(AssertionError):
            np_util.assign_subsequences(M, new_values, start_indices, length,
                                        start_offset=1)

        start_indices = [-1, 1]
        with self.assertRaises(AssertionError):
            np_util.assign_subsequences(M, new_values, start_indices, length,
                                        start_offset=1)

        start_indices = [0, 1]
        # Now everything is right
        np_util.assign_subsequences(M, new_values, start_indices, length,
                                    start_offset=1)

    def test_batch_discounted_cumsum(self):
        values = np.array([
            [1, 1, 1],
            [2, 0, 1],
            [2, 1, 0],
        ])
        discount = 0.5
        expected = np.array([
            [1.75, 1.5, 1],
            [2.25, 0.5, 1],
            [2.5, 1, 0],
        ])
        actual = np_util.batch_discounted_cumsum(values, discount)
        self.assertNpEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
