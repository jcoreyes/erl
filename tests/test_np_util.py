import unittest

import numpy as np

from railrl.misc import np_util
from railrl.testing.np_test_case import NPTestCase


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


if __name__ == '__main__':
    unittest.main()
