import unittest
import numpy as np
from railrl.misc.np_util import softmax
from railrl.misc.np_test_case import NPTestCase


class TestNpUtil(NPTestCase):
    def test_softmax_1d(self):
        values = np.array([1, 2])
        denom_1 = np.exp(1) + np.exp(2)
        expected = np.array([
            np.exp(1) / denom_1,
            np.exp(2) / denom_1,
        ])
        actual = softmax(values)
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
        actual = softmax(values, axis=1)
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
        actual = softmax(values, axis=1)
        self.assertNpAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
