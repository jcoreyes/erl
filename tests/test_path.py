import unittest
import numpy as np

from rlkit.data_management.path_builder import PathBuilder
from rlkit.testing.np_test_case import NPTestCase

class TestPath(NPTestCase):

    def test_add_and_get_all(self):
        path = PathBuilder()
        path.add_all(
            action=np.array([1, 2, 3]),
            obs=-np.array([1, 2, 3]),
        )
        path.add_all(
            action=np.array([10, 2, 3]),
            obs=-np.array([10, 2, 3]),
        )
        result = path.get_all_stacked()
        self.assertNpArraysEqual(
            result['action'],
            np.array([
                [1, 2, 3],
                [10, 2, 3],
            ])
        )
        self.assertNpArraysEqual(
            result['obs'],
            -np.array([
                [1, 2, 3],
                [10, 2, 3],
            ])
        )

    def test_path_length(self):
        path = PathBuilder()
        for _ in range(10):
            path.add_all(
                action=np.array([1, 2, 3]),
                obs=-np.array([1, 2, 3]),
            )
        self.assertEqual(len(path), 10)


if __name__ == '__main__':
    unittest.main()