import unittest

import numpy as np

import rlkit.testing.testing_utils as tu


class TestAreNpArraysEqual(unittest.TestCase):
    def test_equal(self):
        a = np.array([1, 5])
        b = np.array([1.0000000001, 5])
        self.assertTrue(tu.are_np_arrays_equal(a, b))

    def test_not_equal(self):
        a = np.array([1, 5])
        b = np.array([1.0000000001, 5])
        self.assertFalse(tu.are_np_arrays_equal(a, b, threshold=1e-20))


class TestAreDictListsEqual(unittest.TestCase):
    def test_order_does_not_matter(self):
        d1 = [
            {
                "a": 1,
                "b": -1,
            },
            {
                "a": 2,
                "b": -1,
            },
        ]
        d2 = [
            {
                "a": 2,
                "b": -1,
            },
            {
                "a": 1,
                "b": -1,
            },
        ]
        self.assertTrue(tu.are_dict_lists_equal(d1, d2))

    def test_values_matter(self):
        d1 = [
            {
                "a": 1,
                "b": -1,
            },
            {
                "a": 2,
                "b": -1,
            },
        ]
        d2 = [
            {
                "a": 2,
                "b": -1,
            },
            {
                "a": 2,
                "b": -1,
            },
        ]
        self.assertFalse(tu.are_dict_lists_equal(d1, d2))

    def test_keys_matter(self):
        d1 = [
            {
                "a": 1,
                "b": -1,
            },
            {
                "a": 2,
                "b": -1,
            },
        ]
        d2 = [
            {
                "a": 1,
                "b": -1,
            },
            {
                "a": 2,
                "c": -1,
            },
        ]
        self.assertFalse(tu.are_dict_lists_equal(d1, d2))


if __name__ == '__main__':
    unittest.main()
