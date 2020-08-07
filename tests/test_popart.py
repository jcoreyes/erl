import unittest

import numpy as np
from rlkit.testing.np_test_case import NPTestCase
from rlkit.torch.popart import popart


class TestPopArt(NPTestCase):

    def test_normalize_target(self):
        target = np.array([1, 2, 3, 4])
        params = popart.create_popart_params(1)
        new_params = popart.update(params, target)

        mean = target.mean()
        stddev = target.std()
        expected = (target - mean) / stddev
        normalized_target = popart.normalize_target(target, new_params)

        self.assertNpArraysEqual(normalized_target, expected)

    def test_min_stddev(self):
        target = np.array([1, 2, 3, 4])
        params = popart.create_popart_params(1, min_stddev=100)
        new_params = popart.update(params, target)

        mean = target.mean()
        stddev = 100
        expected = (target - mean) / stddev
        normalized_target = popart.normalize_target(target, new_params)

        self.assertNpArraysEqual(normalized_target, expected)

    def test_normalize_target_two_updates(self):
        beta = 0.5
        params = popart.create_popart_params(1, beta=beta)
        target1 = np.array([1, 2, 3, 4])
        target2 = np.array([5, 6, 7, 8])
        params = popart.update(params, target1)
        params = popart.update(params, target2)

        # target1 will effective have a weight 0.5^2 whereas target2 will have
        # a weight of 0.5
        mean = target1.mean() * 1/3 + target2.mean() * 2/3
        second_moment = (target1**2).mean() * 1/3 + (target2**2).mean() * 2/3
        stddev = np.sqrt(second_moment - mean**2)

        self.assertNpArraysEqual(
            popart.normalize_target(target1, params),
            (target1 - mean) / stddev,
        )
        self.assertNpArraysEqual(
            popart.normalize_target(target2, params),
            (target2 - mean) / stddev,
        )

    def test_compute_prediction_preserves_output(self):
        target_orig = np.array([4, 6])
        params = popart.create_popart_params(1, beta=0.5)
        raw_y_hat = np.array(1)
        y_orig = popart.compute_prediction(raw_y_hat, params)
        normalized_target_orig = popart.normalize_target(target_orig, params)

        sample1 = np.array([1, 2, 3])
        params = popart.update(params, sample1)
        y1 = popart.compute_prediction(raw_y_hat, params)
        normalized_target1 = popart.normalize_target(target_orig, params)
        self.assertNpArraysEqual(y_orig, y1)
        self.assertNpArraysNotAlmostEqual(
            normalized_target_orig,
            normalized_target1
        )

        sample2 = np.array([4, 5, 6])
        params = popart.update(params, sample2)
        y2 = popart.compute_prediction(raw_y_hat, params)
        normalized_target2 = popart.normalize_target(target_orig, params)
        self.assertNpArraysEqual(y_orig, y2)
        self.assertNpArraysNotAlmostEqual(
            normalized_target_orig,
            normalized_target2
        )
        self.assertNpArraysNotAlmostEqual(
            normalized_target1,
            normalized_target2,
        )


if __name__ == '__main__':
    unittest.main()
