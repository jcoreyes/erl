import unittest
import numpy as np
import tensorflow as tf

from railrl.misc.testing_utils import (
    are_np_arrays_equal,
    are_np_array_iterables_equal,
)


class TFTestCase(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.sess = tf.get_default_session() or tf.Session()
        self.sess_context = self.sess.as_default()
        self.sess_context.__enter__()

    def tearDown(self):
        self.sess_context.__exit__(None, None, None)
        self.sess.close()

    def assertNpEqual(self, np_arr1, np_arr2, msg="Numpy arrays not equal."):
        self.assertTrue(are_np_arrays_equal(np_arr1, np_arr2), msg)

    def assertNpAlmostEqual(
            self,
            np_arr1,
            np_arr2,
            msg="Numpy arrays not equal.",
            threshold=1e-5,
    ):
        self.assertTrue(
            are_np_arrays_equal(np_arr1, np_arr2, threshold=threshold),
            msg
        )

    def assertNpNotEqual(self, np_arr1, np_arr2, msg="Numpy arrays equal"):
        self.assertFalse(are_np_arrays_equal(np_arr1, np_arr2), msg)

    def assertNpArraysEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg="Numpy array lists are not equal.",
    ):
        self.assertTrue(
            are_np_array_iterables_equal(
                np_arrays1,
                np_arrays2,
            ),
            msg
        )

    # TODO(vpong): see why such a high threshold is needed
    def assertNpArraysAlmostEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg="Numpy array lists are not almost equal.",
            threshold=1e-4,
    ):
        self.assertTrue(
            are_np_array_iterables_equal(
                np_arrays1,
                np_arrays2,
                threshold=threshold,
            ),
            msg
        )

    def assertNpArraysNotEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg="Numpy array lists are equal."
    ):
        self.assertFalse(are_np_array_iterables_equal(np_arrays1, np_arrays2), msg)

    def assertNpArraysNotAlmostEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg="Numpy array lists are equal.",
            threshold=1e-4,
    ):
        self.assertFalse(
            are_np_array_iterables_equal(
                np_arrays1,
                np_arrays2,
                threshold=threshold,
            ),
            msg
        )

    def assertParamsEqual(self, network1, network2):
        self.assertNpArraysEqual(
            network1.get_param_values(),
            network2.get_param_values(),
            msg="Parameters are not equal.",
        )

    def assertParamsNotEqual(self, network1, network2):
        self.assertNpArraysNotEqual(
            network1.get_param_values(),
            network2.get_param_values(),
            msg="Parameters are equal.",
        )

    def randomize_param_values(self, network):
        for v in network.get_params():
            self.sess.run(
                v.assign(np.random.rand(*v.get_shape()))
            )
