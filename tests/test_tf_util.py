import unittest

import numpy as np
import tensorflow as tf
from railrl.misc.tf_test_case import TFTestCase

from railrl.core import tf_util
from railrl.misc.testing_utils import are_np_arrays_equal


class TestUtil(TFTestCase):
    def assertNpEqual(self, np_arr1, np_arr2):
        self.assertTrue(
            are_np_arrays_equal(np_arr1, np_arr2, threshold=1e-4),
            "Numpy arrays not equal")

    def test_linear_shape(self):
        input_placeholder = tf.placeholder(tf.float32, [None, 4])
        linear_output = tf_util.linear(
            input_placeholder,
            4,
            3,
        )
        self.sess.run(tf.global_variables_initializer())
        # y = xW + b
        x = np.random.rand(13, 4)
        y = self.sess.run(linear_output,
                          feed_dict={
                              input_placeholder: x,
                          })
        self.assertEqual(y.shape, (13, 3))

    def test_linear_output(self):
        input_placeholder = tf.placeholder(tf.float32, [None, 4])
        linear_output = tf_util.linear(
            input_placeholder,
            4,
            3,
            W_initializer=tf.constant_initializer(1.),
            b_initializer=tf.constant_initializer(0.),
        )
        self.sess.run(tf.global_variables_initializer())
        # y = xW + b
        x = np.random.rand(13, 4)
        y = self.sess.run(linear_output,
                          feed_dict={
                              input_placeholder: x,
                          })
        expected = np.matmul(x, np.ones((4, 3)))
        self.assertNpEqual(y, expected)

    def test_vec2lower_triangle(self):
        batchsize = 2
        vec_placeholder = tf.placeholder(tf.float32, [batchsize, 9])
        mat = tf_util.vec2lower_triangle(vec_placeholder, 3)
        vec_value = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [-1, -2, -3, -4, -5, -6, -7, -8, -9],
        ])
        actual = self.sess.run(mat,
                               feed_dict={
                                   vec_placeholder: vec_value,
                               })
        expected = np.array([
            [
                [np.exp(1), 0, 0],
                [4, np.exp(5), 0],
                [7, 8, np.exp(9)],
            ],
            [
                [np.exp(-1), 0, 0],
                [-4, np.exp(-5), 0],
                [-7, -8, np.exp(-9)],
            ]
        ])
        self.assertNpEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
