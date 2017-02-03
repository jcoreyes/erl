import unittest

import tensorflow as tf

from railrl.misc.tf_test_case import TFTestCase
from railrl.predictors.perceptron import Perceptron


class TestTFTestCase(TFTestCase):
    def test_randomize_variables(self):
        in_size = 5
        out_size = 1
        input = tf.placeholder(tf.float32, shape=(None, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
        )
        self.sess.run(tf.global_variables_initializer())

        vars = perceptron.get_params()
        vars_old = self.sess.run(vars)
        self.randomize_param_values(perceptron)
        vars_new = self.sess.run(vars)
        for v1, v2 in zip(vars_old, vars_new):
            self.assertNpArraysNotAlmostEqual(v1, v2)

if __name__ == '__main__':
    unittest.main()
