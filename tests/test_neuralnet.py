import unittest
import numpy as np
import tensorflow as tf

from railrl.core.perceptron import Perceptron
from railrl.core.tf_util import BatchNormConfig
from railrl.misc.tf_test_case import TFTestCase


class TestNeuralNetwork(TFTestCase):
    def var_names(self, vars):
        names = [var.name for var in vars]
        unique_names = set(names)
        self.assertEqual(
            len(names),
            len(unique_names),
            "Some variable names are repeated!"
        )
        return unique_names

    def test_get_params(self):
        in_size = 5
        out_size = 1
        W_name = "w"
        b_name = "b"
        input = tf.placeholder(tf.float32, shape=(1, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            b_name=b_name,
        )

        all_vars = perceptron.get_params()
        print(all_vars)
        names = self.var_names(all_vars)
        print(names)

        expected_names = {
            "perceptron/w:0",
            "perceptron/b:0",
        }
        self.assertEqual(names, expected_names)

    def test_regularize_only_w(self):
        in_size = 5
        out_size = 1
        W_name = "w"
        b_name = "b"
        input = tf.placeholder(tf.float32, shape=(1, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            b_name=b_name,
        )

        all_vars = perceptron.get_params(regularizable=True)
        names = self.var_names(all_vars)

        expected_names = {
            "perceptron/w:0",
        }
        self.assertEqual(names, expected_names)

    def test_not_regularize_only_b(self):
        in_size = 5
        out_size = 1
        W_name = "w"
        b_name = "b"
        input = tf.placeholder(tf.float32, shape=(1, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            b_name=b_name,
        )

        all_vars = perceptron.get_params(regularizable=False)
        names = self.var_names(all_vars)

        expected_names = {
            "perceptron/b:0",
        }
        self.assertEqual(names, expected_names)

    def test_batch_norm_whitens_data(self):
        in_size = 2
        out_size = 2
        W_name = "w"
        W_initializer = tf.constant_initializer(value=np.eye(2))
        b_name = "b"
        b_initializer = tf.constant_initializer(value=np.array([0, 0]))
        input = tf.placeholder(tf.float32, shape=(1, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            W_initializer=W_initializer,
            b_name=b_name,
            b_initializer=b_initializer,
            batch_norm=BatchNormConfig(True),
        )

        input_values = np.array([[-2, 2]])

        output = perceptron.output
        self.sess.run(tf.global_variables_initializer())
        values = self.sess.run(output, {input: input_values})
        expected_values = np.array([[-2, 2]])
        self.assertNpArraysEqual(values, expected_values)

        training_output = perceptron.training_output
        training_values = self.sess.run(training_output, {input: input_values})
        expected_training_values = [-1, 1]
        self.assertNpArraysEqual(training_values, expected_training_values)

    def test_batch_norm_stores_moving_average_std(self):
        in_size = 2
        out_size = 2
        W_name = "w"
        W_initializer = tf.constant_initializer(value=np.eye(2))
        b_name = "b"
        b_initializer = tf.constant_initializer(value=np.array([0, 0]))
        input = tf.placeholder(tf.float32, shape=(1, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            W_initializer=W_initializer,
            b_name=b_name,
            b_initializer=b_initializer,
            batch_norm_config=BatchNormConfig(True),
        )
        self.sess.run(tf.global_variables_initializer())

        input_values = np.array([[-2, 2]])
        # std ~= 2

        update_batch_norm_stats = perceptron.batch_norm_update_stats_op
        self.sess.run(update_batch_norm_stats)

        training_values = self.sess.run(
            [perceptron.training_output] + update_batch_norm_stats,
            {input: input_values}
        )[0]
        expected_training_values = [0, 0]
        self.assertNpArraysEqual(training_values, expected_training_values)

        # At this point, the population std is 2
        output = perceptron.output
        values = self.sess.run(output, {input: input_values})
        expected_values = np.array([[-1, 1]])
        self.assertNpArraysEqual(values, expected_values)



if __name__ == '__main__':
    unittest.main()
