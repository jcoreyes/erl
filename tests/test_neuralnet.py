import unittest

import numpy as np
import tensorflow as tf

from railrl.core.tf_util import BatchNormConfig
from railrl.misc.tf_test_case import TFTestCase
from railrl.predictors.perceptron import Perceptron


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
        names = self.var_names(all_vars)

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

    def test_batch_norm_off_is_a_noop(self):
        in_size = 1
        out_size = 1
        W_name = "w"
        W_initializer = tf.constant_initializer(value=np.eye(1))
        b_name = "b"
        b_initializer = tf.constant_initializer(value=np.array([0]))
        input = tf.placeholder(tf.float32, shape=(None, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            W_initializer=W_initializer,
            b_name=b_name,
            b_initializer=b_initializer,
            batch_norm_config=None,
        )

        input_values = np.array([[-2], [2]])

        output = perceptron.output
        self.sess.run(tf.global_variables_initializer())
        values = self.sess.run(output, {input: input_values})
        expected_values = np.array([[-2], [2]])
        self.assertNpArraysEqual(values, expected_values)

        training_output = perceptron.training_output
        training_values = self.sess.run(training_output, {input: input_values})
        expected_training_values = np.array([[-2], [2]])
        self.assertNpArraysEqual(training_values, expected_training_values)

    def test_batch_norm_untrained_is_a_noop(self):
        in_size = 1
        out_size = 1
        W_name = "w"
        W_initializer = tf.constant_initializer(value=np.eye(1))
        b_name = "b"
        b_initializer = tf.constant_initializer(value=np.array([0]))
        input = tf.placeholder(tf.float32, shape=(None, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            W_initializer=W_initializer,
            b_name=b_name,
            b_initializer=b_initializer,
            batch_norm_config=BatchNormConfig(),
        )

        input_values = np.array([[-2, 2]]).T

        eval_output = perceptron.output
        self.sess.run(tf.global_variables_initializer())
        eval_values = self.sess.run(
            eval_output,
            {input: input_values}
        )
        expected_eval_values = np.array([[-2, 2]]).T
        self.assertNpArraysAlmostEqual(expected_eval_values, eval_values)

    def test_batch_norm_whitens_training_data(self):
        in_size = 1
        out_size = 1
        W_name = "w"
        W_initializer = tf.constant_initializer(value=np.eye(1))
        b_name = "b"
        b_initializer = tf.constant_initializer(value=np.array([0]))
        input = tf.placeholder(tf.float32, shape=(None, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            W_initializer=W_initializer,
            b_name=b_name,
            b_initializer=b_initializer,
            batch_norm_config=BatchNormConfig(),
        )

        input_values = np.array([[-2, 2]]).T

        training_output = perceptron.training_output
        self.sess.run(tf.global_variables_initializer())
        training_values = self.sess.run(
            training_output,
            {input: input_values}
        )
        expected_training_values = np.array([[-1, 1]]).T
        self.assertNotEqual(perceptron.training_output, perceptron.output)
        self.assertNpArraysAlmostEqual(expected_training_values,
                                       training_values)

    def test_batch_norm_stores_moving_average_std(self):
        in_size = 1
        out_size = 1
        W_name = "w"
        W_initializer = tf.constant_initializer(value=np.eye(1))
        b_name = "b"
        b_initializer = tf.constant_initializer(value=np.array([0]))
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))

        epsilon = 1e-5
        perceptron = Perceptron(
            "perceptron",
            input_layer,
            in_size,
            out_size,
            W_name=W_name,
            W_initializer=W_initializer,
            b_name=b_name,
            b_initializer=b_initializer,
            batch_norm_config=BatchNormConfig(epsilon=epsilon, decay=0.),
        )
        self.sess.run(tf.global_variables_initializer())

        input_values = np.array([[1, 2, 3, 4]]).T
        # variance([1, 2, 3, 4]) = 1.25
        # mean([1, 2, 3, 4]) = 2.5

        self.sess.run(
            perceptron.batch_norm_update_stats_op,
            {input_layer: input_values}
        )

        expected_eval_values = np.array([[-2, 2]]).T
        expected_pop_mean = 2.5
        expected_pop_var = 1.25
        eval_input_values = (
            expected_eval_values * np.sqrt(expected_pop_var + epsilon)
            + expected_pop_mean
        )
        eval_values = self.sess.run(
            perceptron.output,
            {input_layer: eval_input_values}
        )
        self.assertNpArraysAlmostEqual(expected_eval_values, eval_values)

    def test_batch_norm_variables_are_saved(self):
        in_size = 1
        out_size = 1
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))

        perceptron = Perceptron(
            "perceptron",
            input_layer,
            in_size,
            out_size,
            batch_norm_config=BatchNormConfig(),
        )
        self.sess.run(tf.global_variables_initializer())

        params = perceptron.get_params()
        self.assertEqual(6, len(params))
        # 2 for the network
        # 2 for the pop mean / variance
        # 2 for the scale / offset

    def test_batch_norm_offset_and_scale_variables_change_correctly(self):
        in_size = 1
        out_size = 1
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))
        learning_rate = 0.5
        input_value = 0.75
        W_initializer = tf.constant_initializer(value=np.eye(1))

        scale_name = "test_scale"
        offset_name = "test_offset"
        perceptron = Perceptron(
            "perceptron",
            input_layer,
            in_size,
            out_size,
            W_initializer=W_initializer,
            batch_norm_config=BatchNormConfig(
                bn_scale_name=scale_name,
                bn_offset_name=offset_name,
            ),
        )
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(
            tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate
            ).minimize(
                perceptron.output
            ),
            feed_dict={
                input_layer: np.array([[input_value]]),
            }
        )

        params = perceptron.get_params()
        scale_values = self.sess.run(
            [v for v in params if scale_name in v.name]
        )
        assert len(scale_values) == 1
        scale_value = scale_values[0][0]
        offset_values = self.sess.run(
            [v for v in params if offset_name in v.name]
        )
        assert len(offset_values) == 1
        offset_value = offset_values[0][0]

        self.assertAlmostEqual(offset_value, -learning_rate)
        # Since it's just an offset, it increases by learning_rate
        self.assertAlmostEqual(scale_value,
                               1 - input_value * learning_rate,
                               delta=1e-4)

    def test_output_mode_switches(self):
        in_size = 1
        out_size = 1
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))

        perceptron = Perceptron(
            "perceptron",
            input_layer,
            in_size,
            out_size,
            batch_norm_config=BatchNormConfig(),
        )
        self.sess.run(tf.global_variables_initializer())
        training_output = perceptron.training_output
        eval_output = perceptron._eval_output
        self.assertNotEqual(training_output, eval_output)
        self.assertEqual(perceptron.output, eval_output)
        perceptron.switch_to_training_mode()
        self.assertEqual(perceptron.output, training_output)
        perceptron.switch_to_eval_mode()
        self.assertEqual(perceptron.output, eval_output)
        perceptron.switch_to_eval_mode()
        self.assertEqual(perceptron.output, eval_output)


if __name__ == '__main__':
    unittest.main()
