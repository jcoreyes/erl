import unittest

import numpy as np
import tensorflow as tf
from railrl.misc.tf_test_case import TFTestCase

from railrl.core import tf_util


class TestUtil(TFTestCase):
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
        self.assertNpAlmostEqual(actual, expected, threshold=1e-4)

    def test_batch_norm_untrained_is_a_noop(self):
        in_size = 1
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))
        bn_config = tf_util.BatchNormConfig(epsilon=0., decay=0.5)
        scope_name = 'test_bn_scope'

        with tf.variable_scope(scope_name) as scope:
            output_train = tf_util.batch_norm(
                input_layer,
                True,
                batch_norm_config=bn_config
            )[0]
            scope.reuse_variables()
            output_eval = tf_util.batch_norm(
                input_layer,
                False,
                batch_norm_config=bn_config
            )[0]

        input_values = np.array([[-2], [2]])

        self.sess.run(tf.global_variables_initializer())
        eval_values, _ = self.sess.run(
            [output_eval, output_train],
            {input_layer: input_values}
        )
        expected_eval_values = np.array([[-2], [2]])
        self.assertNpArraysEqual(eval_values, expected_eval_values)

    def test_batch_norm_whitens_training_data(self):
        in_size = 1
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))
        bn_config = tf_util.BatchNormConfig(epsilon=0.)
        scope_name = 'test_bn_scope'

        with tf.variable_scope(scope_name):
            output_train = tf_util.batch_norm(
                input_layer,
                True,
                batch_norm_config=bn_config
            )[0]

        input_values = np.array([[-2], [2]])

        self.sess.run(tf.global_variables_initializer())
        training_values = self.sess.run(
            output_train,
            {input_layer: input_values}
        )
        expected_training_values = np.array([[-1], [1]])
        self.assertNpArraysEqual(training_values, expected_training_values)

    def test_batch_norm_pop_stats_are_computed_correctly(self):
        in_size = 1
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))
        bn_config = tf_util.BatchNormConfig(decay=0.)
        scope_name = 'test_bn_scope'

        with tf.variable_scope(scope_name):
            output_train, batch_ops = tf_util.batch_norm(
                input_layer,
                True,
                batch_norm_config=bn_config
            )

        input_values = np.array([[1, 2, 3, 4]]).T
        # variance([1, 2, 3, 4]) = 1.25
        # mean([1, 2, 3, 4]) = 2.5

        self.sess.run(tf.global_variables_initializer())
        training_values, mean, variance = self.sess.run(
            [
                output_train,
                batch_ops.update_pop_mean_op,
                batch_ops.update_pop_var_op
            ],
            {input_layer: input_values}
        )
        self.assertAlmostEqual(mean[0], 2.5)
        self.assertAlmostEqual(variance[0], 1.25)

    def test_batch_norm_pop_stats_decays_correctly(self):
        in_size = 1
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))
        bn_config = tf_util.BatchNormConfig(decay=0.5)
        scope_name = 'test_bn_scope'

        with tf.variable_scope(scope_name):
            output_train, batch_ops = tf_util.batch_norm(
                input_layer,
                True,
                batch_norm_config=bn_config
            )

        input_values = np.array([[1, 2, 3, 4]]).T
        # variance([1, 2, 3, 4]) = 1.25
        # mean([1, 2, 3, 4]) = 2.5

        self.sess.run(tf.global_variables_initializer())
        training_values, mean, variance = self.sess.run(
            [
                output_train,
                batch_ops.update_pop_mean_op,
                batch_ops.update_pop_var_op
            ],
            {input_layer: input_values}
        )
        self.assertAlmostEqual(mean[0], 2.5 / 2)
        self.assertAlmostEqual(variance[0], (1.25 + 1) / 2)

    def test_batch_norm_pop_stats_decays_correctly_2_itrs(self):
        in_size = 1
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))
        bn_config = tf_util.BatchNormConfig(decay=0.5)
        scope_name = 'test_bn_scope'

        with tf.variable_scope(scope_name):
            output_train, batch_ops = tf_util.batch_norm(
                input_layer,
                True,
                batch_norm_config=bn_config
            )

        input_values = np.array([[1, 2, 3, 4]]).T
        # variance([1, 2, 3, 4]) = 1.25
        # mean([1, 2, 3, 4]) = 2.5

        input_values2 = np.array([[2, 4]]).T
        # variance([2, 4]) = 1
        # mean([2, 4]) = 3

        self.sess.run(tf.global_variables_initializer())
        mean = None
        variance = None
        for values in [input_values, input_values2]:
            training_values, mean, variance = self.sess.run(
                [
                    output_train,
                    batch_ops.update_pop_mean_op,
                    batch_ops.update_pop_var_op
                ],
                {input_layer: values}
            )
        self.assertAlmostEqual(mean[0], ((2.5 / 2) + 3) / 2)
        self.assertAlmostEqual(variance[0], (((1.25 + 1) / 2) + 1) / 2)

    def test_batch_norm_eval_uses_pop_stats_correctly(self):
        in_size = 1
        input_layer = tf.placeholder(tf.float32, shape=(None, in_size))
        epsilon = 1e-5
        bn_config = tf_util.BatchNormConfig(decay=0, epsilon=epsilon)
        scope_name = 'test_bn_scope'

        with tf.variable_scope(scope_name) as scope:
            output_train, batch_ops = tf_util.batch_norm(
                input_layer,
                True,
                batch_norm_config=bn_config
            )
            scope.reuse_variables()
            output_eval = tf_util.batch_norm(
                input_layer,
                False,
                batch_norm_config=bn_config
            )[0]

        input_values = np.array([[1, 2, 3, 4]]).T
        # variance([1, 2, 3, 4]) = 1.25
        # mean([1, 2, 3, 4]) = 2.5

        self.sess.run(tf.global_variables_initializer())
        training_values, mean, variance, gamma, beta = self.sess.run(
            [
                output_train,
                batch_ops.update_pop_mean_op,
                batch_ops.update_pop_var_op,
                batch_ops.scale,
                batch_ops.offset,
            ],
            {input_layer: input_values}
        )
        self.assertEqual(mean[0], 2.5)
        self.assertEqual(variance[0], 1.25)
        self.assertEqual(gamma[0], 1)
        self.assertEqual(beta[0], 0)
        expected_eval_values = np.array([[-2, 2]]).T
        expected_pop_mean = 2.5
        expected_pop_var = 1.25
        eval_input_values = (
            expected_eval_values * np.sqrt(expected_pop_var + epsilon)
            + expected_pop_mean
        )
        eval_values = self.sess.run(
            output_eval,
            {input_layer: eval_input_values}
        )
        self.assertNpArraysAlmostEqual(expected_eval_values, eval_values)

    def test_get_batch_norm_update_pop_stats_ops(self):
        input_layer = tf.placeholder(tf.float32, shape=(None, 1))
        scope_name = 'test_bn_scope'

        with tf.variable_scope(scope_name) as bn_scope:
            output_train, bn_ops = tf_util.batch_norm(
                input_layer,
                True,
            )

        update_ops = bn_ops.update_pop_stats_ops
        update_ops_v2 = tf_util.get_batch_norm_update_pop_stats_ops(bn_scope)

        self.assertEqual(set(update_ops), set(update_ops_v2))

    def test_scope_does_not_affect_get_batch_norm_update_pop_stats_ops(self):
        scope_name = 'test_bn_scope'

        with tf.variable_scope('test'):
            with tf.variable_scope('foo'):
                input_layer = tf.placeholder(tf.float32, shape=(None, 1))
            with tf.variable_scope(scope_name) as bn_scope:
                output_train, bn_ops = tf_util.batch_norm(
                    input_layer,
                    True,
                )

        with tf.variable_scope('bar'):
            update_ops = bn_ops.update_pop_stats_ops
        with tf.variable_scope('zoo'):
            update_ops_v2 = tf_util.get_batch_norm_update_pop_stats_ops(bn_scope)

        self.assertEqual(set(update_ops), set(update_ops_v2))

    def test_get_untrainable_batch_norm_vars(self):
        input_layer = tf.placeholder(tf.float32, shape=(None, 1))
        scope_name = 'test_bn_scope'

        with tf.variable_scope(scope_name) as bn_scope:
            output_train, bn_ops = tf_util.batch_norm(
                input_layer,
                True,
            )

        update_ops = [bn_ops.pop_mean, bn_ops.pop_var]
        update_ops_v2 = tf_util.get_untrainable_batch_norm_vars(bn_scope)

        self.assertEqual(set(update_ops), set(update_ops_v2))

    def test_untrainable_batch_norm_vars_not_trainable(self):
        input_layer = tf.placeholder(tf.float32, shape=(None, 1))
        scope_name = 'test_bn_scope'

        with tf.variable_scope(scope_name):
            tf_util.batch_norm(input_layer, True)
        with tf.variable_scope(scope_name, reuse=True) as bn_scope:
            tf_util.batch_norm(input_layer, False)

        self.sess.run(tf.global_variables_initializer())
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        bn_params = tf_util.get_untrainable_batch_norm_vars(bn_scope)

        self.assertEqual(0, len(set(trainable_params).intersection(bn_params)))

if __name__ == '__main__':
    unittest.main()
