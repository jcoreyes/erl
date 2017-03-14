import unittest

import numpy as np
import tensorflow as tf

from railrl.core import tf_util
from railrl.testing.tf_test_case import TFTestCase


def create_network(in_size):
    hidden_sizes = (32, 4)
    nonlinearity = tf.nn.relu
    input_ph = tf.placeholder(tf.float32, shape=[None, in_size])
    last_layer = tf_util.mlp(input_ph, in_size, hidden_sizes, nonlinearity)
    return input_ph, last_layer


class TestTensorFlow(TFTestCase):
    def test_copy_values(self):
        in_size = 10
        with tf.variable_scope('a') as _:
            in_a, out_a = create_network(in_size)
        with tf.variable_scope('b') as _:
            in_b, out_b = create_network(in_size)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        x = np.random.rand(1, in_size)
        feed_a = {in_a: x}
        feed_b = {in_b: x}
        val_a = self.sess.run(out_a, feed_dict=feed_a)
        val_b = self.sess.run(out_b, feed_dict=feed_b)
        self.assertFalse((val_a == val_b).all())

        # Try copying
        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "a")
        b_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "b")
        assign_ops = [tf.assign(a_vars[i], b_vars[i]) for i in
                      range(len(a_vars))]
        self.sess.run(assign_ops)
        val_a = self.sess.run(out_a, feed_dict=feed_a)
        val_b = self.sess.run(out_b, feed_dict=feed_b)
        self.assertTrue((val_a == val_b).all())

    def test_get_collections(self):
        in_size = 5
        out_size = 10
        input_placeholder = tf.placeholder(tf.float32, [None, in_size])
        scope = 'abc'
        with tf.variable_scope(scope) as _:
            _ = tf_util.linear(input_placeholder,
                               in_size,
                               out_size)
        # TODO(vpong): figure out why this line doesn't work
        # variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        self.assertEqual(2, len(variables))
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "nope")
        self.assertEqual(0, len(variables))

    def test_batch_matmul(self):
        batchsize = 5
        dim = 3
        M = np.random.rand(batchsize, dim, dim)
        x = np.random.rand(batchsize, dim)
        x = np.expand_dims(x, axis=1)
        x_shape = x.shape
        M_shape = M.shape
        x_placeholder = tf.placeholder(tf.float32, x_shape)
        M_placeholder = tf.placeholder(tf.float32, M_shape)

        expected = np.zeros((batchsize, 1, dim))
        for i in range(batchsize):
            expected[i] = np.matmul(x[i], M[i])

        actual = self.sess.run(
            tf.batch_matmul(x_placeholder, M_placeholder),
            feed_dict={
                x_placeholder: x,
                M_placeholder: M,
            })
        self.assertNpEqual(actual, expected)


    def test_batch_matmul2(self):
        batchsize = 5
        dim = 3
        M = np.random.rand(batchsize, dim, dim)
        x = np.random.rand(batchsize, dim)
        x = np.expand_dims(x, axis=1)
        x_shape = x.shape
        M_shape = M.shape
        x_placeholder = tf.placeholder(tf.float32, x_shape)
        M_placeholder = tf.placeholder(tf.float32, M_shape)

        expected = np.zeros((batchsize, 1))
        for i in range(batchsize):
            vec = np.matmul(x[i], M[i])
            expected[i] = np.matmul(vec, vec.T)

        batch = tf.batch_matmul(x_placeholder, M_placeholder)
        actual_op = tf.batch_matmul(
            batch,
            batch,
            adj_y=True,
        )
        actual_op_flat = tf.squeeze(actual_op, [1])
        actual = self.sess.run(
            actual_op_flat,
            feed_dict={
                x_placeholder: x,
                M_placeholder: M,
            })
        self.assertNpEqual(actual, expected)

    def test_argmax(self):
        input_layer = tf.placeholder(tf.float32, shape=(None, 2))
        argmax = tf.argmax(input_layer, axis=1)
        x = np.array([
            [0, 1],
            [-5, -20],
            [100, 101],
        ])
        actual = self.sess.run(argmax,
                               feed_dict={
                                   input_layer: x,
                               })
        expected = np.array([1, 0, 1])
        self.assertNpEqual(actual, expected)

    def test_argmax_none_axis(self):
        input_layer = tf.placeholder(tf.float32, shape=(None, 2))
        argmax = tf.argmax(input_layer, axis=0)
        x = np.array([
            [0, 1],
            [-5, -20],
            [100, 101],
        ])
        actual = self.sess.run(argmax,
                               feed_dict={
                                   input_layer: x,
                               })
        expected = np.array([2, 2])
        self.assertNpEqual(actual, expected)

    def test_argmax_no_gradients(self):
        x_ph = tf.placeholder(tf.float32, shape=(None, 2))
        argmax = tf.argmax(x_ph, axis=1)
        error_found = False
        try:
            tf.gradients(argmax, x_ph)
        except LookupError:
            error_found = True
        self.assertTrue(error_found)

    def test_max(self):
        x_ph = tf.placeholder(tf.float32, shape=(None, 2))
        max = tf.reduce_max(x_ph, reduction_indices=[1])
        x = np.array([
            [0, 1],
            [-5, -20],
            [100, 101],
        ])
        actual = self.sess.run(max,
                               feed_dict={
                                   x_ph: x,
                               })
        expected = np.array([1, -5, 101])
        self.assertNpEqual(actual, expected)

    def test_max_none_axis(self):
        x_ph = tf.placeholder(tf.float32, shape=(None, 2))
        max = tf.reduce_max(x_ph, reduction_indices=[0])
        x = np.array([
            [0, 1],
            [-5, -20],
            [100, 101],
        ])
        actual = self.sess.run(max,
                               feed_dict={
                                   x_ph: x,
                               })
        expected = np.array([100, 101])
        self.assertNpEqual(actual, expected)

    def test_max_has_gradients(self):
        x_ph = tf.placeholder(tf.float32, shape=(None, 2))
        max = tf.reduce_max(x_ph, reduction_indices=[1])
        grad = tf.gradients(max, x_ph)
        self.assertTrue(grad is not None)
        self.assertTrue(grad[0] is not None)

if __name__ == '__main__':
    unittest.main()
