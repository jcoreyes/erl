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


class TestTensorFlowRnns(TFTestCase):
    class _AddOneRnn(tf.nn.rnn_cell.RNNCell):
        """
        A simple RNN that just adds one to the state. The output is input +
        state.
        """
        def __init__(self, dim):
            self._dim = dim

        def __call__(self, inputs, state, scope=None):
            """Run this RNN cell on inputs, starting from the given state.

            Args:
              inputs: `2-D` tensor with shape `[batch_size x input_size]`.
              state: if `self.state_size` is an integer, this should be a `2-D Tensor`
                with shape `[batch_size x self.state_size]`.  Otherwise, if
                `self.state_size` is a tuple of integers, this should be a tuple
                with shapes `[batch_size x s] for s in self.state_size`.
              scope: VariableScope for the created subgraph; defaults to class name.

            Returns:
              A pair containing:

              - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
              - New state: Either a single `2-D` tensor, or a tuple of tensors matching
                the arity and shapes of `state`.
            """
            return inputs + state, state + 1

        @property
        def state_size(self):
            """
            size(s) of state(s) used by this cell.
            """
            return self._dim

        @property
        def output_size(self):
            """Integer or TensorShape: size of outputs produced by this cell."""
            return self._dim

    class _AddOneRnnLastActionInState(tf.nn.rnn_cell.RNNCell):
        """
        Add the last action to the state
        state.
        """
        def __init__(self, dim):
            self._dim = dim

        def __call__(self, inputs, state, scope=None):
            """Run this RNN cell on inputs, starting from the given state.

            Args:
              inputs: `2-D` tensor with shape `[batch_size x input_size]`.
              state: if `self.state_size` is an integer, this should be a `2-D Tensor`
                with shape `[batch_size x self.state_size]`.  Otherwise, if
                `self.state_size` is a tuple of integers, this should be a tuple
                with shapes `[batch_size x s] for s in self.state_size`.
              scope: VariableScope for the created subgraph; defaults to class name.

            Returns:
              A pair containing:

              - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
              - New state: Either a single `2-D` tensor, or a tuple of tensors matching
                the arity and shapes of `state`.
            """
            real_state = state[0]
            output = inputs + real_state
            one = tf.get_variable(
                "add_var",
                shape=(self._dim, ),
                initializer=tf.constant_initializer(1, dtype=tf.float32),
            )
            return output, (real_state + one, output)

        @property
        def state_size(self):
            """
            size(s) of state(s) used by this cell.
            """
            return self._dim

        @property
        def output_size(self):
            """Integer or TensorShape: size of outputs produced by this cell."""
            return self._dim

    def test_sequence_length(self):
        rnn_cell = TestTensorFlowRnns._AddOneRnn(1)
        input_ph = tf.placeholder(tf.float32, shape=(None, 3, 1))
        rnn_inputs = tf.unpack(input_ph, axis=1)
        init_state_ph = tf.placeholder(tf.float32, shape=(None, 1))
        sequence_length_ph = tf.placeholder(tf.int32, shape=(None,))
        rnn_outputs, rnn_final_state = tf.nn.rnn(
            rnn_cell,
            rnn_inputs,
            initial_state=init_state_ph,
            sequence_length=sequence_length_ph,
            dtype=tf.float32,
        )
        x_values = np.array([
            np.zeros((3, 1)),
            np.ones((3, 1)),
            2 * np.ones((3, 1)),
            7 * np.ones((3, 1)),
        ])
        init_state_values = np.array([[10], [20], [30], [40]])
        sequence_length_values = np.array([0, 1, 2, 3])
        output_values, final_state_values = self.sess.run(
            [rnn_outputs, rnn_final_state],
            feed_dict={
                input_ph: x_values,
                init_state_ph: init_state_values,
                sequence_length_ph: sequence_length_values,
            }
        )
        output_expected = [
            # batch =  0    1     2     3
            np.array([[0], [21], [32], [47]]),  # T = 0
            np.array([[0], [0], [33], [48]]),  # T = 1
            np.array([[0], [0], [0], [49]]),  # T = 2
        ]
        final_state_expected = np.array([[10], [21], [32], [43]])
        self.assertNpArraysEqual(output_expected, output_values)
        self.assertNpEqual(np.array(final_state_values), final_state_expected)

    def test_dynamics_rnn(self):
        rnn_cell = TestTensorFlowRnns._AddOneRnn(1)
        input_ph = tf.placeholder(tf.float32, shape=(None, 3, 1))
        init_state_ph = tf.placeholder(tf.float32, shape=(None, 1))
        sequence_length_ph = tf.placeholder(tf.int32, shape=(None,))
        rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(
            rnn_cell,
            input_ph,
            initial_state=init_state_ph,
            sequence_length=sequence_length_ph,
            dtype=tf.float32,
            time_major=False,
        )
        x_values = np.array([
            np.zeros((3, 1)),
            np.ones((3, 1)),
            2 * np.ones((3, 1)),
            7 * np.ones((3, 1)),
        ])
        init_state_values = np.array([[10], [20], [30], [40]])
        sequence_length_values = np.array([0, 1, 2, 3])
        output_values, final_state_values = self.sess.run(
            [rnn_outputs, rnn_final_state],
            feed_dict={
                input_ph: x_values,
                init_state_ph: init_state_values,
                sequence_length_ph: sequence_length_values,
            }
        )
        output_expected = np.array([
            # batch =  0    1     2     3
            np.array([[0], [21], [32], [47]]),  # T = 0
            np.array([[0], [0], [33], [48]]),  # T = 1
            np.array([[0], [0], [0], [49]]),  # T = 2
        ])
        output_expected = np.swapaxes(output_expected, 0, 1)
        final_state_expected = np.array([[10], [21], [32], [43]])
        self.assertNpArraysEqual(output_expected, output_values)
        self.assertNpEqual(np.array(final_state_values), final_state_expected)

    def test_last_action_sequence_length(self):
        rnn_cell = TestTensorFlowRnns._AddOneRnnLastActionInState(1)
        input_ph = tf.placeholder(tf.float32, shape=(None, 3, 1))
        init_state_ph = (tf.placeholder(tf.float32, shape=(None, 1)),
                         tf.placeholder(tf.float32, shape=(None, 1)))
        sequence_length_ph = tf.placeholder(tf.int32, shape=(None,))
        rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(
            rnn_cell,
            input_ph,
            initial_state=init_state_ph,
            sequence_length=sequence_length_ph,
            dtype=tf.float32,
            time_major=False,
        )
        x_values = np.array([
            np.zeros((3, 1)),
            np.ones((3, 1)),
            2 * np.ones((3, 1)),
            7 * np.ones((3, 1)),
        ])
        init_state_values = (np.array([[10], [20], [30], [40]]),
                             np.zeros((4, 1)))
        sequence_length_values = np.array([0, 1, 2, 3])
        self.sess.run(tf.global_variables_initializer())
        output_values, final_state_values = self.sess.run(
            [rnn_outputs, rnn_final_state],
            feed_dict={
                input_ph: x_values,
                init_state_ph: init_state_values,
                sequence_length_ph: sequence_length_values,
            }
        )
        output_expected = np.array([
            # batch =  0    1     2     3
            np.array([[0], [21], [32], [47]]),  # T = 0
            np.array([[0], [0], [33], [48]]),  # T = 1
            np.array([[0], [0], [0], [49]]),  # T = 2
        ])
        output_expected = np.swapaxes(output_expected, 0, 1)
        final_state_expected = (np.array([[10], [21], [32], [43]]),
                                np.array([[0], [21], [33], [49]]))
        self.assertNpArraysEqual(output_expected, output_values)
        self.assertNpArraysEqual(np.array(final_state_values),
                                 final_state_expected)

if __name__ == '__main__':
    unittest.main()
