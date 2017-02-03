import unittest

import numpy as np
import tensorflow as tf

from railrl.core.neuralnet import NeuralNetwork
from railrl.core.tf_util import BatchNormConfig
from railrl.misc.tf_test_case import TFTestCase
from railrl.predictors.mlp import Mlp
from railrl.predictors.perceptron import Perceptron
from rllab.misc.overrides import overrides


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

    def test_get_params_internal(self):
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

        all_vars = perceptron.get_params_internal()
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

        all_vars = perceptron.get_params_internal(regularizable=True)
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

        all_vars = perceptron.get_params_internal(regularizable=False)
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

    def test_batch_norm(self):
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

        perceptron.switch_to_training_mode()
        self.sess.run(
            perceptron.batch_norm_update_stats_op,
            {input_layer: input_values}
        )
        perceptron.switch_to_eval_mode()

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

    def test_subnetwork_walk(self):
        mmlp = TestMmlp(
            "mmlp",
            batch_norm_config=BatchNormConfig(),
        )
        self.assertEqual(6, len(list(mmlp._iter_sub_networks())))
        self.assertEqual(6, len(set(n.full_scope_name for n
                                in mmlp._iter_sub_networks())))

    def test_copy(self):
        in_size = 3
        out_size = 3
        x = tf.placeholder(tf.float32, [None, in_size])
        net1 = Perceptron(
            name_or_scope="p1",
            input_tensor=x,
            input_size=in_size,
            output_size=out_size,
        )

        self.sess.run(tf.global_variables_initializer())

        net2 = net1.get_copy(name_or_scope="p2")
        input_value = np.random.rand(1, in_size)

        feed = {
            x: input_value,
        }

        self.sess.run(tf.global_variables_initializer())

        out1 = self.sess.run(net1.output, feed)
        out2 = self.sess.run(net2.output, feed)
        self.assertNpArraysNotAlmostEqual(out1, out2)

        net2.set_param_values(net1.get_param_values())
        out1 = self.sess.run(net1.output, feed)
        out2 = self.sess.run(net2.output, feed)
        self.assertNpArraysAlmostEqual(out1, out2)

    def test_get_weight_tied_copy(self):
        in_size = 3
        out_size = 3
        net1_input = tf.placeholder(tf.float32, [None, in_size])
        net1 = Perceptron(
            name_or_scope="p1",
            input_tensor=net1_input,
            input_size=in_size,
            output_size=out_size,
        )

        self.sess.run(tf.global_variables_initializer())

        net2_input = tf.placeholder(tf.float32, [None, in_size])
        net2 = net1.get_weight_tied_copy(
            input_tensor=net2_input,
        )
        input_value = np.random.rand(1, in_size)

        feed_1 = {
            net1_input: input_value,
        }
        feed_2 = {
            net2_input: input_value,
        }

        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertNpArraysAlmostEqual(out1, out2)

        # Output should be the same even after re-initializing parameters
        self.sess.run(tf.global_variables_initializer())

        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertNpArraysAlmostEqual(out1, out2)

        params1 = net1.get_params_internal()
        params2 = net2.get_params_internal()
        self.assertEqual(params1, params2)


class TestMmlp(NeuralNetwork):
    """
    Multi-multi-layer perceptron
    2 MLPs with 2 perceptrons each
    """
    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, 1])
        super(TestMmlp, self).__init__(name_or_scope, **kwargs)
        self._create_network(input_tensor=self.input_tensor)

    @overrides
    def _create_network_internal(self, input_tensor=None):
        with tf.variable_scope('a'):
            self._add_subnetwork_and_get_output(
                Mlp('mlp1', input_tensor, 1, 1, (1, 1))
            )
        with tf.variable_scope('b'):
            return self._add_subnetwork_and_get_output(
                Mlp('mlp2', input_tensor, 1, 1, (1, 1))
            )

    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            input_tensor=self.input_tensor,
        )

if __name__ == '__main__':
    unittest.main()
