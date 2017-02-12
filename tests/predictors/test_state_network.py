import numpy as np
import tensorflow as tf

from railrl.misc.tf_test_case import TFTestCase
from railrl.predictors.mlp_state_network import MlpStateNetwork


class TestStateNetwork(TFTestCase):
    def test_set_and_get_params(self):
        obs_dim = 7
        output_dim = 3
        net1 = MlpStateNetwork(name_or_scope="qf_a",
                               observation_dim=obs_dim,
                               output_dim=output_dim)
        net2 = MlpStateNetwork(name_or_scope="qf_b",
                               observation_dim=obs_dim,
                               output_dim=output_dim)

        o = np.random.rand(1, obs_dim)

        feed_1 = {
            net1.observation_input: o,
        }
        feed_2 = {
            net2.observation_input: o,
        }

        self.sess.run(tf.global_variables_initializer())

        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertFalse((out1 == out2).all())

        net2.set_param_values(net1.get_param_values())
        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertTrue((out1 == out2).all())

    def test_copy(self):
        obs_dim = 7
        output_dim = 3
        net1 = MlpStateNetwork(name_or_scope="qf_a",
                               observation_dim=obs_dim,
                               output_dim=output_dim)
        self.sess.run(tf.global_variables_initializer())
        net2 = net1.get_copy(name_or_scope="qf_b")

        o = np.random.rand(1, obs_dim)

        feed_1 = {
            net1.observation_input: o,
        }
        feed_2 = {
            net2.observation_input: o,
        }

        self.sess.run(tf.global_variables_initializer())

        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertFalse((out1 == out2).all())

        net2.set_param_values(net1.get_param_values())
        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertTrue((out1 == out2).all())

    def test_get_weight_tied_copy(self):
        obs_dim = 7
        output_dim = 3
        net1 = MlpStateNetwork(name_or_scope="qf_a",
                               observation_dim=obs_dim,
                               output_dim=output_dim)
        self.sess.run(tf.global_variables_initializer())
        net2_observation_input = tf.placeholder(tf.float32, [None, obs_dim])
        net2 = net1.get_weight_tied_copy(
            observation_input=net2_observation_input
        )

        params1 = net1.get_params_internal()
        params2 = net2.get_params_internal()
        self.assertEqual(params1, params2)

        o = np.random.rand(1, obs_dim)
        feed_1 = {
            net1.observation_input: o,
        }
        feed_2 = {
            net2.observation_input: o,
        }

        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertTrue((out1 == out2).all())


