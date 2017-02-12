import numpy as np
import tensorflow as tf

from railrl.misc.tf_test_case import TFTestCase
from railrl.predictors.mlp_state_action_network import MlpStateActionNetwork


class TestStateActionNetwork(TFTestCase):
    def test_set_and_get_params(self):
        action_dim = 5
        obs_dim = 7
        output_dim = 3
        net1 = MlpStateActionNetwork(name_or_scope="qf_a",
                                     observation_dim=obs_dim,
                                     action_dim=action_dim,
                                     output_dim=output_dim)
        net2 = MlpStateActionNetwork(name_or_scope="qf_b",
                                     observation_dim=obs_dim,
                                     action_dim=action_dim,
                                     output_dim=output_dim)

        a = np.random.rand(1, action_dim)
        o = np.random.rand(1, obs_dim)

        feed_1 = {
            net1.action_input: a,
            net1.observation_input: o,
        }
        feed_2 = {
            net2.action_input: a,
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
        action_dim = 5
        obs_dim = 7
        output_dim = 3
        net1 = MlpStateActionNetwork(name_or_scope="qf_a",
                                     observation_dim=obs_dim,
                                     action_dim=action_dim,
                                     output_dim=output_dim)
        self.sess.run(tf.global_variables_initializer())
        net2 = net1.get_copy(name_or_scope="qf_b")
        self.sess.run(tf.global_variables_initializer())

        a = np.random.rand(1, action_dim)
        o = np.random.rand(1, obs_dim)

        feed_1 = {
            net1.action_input: a,
            net1.observation_input: o,
        }
        feed_2 = {
            net2.action_input: a,
            net2.observation_input: o,
        }

        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertFalse((out1 == out2).all())

        net2.set_param_values(net1.get_param_values())
        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertTrue((out1 == out2).all())

    def test_get_weight_tied_copy_obs_only(self):
        action_dim = 5
        obs_dim = 7
        net2_observation_input = tf.placeholder(tf.float32, [None, obs_dim])
        self.finish_test_get_weight_tied_copy(
            action_dim,
            obs_dim,
            net2_observation_input=net2_observation_input,
        )

    def test_get_weight_tied_copy_action_only(self):
        action_dim = 5
        obs_dim = 7
        net2_action_input = tf.placeholder(tf.float32, [None, action_dim])
        self.finish_test_get_weight_tied_copy(
            action_dim,
            obs_dim,
            net2_action_input=net2_action_input,
        )

    def test_get_weight_tied_copy_obs_and_action(self):
        action_dim = 5
        obs_dim = 7
        net2_observation_input = tf.placeholder(tf.float32, [None, obs_dim])
        net2_action_input = tf.placeholder(tf.float32, [None, action_dim])
        self.finish_test_get_weight_tied_copy(
            action_dim,
            obs_dim,
            net2_observation_input=net2_observation_input,
            net2_action_input=net2_action_input
        )

    def finish_test_get_weight_tied_copy(self,
                                         action_dim,
                                         obs_dim,
                                         net2_observation_input=None,
                                         net2_action_input=None):
        output_dim = 3
        net1 = MlpStateActionNetwork(name_or_scope="qf_a",
                                     observation_dim=obs_dim,
                                     action_dim=action_dim,
                                     output_dim=output_dim)
        self.sess.run(tf.global_variables_initializer())
        net2 = net1.get_weight_tied_copy(
            observation_input=net2_observation_input,
            action_input=net2_action_input)

        self.sess.run(tf.global_variables_initializer())
        a = np.random.rand(1, action_dim)
        o = np.random.rand(1, obs_dim)

        feed_1 = {
            net1.action_input: a,
            net1.observation_input: o,
        }
        feed_2 = {
            net2.action_input: a,
            net2.observation_input: o,
        }

        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertTrue((out1 == out2).all())

        self.randomize_param_values(net1)

        out1 = self.sess.run(net1.output, feed_1)
        out2 = self.sess.run(net2.output, feed_2)
        self.assertTrue((out1 == out2).all())