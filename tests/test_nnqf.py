import numpy as np
import tensorflow as tf

from railrl.misc.tf_test_case import TFTestCase
from railrl.qfunctions.nn_qfunction import FeedForwardCritic


class TestFeedForwardCritic(TFTestCase):
    def test_copy(self):
        action_dim = 5
        obs_dim = 7
        critic1 = FeedForwardCritic(name_or_scope="qf_a",
                                    observation_dim=obs_dim,
                                    action_dim=action_dim)
        critic2 = FeedForwardCritic(name_or_scope="qf_b",
                                    observation_dim=obs_dim,
                                    action_dim=action_dim)
        critic1.sess = self.sess
        critic2.sess = self.sess

        a = np.random.rand(1, action_dim)
        o = np.random.rand(1, obs_dim)

        feed_1 = {
            critic1.action_input: a,
            critic1.observation_input: o,
        }
        feed_2 = {
            critic2.action_input: a,
            critic2.observation_input: o,
        }

        self.sess.run(tf.global_variables_initializer())

        out1 = self.sess.run(critic1.output, feed_1)
        out2 = self.sess.run(critic2.output, feed_2)
        self.assertFalse((out1 == out2).all())

        critic2.set_param_values(critic1.get_param_values())
        out1 = self.sess.run(critic1.output, feed_1)
        out2 = self.sess.run(critic2.output, feed_2)
        self.assertTrue((out1 == out2).all())

    def test_output_len(self):
        action_dim = 5
        obs_dim = 7
        critic = FeedForwardCritic(name_or_scope="1",
                                   observation_dim=obs_dim,
                                   action_dim=action_dim)
        critic.sess = self.sess

        a = np.random.rand(1, action_dim)
        o = np.random.rand(1, obs_dim)
        feed = {
            critic.action_input: a,
            critic.observation_input: o,
        }

        self.sess.run(tf.global_variables_initializer())

        out = self.sess.run(critic.output, feed)
        self.assertEqual(1, out.size)
