import unittest
import pickle

import tensorflow as tf
from misc.tf_test_case import TFTestCase
from policies.nn_policy import FeedForwardPolicy
from qfunctions.nn_qfunction import FeedForwardCritic
from qfunctions.quadratic_naf_qfunction import QuadraticNAF


class TestSerialization(TFTestCase):
    def setUp(self):
        super().setUp()
        self.action_dim = 1
        self.observation_dim = 1

    def test_serialize_feedforward_critic(self):
        f = FeedForwardCritic(
            name_or_scope="a",
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
        )
        self.sess.run(tf.initialize_all_variables())
        pickle.dumps(f)

    def test_serialize_feedforward_policy(self):
        policy = FeedForwardPolicy(
            name_or_scope="b",
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
        )
        self.sess.run(tf.initialize_all_variables())
        pickle.dumps(policy)

    def test_serialize_quadratic_naf(self):
        qf = QuadraticNAF(
            name_or_scope="qf",
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
        )
        self.sess.run(tf.initialize_all_variables())
        pickle.dumps(qf)


if __name__ == '__main__':
    unittest.main()
