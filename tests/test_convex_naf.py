import unittest
import numpy as np

from algos.convex_naf import ConvexNAFAlgorithm
from misc.tf_test_case import TFTestCase
from qfunctions.convex_naf_qfunction import ConvexNAF
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.envs.base import TfEnv


class TestConvexNAF(TFTestCase):
    def setUp(self):
        super().setUp()
        self.env = TfEnv(CartpoleEnv())
        self.es = OUStrategy(env_spec=self.env.spec)

    def test_af_observation_inputs_same(self):
        qf = ConvexNAF(
            name_or_scope="qf",
            env_spec=self.env.spec,
        )
        algo = ConvexNAFAlgorithm(
            self.env,
            self.es,
            qf,
        )
        qf = algo.qf
        af = qf.af
        af1 = qf.get_implicit_policy().qfunction
        af2 = qf.af_copy_with_policy_input

        self.assertEqual(af.observation_input, af1.observation_input)
        self.assertEqual(af2.observation_input, af1.observation_input)

    def test_af_outputs_same(self):
        qf = ConvexNAF(
            name_or_scope="qf",
            env_spec=self.env.spec,
        )
        algo = ConvexNAFAlgorithm(
            self.env,
            self.es,
            qf,
        )
        qf = algo.qf
        af = qf.af
        af1 = qf.get_implicit_policy().qfunction
        af2 = qf.af_copy_with_policy_input

        self.assertParamsEqual(af, af1)
        self.assertParamsEqual(af1, af2)

    def test_af_W_weights_nonnegative(self):
        qf = ConvexNAF(
            name_or_scope="qf",
            env_spec=self.env.spec,
        )
        algo = ConvexNAFAlgorithm(
            self.env,
            self.es,
            qf,
            n_epochs=1,
            epoch_length=5,
            min_pool_size=2,
            eval_samples=0,
        )
        algo.train()
        af = algo.qf.af

        action_param_values = self.sess.run([param for param in
                                             af.get_action_W_params()])
        for param in action_param_values:
            self.assertTrue(np.min(param) >= 0.)

    def test_af_b_weights_can_be_negatative(self):
        qf = ConvexNAF(
            name_or_scope="qf",
            env_spec=self.env.spec,
        )
        algo = ConvexNAFAlgorithm(
            self.env,
            self.es,
            qf,
            n_epochs=1,
            epoch_length=5,
            min_pool_size=2,
            eval_samples=0,
        )
        algo.train()
        af = algo.qf.af

        af_params = self.sess.run([param for param in af.get_params_internal()])
        none_are_neg = True
        for param in af_params:
            if np.min(param) < 0.:
                none_are_neg = False

        self.assertFalse(none_are_neg)


if __name__ == '__main__':
    unittest.main()
