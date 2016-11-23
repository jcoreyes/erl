import unittest

import numpy as np

from algos.naf import NAF
from misc.testing_utils import are_np_array_lists_equal
from misc.tf_test_case import TFTestCase
from qfunctions.optimizable_qfunction import QuadraticNAF
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.envs.base import TfEnv


class TestNAF(TFTestCase):

    def setUp(self):
        super().setUp()
        self.env = TfEnv(CartpoleEnv())
        self.es = OUStrategy(env_spec=self.env.spec)

    def assertParamsEqual(self, network1, network2):
        self.assertTrue(are_np_array_lists_equal(
            network1.get_param_values(),
            network2.get_param_values(),
        ))

    def assertParamsNotEqual(self, network1, network2):
        self.assertFalse(are_np_array_lists_equal(
            network1.get_param_values(),
            network2.get_param_values(),
        ))

    def test_target_params_copied(self):
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF('qf', self.env.spec),
            n_epochs=0,
        )
        target_vf = algo.target_vf
        vf = algo.qf.vf

        # Make sure they're different to start
        random_values = [np.random.rand(*values.shape)
                         for values in vf.get_param_values()]
        vf.set_param_values(random_values)

        self.assertParamsNotEqual(target_vf, vf)

        algo.train()
        self.assertParamsEqual(target_vf, vf)

    def test_target_params_update(self):
        tau = 0.2
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF('qf', self.env.spec),
            n_epochs=0,
            soft_target_tau=tau,
        )
        target_vf = algo.target_vf
        vf = algo.qf.vf

        algo.train()

        orig_target_vals = target_vf.get_param_values()
        orig_vals = vf.get_param_values()
        algo.sess.run(algo.update_target_vf_op)
        new_target_vals = target_vf.get_param_values()

        for orig_target_val, orig_val, new_target_val in zip(
                orig_target_vals, orig_vals, new_target_vals):
            self.assertNpEqual(
                new_target_val,
                tau * orig_val + (1-tau) * orig_target_val
            )

    def test_target_params_hard_update(self):
        tau = 1.
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF('qf', self.env.spec),
            n_epochs=0,
            soft_target_tau=tau,
        )
        target_vf = algo.target_vf
        vf = algo.qf.vf

        # Make sure they're different to start
        random_values = [np.random.rand(*values.shape)
                         for values in vf.get_param_values()]
        vf.set_param_values(random_values)
        self.assertParamsNotEqual(target_vf, vf)
        algo.sess.run(algo.update_target_vf_op)
        self.assertParamsEqual(target_vf, vf)

    def test_target_params_no_update(self):
        tau = 0.
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF('qf', self.env.spec),
            n_epochs=0,
            soft_target_tau=tau,
        )
        target_vf = algo.target_vf
        vf = algo.qf.vf

        # Make sure they're different to start
        random_values = [np.random.rand(*values.shape)
                         for values in vf.get_param_values()]
        vf.set_param_values(random_values)
        self.assertParamsNotEqual(target_vf, vf)
        algo.sess.run(algo.update_target_vf_op)
        self.assertParamsNotEqual(target_vf, vf)

if __name__ == '__main__':
    unittest.main()
