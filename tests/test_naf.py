import unittest
import math

import numpy as np
import tensorflow as tf

from algos.naf import NAF
from misc.tf_test_case import TFTestCase
from qfunctions.quadratic_naf_qfunction import QuadraticNAF
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.envs.base import TfEnv


class TestNAF(TFTestCase):
    def setUp(self):
        super().setUp()
        self.env = TfEnv(CartpoleEnv())
        self.es = OUStrategy(env_spec=self.env.spec)

    def test_target_params_copied(self):
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
        )
        target_vf = algo.target_vf
        vf = algo.qf.value_function

        # Make sure they're different to start
        random_values = [np.random.rand(*values.shape)
                         for values in vf.get_param_values()]
        vf.set_param_values(random_values)

        self.assertParamsNotEqual(target_vf, vf)

        algo.train()
        self.assertParamsEqual(target_vf, vf)

    def test_dead_grads(self):
        self.env = HalfCheetahEnv()
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
        )
        qf = algo.qf
        af = qf.advantage_function
        L_param_gen = af.L_params
        L = af.L
        last_bs = L_param_gen.get_params_internal()[-1]
        grads_ops = tf.gradients(af.output, last_bs)
        a = np.random.rand(1, algo.action_dim)
        o = np.random.rand(1, algo.observation_dim)
        grads = self.sess.run(
            grads_ops,
            {
                qf.action_input: a,
                qf.observation_input: o,
            }
        )[0]
        bs = self.sess.run(last_bs)
        num_elems = bs.size
        length = int(math.sqrt(float(num_elems)))
        expected_zero = length * (length - 1) / 2
        num_zero = np.sum((grads == 0.))
        self.assertAlmostEqual(expected_zero, num_zero)

    def test_target_params_update(self):
        tau = 0.2
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
            soft_target_tau=tau,
        )
        target_vf = algo.target_vf
        vf = algo.qf.value_function

        algo.train()

        orig_target_vals = target_vf.get_param_values()
        orig_vals = vf.get_param_values()
        algo.sess.run(algo.update_target_vf_op)
        new_target_vals = target_vf.get_param_values()

        for orig_target_val, orig_val, new_target_val in zip(
                orig_target_vals, orig_vals, new_target_vals):
            self.assertNpEqual(
                new_target_val,
                tau * orig_val + (1 - tau) * orig_target_val
            )

    def test_target_params_hard_update(self):
        tau = 1.
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
            soft_target_tau=tau,
        )
        target_vf = algo.target_vf
        vf = algo.qf.value_function

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
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
            soft_target_tau=tau,
        )
        target_vf = algo.target_vf
        vf = algo.qf.value_function
        self.assertNotEqual(target_vf.get_params_internal(),
                            vf.get_params_internal())

        # Make sure they're different to start
        random_values = [np.random.rand(*values.shape)
                         for values in vf.get_param_values()]
        vf.set_param_values(random_values)
        self.assertParamsNotEqual(target_vf, vf)
        algo.sess.run(algo.update_target_vf_op)
        self.assertParamsNotEqual(target_vf, vf)

    def test_policy_params_updated(self):
        tau = 0.2
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=1,
            epoch_length=3,
            soft_target_tau=tau,
            min_pool_size=2,
            eval_samples=0,
            max_path_length=5,
        )
        policy = algo.policy
        old_policy_values = policy.get_param_values()
        algo.train()
        new_policy_values = policy.get_param_values()

        self.assertNpArraysNotEqual(old_policy_values, new_policy_values)


class TestNormalizedAdvantageFunction(TFTestCase):
    """
    Test Q function used for NAF algorithm.
    """

    def setUp(self):
        super().setUp()
        self.env = TfEnv(CartpoleEnv())
        self.es = OUStrategy(env_spec=self.env.spec)

    def test_policy_params_are_in_q_params(self):
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
        )
        policy = algo.policy
        qf = algo.qf

        qf_params = qf.get_params_internal()
        policy_params = policy.get_params_internal()
        for param in policy.get_params_internal():
            self.assertTrue(param in qf_params)

    def test_vf_params_are_in_q_params(self):
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
        )
        vf = algo.qf.value_function
        qf = algo.qf

        qf_params = qf.get_params_internal()
        for param in vf.get_params_internal():
            self.assertTrue(param in qf_params)

    def test_target_vf_params_are_not_in_q_params(self):
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
        )
        target_vf = algo.target_vf
        qf = algo.qf

        qf_params = qf.get_params_internal()
        for param in target_vf.get_params_internal():
            self.assertFalse(param in qf_params)


if __name__ == '__main__':
    unittest.main()
