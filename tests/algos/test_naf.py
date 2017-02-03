import math
import unittest

import numpy as np
import tensorflow as tf
from railrl.misc.tf_test_case import TFTestCase
from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF

from railrl.algos.naf import NAF
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
        L_param_gen = af._L_computer
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

    def test_regularized_variables_are_correct(self):
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
        )
        qf = algo.qf
        vars = qf.get_params_internal(regularizable=True)
        expected_vars = [
            v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'qf')
            if 'weight' in v.name
        ]

        for v in vars:
            self.assertTrue(v in expected_vars)
        for v in expected_vars:
            self.assertTrue(v in vars)

    def test_l2_regularization(self):
        algo = NAF(
            self.env,
            self.es,
            QuadraticNAF(name_or_scope='qf', env_spec=self.env.spec),
            n_epochs=0,
        )
        Q_weight_norm = algo.Q_weights_norm
        expected_vars = [
            v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'qf')
            if 'weight' in v.name
        ]
        variable_values = self.sess.run(expected_vars)
        expected_l2 = sum(
            np.linalg.norm(value)**2 / 2 for value in variable_values
        )
        l2 = self.sess.run(Q_weight_norm)

        self.assertAlmostEqual(l2, expected_l2, delta=1e-4)

if __name__ == '__main__':
    unittest.main()
