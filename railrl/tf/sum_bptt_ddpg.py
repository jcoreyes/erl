"""
:author: Vitchyr Pong
"""
import numpy as np
import tensorflow as tf

from railrl.tf.bptt_ddpg import BpttDDPG
from railrl.tf.core.rnn.rnn import OutputStateRnn


class SumBpttDDPG(BpttDDPG):
    """
    BpttDDPT but sum the Q values as you unroll the policy.
    """

    def _init_policy_ops(self):
        self._rnn_inputs_ph = tf.placeholder(
            tf.float32,
            [None, self._num_bptt_unrolls, self._env_obs_dim],
            name='rnn_time_inputs',
        )
        all_env_obs = tf.unstack(self._rnn_inputs_ph, axis=1)
        self._rnn_init_state_ph = self.policy.get_init_state_placeholder()

        self._rnn_cell_scope.reuse_variables()
        wrapper_rnn_cell = OutputStateRnn(self._rnn_cell)
        with tf.variable_scope(self._rnn_cell_scope):
            self._rnn_outputs, self._rnn_final_state = (
                tf.contrib.rnn.static_rnn(
                    wrapper_rnn_cell,
                    all_env_obs,
                    initial_state=self._rnn_init_state_ph,
                    dtype=tf.float32,
                    scope=self._rnn_cell_scope,
                )
            )
        all_env_actions, all_write_actions = zip(*self._rnn_outputs)

        all_rnn_actions = (
            tf.concat(axis=0, values=all_env_actions),
            tf.concat(axis=0, values=all_write_actions),
        )

        self.qf_for_policy_observation_ph = (
            tf.placeholder(
                tf.float32,
                [None, self._env_obs_dim],
                "test",
            ),
            tf.placeholder(
                tf.float32,
                [None, self.observation_dim - self._env_obs_dim],
                "test2",
            ),
        )

        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            action_input=all_rnn_actions,
            observation_input=self.qf_for_policy_observation_ph,
        )
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output)
        self.train_policy_op = tf.train.AdamOptimizer(
            self.policy_learning_rate).minimize(
            self.policy_surrogate_loss,
            var_list=self.policy.get_params_internal())

    def _policy_feed_dict(self, obs, **kwargs):
        """
        :param obs: See output of `self._split_flat_action`.
        :return: Feed dictionary for policy training TensorFlow ops.
        """
        _, initial_memory_obs = _get_time_step(obs, 0)
        env_obs, memory_obs = obs
        """
        The observations need to be stacked in the same order that the q
        function stacks its own actions. The swap axes is necessary to make
        these two match.
        """
        all_obs = (
            np.concatenate(np.swapaxes(env_obs, 0, 1), axis=0),
            np.concatenate(np.swapaxes(memory_obs, 0, 1), axis=0),
        )
        return {
            self._rnn_inputs_ph: env_obs,
            self._rnn_init_state_ph: initial_memory_obs,
            self.qf_for_policy_observation_ph: all_obs,
        }
