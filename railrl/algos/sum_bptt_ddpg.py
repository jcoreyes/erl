"""
:author: Vitchyr Pong
"""
import tensorflow as tf

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.core.rnn.rnn import OutputStateRnn


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
        rnn_inputs = tf.unpack(self._rnn_inputs_ph, axis=1)
        self._rnn_init_state_ph = self.policy.get_init_state_placeholder()

        self._rnn_cell_scope.reuse_variables()
        wrapper_rnn_cell = OutputStateRnn(self._rnn_cell)
        with tf.variable_scope(self._rnn_cell_scope):
            self._rnn_outputs, self._rnn_final_state = tf.nn.rnn(
                wrapper_rnn_cell,
                rnn_inputs,
                initial_state=self._rnn_init_state_ph,
                dtype=tf.float32,
                scope=self._rnn_cell_scope,
            )
        # self._final_rnn_action = self._rnn_outputs[-1]
        all_env_actions, all_write_actions = zip(*self._rnn_outputs)
        all_rnn_actions = (
            tf.concat(0, all_env_actions),
            tf.concat(0, all_write_actions),
        )

        all_memory_obs_list = (
            [self._rnn_init_state_ph] + list(all_write_actions[:-1])
        )
        all_rnn_observations = (
            tf.concat(0, rnn_inputs),
            tf.concat(0, all_memory_obs_list),
        )

        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            action_input=all_rnn_actions,
            observation_input=all_rnn_observations,
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
        first_obs = self._get_time_step(obs, 0)
        env_obs, _ = obs
        initial_memory_obs = first_obs[1]
        return {
            self._rnn_inputs_ph: env_obs,
            self._rnn_init_state_ph: initial_memory_obs,
        }
