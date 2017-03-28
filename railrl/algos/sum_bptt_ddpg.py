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
        # all_rnn_actions = tf.concat(0, all_env_actions)

        import ipdb
        ipdb.set_trace()
        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            action_input=all_rnn_actions,
        )
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output)
        self.train_policy_op = tf.train.AdamOptimizer(
            self.policy_learning_rate).minimize(
            self.policy_surrogate_loss,
            var_list=self.policy.get_params_internal())
