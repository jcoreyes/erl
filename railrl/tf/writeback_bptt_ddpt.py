"""
:author: Vitchyr Pong
"""
import tensorflow as tf

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.core import tf_util


class WritebackBpttDDPG(BpttDDPG):
    """
    Same as BPTT, but allow the loss of the critic to backpropagate through
    to the critic via the write action.
    """
    def _init_qf_ops(self):
        qf_env_action, _ = self.qf.action_input
        _, policy_write_action = self.policy.output
        new_action = qf_env_action, policy_write_action
        self.qf_with_memory_input = self.qf.get_weight_tied_copy(
            action_input=new_action
        )
        self.ys = (
            self.rewards_placeholder +
            (1. - self.terminals_placeholder) *
            self.discount * self.target_qf.output)
        self.qf_loss = tf_util.mse(self.ys, self.qf_with_memory_input.output)
        self.Q_weights_norm = tf.reduce_sum(
            tf.stack(
                [tf.nn.l2_loss(v)
                 for v in
                 self.qf.get_params_internal(regularizable=True)]
            ),
            name='weights_norm'
        )
        self.qf_total_loss = (
            self.qf_loss + self.qf_weight_decay * self.Q_weights_norm)
        self.train_qf_op = tf.train.AdamOptimizer(
            self.qf_learning_rate).minimize(
            self.qf_total_loss,
            var_list=self.qf.get_params() + self.policy.get_params(),
        )

    def _policy_feed_dict(self, obs, **kwargs):
        """
        :param obs: See output of `self._split_flat_action`.
        :return: Feed dictionary for policy training TensorFlow ops.
        """
        last_obs = _get_time_step(obs, -1)
        feed_dict = super()._policy_feed_dict(obs, **kwargs)
        feed_dict[self.policy.observation_input] = last_obs
        return feed_dict
