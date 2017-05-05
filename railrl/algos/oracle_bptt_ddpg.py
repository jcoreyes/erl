"""
:author: Vitchyr Pong
"""
import numpy as np
import tensorflow as tf

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.data_management.ocm_subtraj_replay_buffer import (
    OcmSubtrajReplayBuffer
)
from rllab.misc import special, logger


class OracleUnrollBpttDDPG(BpttDDPG):
    """
    If the environment's loss is only a function of the final output,
    then you need to unroll the current policy to get the final output. This
    is what this class adds.
    """

    def __init__(self, *args, unroll_through_target_policy=False, **kwargs):
        kwargs['replay_buffer_class'] = OcmSubtrajReplayBuffer
        self.unroll_through_target_policy = unroll_through_target_policy
        super().__init__(*args, **kwargs)

    def _init_qf_loss_and_train_ops(self):
        self.qf_loss = tf.zeros([0], tf.float32)
        self.train_qf_op = None

    def _oracle_qf_feed_dict_for_policy_from_batch(self, batch):
        all_target_numbers = batch['target_numbers']
        all_times = batch['times']

        target_numbers = all_target_numbers[:, -1]
        times = all_times[:, -1]

        batch_size = len(times)
        sequence_lengths = np.squeeze(self.env.horizon - 1 - times)
        target_one_hots = special.to_onehot_n(
            target_numbers,
            self.env.wrapped_env.action_space.flat_dim,
        )
        rest_of_obs = np.zeros(
            [
                batch_size,
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        feed_dict = {
            self.qf.sequence_length_placeholder: sequence_lengths,
            self.qf.rest_of_obs_placeholder: rest_of_obs,
            self.qf.target_labels: target_one_hots,
        }
        return feed_dict

    def _policy_feed_dict_from_batch(self, batch):
        policy_feed = super()._policy_feed_dict_from_batch(batch)
        policy_feed.update(self._oracle_qf_feed_dict_for_policy_from_batch(
            batch
        ))
        return policy_feed

    def _init_policy_ops(self):
        super()._init_policy_ops()
        if self.unroll_through_target_policy:
            self.qf_with_action_input = self.qf.get_weight_tied_copy(
                action_input=self._final_rnn_augmented_action,
                observation_input=self._final_rnn_augmented_input,
                policy=self.target_policy,
            )
        else:
            self.qf_with_action_input = self.qf.get_weight_tied_copy(
                action_input=self._final_rnn_augmented_action,
                observation_input=self._final_rnn_augmented_input,
            )

    def _init_policy_loss_and_train_ops(self):
        if self._bpt_bellman_error_weight > 0.:
            logger.log("Setting bpt_bellman_error_weight since QF is oracle.")
            self._bpt_bellman_error_weight = 0
        super()._init_policy_loss_and_train_ops()

    def _qf_statistic_names_and_ops(self):
        return [('None', None)]

    def _statistic_names_and_ops(self):
        names_and_ops = [
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('PolicyOutput', self.policy.output),
            ('TargetPolicyOutput', self.target_policy.output),
            ('QfOutput', self.qf_with_action_input.output),
        ]
        return names_and_ops
