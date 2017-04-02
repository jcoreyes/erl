"""
:author: Vitchyr Pong
"""
import numpy as np
import tensorflow as tf
from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.data_management.ocm_subtraj_replay_buffer import (
    OcmSubtrajReplayBuffer
)
from railrl.qfunctions.memory.hint_mlp_memory_qfunction import (
    HintMlpMemoryQFunction
)
from railrl.qfunctions.memory.oracle_qfunction import OracleQFunction
from railrl.qfunctions.memory.oracle_unroll_qfunction import (
    OracleUnrollQFunction
)
from rllab.misc import special

TARGET_PREFIX = "target_"


class OracleBpttDDPG(BpttDDPG):
    """
    BpttDDPT but with an oracle QFunction.
    """
    def __init__(self, *args, **kwargs):
        kwargs['replay_buffer_class'] = OcmSubtrajReplayBuffer
        super().__init__(*args, **kwargs)

    @property
    def qf_is_trainable(self):
        return len(self.qf.get_params()) > 0

    def _init_qf_ops(self):
        if self.qf_is_trainable:
            super()._init_qf_ops()
        else:
            self.train_qf_op = None

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                      target_numbers=None, times=None):
        indices = target_numbers[:, 0]
        target_one_hots = special.to_onehot_n(
            indices,
            self.env.wrapped_env.action_space.flat_dim,
        )
        qf_feed_dict = super()._qf_feed_dict(
            rewards=rewards,
            terminals=terminals,
            obs=obs,
            actions=actions,
            next_obs=next_obs,
        )
        qf_feed_dict[self.qf.target_labels] = target_one_hots
        qf_feed_dict[self.target_qf.target_labels] = target_one_hots
        return qf_feed_dict

    def _update_feed_dict_from_batch(self, batch):
        return self._update_feed_dict(
            rewards=batch['rewards'],
            terminals=batch['terminals'],
            obs=batch['observations'],
            actions=batch['actions'],
            next_obs=batch['next_observations'],
            target_numbers=batch['target_numbers'],
            times=batch['times'],
        )

    def _statistic_names_and_ops(self):
        names_and_ops = [
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('Ys', self.policy_surrogate_loss),
            ('PolicyOutput', self.policy_surrogate_loss),
            ('TargetPolicyOutput', self.policy_surrogate_loss),
            ('QfOutput', self.policy_surrogate_loss),
            ('TargetQfOutput', self.policy_surrogate_loss),
        ]
        if self.qf_is_trainable:
            names_and_ops.append(
                ('QfLoss', self.qf_loss),
            )
        return names_and_ops

    def _update_feed_dict_from_path(self, paths):
        eval_pool = self._replay_buffer_class(
            len(paths) * self.max_path_length,
            self.env,
            self._num_bptt_unrolls,
            )
        for path in paths:
            eval_pool.add_trajectory(path)

        batch = eval_pool.get_all_valid_subtrajectories()
        return self._update_feed_dict_from_batch(batch)


class OracleUnrollBpttDDPG(OracleBpttDDPG):
    def __init__(self, *args, unroll_through_target_policy=False, **kwargs):
        # TODO(vitchyr): pass this in
        self.unroll_through_target_policy = unroll_through_target_policy
        super().__init__(*args, **kwargs)

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                      target_numbers=None, times=None):
        sequence_lengths = np.squeeze(self.env.horizon - times[:, -1])
        batch_size = len(rewards)
        rest_of_obs = np.zeros(
            [
                batch_size,
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        rest_of_obs[:, :, 0] = 1
        qf_feed_dict = super()._qf_feed_dict(
            rewards=rewards,
            terminals=terminals,
            obs=obs,
            actions=actions,
            next_obs=next_obs,
            target_numbers=target_numbers,
            times=times,
        )
        qf_feed_dict[self.qf.sequence_length_placeholder] = sequence_lengths
        qf_feed_dict[self.qf.rest_of_obs_placeholder] = rest_of_obs
        return qf_feed_dict

    def _init_policy_ops(self):
        self._rnn_inputs_ph = tf.placeholder(
            tf.float32,
            [None, self._num_bptt_unrolls, self._env_obs_dim],
            name='rnn_time_inputs',
        )
        rnn_inputs = tf.unstack(self._rnn_inputs_ph, axis=1)
        self._rnn_init_state_ph = self.policy.get_init_state_placeholder()

        # This call isn't REALLY necessary since OracleUnrollQFunction will
        # probably already make a call this scope's reuse_variable(),
        # but it's good practice to have this here.
        self._rnn_cell_scope.reuse_variables()
        self._rnn_outputs, self._rnn_final_state = tf.contrib.rnn.static_rnn(
            self._rnn_cell,
            rnn_inputs,
            initial_state=self._rnn_init_state_ph,
            dtype=tf.float32,
            scope=self._rnn_cell_scope,
        )
        self._final_rnn_output = self._rnn_outputs[-1]
        self._final_rnn_action = (
            self._final_rnn_output,
            self._rnn_final_state,
        )
        if self.unroll_through_target_policy:
            self.qf_with_action_input = self.qf.get_weight_tied_copy(
                action_input=self._final_rnn_action,
                policy=self.target_policy,
            )
        else:
            self.qf_with_action_input = self.qf.get_weight_tied_copy(
                action_input=self._final_rnn_action,
            )
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output)
        if self._freeze_hidden:
            trainable_policy_params = self.policy.get_params(env_only=True)
        else:
            trainable_policy_params = self.policy.get_params_internal()
        self.train_policy_op = tf.train.AdamOptimizer(
            self.policy_learning_rate
        ).minimize(
            self.policy_surrogate_loss,
            var_list=trainable_policy_params,
        )
