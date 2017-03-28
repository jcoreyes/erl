"""
:author: Vitchyr Pong
"""
import numpy as np
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

    def _get_training_ops(self, **kwargs):
        ops = super()._get_training_ops(**kwargs)
        if None in ops:
            ops.remove(None)
        return ops

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
