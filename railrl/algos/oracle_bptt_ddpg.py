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

    def _init_qf_ops(self):
        self.train_qf_op = None

    def _get_training_ops(self, **kwargs):
        ops = super()._get_training_ops(**kwargs)
        if None in ops:
            ops.remove(None)
        return ops

    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                          target_numbers=None, times=None):
        actions = self._split_flat_actions(actions)
        obs = self._split_flat_obs(obs)
        next_obs = self._split_flat_obs(next_obs)

        # rewards and terminals both have shape [batch_size x sub_traj_length],
        # but they really just need to be [batch_size x 1]. Right now we only
        # care about the reward/terminal at the very end since we're only
        # computing the rewards for the last time step.
        qf_terminals = terminals[:, -1:]
        qf_rewards = rewards[:, -1:]
        # For obs/actions, we only care about the last time step for the critic.
        qf_obs = self._get_time_step(obs, t=-1)
        qf_actions = self._get_time_step(actions, t=-1)
        qf_next_obs = self._get_time_step(next_obs, t=-1)
        feed = self._qf_feed_dict(
            qf_rewards,
            qf_terminals,
            qf_obs,
            qf_actions,
            qf_next_obs,
            target_numbers=target_numbers,
            times=times,
        )

        policy_feed = self._policy_feed_dict(obs)
        feed.update(policy_feed)
        return feed

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                      target_numbers=None, times=None):
        indices = target_numbers[:, 0]
        target_one_hots = special.to_onehot_n(
            indices,
            self.env.wrapped_env.action_space.flat_dim,
        )
        batch_size = len(rewards)
        rest_of_obs = np.zeros(
            [
                batch_size,
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        rest_of_obs[:, :, 0] = 1
        return {
            self.rewards_placeholder: rewards,
            self.terminals_placeholder: terminals,
            self.qf.observation_input: obs,
            self.qf.action_input: actions,
            self.target_qf.observation_input: next_obs,
            self.target_policy.observation_input: next_obs,
            self.qf.target_labels: target_one_hots,
        }

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
        return [
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('Ys', self.policy_surrogate_loss),
            ('PolicyOutput', self.policy_surrogate_loss),
            ('TargetPolicyOutput', self.policy_surrogate_loss),
            ('QfOutput', self.policy_surrogate_loss),
            ('TargetQfOutput', self.policy_surrogate_loss),
        ]

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
        indices = target_numbers[:, 0]
        target_one_hots = special.to_onehot_n(
            indices,
            self.env.wrapped_env.action_space.flat_dim,
        )
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
        if (isinstance(self.qf, OracleQFunction) or
                isinstance(self.qf, OracleUnrollQFunction)):
            return {
                self.rewards_placeholder: rewards,
                self.terminals_placeholder: terminals,
                self.qf.observation_input: obs,
                self.qf.action_input: actions,
                self.target_qf.observation_input: next_obs,
                self.target_policy.observation_input: next_obs,
                self.qf.target_labels: target_one_hots,
                self.qf.sequence_length_placeholder: sequence_lengths,
                self.qf.rest_of_obs_placeholder: rest_of_obs,
            }
        elif isinstance(self.qf, HintMlpMemoryQFunction):
            return {
                self.rewards_placeholder: rewards,
                self.terminals_placeholder: terminals,
                self.qf.observation_input: obs,
                self.qf.action_input: actions,
                self.target_qf.observation_input: next_obs,
                self.target_policy.observation_input: next_obs,
                self.qf.hint_input: target_one_hots,
            }
