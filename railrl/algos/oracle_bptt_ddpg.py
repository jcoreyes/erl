"""
:author: Vitchyr Pong
"""
from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.data_management.ocm_subtraj_replay_buffer import (
    OcmSubtrajReplayBuffer
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
                          debug_info=None):
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
            debug_info=debug_info,
        )

        policy_feed = self._policy_feed_dict(obs)
        feed.update(policy_feed)
        return feed

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                      debug_info=None):
        indices = debug_info[:, 0]
        target_one_hots = special.to_onehot_n(
            indices,
            self.env.wrapped_env.action_space.flat_dim,
        )
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
            debug_info=batch['debug_numbers'],
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
