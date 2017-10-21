import math
from collections import OrderedDict

import numpy as np
import torch

from railrl.torch import pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
    HorizonFedStateDistanceQLearning,
)


class VectorizedSdql(StateDistanceQLearning):
    def compute_rewards(self, obs, actions, next_obs, goal_states):
        return -np.abs(next_obs - goal_states)


class VectorizedTauSdql(HorizonFedStateDistanceQLearning):
    def compute_rewards(self, obs, actions, next_obs, goal_states):
        return -np.abs(next_obs - goal_states)


class VectorizedDeltaTauSdql(HorizonFedStateDistanceQLearning):
    """
    Just.... look at the reward
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.sparse_reward
        assert self.qf_weight_decay == 0

    def compute_rewards(self, obs, actions, next_obs, goal_states):
        return next_obs - obs

    def get_train_dict(self, batch):
        batch = self._modify_batch_for_training(batch)
        rewards = batch['rewards']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goal_states = batch['goal_states']
        terminals = batch['terminals']
        num_steps_left = batch['num_steps_left']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs, goal_states, num_steps_left)
        # qf isn't really a qf anymore. It's a goal-conditioned (delta) model
        q_output = self.qf(obs, policy_actions, goal_states, num_steps_left)
        predicted_distance_to_goal = q_output + obs - goal_states
        policy_loss = (predicted_distance_to_goal**2).mean()

        """
        Critic operations.
        """
        next_actions = self.target_policy(
            next_obs,
            goal_states,
            num_steps_left - 1,
        )
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
            goal_states,
            num_steps_left - 1,  # Important! Else QF will (probably) blow up
        )
        if self.clamp_q_target_values:
            target_q_values = torch.clamp(target_q_values, -math.inf, 0)
        y_target = rewards + (1. - terminals) * target_q_values

        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions, goal_states, num_steps_left)
        bellman_errors = (y_pred - y_target) ** 2
        raw_qf_loss = self.qf_criterion(y_pred, y_target)

        qf_loss = raw_qf_loss

        target_policy_loss = ptu.FloatTensor([0])

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('Unregularized QF Loss', raw_qf_loss),
            ('QF Loss', qf_loss),
            ('Target Policy Loss', target_policy_loss),
        ])
