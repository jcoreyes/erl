from collections import OrderedDict

import numpy as np

from railrl.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
    HorizonFedStateDistanceQLearning,
)
from railrl.torch import pytorch_util as ptu
from railrl.torch.algos.util import np_to_pytorch_batch


class VectorizedSdql(StateDistanceQLearning):
    def compute_rewards(self, obs, actions, next_obs, goals):
        return -np.abs(
            self.env.convert_obs_to_goals_pytorch(next_obs)
            - goals
        )


class VectorizedTauSdql(HorizonFedStateDistanceQLearning):
    def compute_rewards(self, obs, actions, next_obs, goals):
        diff = self.env.convert_obs_to_goals(next_obs) - goals
        weighted_diff = self.env.goal_dim_weights * diff
        return -np.abs(weighted_diff)


class VectorizedDeltaTauSdql(HorizonFedStateDistanceQLearning):
    """
    Just.... look at the reward
    """
    def __init__(
            self,
            *args,
            goal_chooser=None,
            sparse_rewards_learn_diff=True,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert self.qf_weight_decay == 0
        assert not self.clamp_q_target_values
        self.goal_chooser = goal_chooser
        self.sparse_rewards_learn_diff = sparse_rewards_learn_diff

    def compute_rewards(self, obs, actions, next_obs, goals):
        if self.sparse_reward:
            if self.sparse_rewards_learn_diff:
                return self.env.convert_obs_to_goals(
                    next_obs
                ) - goals
            else:
                return next_obs
        else:
            return next_obs - obs

    def get_train_dict(self, batch):
        batch = self._modify_batch_for_training(batch)
        rewards = batch['rewards']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']
        terminals = batch['terminals']
        num_steps_left = batch['num_steps_left']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs, goals, num_steps_left)
        # qf isn't really a qf anymore. It's a goal-conditioned (delta) model
        q_output = self.qf(obs, policy_actions, goals, num_steps_left)
        if self.sparse_reward:
            if self.sparse_rewards_learn_diff:
                predicted_distance_to_goal = q_output
            else:
                predicted_state = q_output
                predicted_goal = self.env.convert_obs_to_goals_pytorch(
                    predicted_state
                )
                predicted_distance_to_goal = predicted_goal - goals
        else:
            predicted_state = q_output + obs
            predicted_goal = self.env.convert_obs_to_goals_pytorch(
                predicted_state
            )
            predicted_distance_to_goal = predicted_goal - goals
        policy_loss = (predicted_distance_to_goal**2).mean()

        """
        Critic operations.
        """
        if self.only_do_sl:
            qf_loss = 0
            raw_qf_loss = ptu.FloatTensor([0])
            y_target = ptu.FloatTensor([0])
            y_pred = ptu.FloatTensor([0])
            bellman_errors = ptu.FloatTensor([0])
        else:
            next_actions = self.target_policy(
                next_obs,
                goals,
                num_steps_left - 1,
            )
            target_q_values = self.target_qf(
                next_obs,
                next_actions,
                goals,
                num_steps_left - 1,  # Important! Else QF will (probably) blow up
            )
            y_target = rewards + (1. - terminals) * target_q_values

            # noinspection PyUnresolvedReferences
            y_target = y_target.detach()
            y_pred = self.qf(obs, actions, goals, num_steps_left)
            bellman_errors = (y_pred - y_target) ** 2
            raw_qf_loss = self.qf_criterion(y_pred, y_target)

            qf_loss = raw_qf_loss

        target_policy_loss = ptu.FloatTensor([0])
        """
        Train optimal control policy
        """
        if self.goal_chooser is not None:
            goal = self.goal_chooser(obs)
            actions = self.policy(
                obs,
                goal,
                num_steps_left
            )
            final_state_predicted = self.qf(
                obs,
                actions,
                goal,
                num_steps_left,
            ) + obs
            rewards = self.goal_chooser.reward_function(final_state_predicted)
            raise NotImplementedError()

        """
        Do some SL supervision
        """
        for _ in range(self.num_sl_batches_per_rl_batch):
            batch = self.replay_buffer.random_batch_for_sl(
                self.batch_size,
                self.discount,
            )
            batch = np_to_pytorch_batch(batch)
            obs = batch['observations']
            actions = batch['actions']
            states_after_tau_steps = batch['states_after_tau_steps']
            goals = self.env.convert_obs_to_goals_pytorch(
                states_after_tau_steps
            )
            num_steps_left = batch['taus']
            y_pred = self.qf(obs, actions, goals, num_steps_left)

            y_target = states_after_tau_steps - obs
            sl_loss = self.qf_criterion(y_pred, y_target)
            qf_loss = qf_loss + sl_loss * self.sl_grad_weight

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
