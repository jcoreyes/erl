import torch
import numpy as np
from collections import OrderedDict

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    MultigoalSimplePathSampler
)
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.misc.rllab_util import split_paths_to_dict
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.ddpg import DDPG
from rllab.misc import logger


class HER(DDPG):
    def __init__(
            self,
            env,
            qf,
            policy,
            exploration_policy,
            replay_buffer,
            eval_sampler=None,
            epsilon=1e-4,
            num_steps_per_eval=1000,
            max_path_length=1000,
            terminate_when_goal_reached=False,
            **kwargs
    ):
        assert isinstance(replay_buffer, SplitReplayBuffer)
        assert isinstance(replay_buffer.train_replay_buffer, HerReplayBuffer)
        assert isinstance(replay_buffer.validation_replay_buffer,
                          HerReplayBuffer)
        assert eval_sampler is None
        eval_sampler = MultigoalSimplePathSampler(
            env=env,
            policy=policy,
            max_samples=num_steps_per_eval,
            max_path_length=max_path_length,
            discount_sampling_function=self._sample_discount_for_rollout,
            goal_sampling_function=self._sample_goal_state_for_rollout,
            cycle_taus_for_rollout=False,
        )
        super().__init__(
            env, qf, policy, exploration_policy,
            replay_buffer=replay_buffer,
            eval_sampler=eval_sampler,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=max_path_length,
            **kwargs
        )
        self.epsilon = epsilon
        assert self.qf_weight_decay == 0
        assert not self.optimize_target_policy
        assert self.residual_gradient_weight == 0
        self.terminate_when_goal_reached = terminate_when_goal_reached

    def _sample_goal_state_for_rollout(self):
        return self.env.sample_goal_state_for_rollout()

    def _sample_discount_for_rollout(self):
        return self.discount

    def get_batch(self, training=True):
        batch = super().get_batch(training=training)
        # The original HER paper says to use obs - goal state, but that doesn't
        # really make sense
        # diff = torch.abs(
        #     self.env.convert_obs_to_goal_states(batch['next_observations'])
        #     - self.env.convert_obs_to_goal_states(batch['goal_states'])
        # )
        # diff_sum = diff.sum(dim=1, keepdim=True)
        # goal_not_reached = (diff_sum >= self.epsilon).float()
        # batch['rewards'] = - goal_not_reached
        # if self.terminate_when_goal_reached:
        #     batch['terminals'] = 1 - (1 - batch['terminals']) * goal_not_reached
        batch['rewards'] = self.env.compute_her_reward_pytorch(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            batch['goal_states'],
        )
        return batch

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = self.env.convert_obs_to_goal_states(batch['goal_states'])

        """
        Policy operations.
        """
        policy_actions = self.policy(obs, goals)
        q_output = self.qf(obs, policy_actions, goals)
        policy_loss = - q_output.mean()

        """
        Critic operations.
        """
        next_actions = self.target_policy(next_obs, goals)
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
            goals,
        )
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        y_target = torch.clamp(y_target, -1./(1-self.discount), 0)
        y_pred = self.qf(obs, actions, goals)
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('Unregularized QF Loss', qf_loss),
            ('QF Loss', qf_loss),
            ('Target Policy Loss', ptu.FloatTensor([0])),       # For DDPG
            ('Target Policy Loss Mean', ptu.FloatTensor([0])),  # For DDPG
        ])

    @staticmethod
    def paths_to_batch(paths):
        """
        Converts
        [
            {
                'rewards': [1, 2],
                'goal_states': [3, 4],
                ...
            },
            {
                'rewards': [5, 6],
                'goal_states': [7, 8],
                ...
            },
        ]
        into
        {
            'rewards': [1, 2, 5, 6],
            'goal_states': [3, 4, 7, 8],
            ...
        },

        :param paths:
        :return:
        """
        np_batch = split_paths_to_dict(paths)
        goal_states = [path["goal_states"] for path in paths]
        np_batch['goal_states'] = np.vstack(goal_states)
        return np_to_pytorch_batch(np_batch)

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self.goal_state = self._sample_goal_state_for_rollout()
        self.training_env.set_goal(self.goal_state)
        self.exploration_policy.set_goal(self.goal_state)
        return self.training_env.reset()

    def _handle_step(
            self,
            num_paths_total,
            observation,
            action,
            reward,
            terminal,
            agent_info,
            env_info,
    ):
        if num_paths_total % self.save_exploration_path_period == 0:
            self._current_path.add_all(
                observations=self.obs_space.flatten(observation),
                rewards=reward,
                terminals=terminal,
                actions=self.action_space.flatten(action),
                agent_infos=agent_info,
                env_infos=env_info,
                goal_states=self.goal_state,  # <- this is where we save the
                                              # current goal state
            )

        self.replay_buffer.add_sample(
            observation,
            action,
            reward,
            terminal,
            agent_info=agent_info,
            env_info=env_info,
            goal_state=self.goal_state,
        )

    def _handle_rollout_ending(
            self,
            n_steps_total,
            final_obs,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path.add_all(
            final_observation=final_obs,
            increment_path_length=False,
        )
        self.replay_buffer.terminate_episode(
            final_obs,
            terminal,
            agent_info=agent_info,
            env_info=env_info,
            goal_state=self.goal_state,
        )
