import torch
from collections import OrderedDict
import railrl.torch.pytorch_util as ptu
from railrl.data_management.path_builder import PathBuilder
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.misc.ml_util import ConstantSchedule
from railrl.state_distance.rollout_util import MultigoalSimplePathSampler
from railrl.torch.core import np_to_pytorch_batch

from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
import numpy as np
from torch.optim import Adam
from torch import nn


class BetaLearning(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            exploration_policy,
            beta_q,
            beta_q2,
            beta_v,
            policy,
            goal_reached_epsilon=1e-3,
            learning_rate=1e-3,
            prioritized_replay=False,

            policy_and_target_update_period=2,
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,
            soft_target_tau=0.005,
            per_beta_schedule=None,
            **kwargs
    ):
        super().__init__(env, exploration_policy, **kwargs)
        self.eval_sampler = MultigoalSimplePathSampler(
            env=self.env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            tau_sampling_function=lambda: 0,
            goal_sampling_function=self.env.sample_goal_for_rollout,
            cycle_taus_for_rollout=False,
            render=self.render_during_eval
        )
        self.goal_reached_epsilon = goal_reached_epsilon
        self.beta_q = beta_q
        self.beta_v = beta_v
        self.beta_q2 = beta_q2
        self.target_beta_q = self.beta_q.copy()
        self.target_beta_q2 = self.beta_q2.copy()
        self.policy = policy
        self.target_policy = policy
        self.prioritized_replay = prioritized_replay
        self.policy_and_target_update_period = policy_and_target_update_period
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        self.soft_target_tau = soft_target_tau
        if per_beta_schedule is None:
            per_beta_schedule = ConstantSchedule(1.0)
        self.per_beta_schedule = per_beta_schedule

        self.beta_q_optimizer = Adam(
            self.beta_q.parameters(), lr=learning_rate
        )
        self.beta_q2_optimizer = Adam(
            self.beta_q2.parameters(), lr=learning_rate
        )
        self.beta_v_optimizer = Adam(
            self.beta_v.parameters(), lr=learning_rate
        )
        self.policy_optimizer = Adam(
            self.policy.parameters(),
            lr=learning_rate,
        )
        self.q_criterion = nn.BCELoss()
        self.v_criterion = nn.BCELoss()

        # For the multitask env
        self._rollout_tau = np.array([0])
        self._current_path_goal = None

    def _do_training(self):
        np_batch = self.replay_buffer.random_batch(
            self.batch_size,
            beta=self.per_beta_schedule.get_value(
                self._n_train_steps_total,
            ),
        )
        next_obs = np_batch['next_observations']
        goals = np_batch['goals']
        terminals = np_batch['terminals']
        indices = np_batch['indices']
        events = self.detect_event(next_obs, goals)
        np_batch['events'] = events
        np_batch['terminals'] = 1 - (1 - terminals) * (1 - events)
        batch = np_to_pytorch_batch(np_batch)

        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        num_steps_left = batch['num_steps_left']
        goals = batch['goals']
        events = batch['events']

        next_actions = self.target_policy(
            observations=next_obs,
            goals=goals,
            num_steps_left=num_steps_left,  #TODO: decrement
        )
        noise = torch.normal(
            torch.zeros_like(next_actions),
            self.target_policy_noise,
        )
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise
        next_beta_1 = self.target_beta_q(
            observations=next_obs,
            actions=noisy_next_actions,
            goals=goals,
            num_steps_left=num_steps_left,  #TODO: decrement
        )
        next_beta_2 = self.target_beta_q2(
            observations=next_obs,
            actions=noisy_next_actions,
            goals=goals,
            num_steps_left=num_steps_left,  #TODO: decrement
        )
        # TODO: maybe add twin back in
        # next_beta = torch.min(next_beta_1, next_beta_2)
        next_beta = next_beta_1
        targets = (
            terminals * events
            + (1 - terminals) * self.discount * next_beta
        ).detach()

        predictions = self.beta_q(obs, actions, goals, num_steps_left)
        if self.prioritized_replay:
            weights = ptu.from_numpy(np_batch['is_weights']).float()
            self.q_criterion.weight = weights
            priorities = ptu.get_numpy(torch.abs(predictions - targets))
            self.replay_buffer.update_priorities(indices, priorities)
        beta_q_loss = self.q_criterion(predictions, targets)

        predictions2 = self.beta_q2(obs, actions, goals, num_steps_left)
        beta_q2_loss = self.q_criterion(predictions2, targets)

        self.beta_q_optimizer.zero_grad()
        beta_q_loss.backward()
        self.beta_q_optimizer.step()

        self.beta_q2_optimizer.zero_grad()
        beta_q2_loss.backward()
        self.beta_q2_optimizer.step()

        beta_v_loss = self.v_criterion(
            self.beta_v(next_obs, goals, num_steps_left),  #TODO: decrement
            targets
        )
        self.beta_v_optimizer.zero_grad()
        beta_v_loss.backward()
        self.beta_v_optimizer.step()

        policy_actions = self.policy(obs, goals, num_steps_left)
        q_output = self.beta_q(
            observations=obs,
            actions=policy_actions,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        # TODO: this definitely gets flattened sometimes...
        policy_loss = - q_output.mean()
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(
                self.policy, self.target_policy, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.beta_q, self.target_beta_q, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.beta_q2, self.target_beta_q2, self.soft_target_tau
            )
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Beta Q Loss'] = np.mean(ptu.get_numpy(
                beta_q_loss
            ))
            self.eval_statistics['Beta V Loss'] = np.mean(ptu.get_numpy(
                beta_v_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Beta Q Targets',
                ptu.get_numpy(targets),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Beta Q Predictions',
                ptu.get_numpy(predictions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Beta Q1 - Q2',
                ptu.get_numpy(next_beta_1 - next_beta_2),
            ))
            real_goal = np.array([0., 4.])
            is_real_goal = (np_batch['goals'] == real_goal).all(axis=1)
            goal_is_corner = (np.abs(np_batch['goals']) == 4).all(axis=1)
            self.eval_statistics['Event Prob'] = np_batch['events'].mean()
            self.eval_statistics['Training Goal is Real Goal Prob'] = (
                is_real_goal.mean()
            )
            self.eval_statistics['Training Goal is Corner'] = (
                goal_is_corner.mean()
            )

    def detect_event(self, next_obs, goals):
        diff = self.env.convert_obs_to_goals(next_obs) - goals
        goal_reached = (
            np.linalg.norm(diff, axis=1, keepdims=True)
            <= self.goal_reached_epsilon
        )
        return goal_reached

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            beta_q=self.beta_q,
            beta_v=self.beta_v,
            policy=self.policy,
            exploration_policy=self.exploration_policy,
        )
        return snapshot

    @property
    def networks(self):
        return [
            self.policy,
            self.beta_q,
            self.beta_q2,
            self.beta_v,
            self.target_policy,
            self.target_beta_q,
            self.target_beta_q2,
        ]

    # Multitask env code

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            num_steps_left=self._rollout_tau,
            goals=self._current_path_goal,
        )

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self._current_path_goal = self.env.sample_goal_for_rollout()
        # self._current_path_goal = (
        #     self._current_path_goal + np.random.normal(
        #         size=self._current_path_goal.shape
        #     )
        # )
        self.training_env.set_goal(self._current_path_goal)
        return self.training_env.reset()

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
            self._current_path_goal,
            self._rollout_tau,
        )
