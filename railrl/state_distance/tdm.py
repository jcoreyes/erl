"""
New implementation of state distance q learning.
"""
import abc
import numpy as np

from railrl.data_management.path_builder import PathBuilder
from railrl.misc.ml_util import ConstantSchedule
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.state_distance.exploration import MakeUniversal
from railrl.state_distance.rollout_util import MultigoalSimplePathSampler


class TemporalDifferenceModel(TorchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            max_tau=10,
            epoch_max_tau_schedule=None,
            sample_train_goals_from='replay_buffer',
            sample_rollout_goals_from='environment',
            vectorized=True,
            cycle_taus_for_rollout=False,
    ):
        """

        :param epoch_max_tau_schedule: A schedule for the maximum planning
        horizon tau.
        """
        if epoch_max_tau_schedule is None:
            epoch_max_tau_schedule = ConstantSchedule(max_tau)

        self.max_tau = max_tau
        self.epoch_max_tau_schedule = epoch_max_tau_schedule
        self.sample_train_goals_from = sample_train_goals_from
        self.sample_rollout_goals_from = sample_rollout_goals_from
        self.vectorized = vectorized
        self.cycle_taus_for_rollout = cycle_taus_for_rollout
        self.goal = None

        self.policy = MakeUniversal(self.policy)
        self.eval_policy = MakeUniversal(self.eval_policy)
        self.exploration_policy = MakeUniversal(self.exploration_policy)
        self.eval_sampler = MultigoalSimplePathSampler(
            env=self.env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            discount_sampling_function=self._sample_max_tau_for_rollout,
            goal_sampling_function=self._sample_goal_for_rollout,
            cycle_taus_for_rollout=self.cycle_taus_for_rollout,
        )

    def _start_epoch(self, epoch):
        self.max_tau = self.epoch_max_tau_schedule.get_value(epoch)
        super()._start_epoch(epoch)

    def get_batch(self, training=True):
        if self.replay_buffer_is_split:
            replay_buffer = self.replay_buffer.get_replay_buffer(training)
        else:
            replay_buffer = self.replay_buffer
        batch = replay_buffer.random_batch(self.batch_size)

        """
        Update the goal states/rewards
        """
        num_steps_left = np.random.randint(
            0, self.max_tau + 1, (self.batch_size, 1)
        )
        terminals = 1 - (1 - batch['terminals']) * (num_steps_left != 0)
        batch['terminals'] = terminals

        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        if self.sample_train_goals_from == 'her':
            goals = batch['goals']
        else:
            goals = self._sample_goals_for_training()
        rewards = self.compute_rewards_np(
            obs,
            actions,
            next_obs,
            goals,
        )
        batch['rewards'] = rewards * terminals

        """
        Update the observations
        """
        batch['observations'] = np.hstack((
            batch['observations'],
            batch['goals'],
            num_steps_left,
        ))
        batch['next_observations'] = np.hstack((
            batch['next_observations'],
            batch['goals'],
            num_steps_left - 1,
        ))

        return np_to_pytorch_batch(batch)

    def compute_rewards_np(self, obs, actions, next_obs, goals):
        if self.vectorized:
            if hasattr(self.env, 'compute_rewards_vectorized'):
                self.env.compute_rewards_vectorized(obs, actions, next_obs, goals)
            else:
                diff = self.env.convert_obs_to_goals(next_obs) - goals
                return -np.abs(diff) * self.reward_scale
        else:
            return self.env.compute_rewards(
                obs,
                actions,
                next_obs,
                goals,
            )

    @property
    def train_buffer(self):
        if self.replay_buffer_is_split:
            return self.replay_buffer.get_replay_buffer(trainig=True)
        else:
            return self.replay_buffer

    def _sample_goals_for_training(self):
        if self.sample_train_goals_from == 'environment':
            return self.env.sample_goals(self.batch_size)
        elif self.sample_train_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(self.batch_size)
            obs = batch['observations']
            return self.env.convert_obs_to_goals(obs)
        elif self.sample_train_goals_from == 'her':
            raise Exception("Take samples from replay buffer.")
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_train_goals_from
            ))

    def _sample_goal_for_rollout(self):
        if self.sample_rollout_goals_from == 'environment':
            return self.env.sample_goal_for_rollout()
        elif self.sample_rollout_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(1)
            obs = batch['observations']
            goal_state = self.env.convert_obs_to_goals(obs)[0]
            return self.env.modify_goal_for_rollout(goal_state)
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_rollout_goals_from
            ))

    def _sample_max_tau_for_rollout(self):
        return np.random.randint(0, self.max_tau + 1)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self.goal = self._sample_goal_for_rollout()
        self.training_env.set_goal(self.goal)
        self.exploration_policy.set_goal(self.goal)
        self.exploration_policy.set_discount(self.discount)
        return self.training_env.reset()

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
            goals=self.goal,
            # taus=self._rollout_discount,
        )

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()
