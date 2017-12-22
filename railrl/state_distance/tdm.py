"""
New implementation of state distance q learning.
"""
import abc

import numpy as np

from railrl.data_management.path_builder import PathBuilder
from railrl.envs.remote import RemoteRolloutEnv
from railrl.misc.ml_util import ConstantSchedule
from railrl.state_distance.exploration import MakeUniversal
from railrl.state_distance.rollout_util import MultigoalSimplePathSampler, \
    multitask_rollout
from railrl.state_distance.util import merge_into_flat_obs
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.algos.util import np_to_pytorch_batch


class TemporalDifferenceModel(TorchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            max_tau=10,
            epoch_max_tau_schedule=None,
            sample_train_goals_from='replay_buffer',
            sample_rollout_goals_from='environment',
            vectorized=True,
            cycle_taus_for_rollout=False,
            dense_rewards=False,
            finite_horizon=True,
            tau_sample_strategy='uniform',
            reward_type='distance',
            goal_reached_epsilon=1e-3,
            terminate_when_goal_reached=False,
    ):
        """

        :param max_tau: Maximum tau (planning horizon) to train with.
        :param epoch_max_tau_schedule: A schedule for the maximum planning
        horizon tau.
        :param sample_train_goals_from: Sampling strategy for goals used in
        training. Can be one of the following strings:
            - environment: Sample from the environment
            - replay_buffer: Sample from anywhere in the replay_buffer
            - her: Sample from a HER-based replay_buffer
            - no_resampling: Use the goals used in the rollout
        :param sample_rollout_goals_from: Sampling strategy for goals used
        during rollout. Can be one of the following strings:
            - environment: Sample from the environment
            - replay_buffer: Sample from the replay_buffer
            - fixed: Do no resample the goal. Just use the one in the
            environment.
        :param vectorized: Train the QF in vectorized form?
        :param cycle_taus_for_rollout: Decrement the tau passed into the
        policy during rollout?
        :param dense_rewards: If True, always give rewards. Otherwise,
        only give rewards when the episode terminates.
        :param finite_horizon: If True, use a finite horizon formulation:
        give the time as input to the Q-function and terminate.
        :param tau_sample_strategy: Sampling strategy for taus used
        during training. Can be one of the following strings:
            - no_resampling: Do not resample the tau. Use the one from rollout.
            - uniform: Sample from [0, max_tau]
            - all_valid: Always use all 0 to max_tau values
        :param reward_type: One of the following:
            - 'distance': Reward is -|s_t - s_g|
            - 'sparse': Reward is -1{||s_t - s_g||_2 > epsilon}
        :param goal_reached_epsilon: Epsilon used to determine if the goal
        has been reached. Used by `sparse` version of `reward_type` and when
        `terminate_whe_goal_reached` is True.
        :param terminate_when_goal_reached: Do you terminate when you have
        reached the goal?
        """
        assert sample_train_goals_from in ['environment', 'replay_buffer',
                                           'her', 'no_resampling']
        assert sample_rollout_goals_from in ['environment', 'replay_buffer',
                                             'fixed']
        assert tau_sample_strategy in ['no_resampling', 'uniform', 'all_valid']
        assert reward_type in ['distance', 'sparse']
        if epoch_max_tau_schedule is None:
            epoch_max_tau_schedule = ConstantSchedule(max_tau)

        if not finite_horizon:
            max_tau = 0
            epoch_max_tau_schedule = ConstantSchedule(max_tau)
            cycle_taus_for_rollout = False

        self.max_tau = max_tau
        self.epoch_max_tau_schedule = epoch_max_tau_schedule
        self.sample_train_goals_from = sample_train_goals_from
        self.sample_rollout_goals_from = sample_rollout_goals_from
        self.vectorized = vectorized
        self.cycle_taus_for_rollout = cycle_taus_for_rollout
        self.dense_rewards = dense_rewards
        self.finite_horizon = finite_horizon
        self.tau_sample_strategy = tau_sample_strategy
        self.reward_type = reward_type
        self.sparse_reward_epsilon = goal_reached_epsilon
        self.terminate_when_goal_reached = terminate_when_goal_reached
        self._current_path_goal = None
        self._rollout_tau = self.max_tau

        self.policy = MakeUniversal(self.policy)
        self.eval_policy = MakeUniversal(self.eval_policy)
        self.exploration_policy = MakeUniversal(self.exploration_policy)
        self.eval_sampler = MultigoalSimplePathSampler(
            env=self.env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            tau_sampling_function=self._sample_max_tau_for_rollout,
            goal_sampling_function=self._sample_goal_for_rollout,
            cycle_taus_for_rollout=self.cycle_taus_for_rollout,
        )
        if self.collection_mode == 'online-parallel':
            # TODO(murtaza): What happens to the eval env?
            # see `eval_sampler` definition above.

            self.training_env = RemoteRolloutEnv(
                env=self.env,
                policy=self.eval_policy,
                exploration_policy=self.exploration_policy,
                max_path_length=self.max_path_length,
                normalize_env=self.normalize_env,
                rollout_function=self.rollout,
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
        num_steps_left = self._sample_taus_for_training(batch)
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = self._sample_goals_for_training(batch)
        rewards = self.compute_rewards_np(obs, actions, next_obs, goals)
        terminals = batch['terminals']

        if self.tau_sample_strategy == 'all_valid':
            obs = np.repeat(obs, self.max_tau + 1, 0)
            actions = np.repeat(actions, self.max_tau + 1, 0)
            next_obs = np.repeat(next_obs, self.max_tau + 1, 0)
            goals = np.repeat(goals, self.max_tau + 1, 0)
            rewards = np.repeat(rewards, self.max_tau + 1, 0)
            terminals = np.repeat(terminals, self.max_tau + 1, 0)

        if self.finite_horizon:
            terminals = 1 - (1 - terminals) * (num_steps_left != 0)
        if self.terminate_when_goal_reached:
            diff = self.env.convert_obs_to_goals(next_obs) - goals
            goal_not_reached = (
                np.linalg.norm(diff, axis=1, keepdims=True)
                > self.sparse_reward_epsilon
            )
            terminals = 1 - (1 - terminals) * goal_not_reached

        if not self.dense_rewards:
            rewards = rewards * terminals

        """
        Update the batch
        """
        batch['rewards'] = rewards
        batch['terminals'] = terminals
        batch['actions'] = actions
        batch['observations'] = merge_into_flat_obs(
            obs=obs,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        if self.finite_horizon:
            batch['next_observations'] = merge_into_flat_obs(
                obs=next_obs,
                goals=goals,
                num_steps_left=num_steps_left-1,
            )
        else:
            batch['next_observations'] = merge_into_flat_obs(
                obs=next_obs,
                goals=goals,
                num_steps_left=num_steps_left,
            )

        return np_to_pytorch_batch(batch)

    def compute_rewards_np(self, obs, actions, next_obs, goals):
        if self.reward_type == 'sparse':
            diff = self.env.convert_obs_to_goals(next_obs) - goals
            if self.vectorized:
                return -self.reward_scale * (diff > self.sparse_reward_epsilon)
            else:
                return -self.reward_scale * (
                    np.linalg.norm(diff, axis=1, keepdims=True)
                    > self.sparse_reward_epsilon
                )
        elif self.reward_type == 'distance':
            if self.vectorized:
                diff = self.env.convert_obs_to_goals(next_obs) - goals
                return -np.abs(diff) * self.reward_scale
            else:
                return self.env.compute_rewards(
                    obs,
                    actions,
                    next_obs,
                    goals,
                ) * self.reward_scale
        else:
            raise TypeError("Invalid reward type: {}".format(self.reward_type))

    @property
    def train_buffer(self):
        if self.replay_buffer_is_split:
            return self.replay_buffer.get_replay_buffer(trainig=True)
        else:
            return self.replay_buffer

    def _sample_taus_for_training(self, batch):
        if self.finite_horizon:
            if self.tau_sample_strategy == 'uniform':
                num_steps_left = np.random.randint(
                    0, self.max_tau + 1, (self.batch_size, 1)
                )
            elif self.tau_sample_strategy == 'no_resampling':
                num_steps_left = batch['num_steps_left']
            elif self.tau_sample_strategy == 'all_valid':
                num_steps_left = np.tile(
                    np.arange(0, self.max_tau+1),
                    self.batch_size
                )
                num_steps_left = np.expand_dims(num_steps_left, 1)
            else:
                raise TypeError("Invalid tau_sample_strategy: {}".format(
                    self.tau_sample_strategy
                ))
        else:
            num_steps_left = np.zeros((self.batch_size, 1))
        return num_steps_left

    def _sample_goals_for_training(self, batch):
        if self.sample_train_goals_from == 'her':
            return batch['resampled_goals']
        elif self.sample_train_goals_from == 'no_resampling':
            return batch['goals_used_for_rollout']
        elif self.sample_train_goals_from == 'environment':
            return self.env.sample_goals(self.batch_size)
        elif self.sample_train_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(self.batch_size)
            obs = batch['observations']
            return self.env.convert_obs_to_goals(obs)
        else:
            raise Exception("Invalid `sample_train_goals_from`: {}".format(
                self.sample_train_goals_from
            ))

    def _sample_goal_for_rollout(self):
        if self.sample_rollout_goals_from == 'environment':
            return self.env.sample_goal_for_rollout()
        elif self.sample_rollout_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(1)
            obs = batch['observations']
            goal = self.env.convert_obs_to_goals(obs)[0]
            return self.env.modify_goal_for_rollout(goal)
        elif self.sample_rollout_goals_from == 'fixed':
            return self.env.multitask_goal
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_rollout_goals_from
            ))

    def _sample_max_tau_for_rollout(self):
        if self.finite_horizon:
            return self.max_tau
        else:
            return 0

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self._current_path_goal = self._sample_goal_for_rollout()
        self.training_env.set_goal(self._current_path_goal)
        self.exploration_policy.set_goal(self._current_path_goal)
        self._rollout_tau = self.max_tau
        self.exploration_policy.set_tau(self._rollout_tau)
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
            num_steps_left=np.array([self._rollout_tau]),
            goals=self._current_path_goal,
        )
        if self.cycle_taus_for_rollout:
            self._rollout_tau -= 1
            if self._rollout_tau < 0:
                self._rollout_tau = self.max_tau
            self.exploration_policy.set_tau(self._rollout_tau)

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _handle_path(self, path):
        self._n_rollouts_total += 1
        self.replay_buffer.add_path(path)
