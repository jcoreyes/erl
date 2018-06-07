"""
New implementation of state distance q learning.
"""
import abc

import numpy as np

from railrl.data_management.path_builder import PathBuilder
from railrl.envs.remote import RemoteRolloutEnv
from railrl.misc.np_util import truncated_geometric
from railrl.misc.ml_util import ConstantSchedule
from railrl.policies.base import SerializablePolicy
from railrl.state_distance.policies import UniversalPolicy
from railrl.state_distance.rollout_util import MultigoalSimplePathSampler, \
    multitask_rollout
from railrl.state_distance.tdm_networks import TdmNormalizer
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.core import np_to_pytorch_batch

class TemporalDifferenceModel(TorchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            max_tau=10,
            epoch_max_tau_schedule=None,
            sample_train_goals_from='her',
            sample_rollout_goals_from='environment',
            vectorized=True,
            cycle_taus_for_rollout=True,
            dense_rewards=False,
            finite_horizon=True,
            tau_sample_strategy='uniform',
            reward_type='distance',
            goal_reached_epsilon=1e-3,
            terminate_when_goal_reached=False,
            truncated_geom_factor=2.,
            norm_order=1,
            square_distance=False,
            goal_weights=None,
            tdm_normalizer: TdmNormalizer = None,
            num_pretrain_paths=0,
            normalize_distance=False,
            env_samples_goal_on_reset=False,
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
            - fixed: Do not resample the goal. Just use the one in the
            environment.
            - pretrain_paths: Resample goals from paths collected with a
            random policy.
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
            - uniform: Sample uniformly from [0, max_tau]
            - truncated_geometric: Sample from a truncated geometric
            distribution, truncated at max_tau.
            - all_valid: Always use all 0 to max_tau values
        :param reward_type: One of the following:
            - 'distance': Reward is -|s_t - s_g|
            - 'indicator': Reward is -1{||s_t - s_g||_2 > epsilon}
            - 'env': Use the environment reward
        :param goal_reached_epsilon: Epsilon used to determine if the goal
        has been reached. Used by `indicator` version of `reward_type` and when
        `terminate_whe_goal_reached` is True.
        :param terminate_when_goal_reached: Do you terminate when you have
        reached the goal?
        :param norm_order: If vectorized=False, do you use L1, L2,
        etc. for distance?
        :param goal_weights: None or the weights for the different goal
        dimensions. These weights are used to compute the distances to the goal.
        """
        assert sample_train_goals_from in ['environment', 'replay_buffer',
                                           'her', 'no_resampling']
        assert sample_rollout_goals_from in [
            'environment',
            'replay_buffer',
            'fixed',
            'pretrain_paths',
        ]
        assert tau_sample_strategy in [
            'no_resampling',
            'uniform',
            'truncated_geometric',
            'all_valid',
        ]
        assert reward_type in ['distance', 'indicator', 'env']
        # Just for NIPS 2018
        assert reward_type == 'env'
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
        self.goal_reached_epsilon = goal_reached_epsilon
        self.terminate_when_goal_reached = terminate_when_goal_reached
        self.norm_order = norm_order
        self.square_distance = square_distance
        self._current_path_goal = None
        self._rollout_tau = np.array([self.max_tau])
        self.truncated_geom_factor = float(truncated_geom_factor)
        self.goal_weights = goal_weights
        if self.goal_weights is not None:
            # In case they were passed in as (e.g.) tuples or list
            self.goal_weights = np.array(self.goal_weights)
            assert self.goal_weights.size == self.env.goal_dim
        self.tdm_normalizer = tdm_normalizer
        self.num_pretrain_paths = num_pretrain_paths
        self.normalize_distance = normalize_distance
        self.env_samples_goal_on_reset = env_samples_goal_on_reset

        self.eval_sampler = MultigoalSimplePathSampler(
            env=self.env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            tau_sampling_function=self._sample_max_tau_for_rollout,
            goal_sampling_function=self._sample_goal_for_rollout,
            cycle_taus_for_rollout=self.cycle_taus_for_rollout,
            render=self.render_during_eval,
            env_samples_goal_on_reset=env_samples_goal_on_reset,
        )
        self.pretrain_obs = None
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

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
        """
        Update the goal states/rewards
        """
        num_steps_left = self._sample_taus_for_training(batch)
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = self._sample_goals_for_training(batch)
        env_infos = batch.get('env_infos', None)
        rewards = self._compute_scaled_rewards_np(
            batch, obs, actions, next_obs, goals, env_infos
        )
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
                    > self.goal_reached_epsilon
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
        batch['num_steps_left'] = num_steps_left
        batch['goals'] = goals
        batch['observations'] = obs
        batch['next_observations'] = next_obs

        return np_to_pytorch_batch(batch)

    def _compute_scaled_rewards_np(self, batch, obs, actions, next_obs,
                                   goals, env_infos):
        """
        Rewards should be already multiplied by the reward scale and/or other
        factors. In other words, the rewards returned here should be
        immediately ready for any down-stream learner to consume.
        """
        if self.reward_type == 'indicator':
            diff = self.env.convert_obs_to_goals(next_obs) - goals
            if self.vectorized:
                return -self.reward_scale * (diff > self.goal_reached_epsilon)
            else:
                return -self.reward_scale * (
                        np.linalg.norm(diff, axis=1, keepdims=True)
                        > self.goal_reached_epsilon
                )
        elif self.reward_type == 'distance':
            neg_distances = self._compute_unscaled_neg_distances(next_obs,
                                                                 goals)
            return neg_distances * self.reward_scale
        elif self.reward_type == 'env':
            rewards = batch['rewards']
            # Hacky/inefficient for NIPS 2018
            for i in range(len(rewards)):
                if env_infos is None:
                    env_info = None
                else:
                    env_info = env_infos[i]
                rewards[i] = self.training_env.compute_her_reward_np(
                    obs[i],
                    actions[i],
                    next_obs[i],
                    goals[i],
                    env_info,
                )
            return rewards
        else:
            raise TypeError("Invalid reward type: {}".format(self.reward_type))

    def _compute_unscaled_neg_distances(self, next_obs, goals):
        diff = self.env.convert_obs_to_goals(next_obs) - goals
        if self.goal_weights is not None:
            diff = diff * self.goal_weights
        else:
            diff = diff * self.env.goal_dim_weights
        if self.vectorized:
            if self.square_distance:
                raw_neg_distances = - diff ** 2
            else:
                raw_neg_distances = -np.abs(diff)
        else:
            if self.square_distance:
                raw_neg_distances = -(diff ** 2).sum(1, keepdims=True)
            else:
                raw_neg_distances = -np.linalg.norm(
                    diff,
                    ord=self.norm_order,
                    axis=1,
                    keepdims=True,
                )
        return raw_neg_distances

    def _sample_taus_for_training(self, batch):
        if self.finite_horizon:
            if self.tau_sample_strategy == 'uniform':
                num_steps_left = np.random.randint(
                    0, self.max_tau + 1, (self.batch_size, 1)
                )
            elif self.tau_sample_strategy == 'truncated_geometric':
                num_steps_left = truncated_geometric(
                    p=self.truncated_geom_factor / self.max_tau,
                    truncate_threshold=self.max_tau,
                    size=(self.batch_size, 1),
                    new_value=0
                )
            elif self.tau_sample_strategy == 'no_resampling':
                num_steps_left = batch['num_steps_left']
            elif self.tau_sample_strategy == 'all_valid':
                num_steps_left = np.tile(
                    np.arange(0, self.max_tau + 1),
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
            batch = self.replay_buffer.random_batch(self.batch_size)
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
            if self.replay_buffer.num_steps_can_sample() == 0:
                return np.zeros(self.env.goal_dim)
            batch = self.replay_buffer.random_batch(1)
            obs = batch['observations']
            goal = self.env.convert_obs_to_goals(obs)[0]
            return self.env.modify_goal_for_rollout(goal)
        elif self.sample_rollout_goals_from == 'fixed':
            return self.env.multitask_goal
        elif self.sample_rollout_goals_from == 'pretrain_paths':
            random_i = np.random.randint(0, len(self.pretrain_obs))
            ob = self.pretrain_obs[random_i]
            return self.env.convert_ob_to_goal(ob)
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

    def _start_new_rollout(self, terminal=True, previous_rollout_last_ob=None):
        self.exploration_policy.reset()
        self._rollout_tau = np.array([self.max_tau])
        if self.env_samples_goal_on_reset:
            obs = self.training_env.reset()
            self._current_path_goal = self.training_env.get_goal()
        else:
            self._current_path_goal = self._sample_goal_for_rollout()
            self.training_env.set_goal(self._current_path_goal)
            obs = self.training_env.reset()
            assert (self.training_env.get_goal() == self._current_path_goal).all()
        return obs

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
        if self.cycle_taus_for_rollout:
            self._rollout_tau -= 1
            if self._rollout_tau[0] < 0:
                self._rollout_tau = np.array([self.max_tau])

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

    def evaluate(self, epoch):
        self.eval_statistics['Max Tau'] = self.max_tau
        super().evaluate(epoch)

    def pretrain(self):
        if (
                self.num_pretrain_paths == 0 and
                self.sample_rollout_goals_from != 'pretrain_paths'
        ):
            return

        paths = []
        random_policy = RandomUniveralPolicy(self.env.action_space)
        for _ in range(self.num_pretrain_paths):
            goal = self.env.sample_goal_for_rollout()
            path = multitask_rollout(
                self.training_env,
                random_policy,
                goal=goal,
                init_tau=0,
                max_path_length=self.max_path_length,
            )
            paths.append(path)

        obs = np.vstack([path["observations"] for path in paths])
        self.pretrain_obs = obs
        if self.num_pretrain_paths == 0:
            return
        next_obs = np.vstack([path["next_observations"] for path in paths])
        actions = np.vstack([path["actions"] for path in paths])
        goals = np.vstack([path["goals"] for path in paths])
        neg_distances = self._compute_unscaled_neg_distances(next_obs, goals)

        ob_mean = np.mean(obs, axis=0)
        ob_std = np.std(obs, axis=0)
        ac_mean = np.mean(actions, axis=0)
        ac_std = np.std(actions, axis=0)
        new_goals = np.vstack([
            self._sample_goal_for_rollout()
            for _ in range(
                self.num_pretrain_paths * self.max_path_length
            )
        ])
        goal_mean = np.mean(new_goals, axis=0)
        goal_std = np.std(new_goals, axis=0)
        distance_mean = np.mean(neg_distances, axis=0)
        distance_std = np.std(neg_distances, axis=0)

        if self.tdm_normalizer is not None:
            self.tdm_normalizer.obs_normalizer.set_mean(ob_mean)
            self.tdm_normalizer.obs_normalizer.set_std(ob_std)
            self.tdm_normalizer.action_normalizer.set_mean(ac_mean)
            self.tdm_normalizer.action_normalizer.set_std(ac_std)
            self.tdm_normalizer.goal_normalizer.set_mean(goal_mean)
            self.tdm_normalizer.goal_normalizer.set_std(goal_std)
            if self.normalize_distance:
                self.tdm_normalizer.distance_normalizer.set_mean(distance_mean)
                self.tdm_normalizer.distance_normalizer.set_std(distance_std)

class RandomUniveralPolicy(UniversalPolicy, SerializablePolicy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def get_action(self, *args, **kwargs):
        return self.action_space.sample(), {}
