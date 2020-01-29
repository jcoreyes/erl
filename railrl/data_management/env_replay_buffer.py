from gym.spaces import Discrete
from railrl.data_management.simple_replay_buffer import SimpleReplayBuffer
from railrl.envs.env_utils import get_dim
import numpy as np
import railrl.torch.pytorch_util as ptu
import torch.nn.functional as F
import torch

class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

class AWREnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
            use_weights=False,
            policy=None,
            qf1=None,
            beta=0,
            weight_update_period=10000,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()
        self._use_weights = use_weights
        self.policy = policy
        self.qf1 = qf1
        self.beta = beta
        self.weight_update_period = weight_update_period
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )
        self.weights = torch.zeros((self._max_replay_buffer_size, 1), dtype=torch.float32)
        self.actual_weights = None
        self.counter = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def refresh_weights(self):
        if self.counter % self.weight_update_period == 0 and self._use_weights:
            batch_size = 1024
            next_idx = min(batch_size, self._size)

            cur_idx = 0
            while cur_idx < self._size:
                idxs = np.arange(cur_idx, next_idx)
                obs = ptu.from_numpy(self._observations[idxs])
                actions = ptu.from_numpy(self._actions[idxs])

                new_obs_actions, policy_mean, policy_log_std, log_pi, entropy, policy_std, *_ = self.policy(
                    obs, reparameterize=True, return_log_prob=True,
                )
                q1_pred = self.qf1(obs, actions)
                v_pi = self.qf1(obs, new_obs_actions)

                advantage = q1_pred - v_pi
                self.weights[idxs] = (advantage/self.beta).detach()

                cur_idx = next_idx
                next_idx += batch_size
                next_idx = min(next_idx, self._size)

            self.actual_weights = ptu.get_numpy(F.softmax(self.weights[:self._size], dim=0).reshape(-1))
        self.counter += 1

    def sample_weighted_indices(self, batch_size):
        if self._use_weights:
            indices = np.random.choice(
                len(self.actual_weights),
                batch_size,
                p=self.actual_weights,
            )
        else:
            indices = self._sample_indices(batch_size)
        return indices

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        self.refresh_weights()
        indices = self.sample_weighted_indices(batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            weights=self.actual_weights[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

class VPGEnvReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            discount_factor,
    ):
        super().__init__(max_replay_buffer_size, env)
        self._returns = np.zeros((max_replay_buffer_size, 1))
        self.current_trajectory_rewards = np.zeros((max_replay_buffer_size, 1))
        self._max_replay_buffer_size = max_replay_buffer_size
        self.discount_factor = discount_factor
        self._bottom = 0

    def terminate_episode(self):
        returns = []
        return_so_far = 0
        for t in range(len(self._rewards[self._bottom:self._top]) - 1, -1, -1):
            return_so_far = self._rewards[t][0] + self.discount_factor * return_so_far
            returns.append(return_so_far)

        returns = returns[::-1]
        returns = np.reshape(np.array(returns),(len(returns), 1))
        self._returns[self._bottom:self._top] = returns
        self._bottom = self._top

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if self._top == self._max_replay_buffer_size:
            raise EnvironmentError('Replay Buffer Overflow, please reduce the number of samples added!')
        super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

    def get_training_data(self):
        batch= dict(
            observations=self._observations[0:self._top],
            actions=self._actions[0:self._top],
            rewards=self._rewards[0:self._top],
            terminals=self._terminals[0:self._top],
            next_observations=self._next_obs[0:self._top],
            returns = self._returns[0:self._top],
        )
        return batch

    def empty_buffer(self):
        self._observations = np.zeros(self._observations.shape)
        self._next_obs = np.zeros(self._next_obs.shape)
        self._actions = np.zeros(self._actions.shape)
        self._rewards = np.zeros(self._rewards.shape)
        self._terminals = np.zeros(self._terminals.shape, dtype='uint8')
        self._returns = np.zeros(self._returns.shape)
        self._size = 0
        self._top = 0
        self._bottom = 0

