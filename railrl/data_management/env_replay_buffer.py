from gym.spaces import Discrete

from railrl.data_management.simple_replay_buffer import SimpleReplayBuffer
from railrl.envs.env_utils import get_dim
import numpy as np

class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation, new_action, reward, terminal,
            next_observation, **kwargs
        )

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