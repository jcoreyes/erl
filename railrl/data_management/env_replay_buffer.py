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

class VPGEnvReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            discount_factor,
    ):
        super().__init__(max_replay_buffer_size, env)
        self.current_return = 0
        self._returns = np.zeros((max_replay_buffer_size, 1))
        self._max_replay_buffer_size = max_replay_buffer_size
        self.discount_factor = discount_factor

    def terminate_episode(self):
        self.current_return = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self.current_return = self.current_return*self.discount_factor + reward
        self._returns[self._top] = self.current_return
        super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

    def get_training_data(self):
        batch= dict(
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            terminals=self._terminals,
            next_observations=self._next_obs,
            returns = self._returns,
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