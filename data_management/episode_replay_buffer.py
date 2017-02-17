from collections import deque
import random

import numpy as np

from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.pythonplusplus import sample_with_replacement


class EpisodeReplayBuffer(ReplayBuffer):
    """
    A replay buffer that stores episodes rather than simple transition tuples
    """
    def __init__(
            self,
            max_num_episodes,
            observation_dim,
            action_dim,
    )
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._episodes = deque([], maxlen=max_num_episodes)

        self._current_episode = SingleEpisode()

    def terminate_epsiode(self, terminal_observation):
        self._current_episode.terminate_epsiode(terminal_observation)
        self._episodes.append(self._current_episode)
        self._current_episode = SingleEpisode()

    def add_sample(self, observation, action, reward, terminal):
        self._current_episode.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
        )

    def random_subtrajectories(self, batch_size, sub_traj_length):
        """
        Return a list of
        :param batch_size:
        :return: Dictionary with the following key and np array values:
            - observations: [batch_size x sub_traj_length x obs_dim]
            - actions: [batch_size x sub_traj_length x action_dim]
            - rewards: [batch_size x sub_traj_length]
            - terminals: [batch_size x sub_traj_length]
            - next_observations: [batch_size x sub_traj_length x obs_dim]
        """
        observations = np.zeros(
            (batch_size, sub_traj_length,self._observation_dim)
        )
        next_obs = np.zeros(
            (batch_size, sub_traj_length, self._observation_dim)
        )
        actions = np.zeros(
            (batch_size, sub_traj_length, self._action_dim)
        )
        rewards = np.zeros((batch_size, sub_traj_length))
        terminals = np.zeros((batch_size, sub_traj_length))

        episodes = sample_with_replacement(self._episodes, batch_size)
        for i, episode in enumerate(episodes):
            all_obs = None
            observations[i, :, :] = all_obs[:, :-1]
            next_obs[i, :, :] = all_obs[:, 1:]
            actions[i, :, :] = None
            rewards[i, :, :] = None
            terminals[i, :, :] = None

        return dict(
            observations=observations,
            actions=actions,
            next_observations=next_obs,
            rewards=rewards,
            terminals=terminals,
        )

class SingleEpisode(object):
    def __init__(self):
        self._actions = []
        self._observations = []
        self._rewards = []
        self._terminals = []
        self._terminal_observation = None

    def add_sample(self, observation, action, reward, terminal):
        self._actions.append(action)
        self._observations.append(observation)
        self._rewards.append(reward)
        self._terminals.append(terminal)

    def terminate_epsiode(self, terminal_observation):
        self._terminal_observation = terminal_observation


