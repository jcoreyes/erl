from collections import deque

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
            env,
    ):
        self._observation_dim = env.observation_space.flat_dim
        self._action_dim = env.action_space.flat_dim
        self._env = env
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
            - observations: [batch_size x sub_traj_length x flat_obs_dim]
            - actions: [batch_size x sub_traj_length x flat_action_dim]
            - rewards: [batch_size x sub_traj_length]
            - terminals: [batch_size x sub_traj_length]
            - next_observations: [batch_size x sub_traj_length x obs_dim]
        """
        observations = np.zeros(
            (batch_size, sub_traj_length, self._observation_dim)
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
    def __init__(self, env):
        self._env = env
        self._flat_actions = []
        self._flat_observations = []
        self._rewards = []
        self._terminals = []
        self._terminal_observation = None
        self._length = 0  # = number of actions taken
        self._episode_terminated = False

        self._flat_actions_np = None
        self._flat_observations_np = None
        self._rewards_np = None
        self._terminals_np = None

    def add_sample(self, observation, action, reward, terminal):
        assert not self._episode_terminated
        flat_action = self._env.action_space.flatten(action)
        flat_obs = self._env.observation_space.flatten(observation)
        self._flat_actions.append(flat_action)
        self._flat_observations.append(flat_obs)
        self._rewards.append(reward)
        self._terminals.append(terminal)
        self._length += 1

    def terminate_epsiode(self, terminal_observation):
        assert not self._episode_terminated
        self._terminal_observation = terminal_observation
        self._episode_terminated = True

        self._flat_actions_np = np.array(self._flat_actions)
        self._flat_observations_np = np.array(self._flat_observations)
        self._rewards_np = np.array(self._rewards)
        self._terminals_np = np.array(self._terminals)

    @property
    def length(self):
        return self._length

    def sample_subtrajectory(self, length):
        assert length <= self.length
