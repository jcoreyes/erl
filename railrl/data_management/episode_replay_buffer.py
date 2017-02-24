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
            env,
    ):
        self._observation_dim = env.observation_space.flat_dim
        self._action_dim = env.action_space.flat_dim
        self._env = env
        self._size = 0

        self._current_episode = SingleEpisode(self._env)
        self._episodes = deque([], maxlen=max_num_episodes)
        self._episodes.append(self._current_episode)

    def terminate_episode(self, terminal_observation):
        self._current_episode.terminate_epsiode(terminal_observation)
        self._episodes.append(self._current_episode)
        self._current_episode = SingleEpisode(self._env)
        self._size += 1

    def add_sample(self, observation, action, reward, terminal):
        self._current_episode.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
        )
        self._size += 1

    @property
    def size(self):
        return self._size

    def random_subtrajectories(self, batch_size, sub_traj_length):
        """
        Return a list of
        :param batch_size: How many sub-trajectories to sample.
        :param sub_traj_length: The length of each sub-trajectory.
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

        episode_can_sample_from = [
            episode for episode in self._episodes
            if episode.is_long_enough_to_sample_subtraj_of_length(
                sub_traj_length
            )
        ]
        episodes = sample_with_replacement(episode_can_sample_from, batch_size)
        for i, episode in enumerate(episodes):
            subtraj = episode.sample_subtrajectory(sub_traj_length)
            observations[i, :, :] = subtraj["observations"]
            next_obs[i, :, :] = subtraj["next_observations"]
            actions[i, :, :] = subtraj["actions"]
            rewards[i, :] = subtraj["rewards"]
            terminals[i, :] = subtraj["terminals"]

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
        # length = number of observations seen, so when the episode has
        # terminated, there should be self._length - 1 actions since the
        # last observation doesn't have a corresponding action
        self._length = 0
        self._episode_terminated = False

        self._flat_actions_np = None
        self._flat_observations_np = None
        self._rewards_np = None
        self._terminals_np = None

    def add_sample(self, observation, action, reward, terminal):
        assert not self._episode_terminated
        self._length += 1
        flat_action = self._env.action_space.flatten(action)
        flat_obs = self._env.observation_space.flatten(observation)
        self._flat_actions.append(flat_action)
        self._flat_observations.append(flat_obs)
        self._rewards.append(reward)
        self._terminals.append(terminal)

    def terminate_epsiode(self, terminal_observation):
        assert not self._episode_terminated
        self._length += 1
        self._episode_terminated = True
        flat_obs = self._env.observation_space.flatten(terminal_observation)
        self._flat_observations.append(flat_obs)

        self._flat_observations_np = np.array(self._flat_observations)
        self._flat_actions_np = np.array(self._flat_actions)
        self._rewards_np = np.array(self._rewards)
        self._terminals_np = np.array(self._terminals)

    @property
    def length(self):
        return self._length

    def is_long_enough_to_sample_subtraj_of_length(self, length):
        return 0 <= self.length - length - 1

    def sample_subtrajectory(self, length):
        """

        :param length:
        :return: Dictionary with the following key and values:
         "observations": np.ndarray of shape (length x observation_dim)
         "next_observations": np.ndarray of shape (length x observation_dim)
         "actions": np.ndarray of shape (length x action_dim)
         "rewards": np.ndarray of shape (length)
         "terminals": np.ndarray of shape (length)
        """
        # Since next_observations is indexed with `last_index + 1`, we want
        #     last_index + 1 <= self.length
        # i.e.
        #     start_index + length + 1 <= self.length
        # So we need
        #     0 <= start_index <= self.length - length - 1
        # i.e.
        #     0 <= self.length - length - 1
        assert 0 <= self.length - length - 1
        if not self._episode_terminated:
            self._flat_observations_np = np.array(self._flat_observations)
            self._flat_actions_np = np.array(self._flat_actions)
            self._rewards_np = np.array(self._rewards)
            self._terminals_np = np.array(self._terminals)
        start_index = random.randint(0, self.length - length - 1)
        last_index = start_index + length
        return dict(
            observations=self._flat_observations_np[start_index:last_index],
            next_observations=(
                self._flat_observations_np[start_index + 1:last_index + 1]
            ),
            actions=self._flat_actions_np[start_index:last_index],
            rewards=self._rewards_np[start_index:last_index],
            terminals=self._terminals_np[start_index:last_index],
        )
