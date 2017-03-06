import numpy as np

from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.misc.np_util import subsequences


class FlatEpisodeReplayPool(ReplayBuffer):
    """
    Combine all the episode data into one big replay buffer.
    """
    def __init__(
            self, max_pool_size, env,
            replacement_policy='stochastic', replacement_prob=1.0,
            max_skip_episode=10):
        observation_dim = env.observation_space.flat_dim
        action_dim = env.action_space.flat_dim
        self._env = env
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._replacement_policy = replacement_policy
        self._replacement_prob = replacement_prob
        self._max_skip_episode = max_skip_episode
        self._observations = np.zeros((max_pool_size, observation_dim))
        self._actions = np.zeros((max_pool_size, action_dim))
        self._rewards = np.zeros(max_pool_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        # self._final_state[i] = state i was the final state in a rollout,
        # so it should never be sampled since it has no correspond next state
        # In other words, we're saving the s_{t+1} after sampling a tuple of
        # (s_t, a_t, r_t, s_{t+1}, TERMINAL=TRUE)
        self._final_state = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def _add_sample(self, observation, action_, reward, terminal,
                    final_state):
        action = self._env.action_space.flatten(action_)
        observation = self._env.observation_space.flatten(observation)
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._final_state[self._top] = final_state
        self.advance()

    def add_sample(self, observation, action, reward, terminal):
        self._add_sample(
            observation,
            action,
            reward,
            terminal,
            False,
        )
        self._example_action = action

    def terminate_episode(self, terminal_observation):
        self._add_sample(
            terminal_observation,
            self._example_action,
            0,
            0,
            True,
        )

    def advance(self):
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        assert self._size > 1
        transition_indices = np.zeros(batch_size, dtype='uint64')
        # make sure that the transition is valid: if we are at the end of
        # the pool, we need to discard this sample
        current_i = (self._top - 1) % self._max_pool_size
        # TODO(vitchyr): this takes up a non-trivial amount of computation.
        # Consider caching this.
        valid_indices = [i for i in range(min(self._size, self._max_pool_size))
                         if not self._final_state[i] and i != current_i]
        indices = np.random.choice(valid_indices, batch_size, replace=False)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    def _can_sample_subtraj_at(self, idx, length):
        """
        Return whether or not you can sample a subtrajectory of length
        `length` starting at index `idx`.
        :param idx:
        :param length:
        :return:
        """
        if idx == (self._top - 1) % self._max_pool_size:
            return False
        # This shouldn't be range(length - 1) because
        #   final_state[idx + length] == True
        # means that the last state in the trajectory is a final state.
        # However, the last next_observation will be at index
        #   idx + length + 1
        # which would be the next episode.
        for i in range(length):
            if (idx + i + 1 >= len(self._final_state) or
                    self._final_state[idx + i]):
                return False
        return True

    def num_can_sample(self, sub_traj_length):
        return len(self._valid_start_indices(sub_traj_length))

    def _valid_start_indices(self, sub_traj_length):
        return [
            i for i in range(min(self._size, self._max_pool_size))
            if self._can_sample_subtraj_at(i, sub_traj_length)
        ]

    def random_subtrajectories(self, batch_size, sub_traj_length,
                               replace=True):
        assert self._size > 1
        # TODO(vitchyr): this takes up a non-trivial amount of computation.
        # Consider caching this.
        valid_start_indices = self._valid_start_indices(sub_traj_length)
        start_indices = np.random.choice(valid_start_indices, batch_size,
                                         replace=replace)
        return dict(
            observations=subsequences(self._observations, start_indices,
                                      sub_traj_length),
            actions=subsequences(self._actions, start_indices,
                                 sub_traj_length),
            next_observations=subsequences(self._observations, start_indices,
                                           sub_traj_length, start_offset=1),
            rewards=subsequences(self._rewards, start_indices,
                                 sub_traj_length),
            terminals=subsequences(self._terminals, start_indices,
                                   sub_traj_length),
        )

    @property
    def size(self):
        return self._size

    def add_trajectory(self, path):
        for observation, action, reward in zip(
                path["observations"],
                path["actions"],
                path["rewards"],
        ):
            observation = self._env.observation_space.unflatten(observation)
            action = self._env.observation_space.unflatten(action)
            self.add_sample(observation, action, reward, False)
        terminal_observation = self._env.observation_space.unflatten(
            path["observations"][-1]
        )
        self.terminate_episode(terminal_observation)
