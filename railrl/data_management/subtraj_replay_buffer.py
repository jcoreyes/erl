from collections import deque
import numpy as np

from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.misc.np_util import subsequences


class SubtrajReplayBuffer(ReplayBuffer):
    """
    Combine all the episode data into one big replay buffer and sample
    sub-trajectories
    """
    def __init__(
            self,
            max_pool_size,
            env,
            subtraj_length,
    ):
        self._max_pool_size = max_pool_size
        self._env = env
        self._subtraj_length = subtraj_length
        observation_dim = env.observation_space.flat_dim
        action_dim = env.action_space.flat_dim
        self._observation_dim = observation_dim
        self._action_dim = action_dim
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

        self._valid_start_indices = []
        self._previous_indices = deque(maxlen=self._subtraj_length)

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
        self._previous_indices = deque(maxlen=self._subtraj_length)

    def advance(self):
        # +1 since we're about to add an index
        if len(self._previous_indices) + 1>= self._subtraj_length:
            previous_idx = self._previous_indices[0]
            # The first condition isn't stictly needed, but this makes it so
            # that we don't have to reason about when the circular buffer
            # loops back to the start. At worse, we throw away a few
            # transitions, but we get to greatly simplfy the code. Otherwise,
            # the `subsequence` method would need to reason about circular
            # indices.
            if (previous_idx + self._subtraj_length < self._max_pool_size and
                    previous_idx not in self._valid_start_indices):
                self._valid_start_indices.append(previous_idx)
        # Current self._top is NOT a valid transition index since the next time
        # step is either garbage or from another episode
        if self._top in self._valid_start_indices:
            self._valid_start_indices.remove(self._top)

        self._previous_indices.append(self._top)
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_subtrajectories(self, batch_size, replace=False):
        start_indices = np.random.choice(self._valid_start_indices, batch_size,
                                         replace=replace)
        return dict(
            observations=subsequences(self._observations, start_indices,
                                      self._subtraj_length),
            actions=subsequences(self._actions, start_indices,
                                 self._subtraj_length),
            next_observations=subsequences(self._observations, start_indices,
                                           self._subtraj_length,
                                           start_offset=1),
            rewards=subsequences(self._rewards, start_indices,
                                 self._subtraj_length),
            terminals=subsequences(self._terminals, start_indices,
                                   self._subtraj_length),
        )

    @property
    def num_can_sample(self):
        return len(self._valid_start_indices)

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

    def get_all_valid_subtrajectories(self):
        start_indices = self._valid_start_indices
        return dict(
            observations=subsequences(self._observations, start_indices,
                                      self._subtraj_length),
            actions=subsequences(self._actions, start_indices,
                                 self._subtraj_length),
            next_observations=subsequences(self._observations, start_indices,
                                           self._subtraj_length, start_offset=1),
            rewards=subsequences(self._rewards, start_indices,
                                 self._subtraj_length),
            terminals=subsequences(self._terminals, start_indices,
                                   self._subtraj_length),
        )
