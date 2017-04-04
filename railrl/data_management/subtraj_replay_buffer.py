from collections import deque
import numpy as np

from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.misc.np_util import subsequences


def dict_of_list__to__list_of_dicts(dict, n_items):
    new_dicts = [{} for _ in range(n_items)]
    for key, values in dict.items():
        for i in range(n_items):
            new_dicts[i][key] = values[i]
    return new_dicts



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
            only_sample_at_start_of_episode=False,
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

        # placeholder for when saving a terminal observation
        self._example_action = None
        self._bottom = 0
        self._top = 0
        self._size = 0

        self._all_valid_start_indices = []
        self._previous_indices = deque(maxlen=self._subtraj_length)

        self._only_sample_at_start_of_episode = only_sample_at_start_of_episode
        self._episode_start_indices = np.zeros(max_pool_size, dtype='uint8')
        self._starting_episode = True
        self._valid_start_episode_start_indices = []

    def _add_sample(self, observation, action_, reward, terminal,
                    final_state, **kwargs):
        action = self._env.action_space.flatten(action_)
        observation = self._env.observation_space.flatten(observation)
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._final_state[self._top] = final_state
        self._episode_start_indices[self._top] = self._starting_episode
        self._starting_episode = False

        self.advance()

    def add_sample(self, observation, action, reward, terminal, **kwargs):
        self._add_sample(
            observation,
            action,
            reward,
            terminal,
            False,
            **kwargs
        )
        self._example_action = action

    def terminate_episode(self, terminal_observation, **kwargs):
        self._add_sample(
            terminal_observation,
            self._example_action,
            0,
            0,
            True,
        )
        self._previous_indices = deque(maxlen=self._subtraj_length)
        self._starting_episode = True

    def advance(self):
        if len(self._previous_indices) >= self._subtraj_length:
            previous_idx = self._previous_indices[0]
            # The first condition isn't stictly needed, but this makes it so
            # that we don't have to reason about when the circular buffer
            # loops back to the start. At worse, we throw away a few
            # transitions, but we get to greatly simplfy the code. Otherwise,
            # the `subsequence` method would need to reason about circular
            # indices.
            if (previous_idx + self._subtraj_length < self._max_pool_size and
                    previous_idx not in self._all_valid_start_indices):
                self._all_valid_start_indices.append(previous_idx)
                if (self._only_sample_at_start_of_episode
                        and self._episode_start_indices[previous_idx]):
                    self._valid_start_episode_start_indices.append(previous_idx)
        # Current self._top is NOT a valid transition index since the next time
        # step is either garbage or from another episode
        if self._top in self._all_valid_start_indices:
            self._all_valid_start_indices.remove(self._top)
        if self._top in self._valid_start_episode_start_indices:
            self._valid_start_episode_start_indices.remove(self._top)

        self._previous_indices.append(self._top)

        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_subtrajectories(self, batch_size, replace=False):
        start_indices = np.random.choice(self._valid_start_indices, batch_size,
                                         replace=replace)
        return self._get_trajectories(start_indices)

    @property
    def _valid_start_indices(self):
        if self._only_sample_at_start_of_episode:
            return self._valid_start_episode_start_indices
        else:
            return self._all_valid_start_indices

    @property
    def num_can_sample(self):
        return len(self._valid_start_indices)

    def add_trajectory(self, path):
        agent_infos = path['agent_infos']
        env_infos = path['env_infos']
        n_items = len(path["observations"])
        list_of_agent_infos = dict_of_list__to__list_of_dicts(agent_infos, n_items)
        list_of_env_infos = dict_of_list__to__list_of_dicts(env_infos, n_items)
        for (
            observation,
            action,
            reward,
            agent_info,
            env_info,
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            list_of_agent_infos,
            list_of_env_infos
        ):
            observation = self._env.observation_space.unflatten(observation)
            action = self._env.observation_space.unflatten(action)
            self.add_sample(observation, action, reward, False,
                            agent_info=agent_info, env_info=env_info)
        terminal_observation = self._env.observation_space.unflatten(
            path["observations"][-1]
        )
        self.terminate_episode(terminal_observation)

    def get_all_valid_subtrajectories(self):
        start_indices = self._valid_start_indices
        return self._get_trajectories(start_indices)

    def _get_trajectories(self, start_indices):
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
