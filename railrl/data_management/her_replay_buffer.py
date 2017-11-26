import numpy as np

from railrl.data_management.env_replay_buffer import EnvReplayBuffer


class HerReplayBuffer(EnvReplayBuffer):
    """
    Save goal states from the same trajectory into the replay buffer

    Implementation details:
     - Ignore any trajectory that wraps around the buffer, so the final index
     for a trajectory is always larger than the initial index
     - Unlike SimpleReplayBuffer, only add transitions to
     _valid_transition_indices after the corresponding trajectory terminates
    """
    def __init__(
            self,
            max_size,
            env,
            num_goals_to_sample=8,
            fraction_goals_are_rollout_goals=None,
            goal_sample_strategy='store',
    ):
        """

        :param max_size:
        :param observation_dim:
        :param action_dim:
        :param num_goals_to_sample:
        :param fraction_goals_are_rollout_goals:
        :param goal_sample_strategy:
            'store': Just sample `num_goals_to_sample` when you save the
            tuple as in HER
            'online': Sample on the fly with every batch
        """
        super().__init__(max_size, env)
        self._current_episode_start_index = 0
        # Sample any value in this list
        self._index_to_sampled_goals_idxs = (
            [None] * max_size
        )
        # Do NOT sample self._index_to_goals_interval[i][1].
        # It's exclusive, like how range(3, 10) does not return 10.
        self._index_to_goals_interval = (
            [None] * max_size
        )
        self.num_goals_to_sample = num_goals_to_sample
        self._goals = np.zeros((max_size, self._env.goal_dim))
        if fraction_goals_are_rollout_goals is None:
            fraction_goals_are_rollout_goals = (
                1. / num_goals_to_sample
            )
        self.fraction_goals_are_rollout_goals = (
            fraction_goals_are_rollout_goals
        )
        self.goal_sample_strategy = goal_sample_strategy
        self._last_top_index = -1
        # Completely ignore trajectories that wrap the replay buffer
        self._wrapped_buffer = False
        self._min_valid_idx = 0
        self._max_valid_idx = 0

    def add_sample(self, observation, action, reward, terminal,
                    next_observation, goal_state=None, **kwargs):
        assert goal_state is not None
        self._goals[self._top] = goal_state
        super().add_sample(observation, action, reward, terminal, next_observation)

    def _advance(self):
        # In theory not necessary since they won't be sampled... but you
        # just to be safe
        self._index_to_goals_interval[self._top] = None
        self._index_to_sampled_goals_idxs[self._top] = None

        self._last_top_index = self._top

        if self._top + 1 < self._max_replay_buffer_size:
            self._top += 1
        else:
            self._top = 0
            self._wrapped_buffer = True
            self._max_valid_idx = self._current_episode_start_index - 1

        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def terminate_episode(self):
        # Completely ignore trajectories that wrap the replay buffer
        if not self._wrapped_buffer:
            self._max_valid_idx = max(self._last_top_index, self._max_valid_idx)
            self._add_goals_to_last_trajectory()
        self._current_episode_start_index = self._top
        self._wrapped_buffer = False

    def _add_goals_to_last_trajectory(self):
        """
        Preconditions: the last trajectory did not wrap the replay buffer.
        Therefore, the indices of the last trajectory are
        [self._current_episode_start_index, self._last_top_index]
        """
        for i in range(self._current_episode_start_index,
                       self._last_top_index + 1):
            self._index_to_goals_interval[i] = (i+1, self._last_top_index+1)
            potential_goal_state_indices = list(range(i+1, self._last_top_index+1))
            if len(potential_goal_state_indices) <= self.num_goals_to_sample:
                self._index_to_sampled_goals_idxs[i] = (
                    potential_goal_state_indices
                )
            else:
                self._index_to_sampled_goals_idxs[i] = np.random.choice(
                    potential_goal_state_indices,
                    size=self.num_goals_to_sample,
                    replace=False,
                )

            if i == self._last_top_index:
                self._index_to_goals_interval[i] = None
                self._index_to_sampled_goals_idxs[i] = None

    def random_batch(self, batch_size):
        indices = np.random.randint(
            self._min_valid_idx, self._max_valid_idx, batch_size
        )
        next_indices = (indices + 1) % self._size
        if self.goal_sample_strategy == 'store':
            goal_state_indices = [
                # This is generally faster than random.choice.
                # random.choice is only fast if you give it a np array.
                self._index_to_sampled_goals_idxs[np.random.randint(0, i)]
                for i in indices
            ]
        else:
            goal_state_indices = [
                np.random.randint(*self._index_to_goals_interval[i])
                for i in indices
            ]
        goals = self._env.convert_obs_to_goals(
            self._observations[goal_state_indices]
        )
        num_goals_are_from_rollout = int(
            batch_size * self.fraction_goals_are_rollout_goals
        )
        if num_goals_are_from_rollout > 0:
            goals[:num_goals_are_from_rollout] = self._goals[
                indices[:num_goals_are_from_rollout]
            ]
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[next_indices],
            goals=goals,
        )

    def random_batch_for_sl(self, batch_size, max_tau):
        indices = np.random.choice(
            self._valid_transition_indices,
            batch_size,
            replace=False
        )
        next_indices = (indices + 1) % self._size
        goal_state_indices = []
        """
        This +1 -1 business is because tau is defined as "the number of time
        steps you have left AFTER taking the current action."

        Say you're at state s_t and take action a_t. You end up at state
        s_{t+1}. If tau = 2, then we want to look at s_{t+3}, not s_{t+2}.

        The difference in indices is (t+3) - (t) = 3, but tau = 2.
        """
        for i in indices:
            min_i, max_i = self._index_to_goals_interval[i]
            # +1 for funny tau stuff. See above
            max_i = min(max_i, i + max_tau + 1)
            # This +1 is because randint exclused the last element
            goal_state_indices.append(np.random.randint(min_i, max_i+1))
        goals = self._env.convert_obs_to_goals(
            self._observations[goal_state_indices]
        )
        taus = np.array(goal_state_indices) - np.array(indices) - 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[next_indices],
            states_after_tau_steps=goals,
            taus=np.expand_dims(taus, 1),
        )
