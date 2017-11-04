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
            fraction_goal_states_are_rollout_goal_states=None,
            goal_sample_strategy='store',
    ):
        """

        :param max_size:
        :param observation_dim:
        :param action_dim:
        :param num_goals_to_sample:
        :param fraction_goal_states_are_rollout_goal_states:
        :param goal_sample_strategy:
            'store': Just sample `num_goals_to_sample` when you save the
            tuple as in HER
            'online': Sample on the fly with every batch
        """
        super().__init__(max_size, env)
        self._current_episode_start_index = 0
        # Sample any value in this list
        self._index_to_sampled_goal_states_idxs = (
            [None] * max_size
        )
        # Do NOT sample self._index_to_goal_states_interval[i][1].
        # It's exclusive, like how range(3, 10) does not return 10.
        self._index_to_goal_states_interval = (
            [None] * max_size
        )
        self.num_goals_to_sample = num_goals_to_sample
        self._goal_states = np.zeros((max_size, self._env.goal_dim))
        if fraction_goal_states_are_rollout_goal_states is None:
            fraction_goal_states_are_rollout_goal_states = (
                1. / num_goals_to_sample
            )
        self.fraction_goal_states_are_rollout_goal_states = (
            fraction_goal_states_are_rollout_goal_states
        )
        self.goal_sample_strategy = goal_sample_strategy

    def _add_sample(self, observation, action, reward, terminal,
                    final_state, goal_state=None, **kwargs):
        assert goal_state is not None
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._final_state[self._top] = final_state
        self._goal_states[self._top] = goal_state
        self.her_advance(final_state)

    def her_advance(self, traj_is_done):
        # Current self._top is NOT a valid transition index since the next time
        # step is either garbage or from another episode
        if self._top in self._valid_transition_indices:
            self._valid_transition_indices.remove(self._top)
            # In theory not necessary since they won't be sampled... but you
            # just to be safe
            self._index_to_goal_states_interval[self._top] = None
            self._index_to_sampled_goal_states_idxs[self._top] = None

        last_top_index = self._top
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size >= self._max_replay_buffer_size:
            self._bottom = (self._bottom + 1) % self._max_replay_buffer_size
        else:
            self._size += 1

        # Completely ignore trajectories that wrap the replay buffer
        if traj_is_done:
            if last_top_index >= self._current_episode_start_index:
                self._make_last_trajectory_sample_able(last_top_index)
                self._add_goal_states_to_last_trajectory(last_top_index)
            self._current_episode_start_index = self._top

    def _make_last_trajectory_sample_able(self, last_top_index):
        """
        Preconditions: the last trajectory did not wrap the replay buffer.
        Therefore, the indices of the last trajectory are
        [self._current_episode_start_index, last_top_index]
        """
        # Note: do not include last_top_index since that is the last
        # "next_observation"
        for i in range(self._current_episode_start_index, last_top_index):
            if i not in self._valid_transition_indices:
                self._valid_transition_indices.append(i)

    def _add_goal_states_to_last_trajectory(self, last_top_index):
        """
        Preconditions: the last trajectory did not wrap the replay buffer.
        Therefore, the indices of the last trajectory are
        [self._current_episode_start_index, last_top_index]
        """
        for i in range(self._current_episode_start_index, last_top_index):
            self._index_to_goal_states_interval[i] = (i+1, last_top_index+1)
            potential_goal_state_indices = list(range(i+1, last_top_index+1))
            if len(potential_goal_state_indices) <= self.num_goals_to_sample:
                self._index_to_sampled_goal_states_idxs[i] = (
                    potential_goal_state_indices
                )
            else:
                self._index_to_sampled_goal_states_idxs[i] = np.random.choice(
                    potential_goal_state_indices,
                    size=self.num_goals_to_sample,
                    replace=False,
                )

            if i == last_top_index:
                self._index_to_goal_states_interval[i] = None
                self._index_to_sampled_goal_states_idxs[i] = None

    def random_batch(self, batch_size):
        # This is *much* faster than np.random.choice, because np.random.choice
        # converts the inputted array into a np.array internally. But
        # self._valid_transition_indices may be huge!
        length = len(self._valid_transition_indices)
        indices = np.array([
            self._valid_transition_indices[i]
            for i in np.random.randint(0, length, batch_size)
        ])
        next_indices = (indices + 1) % self._size
        if self.goal_sample_strategy == 'store':
            goal_state_indices = [
                np.random.choice(self._index_to_sampled_goal_states_idxs[i])
                for i in indices
            ]
        else:
            goal_state_indices = [
                np.random.choice(list(range(
                    *self._index_to_goal_states_interval[i]
                )))
                for i in indices
            ]
        goal_states = self._env.convert_obs_to_goal_states(
            self._observations[goal_state_indices]
        )
        taus = np.array(goal_state_indices) - np.array(indices)
        taus = taus.astype(float)
        num_goal_states_are_from_rollout = int(
            batch_size * self.fraction_goal_states_are_rollout_goal_states
        )
        if num_goal_states_are_from_rollout > 0:
            goal_states[:num_goal_states_are_from_rollout] = self._goal_states[
                indices[:num_goal_states_are_from_rollout]
            ]
            taus[:num_goal_states_are_from_rollout] = None
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[next_indices],
            goal_states=goal_states,
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
            min_i, max_i = self._index_to_goal_states_interval[i]
            # +1 for funny tau stuff. See above
            max_i = min(max_i, i + max_tau + 1)
            # This +1 is because randint exclused the last element
            goal_state_indices.append(np.random.randint(min_i, max_i+1))
        goal_states = self._env.convert_obs_to_goal_states(
            self._observations[goal_state_indices]
        )
        taus = np.array(goal_state_indices) - np.array(indices) - 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[next_indices],
            states_after_tau_steps=goal_states,
            taus=np.expand_dims(taus, 1),
        )
