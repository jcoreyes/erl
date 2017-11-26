import numpy as np

from railrl.data_management.env_replay_buffer import EnvReplayBuffer


class HerReplayBuffer(EnvReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer

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
    ):
        """

        :param max_size:
        :param observation_dim:
        :param action_dim:
        :param num_goals_to_sample:
        :param fraction_goals_are_rollout_goals:
        """
        super().__init__(max_size, env)
        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size
        self.num_goals_to_sample = num_goals_to_sample
        self._goals = np.zeros((max_size, self._env.goal_dim))
        if fraction_goals_are_rollout_goals is None:
            fraction_goals_are_rollout_goals = (
                1. / num_goals_to_sample
            )
        self.fraction_goals_are_rollout_goals = (
            fraction_goals_are_rollout_goals
        )

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        goals = path["goals"]
        path_len = len(obs)

        if self._top + path_len >= self._max_replay_buffer_size:
            num_pre_wrap_steps = self._max_replay_buffer_size - self._top
            pre_wrap_buffer_slice = slice(
                self._top, self._top + num_pre_wrap_steps
            )
            pre_wrap_path_slice = slice(0, num_pre_wrap_steps)

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._observations[buffer_slice] = obs[path_slice]
                self._actions[buffer_slice] = actions[path_slice]
                self._rewards[buffer_slice] = rewards[path_slice]
                self._next_obs[buffer_slice] = next_obs[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._goals[buffer_slice] = goals[path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self._max_replay_buffer_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self._max_replay_buffer_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = slice(self._top, self._top + path_len)
            self._observations[slc] = obs
            self._actions[slc] = actions
            self._rewards[slc] = rewards
            self._next_obs[slc] = next_obs
            self._terminals[slc] = terminals
            self._goals[slc] = goals
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(i, i + path_len)
        self._top = (self._top + path_len) % self._max_replay_buffer_size
        self._size = min(self._size + path_len, self._max_replay_buffer_size)

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        next_obs_idxs = []
        for i in indices:
            possible_next_obs = self._idx_to_future_obs_idx[i]
            # This is generally faster than random.choice. Makes you wonder what
            # random.choice is doing
            next_obs_idxs.append(
                possible_next_obs[np.random.randint(0, len(possible_next_obs))]
            )
        next_obs_idxs = np.array(next_obs_idxs)
        goals = self._env.convert_obs_to_goals(self._next_obs[next_obs_idxs])
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
            next_observations=self._next_obs[indices],
            goals=goals,
        )

    def random_batch_for_sl(self, batch_size, max_tau):
        indices = np.random.randint(
            self._min_valid_idx, self._max_valid_idx, batch_size
        )
        next_indices = (indices + 1) % self._size
        goal_indices = []
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
            goal_indices.append(np.random.randint(min_i, max_i+1))
        obs_after_tau_steps = self._observations[goal_indices]
        taus = np.array(goal_indices) - np.array(indices) - 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[next_indices],
            obs_after_tau_steps=obs_after_tau_steps,
            taus=np.expand_dims(taus, 1),
        )
