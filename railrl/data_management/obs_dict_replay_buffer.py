import numpy as np
from gym.spaces import Dict

from railrl.data_management.replay_buffer import ReplayBuffer
from multiworld.core.image_env import unormalize_image, normalize_image

class ObsDictRelabelingBuffer(ReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.

    Implementation details:
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            fraction_goals_are_rollout_goals=1.0,
            fraction_resampled_goals_are_env_goals=0.0,
            ob_keys_to_save=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            vectorized=False,
    ):
        """

        :param max_size:
        :param env:
        :param fraction_goals_rollout_goals: Default, no her
        :param fraction_resampled_goals_env_goals:  of the resampled
        goals, what fraction are sampled from the env. Reset are sampled from future.
        :param ob_keys_to_save: List of keys to save
        """
        if ob_keys_to_save is None:
            ob_keys_to_save = ['observation', 'desired_goal', 'achieved_goal']
        else:  # in case it's a tuple
            ob_keys_to_save = list(ob_keys_to_save)
        assert isinstance(env.observation_space, Dict)
        self.max_size = max_size
        self.env = env
        self.fraction_goals_rollout_goals = fraction_goals_are_rollout_goals
        self.fraction_resampled_goals_env_goals = fraction_resampled_goals_are_env_goals
        self.ob_keys_to_save = ob_keys_to_save
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key

        self._action_dim = env.action_space.low.size
        self._actions = np.zeros((max_size, self._action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self.vectorized = vectorized
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_size, 1), dtype='uint8')
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces
        for key in [observation_key, desired_goal_key, achieved_goal_key]:
            assert key in self.ob_spaces
            if key not in ob_keys_to_save:
                ob_keys_to_save.append(key)
        for key in ob_keys_to_save:
            assert key in self.ob_spaces
            type = np.float64
            if key.startswith('image'):
                type = np.uint8
            self._obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)
            self._next_obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)

        self._top = 0
        self._size = 0

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)

        actions = flatten_n(actions)
        obs = flatten_dict(obs, self.ob_keys_to_save)
        next_obs = flatten_dict(next_obs, self.ob_keys_to_save)

        if self._top + path_len >= self.max_size:
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                for key in self.ob_keys_to_save:
                    if key.startswith('image'):
                        if key.endswith('observation'):
                            self._obs[key][buffer_slice] = \
                                    unormalize_image(obs[key][path_slice])
                        self._next_obs[key][buffer_slice] = \
                                unormalize_image(next_obs[key][path_slice])
                    else:
                        self._obs[key][buffer_slice] = obs[key][path_slice]
                        self._next_obs[key][buffer_slice] = next_obs[key][path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self.max_size),
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
            slc = np.s_[self._top:self._top + path_len, :]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            for key in self.ob_keys_to_save:
                assign_obs = True
                if key.startswith('image'):
                    if not key.endswith('observation'):
                        assign_obs = False
                    obs[key] = unormalize_image(obs[key])
                    next_obs[key] = unormalize_image(next_obs[key])
                if assign_obs:
                    self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_rollout_goals = int(
            batch_size * self.fraction_goals_rollout_goals
        )
        num_env_goals = int(
            batch_size * (1 - self.fraction_goals_rollout_goals)
            * self.fraction_resampled_goals_env_goals
        )
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        if num_env_goals > 0:
            resampled_goals[num_rollout_goals:num_rollout_goals + num_env_goals] = (
                self.env.sample_goals(num_env_goals)[self.desired_goal_key]
            )

        if num_future_goals > 0:
            future_obs_idxs = []
            for i in indices[-num_future_goals:]:
                possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
                # This is generally faster than random.choice. Makes you wonder what
                # random.choice is doing
                num_options = len(possible_future_obs_idxs)
                next_obs_i = int(np.random.randint(0, num_options))
                future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
            future_obs_idxs = np.array(future_obs_idxs)
            resampled_goals[-num_future_goals:] = self._next_obs[
                self.achieved_goal_key
            ][future_obs_idxs]

        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)
        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        new_actions = self._actions[indices]
        new_rewards = self.env.compute_rewards(
            new_actions,
            new_obs_dict,
        )
        if not self.vectorized:
            new_rewards = new_rewards.reshape(-1, 1)

        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]
        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': self._terminals[indices],
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
        }
        return batch

    def _batch_obs_dict(self, indices):
        return {
            key: self._obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, indices):
        return {
            key: self._next_obs[key][indices]
            for key in self.ob_keys_to_save
        }


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }
