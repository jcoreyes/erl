import numpy as np
from cached_property import cached_property

from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from railrl.misc.np_util import subsequences, assign_subsequences
from railrl.data_management.subtraj_replay_buffer import SubtrajReplayBuffer


class UpdatableSubtrajReplayBuffer(SubtrajReplayBuffer):
    def __init__(
            self,
            max_pool_size,
            env: ContinuousMemoryAugmented,
            subtraj_length,
            memory_dim,
            keep_old_fraction=0.,
            **kwargs
    ):
        super().__init__(
            max_pool_size=max_pool_size,
            env=env,
            subtraj_length=subtraj_length,
            **kwargs
        )
        self._action = None
        self.observations = None

        self.memory_dim = memory_dim
        # Note that this must be computed from the next time step.
        """
        self._dloss_dwrite[t] = dL/dw_t     (zero-indexed)
        """
        self._dloss_dmemories = np.zeros((self._max_pool_size,
                                          self.memory_dim))
        self._env_obs_dim = env.env_spec.observation_space.flat_dim
        self._env_action_dim = env.env_spec.action_space.flat_dim
        self._env_obs = np.zeros((max_pool_size, self._env_obs_dim))
        self._env_actions = np.zeros((max_pool_size, self._env_action_dim))

        self._memory_dim = env.memory_dim
        self._memories = np.zeros((max_pool_size, self._memory_dim))
        self.keep_old_fraction = keep_old_fraction

    def random_subtrajectories(self, batch_size, replace=False,
                               validation=False, _fixed_start_indices=None):
        if _fixed_start_indices is None:
            start_indices = np.random.choice(
                self._valid_start_indices(validation=validation),
                batch_size,
                replace=replace,
            )
        else:
            start_indices = _fixed_start_indices
        return self._get_trajectories(start_indices), start_indices

    def _add_sample(self, observation, action, reward, terminal,
                    final_state, **kwargs):
        env_action, write = action  # write should be saved as next memory
        env_obs, memory = observation
        self._env_obs[self._top] = env_obs
        self._env_actions[self._top] = env_action
        self._memories[self._top] = memory
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._final_state[self._top] = final_state
        self._episode_start_indices[self._top] = self._starting_episode
        self._starting_episode = False

        self.advance()

    def _get_trajectories(self, start_indices):
        next_memories = subsequences(self._memories, start_indices,
                                     self._subtraj_length, start_offset=1)
        return dict(
            env_obs=subsequences(self._env_obs, start_indices,
                                 self._subtraj_length),
            env_actions=subsequences(self._env_actions, start_indices,
                                     self._subtraj_length),
            next_env_obs=subsequences(self._env_obs, start_indices,
                                      self._subtraj_length,
                                      start_offset=1),
            memories=subsequences(self._memories, start_indices,
                                  self._subtraj_length),
            writes=next_memories,
            next_memories=next_memories,
            rewards=subsequences(self._rewards, start_indices,
                                 self._subtraj_length),
            terminals=subsequences(self._terminals, start_indices,
                                   self._subtraj_length),
            dloss_dwrites=subsequences(self._dloss_dmemories, start_indices,
                                         self._subtraj_length, start_offset=1),
        )

    @cached_property
    def _stub_action(self):
        # Technically, the parent's method should work, but I think this is more
        # clear.
        return np.zeros(self._env_action_dim), np.zeros(self.memory_dim)

    def update_write_subtrajectories(self, updated_writes, start_indices):
        assign_subsequences(
            tensor=self._memories,
            new_values=updated_writes,
            start_indices=start_indices,
            length=self._subtraj_length,
            start_offset=1
        )

    def update_dloss_dmemories_subtrajectories(
            self,
            updated_dloss_dmemories,
            start_indices
    ):
        assign_subsequences(
            tensor=self._dloss_dmemories,
            new_values=updated_dloss_dmemories,
            start_indices=start_indices,
            length=self._subtraj_length,
            keep_old_fraction=self.keep_old_fraction,
        )

    def fraction_dloss_dmemories_zero(self):
        dloss_dmemories_loaded = self._dloss_dmemories[:self._size]
        return np.mean(dloss_dmemories_loaded == 0)
