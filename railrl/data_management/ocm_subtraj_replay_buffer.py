import numpy as np

from railrl.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from railrl.misc.np_util import subsequences


class OcmSubtrajReplayBuffer(SubtrajReplayBuffer):
    """
    A replay buffer desired specifically for OneCharMem
    sub-trajectories
    """

    def __init__(
            self,
            max_pool_size,
            env,
            subtraj_length,
    ):
        self._target_numbers = np.zeros(max_pool_size, dtype='uint8')
        self._times = np.zeros(max_pool_size, dtype='uint8')
        super().__init__(
            max_pool_size,
            env,
            subtraj_length,
            only_sample_at_start_of_episode=True,
        )

    def _add_sample(self, observation, action, reward, terminal,
                    final_state, debug_info=None):
        if debug_info is not None:
            self._target_numbers[self._top] = debug_info['target_number']
            self._times[self._top] = debug_info['time']
        super()._add_sample(
            observation,
            action,
            reward,
            terminal,
            final_state
        )

    def _get_trajectories(self, start_indices):
        trajs = super()._get_trajectories(start_indices)
        trajs['target_numbers'] = subsequences(
            self._target_numbers,
            start_indices,
            self._subtraj_length,
        )
        trajs['times'] = subsequences(
            self._times,
            start_indices,
            self._subtraj_length,
        )
        return trajs
