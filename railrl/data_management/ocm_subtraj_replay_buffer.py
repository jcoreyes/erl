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
        self._debug_number = np.zeros(max_pool_size, dtype='uint8')
        self._time = np.zeros(max_pool_size, dtype='uint8')
        super().__init__(max_pool_size, env, subtraj_length)

    def _add_sample(self, observation, action, reward, terminal,
                    final_state, debug_info=None):
        if debug_info is not None:
            self._debug_number[self._top] = debug_info['target_number']
            self._time[self._top] = debug_info['time']
        super()._add_sample(
            observation,
            action,
            reward,
            terminal,
            final_state
        )

    def _get_trajectories(self, start_indices):
        trajs = super()._get_trajectories(start_indices)
        trajs['debug_numbers'] = subsequences(
            self._debug_number,
            start_indices,
            self._subtraj_length,
        )
        trajs['time'] = subsequences(
            self._time,
            start_indices,
            self._subtraj_length,
        )
        return trajs
