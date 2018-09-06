import numpy as np

from gym.spaces import Dict

from railrl.core.serializable import Serializable
from railrl.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer

from multiworld.core.image_env import unormalize_image, normalize_image

import torch.multiprocessing as mp
import ctypes

class SharedObsDictRelabelingBuffer(Serializable, ObsDictRelabelingBuffer):
    """
    Same as an ObsDictRelabelingBuffer but the obs and next_obs are backed
    by multiprocessing arrays. The replay buffer size is also shared. The
    intended use case is for if one wants obs/next_obs to be shared between
    processes. Accesses are synchronized internally by locks (mp takes care
    of that). Technically, putting such large arrays in shared memory/requiring
    synchronized access can be extremely slow, but it seems ok empirically.

    This code also breaks a lot of functionality for the subprocess. For example,
    random_batch is incorrect as actions and _idx_to_future_obs_idx are not
    shared. If the subprocess needs all of the functionality, a mp.Array
    must be used for all numpy arrays in the replay buffer.

    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        self._shared_size = mp.Value(ctypes.c_long, 0)
        ObsDictRelabelingBuffer.__init__(self, *args, **kwargs)
        self._shared_obs_info = {}
        self._shared_next_obs_info = {}

        for obs_key, obs_arr in self._obs.items():
            ctype = ctypes.c_double
            if obs_arr.dtype == np.uint8:
               ctype = ctypes.c_uint8

            self._shared_obs_info[obs_key] = (
                mp.Array(ctype, obs_arr.size),
                obs_arr.dtype,
                obs_arr.shape,
            )
            self._shared_next_obs_info[obs_key] = (
                mp.Array(ctype, obs_arr.size),
                obs_arr.dtype,
                obs_arr.shape,
            )

            self._obs[obs_key] = to_np(*self._shared_obs_info[obs_key])
            self._next_obs[obs_key] = to_np(*self._shared_next_obs_info[obs_key])


    def add_path(self, path):
        super().add_path(path)

    def init_from(self, shared_obs_info, shared_next_obs_info, shared_size):
        """
        The intended use is to have a subprocess serialize/copy a
        SharedObsDictRelabelingBuffer instance and call init_from on the
        instance's shared variables. This can't be done during serialization
        since multiprocessing shared objects can't be serialized and must be
        passed directly to the subprocess as an argument to the fork call.
        """
        self._shared_obs_info = shared_obs_info
        self._shared_next_obs = shared_next_obs_info
        for obs_key in self._shared_obs_info.keys():
            self._obs[obs_key] = to_np(*self._shared_obs_info[obs_key])
            self._next_obs[obs_key] = to_np(*self._shared_next_obs_info[obs_key])
        self._shared_size = shared_size

    @property
    def _size(self):
        return self._shared_size.value

    @_size.setter
    def _size(self, size):
        self._shared_size.value = size


def to_np(shared_arr, np_dtype, shape):
    return np.frombuffer(shared_arr.get_obj(), dtype=np_dtype).reshape(shape)
