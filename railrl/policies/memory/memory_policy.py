import abc
import numpy as np

from railrl.policies.nn_policy import NNPolicy


class MemoryPolicy(NNPolicy, metaclass=abc.ABCMeta):
    """
    A policy with memory states. The only difference is how this policy expands
    the observations individually.
    """
    @staticmethod
    def _unflatten_observation(observation):
        return tuple(np.expand_dims(o, axis=0) for o in observation)

    @staticmethod
    def _flatten_action(action):
        return tuple(np.squeeze(a) for a in action)
