import abc
import numpy as np

from railrl.policies.nn_policy import NNPolicy


class MemoryPolicy(NNPolicy, metaclass=abc.ABCMeta):
    """
    A policy with memory states. The main thing is that it needs to expand
    the observations individually.
    """

    def get_action(self, observation):
        new_observation = self._preprocess_observation(observation)
        action = self.sess.run(
            self.output,
            {
                self.observation_input: new_observation,
            }
        )
        return action, {}

    @staticmethod
    def _preprocess_observation(observation):
        return tuple(np.expand_dims(o, axis=0) for o in observation)
