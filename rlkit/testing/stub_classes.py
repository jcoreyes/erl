import numpy as np
from gym.spaces import Box

from rlkit.exploration_strategies.base import RawExplorationStrategy


class StubEnv(object):
    def __init__(self):
        low = np.array([0.])
        high = np.array([1.])
        self._action_space = Box(low, high)
        self._observation_space = Box(low, high)

    def reset(self):
        pass

    def step(self, action):
        return 0, 0, 0, {}

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return 99999

    @property
    def observation_space(self):
        return self._observation_space


class StubPolicy(object):
    def __init__(self, action):
        self._action = action

    def get_action(self, *arg, **kwargs):
        return self._action, {}


class AddEs(RawExplorationStrategy):
    """
    return action + constant
    """
    def __init__(self, number):
        self._number = number

    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        return self.get_action_from_raw_action(action)

    def get_action_from_raw_action(self, action, **kwargs):
        return self._number + action