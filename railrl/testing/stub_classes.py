import numpy as np
from rllab.envs.base import Env
from rllab.spaces.box import Box


class StubEnv(Env):
    def reset(self):
        pass

    def step(self, action):
        pass

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return 99999

    @property
    def observation_space(self):
        return self._observation_space

    def __init__(self):
        low = np.array([0.])
        high = np.array([1.])
        self._action_space = Box(low, high)
        self._observation_space = Box(low, high)
