import itertools

import numpy as np
from gym import spaces

from railrl.envs.multitask.reacher_env import GoalXYStateXYAndCosSinReacher2D
from railrl.core.serializable import Serializable


class DiscreteReacher2D(GoalXYStateXYAndCosSinReacher2D, Serializable):
    def __init__(self, num_bins=5):
        Serializable.quick_init(self, locals())
        super().__init__()
        self.num_bins = num_bins
        joint_ranges = []
        for low, high in zip(self.action_space.low, self.action_space.high):
            joint_ranges.append(
                np.linspace(low, high, num_bins)

            )
        self.idx_to_continuous_action = itertools.product(*joint_ranges)
        self.idx_to_continuous_action = list(map(
            lambda lst: np.array(lst),
            self.idx_to_continuous_action,
        ))
        self.action_space = spaces.Discrete(len(self.idx_to_continuous_action))

    def _step(self, a):
        if not self.action_space or not self.action_space.contains(a):
            continuous_action = a
        else:
            continuous_action = self.idx_to_continuous_action[a]
        return super()._step(continuous_action)
