from railrl.exploration_strategies.base import RawExplorationStrategy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import numpy as np


class UniformStrategy(RawExplorationStrategy, Serializable):
    """
    This strategy adds noise sampled uniformly to the action taken by the
    deterministic policy.
    """
    def __init__(self, action_space, low=0., high=1.):
        assert isinstance(action_space, Box)
        assert len(action_space.shape) == 1
        Serializable.quick_init(self, locals())
        self._low = low
        self._high = high
        self._action_space = action_space

    def get_action_from_raw_action(self, action, t=None, **kwargs):
        return np.clip(
            action + np.random.uniform(
                self._low,
                self._high,
                size=action.shape,
            ),
            self._action_space.low,
            self._action_space.high,
        )
