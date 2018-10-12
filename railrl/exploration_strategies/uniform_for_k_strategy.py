import numpy as np
import numpy.random as nr

from railrl.exploration_strategies.base import RawExplorationStrategy
from railrl.core.serializable import Serializable


class UniformforKStrategy(RawExplorationStrategy, Serializable):
    def __init__(
            self,
            action_space,
            num_uniform_steps=0,
    ):
        Serializable.quick_init(self, locals())
        self.action_space=action_space
        self.num_uniform_steps=num_uniform_steps
        self._n_env_steps_total=0
        self.reset()

    def get_action_from_raw_action(self, action, **kwargs):
        if self._n_env_steps_total < self.num_uniform_steps:
            action = self.action_space.sample()
        self._n_env_steps_total+=1
        return action


    def get_actions_from_raw_actions(self, actions, t=0, **kwargs):
        return actions


