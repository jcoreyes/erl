import abc
from rllab.exploration_strategies.base import ExplorationStrategy


class RawExplorationStrategy(ExplorationStrategy, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action_from_raw_action(self, action, **kwargs):
        pass

    @abc.abstractmethod
    def get_action(self, t, observation, policy, **kwargs):
        pass

    def reset(self):
        pass
