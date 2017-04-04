from rllab.exploration_strategies.base import ExplorationStrategy
from rllab.spaces.product import Product


class NoopStrategy(ExplorationStrategy):
    """
    Exploration strategy that does nothing.
    """
    def get_action(self, t, observation, policy, **kwargs):
        return policy.get_action(observation)
