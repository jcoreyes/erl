from rllab.exploration_strategies.base import ExplorationStrategy
from rllab.spaces.product import Product


class NoopStrategy(ExplorationStrategy):
    """
    Exploration strategy that does nothing.
    """
    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        return action

    def get_action_from_raw_action(self, action, **kwargs):
        return action
