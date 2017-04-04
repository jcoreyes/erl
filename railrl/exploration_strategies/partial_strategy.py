from rllab.exploration_strategies.base import ExplorationStrategy
from rllab.spaces.product import Product


# TODO(vpong): test this
class PartialStrategy(ExplorationStrategy):
    """
    Only apply an exploration strategy to part of the action space. This is
    used when you have a product action space, and you only want to add an
    exploration strategy to part of that action.
    """

    def __init__(
            self,
            exploration_strategy: ExplorationStrategy,
            product_space: Product,
            component=0,
    ):
        """

        :param exploration_strategy: ExplorationStrategy
        :param product_space: Product space
        :param component: Apply the ES to the
        `component_to_apply_es_to`th component of the Product space.
        """
        self._wrapped_es = exploration_strategy
        self._product_space = product_space
        self._component = component
        assert isinstance(product_space, Product)
        assert 0 <= self._component <= len(self._product_space.components) - 1

    def get_action(self, t, observation, policy, **kwargs):
        action, agent_info = policy.get_action(observation)
        return self.get_action_from_raw_action(action), agent_info

    def get_action_from_raw_action(self, action, **kwargs):
        actions_split = list(action)
        partial_action = action[self._component]
        new_action = self._wrapped_es.get_action_from_raw_action(partial_action)
        actions_split[self._component] = new_action
        return tuple(actions_split)
