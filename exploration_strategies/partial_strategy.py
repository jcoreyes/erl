import numpy as np
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
        action, _ = policy.get_action(observation)
        return self.get_action_from_raw_action(action)

    def get_action_from_raw_action(self, action, **kwargs):
        action = np.squeeze(action)

        actions_split = list(self._product_space.unflatten(action))
        partial_action = actions_split[self._component]
        new_action = self._wrapped_es.get_action_from_raw_action(partial_action)

        actions_split[self._component] = new_action
        final_action = self._product_space.flatten(actions_split)
        print("---")
        print(final_action.shape)
        final_action.squeeze()
        print(final_action.shape)
        final_action = final_action.squeeze()
        print(final_action.shape)
        return final_action

        # list_of_actions_split = self._product_space.unflatten_n(action)
        # new_list_of_actions_split = [
        #     list(actions_split) for actions_split in list_of_actions_split
        # ]
        # new_partial_actions = []
        #
        # for actions_split in list_of_actions_split:
        #     partial_action = actions_split[self._component]
        #     new_partial_actions.append(
        #         self._wrapped_es.get_action_from_raw_action(partial_action)
        #     )
        #
        # for i, new_partial_action in enumerate(new_partial_actions):
        #     new_list_of_actions_split[i][self._component] = new_partial_action
        #
        # return self._product_space.flatten_n(list_of_actions_split)
