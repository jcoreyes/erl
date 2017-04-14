from railrl.exploration_strategies.base import RawExplorationStrategy


class NoopStrategy(RawExplorationStrategy):
    """
    Exploration strategy that does nothing.
    """

    def __init__(self, **kwargs):
        pass

    def get_action(self, t, observation, policy, **kwargs):
        return policy.get_action(observation)

    def get_action_from_raw_action(self, action, **kwargs):
        return action
