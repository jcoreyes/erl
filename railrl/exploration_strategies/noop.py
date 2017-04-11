from railrl.exploration_strategies.base import RawExplorationStrategy


class NoopStrategy(RawExplorationStrategy):
    """
    Exploration strategy that does nothing.
    """
    def get_action(self, t, observation, policy, **kwargs):
        return policy.get_action(observation)

    def get_action_from_raw_action(self, action, **kwargs):
        return action
