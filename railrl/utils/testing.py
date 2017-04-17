from railrl.exploration_strategies.base import RawExplorationStrategy


class StubPolicy(object):
    def __init__(self, action):
        self._action = action

    def get_action(self, *arg, **kwargs):
        return self._action, {}


class AddEs(RawExplorationStrategy):
    """
    return action + constant
    """
    def __init__(self, number):
        self._number = number

    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        return self.get_action_from_raw_action(action)

    def get_action_from_raw_action(self, action, **kwargs):
        return self._number + action
