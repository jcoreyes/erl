import abc

from railrl.policies.base import Policy
from rllab.exploration_strategies.base import ExplorationStrategy


class RawExplorationStrategy(ExplorationStrategy, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action_from_raw_action(self, action, **kwargs):
        pass

    def get_action(self, t, observation, policy, **kwargs):
        action, agent_info = policy.get_action(observation)
        return self.get_action_from_raw_action(action, **kwargs), agent_info

    def reset(self):
        pass


class PolicyWrappedWithExplorationStrategy(Policy):
    def __init__(
            self,
            exploration_strategy: ExplorationStrategy,
            policy: Policy,
    ):
        self.es = exploration_strategy
        self.policy = policy
        self.t = 0

    def set_num_steps_total(self, t):
        self.t = t

    def get_action(self, obs):
        return self.es.get_action(self.t, obs, self.policy)

    def reset(self):
        self.es.reset()
        self.policy.reset()