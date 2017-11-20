import abc
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.policies.base import ExplorationPolicy
from railrl.policies.state_distance import UniversalPolicy
from rllab.exploration_strategies.base import ExplorationStrategy


class UniversalExplorationPolicy(
    UniversalPolicy,
    ExplorationPolicy,
    metaclass=abc.ABCMeta,
):
    pass


class UniversalPolicyWrappedWithExplorationStrategy(
    PolicyWrappedWithExplorationStrategy,
    UniversalExplorationPolicy,
):
    def __init__(
            self,
            exploration_strategy: ExplorationStrategy,
            policy: UniversalPolicy,
    ):
        super().__init__(exploration_strategy, policy)

    def set_goal(self, goal_np):
        self.policy.set_goal(goal_np)

    def set_discount(self, discount):
        self.policy.set_discount(discount)
