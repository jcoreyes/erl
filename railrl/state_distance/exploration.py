import abc
import numpy as np
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.policies.base import ExplorationPolicy
from railrl.policies.state_distance import UniversalPolicy
from railrl.state_distance.util import merge_into_flat_obs
from railrl.torch.core import PyTorchModule
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

    def set_tau(self, tau):
        self.policy.set_tau(tau)


class MakeUniversal(PyTorchModule, UniversalExplorationPolicy):
    def __init__(self, policy):
        self.save_init_params(locals())
        super().__init__()
        self.policy = policy

    def get_action(self, observation):
        new_obs = merge_into_flat_obs(
            obs=observation,
            goals=self._goal_np,
            num_steps_left=self._tau_np,
        )
        return self.policy.get_action(new_obs)

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)
