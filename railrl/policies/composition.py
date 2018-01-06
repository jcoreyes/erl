import numpy as np

from railrl.policies.base import SerializablePolicy, Policy
from railrl.torch.naf import NafPolicy
from railrl.core.serializable import Serializable


class CombinedNafPolicy(SerializablePolicy, Serializable):
    def __init__(
            self,
            policy1: NafPolicy,
            policy2: NafPolicy,
    ):
        Serializable.quick_init(self, locals())
        self.policy1 = policy1
        self.policy2 = policy2

    def get_action(self, obs):
        mu1, P1 = self.policy1.get_action_and_P_matrix(obs)
        mu2, P2 = self.policy2.get_action_and_P_matrix(obs)
        inv = np.linalg.inv(P1 + P2)
        return inv @ (P1 @ mu1 + P2 @ mu2), {}

    def log_diagnostics(self, paths):
        pass


class AveragerPolicy(Policy, Serializable):
    def __init__(self, policy1, policy2):
        Serializable.quick_init(self, locals())
        self.policy1 = policy1
        self.policy2 = policy2

    def get_action(self, obs):
        action1, info_dict1 = self.policy1.get_action(obs)
        action2, info_dict2 = self.policy2.get_action(obs)
        return (action1 + action2) / 2, dict(info_dict1, **info_dict2)

    def log_diagnostics(self, paths):
        pass
