import numpy as np
from rllab.policies.base import Policy


class ZeroPolicy(Policy):
    """
    Policy that always outputs zero.
    """

    def __init__(
            self,
            action_dim,
    ):
        self.action_dim = action_dim

    def get_params_internal(self, **tags):
        return None

    def get_action(self, obs):
        return np.zeros(self.action_dim), {}

    def reset(self):
        pass
