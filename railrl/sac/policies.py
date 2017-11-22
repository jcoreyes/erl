import torch
from torch.distributions import Normal
import numpy as np

from railrl.torch.networks import Mlp
from railrl.policies.base import Policy
import railrl.torch.pytorch_util as ptu


class TanhGaussianPolicy(Mlp, Policy):
    def get_action(self, obs, return_only_deterministic=False):
        self.foo = None
        obs = ptu.np_to_var(
            np.expand_dims(obs_np, 0)
        )
        action = self.__call__(
            obs,
            return_only_deterministic=return_only_deterministic,
        )
        mean = action.squeeze(0)
        action_np = ptu.get_numpy(action), {}
        action_np =

        return action_np, {}

    def forward(self, input, return_only_deterministic=False, return_log_prob=False):
        """
        :param input:
        :param return_only_deterministic: This takes precedence.
        :param return_log_prob: If true, return a sample and its log probability
        :return: A sample from the Gaussian
        """
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if return_log_prob:


        else:
            return self.output_activation(self.last_fc(h))

