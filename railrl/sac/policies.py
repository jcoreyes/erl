import torch
from torch.distributions import Normal
import numpy as np

from railrl.torch.distributions import TanhNormal
from railrl.torch.networks import Mlp
from railrl.policies.base import Policy
import railrl.torch.pytorch_util as ptu


class TanhGaussianPolicy(Mlp, Policy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            **kwargs
    ):
        if std is None:
            output_size = action_dim * 2
        else:
            output_size = action_dim
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=output_size,
            **kwargs,
        )
        self.std = std

    def get_action(self, obs_np, return_only_deterministic=False):
        self.foo = None
        obs = ptu.np_to_var(
            np.expand_dims(obs_np, 0)
        )
        action = self.__call__(
            obs,
            return_only_deterministic=return_only_deterministic,
        )
        action_np = ptu.get_numpy(action)
        return action_np, {}

    def forward(
            self,
            input,
            return_only_deterministic=False,
            return_log_prob=False,
    ):
        """
        :param input:
        :param return_only_deterministic: This takes precedence.
        :param return_log_prob: If true, return a sample and its log probability
        :return: A sample from the Gaussian
        """
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        if self.std is None:
            mean, std = torch.split(self.last_fc(h), 2, 1)
        else:
            mean = self.last_fc(h)
            std = self.std

        if return_only_deterministic:
            return torch.tanh(mean)

        tanh_normal = TanhNormal(mean, std)
        if return_log_prob:
            action, pre_tanh_value = tanh_normal.sample(
                return_pretanh_value=True
            )
            action, tanh_normal.log_prob(action, pre_tanh_value=mean)
        else:
            return tanh_normal.sample()

