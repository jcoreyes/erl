import numpy as np
import torch
from torch import nn as nn

from railrl.policies.base import ExplorationPolicy, Policy
from railrl.torch.distributions import TanhNormal
from railrl.torch.networks import Mlp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


# noinspection PyMethodOverriding
class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_expected_log_prob=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_expected_log_prob: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                action = tanh_normal.sample()

        if return_expected_log_prob:
            expected_log_prob = - (
                log_std + 0.5 + np.log(2 * np.pi) / 2
            )
            # shoot, idk how to compute the expected log prob for the tanh term
            # TODO(vitchyr): fix
            expected_log_prob = expected_log_prob.sum(dim=1, keepdim=True)
        if return_log_prob_of_mean:
            tanh_normal = TanhNormal(mean, std)
            mean_action_log_prob = tanh_normal.log_prob(
                torch.tanh(mean),
                pre_tanh_value=mean,
            )
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)
        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob
        )


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)
