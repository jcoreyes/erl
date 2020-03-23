import numpy as np
import torch
from torch import nn as nn

from railrl.policies.base import ExplorationPolicy, Policy
from railrl.torch.core import eval_np
from railrl.torch.distributions import TanhNormal, Normal, GaussianMixture
from railrl.torch.networks import Mlp, CNN

import railrl.torch.pytorch_util as ptu
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicyAdapter(nn.Module, ExplorationPolicy):
    """
    Usage:

    ```
    obs_processor = ...
    policy = TanhGaussianPolicyAdapter(obs_processor)
    ```
    """

    def __init__(
            self,
            obs_processor,
            obs_processor_output_dim,
            action_dim,
            hidden_sizes,
    ):
        super().__init__()
        self.obs_processor = obs_processor
        self.obs_processor_output_dim = obs_processor_output_dim
        self.mean_and_log_std_net = Mlp(
            hidden_sizes=hidden_sizes,
            output_size=action_dim*2,
            input_size=obs_processor_output_dim,
        )
        self.action_dim = action_dim

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = self.obs_processor(obs)
        h = self.mean_and_log_std_net(h)
        mean, log_std = torch.split(h, self.action_dim, dim=1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            tanh_normal = TanhNormal(mean, std)
            mean_action_log_prob = tanh_normal.log_prob(
                torch.tanh(mean),
                pre_tanh_value=mean,
            )
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


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
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
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
            std = torch.from_numpy(np.array([self.std, ])).float().to(ptu.device)
            log_std = torch.log(std) # self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        tanh_normal = TanhNormal(mean, std)
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            tanh_normal = TanhNormal(mean, std)
            mean_action_log_prob = tanh_normal.log_prob(
                torch.tanh(mean),
                pre_tanh_value=mean,
            )
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value, tanh_normal
        )

    def logprob(self, action, mean, std):
        # import ipdb; ipdb.set_trace()
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

class GaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(ptu.zeros(action_dim, requires_grad=True))
            else:
                error
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            # log_std = self.last_fc_log_std(h)
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                error
            log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(ptu.device)
            log_std = torch.log(std) # self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        normal = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            if return_log_prob:
                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()
                log_prob = normal.log_prob(action)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            normal = Normal(mean, std)
            mean_action_log_prob = normal.log_prob(mean)
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value, normal,
        )

class GaussianMixturePolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            num_gaussians=1,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim * num_gaussians,
            init_w=init_w,
            # output_activation=torch.tanh,
            **kwargs
        )
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]

            if self.std_architecture == "shared":
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim * num_gaussians)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(ptu.zeros(action_dim * num_gaussians, requires_grad=True))
            else:
                error
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        self.last_fc_weights = nn.Linear(last_hidden_size, num_gaussians)
        self.last_fc_weights.weight.data.uniform_(-init_w, init_w)
        self.last_fc_weights.bias.data.uniform_(-init_w, init_w)

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            # log_std = self.last_fc_log_std(h)
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            # log_std = torch.sigmoid(self.last_fc_log_std(h))
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                error
            log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(self.std)
            log_std = self.log_std

        weights = F.softmax(self.last_fc_weights(h)).reshape((-1, self.num_gaussians, 1))
        mixture_means = mean.reshape((-1, self.action_dim, self.num_gaussians, ))
        mixture_stds = std.reshape((-1, self.action_dim, self.num_gaussians, ))
        dist = GaussianMixture(mixture_means, mixture_stds, weights)

        # normal = Normal(mean, std)
        # import ipdb; ipdb.set_trace()

        mean = dist.mean()

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        if deterministic:
            action = mean
        else:
            # normal = Normal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action = dist.rsample()
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                if reparameterize is True:
                    action = dist.rsample()
                else:
                    action = dist.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            normal = Normal(mean, std)
            mean_action_log_prob = normal.log_prob(mean)
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, None, dist,
        )

# noinspection PyMethodOverriding
class TanhCNNGaussianPolicy(CNN, ExplorationPolicy):
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
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            init_w=init_w,
            **kwargs
        )
        obs_dim = self.input_width * self.input_height
        action_dim = self.output_size
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(self.hidden_sizes) > 0:
                last_hidden_size = self.hidden_sizes[-1]
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
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = super().forward(obs, return_last_activations=True)

        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            tanh_normal = TanhNormal(mean, std)
            mean_action_log_prob = tanh_normal.log_prob(
                torch.tanh(mean),
                pre_tanh_value=mean,
            )
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(Policy, ):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, *args, deterministic=False, **kwargs):
        return self.stochastic_policy.get_action(
            *args, deterministic=True, **kwargs
        )

    def to(self, device):
        self.stochastic_policy.to(device)

    def load_state_dict(self, stochastic_state_dict):
        self.stochastic_policy.load_state_dict(stochastic_state_dict)

    def state_dict(self):
        return self.stochastic_policy.state_dict()
