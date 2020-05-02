"""
Add custom distributions in addition to th existing ones
"""
import torch
from torch.distributions import Categorical, OneHotCategorical
from torch.distributions import Normal as TorchNormal
from torch.distributions import Beta as TorchBeta
from torch.distributions import Distribution as TorchDistribution
from railrl.misc.eval_util import create_stats_ordered_dict
import railrl.torch.pytorch_util as ptu
import numpy as np
from collections import OrderedDict


class Distribution(TorchDistribution):
    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}


class Delta(Distribution):
    """A deterministic distribution"""
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value.detach()

    def rsample(self):
        return self.value

    @property
    def mean(self):
        return self.value

    @property
    def variance(self):
        return 0

    @property
    def entropy(self):
        return 0


class Beta(TorchBeta, Distribution):
    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'alpha',
            ptu.get_numpy(self.concentration0),
        ))
        stats.update(create_stats_ordered_dict(
            'beta',
            ptu.get_numpy(self.concentration1),
        ))
        stats.update(create_stats_ordered_dict(
            'entropy',
            ptu.get_numpy(self.entropy()),
        ))
        return stats


class Beta(TorchBeta, Distribution):
    def get_diagnostics(self, ):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'alpha',
            ptu.get_numpy(self.concentration0),
        ))
        stats.update(create_stats_ordered_dict(
            'beta',
            ptu.get_numpy(self.concentration1),
        ))
        stats.update(create_stats_ordered_dict(
            'entropy',
            ptu.get_numpy(self.entropy()),
        ))
        return stats

    def get_entropy(self, ):
        return self.entropy()

    def get_mle(self, ):
        return self.mean()


class Normal(TorchNormal, Distribution):
    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'mean',
            ptu.get_numpy(self.loc),
        ))
        stats.update(create_stats_ordered_dict(
            'std',
            ptu.get_numpy(self.scale),
        ))
        stats.update(create_stats_ordered_dict(
            'log_std',
            ptu.get_numpy(torch.log(self.scale)),
        ))
        stats.update(create_stats_ordered_dict(
            'entropy',
            ptu.get_numpy(self.get_entropy()),
        ))
        return stats

    def get_entropy(self):
        log_std = torch.log(self.scale)
        entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
        return entropy.sum(dim=1, keepdim=True)



class GaussianMixture(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = Normal(normal_means, normal_stds)
        self.normals = [Normal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = weights
        self.categorical = OneHotCategorical(self.weights[:, :, 0])

    def log_prob(self, value, ):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]
        log_p = torch.stack(log_p, -1)
        log_p = log_p.sum(dim=1)
        log_weights = torch.log(self.weights[:, :, 0])
        lp = log_weights + log_p
        m = lp.max(dim=1, keepdim=True)[0]  # log-sum-exp numerical stability trick
        log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=1, keepdim=True))
        return log_p_mixture

    def sample(self):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def rsample(self):
        z = (
            self.normal_means +
            self.normal_stds *
                Normal(
                    ptu.zeros(self.normal_means.size()),
                    ptu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def mle_estimate(self):
        """Return the mean of the most likely component.

        This often computes the mode of the distribution, but not always.
        """
        c = ptu.zeros(self.weights.shape[:2])
        ind = torch.argmax(self.weights, dim=1) # [:, 0]
        c.scatter_(1, ind, 1)
        s = torch.matmul(self.normal_means, c[:, :, None])
        return torch.squeeze(s, 2)

    def __repr__(self):
        s = "GaussianMixture(normal_means=%s, normal_stds=%s, weights=%s)"
        return s % (self.normal_means, self.normal_stds, self.weights)


epsilon = 0.001


class GaussianMixtureFull(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[-1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = Normal(normal_means, normal_stds)
        self.normals = [Normal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = (weights + epsilon) / (1 + epsilon * self.num_gaussians)
        assert (self.weights > 0).all()
        self.categorical = Categorical(self.weights)

    def log_prob(self, value, ):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]
        log_p = torch.stack(log_p, -1)
        log_weights = torch.log(self.weights)
        lp = log_weights + log_p
        m = lp.max(dim=2, keepdim=True)[0]  # log-sum-exp numerical stability trick
        log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=2, keepdim=True))
        return torch.squeeze(log_p_mixture, 2)

    def sample(self):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.gather(z, dim=2, index=c)
        return s[:, :, 0]

    def rsample(self):
        z = (
            self.normal_means +
            self.normal_stds *
                Normal(
                    ptu.zeros(self.normal_means.size()),
                    ptu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.gather(z, dim=2, index=c)
        return s[:, :, 0]

    def mle_estimate(self):
        """Return the mean of the most likely component.

        This often computes the mode of the distribution, but not always.
        """
        ind = torch.argmax(self.weights, dim=2)[:, :, None]
        means = torch.gather(self.normal_means, dim=2, index=ind)
        return torch.squeeze(means, 2)

    def __repr__(self):
        s = "GaussianMixture(normal_means=%s, normal_stds=%s, weights=%s)"
        return s % (self.normal_means, self.normal_stds, self.weights)


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob_from_pre_tanh(self, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        This formula is mathematically equivalent to log(1 - tanh(x)^2).

        Derivation:

        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        return self.normal.log_prob(pre_tanh_value) - 2. * (
            ptu.from_numpy(np.log([2.]))
            - pre_tanh_value
            - torch.nn.functional.softplus(-2. * pre_tanh_value)
        )

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            # errors or instability at values near 1
            value = torch.clamp(value, -0.999999, 0.999999)
            pre_tanh_value = torch.log(1+value) / 2 - torch.log(1-value) / 2
        return self.log_prob_from_pre_tanh(pre_tanh_value)

    def rsample_with_pretanh(self):
        z = (
            self.normal_mean +
            self.normal_std *
                Normal(
                    ptu.zeros(self.normal_mean.size()),
                    ptu.ones(self.normal_std.size())
                ).sample()
        )
        return torch.tanh(z), z

    def sample(self):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value.detach()

    def rsample(self):
        """
        Sampling in the reparameterization case.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value

    def sample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        value, pre_tanh_value = value.detach(), pre_tanh_value.detach()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)
