"""
Add custom distributions in addition to th existing ones
"""
import torch
from torch.distributions import Distribution, Normal
import railrl.torch.pytorch_util as ptu

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

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
                Normal(
                    torch.zeros(self.normal_mean.size()),
                    torch.ones(self.normal_std.size())
                ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

class SigmoidNormal(Distribution):
    """
    Represent distribution of X where
        X ~ sigmoid(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-8):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_sigmoid_value=False):
        z = self.normal.sample_n(n)
        if return_pre_sigmoid_value:
            return torch.sigmoid(z), z
        else:
            return torch.sigmoid(z)

    def log_prob(self, value, pre_sigmoid_value=None):
        """

        :param value: some value, x
        :param pre_sigmoid_value: arctanh(x)
        :return:
        """
        if pre_sigmoid_value is None:
            pre_sigmoid_value = torch.log((value+self.epsilon) / ((1-value)+self.epsilon))
        return self.normal.log_prob(pre_sigmoid_value) -2 * torch.log(1-value)

    def sample(self, pre_sigmoid_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if pre_sigmoid_value:
            return torch.sigmoid(z), z
        else:
            return torch.sigmoid(z)

    def rsample(self, return_pre_sigmoid_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = self.normal_mean + \
            self.normal_std * \
            ptu.Variable(
                Normal(torch.zeros(self.normal_mean.size()), torch.ones(self.normal_std.size())).sample(),
                requires_grad=False)

        if return_pre_sigmoid_value:
            return torch.sigmoid(z), z
        else:
            return torch.sigmoid(z)
