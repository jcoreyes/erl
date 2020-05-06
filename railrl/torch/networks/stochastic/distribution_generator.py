import abc

from torch import nn

from railrl.torch.distributions import (
    Beta,
    Distribution,
    GaussianMixture,
    GaussianMixtureFull,
    MultivariateDiagonalNormal,
    TanhNormal,
)
from railrl.torch.networks.basic import MultiInputSequential


class DistributionGenerator(nn.Module, metaclass=abc.ABCMeta):
    def forward(self, *input, **kwarg) -> Distribution:
        raise NotImplementedError


class ModuleToDistributionGenerator(
    MultiInputSequential,
    DistributionGenerator,
    metaclass=abc.ABCMeta
):
    pass


class Beta(ModuleToDistributionGenerator):
    def forward(self, *input):
        alpha, beta = super().forward(*input)
        return Beta(alpha, beta)


class Gaussian(ModuleToDistributionGenerator):
    def forward(self, *input):
        mean, log_std = super().forward(*input)
        std = log_std.exp()
        return MultivariateDiagonalNormal(mean, std)


class GaussianMixture(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixture(mixture_means, mixture_stds, weights)


class GaussianMixtureFull(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixtureFull(mixture_means, mixture_stds, weights)


class TanhGaussian(ModuleToDistributionGenerator):
    def forward(self, *input):
        mean, log_std = super().forward(*input)
        std = log_std.exp()
        return TanhNormal(mean, std)
