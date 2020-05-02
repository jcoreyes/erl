import abc

from torch import nn

from railrl.torch.distributions import (
    Beta,
    Distribution,
    GaussianMixture,
    GaussianMixtureFull,
    Normal,
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


class BetaDistributionGenerator(ModuleToDistributionGenerator):
    def forward(self, *input):
        alpha, beta = super().forward(*input)
        return Beta(alpha, beta)


class GaussianDistributionGenerator(ModuleToDistributionGenerator):
    def forward(self, *input):
        mean, log_std = super().forward(*input)
        std = log_std.exp()
        return Normal(mean, std)


class GaussianMixtureDistributionGenerator(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixture(mixture_means, mixture_stds, weights)


class GaussianMixtureFullDistributionGenerator(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixtureFull(mixture_means, mixture_stds, weights)


class TanhGaussianDistributionGenerator(ModuleToDistributionGenerator):
    def forward(self, *input):
        mean, log_std = super().forward(*input)
        std = log_std.exp()
        return TanhNormal(mean, std)
