import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.policies.base import ExplorationPolicy
from railrl.torch.core import torch_ify, elem_or_tuple_to_numpy
from railrl.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from railrl.torch.networks import Mlp, CNN
from railrl.torch.networks.basic import MultiInputSequential
from railrl.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from railrl.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)


class LatentVariableModel(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
