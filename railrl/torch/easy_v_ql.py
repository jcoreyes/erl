from collections import OrderedDict

import numpy as np
import torch.optim as optim
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import init

from railrl.torch.core import PyTorchModule
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.torch.ddpg import DDPG
from railrl.torch.online_algorithm import OnlineAlgorithm
import railrl.torch.pytorch_util as ptu
import railrl.torch.modules as M
from rllab.misc import logger, special


class EasyVQLearning(DDPG):
    """
    Continous action Q learning where the V function is easy:

    Q(s, a) = A(s, a) + V(s)

    The main thing is that the following needs to be enforced:

        max_a A(s, a) = 0

    """
    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        q_output = self.qf(obs, policy_actions)
        policy_loss = - q_output.mean()

        """
        Critic operations.
        """
        next_actions = self.policy(next_obs)
        # TODO: try to get this to work
        # next_actions = None
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
        )
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions)
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('QF Loss', qf_loss),
        ])

    def training_mode(self, mode):
        self.policy.train(mode)
        self.qf.train(mode)
        self.target_qf.train(mode)


class EasyVQFunction(PyTorchModule):
    """
    Parameterize Q function as the follows:

        Q(s, a) = A(s, a) + V(s)

    To ensure that max_a A(s, a) = 0, use the following form:

        A(s, a) = - diff(s, a)^T diag(exp(d(s))) diff(s, a)  *  f(s, a)^2

    where

        diff(s, a) = a - z(s)

    so that a = z(s) is at least one zero.

    d(s) and f(s, a) are arbitrary functions
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            diag_fc1_size,
            diag_fc2_size,
            af_fc1_size,
            af_fc2_size,
            zero_fc1_size,
            zero_fc2_size,
            vf_fc1_size,
            vf_fc2_size,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs_batchnorm = nn.BatchNorm1d(obs_dim)

        self.batch_square = M.BatchSquareDiagonal(action_dim)

        self.diag = nn.Sequential(
            nn.Linear(obs_dim, diag_fc1_size),
            nn.ReLU(),
            nn.Linear(diag_fc1_size, diag_fc2_size),
            nn.ReLU(),
            nn.Linear(diag_fc2_size, action_dim),
        )

        self.zero = nn.Sequential(
            nn.Linear(obs_dim, zero_fc1_size),
            nn.ReLU(),
            nn.Linear(zero_fc1_size, zero_fc2_size),
            nn.ReLU(),
            nn.Linear(zero_fc2_size, action_dim),
        )

        self.f = nn.Sequential(
            M.Concat(),
            nn.Linear(obs_dim + action_dim, af_fc1_size),
            nn.ReLU(),
            nn.Linear(af_fc1_size, af_fc2_size),
            nn.ReLU(),
            nn.Linear(af_fc2_size, 1),
        )

        self.vf = nn.Sequential(
            nn.Linear(obs_dim, vf_fc1_size),
            nn.ReLU(),
            nn.Linear(vf_fc1_size, vf_fc2_size),
            nn.ReLU(),
            nn.Linear(vf_fc2_size, 1),
        )

        self.apply(init_layer)

    def forward(self, obs, action):
        obs = self.obs_batchnorm(obs)
        V = self.vf(obs)
        if action is None:
            return V

        diag_values = torch.exp(self.diag(obs))
        diff = action - self.zero(obs)
        quadratic = self.batch_square(diff, diag_values)
        f = self.f((obs, action))
        AF = - quadratic * (f**2)

        return V + AF


def init_layer(layer):
    if isinstance(layer, nn.Linear):
        init.kaiming_normal(layer.weight)
        layer.bias.data.fill_(0)
    elif isinstance(layer, nn.BatchNorm1d):
        layer.reset_parameters()
