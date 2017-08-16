from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.ddpg import DDPG


class FeedForwardZFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_size,
            embedded_hidden_size,
            num_bins,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            batchnorm_obs=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size
        self.hidden_init = hidden_init

        self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(observation_hidden_size + action_dim,
                                     embedded_hidden_size)
        self.last_fc = nn.Linear(embedded_hidden_size, num_bins)

        self.init_weights(init_w)
        self.batchnorm_obs = batchnorm_obs
        if self.batchnorm_obs:
            self.bn_obs = nn.BatchNorm1d(obs_dim)

    def init_weights(self, init_w):
        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        self.hidden_init(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        if self.batchnorm_obs:
            obs = self.bn_obs(obs)
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return F.softmax(self.last_fc(h))


class DistributionalDDPG(DDPG):
    def __init__(
            self,
            env,
            zf,
            policy,
            exploration_strategy,
            num_bins,
            returns_min,
            returns_max,
            **kwargs
    ):
        super().__init__(
            env,
            zf,
            policy,
            exploration_strategy,
            **kwargs
        )
        self.num_bins = num_bins
        self.returns_min = returns_min
        self.returns_max = returns_max
        self.bin_width = (returns_max - returns_min ) / num_bins
        atom_values_batch = np.expand_dims(
            np.linspace(returns_min, returns_max, num_bins),
            0,
        ).repeat(self.batch_size, 0)
        self.atom_values = ptu.np_to_var(atom_values_batch[0:1, :])
        self.atom_values_batch = ptu.np_to_var(atom_values_batch)

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
        z_output = self.qf(obs, policy_actions)  # BATCH_SIZE x NUM_BINS
        q_output = (z_output * self.atom_values).sum(1)
        policy_loss = - q_output.mean()

        """
        Critic operations.
        """
        next_actions = self.target_policy(next_obs)
        target_qf_histogram = self.target_qf(
            next_obs,
            next_actions,
        )
        z_target = ptu.Variable(self.batch_size, 1)
        for j in range(self.num_bins):
            atom_value = self.atom_values[:, j:]
            projected_returns = rewards + (1. - terminals) * self.discount * (
                atom_value
            )
            bin_values = (projected_returns - self.returns_min) / self.bin_width
            lower_bin_indices = torch.floor(bin_values)
            upper_bin_indices = torch.ceil(bin_values)
            z_target[lower_bin_indices] += target_qf_histogram[j] * (
                upper_bin_indices - bin_values
            )
            z_target[upper_bin_indices] += target_qf_histogram[j] * (
                bin_values - lower_bin_indices
            )

        # noinspection PyUnresolvedReferences
        z_target = z_target.detach()
        z_pred = self.qf(obs, actions)
        qf_loss = self.qf_criterion(z_pred, z_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Z targets', z_target),
            ('Z predictions', z_pred),
            ('QF Loss', qf_loss),
        ])

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = OrderedDict()

        train_dict = self.get_train_dict(batch)
        for name in [
            'QF Loss',
            'Policy Loss',
        ]:
            tensor = train_dict[name]
            statistics_name = "{} {} Mean".format(stat_prefix, name)
            statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

        # for name in [
        #     'Bellman Errors',
        # ]:
        #     tensor = train_dict[name]
        #     statistics.update(create_stats_ordered_dict(
        #         '{} {}'.format(stat_prefix, name),
        #         ptu.get_numpy(tensor)
        #     ))

        return statistics
