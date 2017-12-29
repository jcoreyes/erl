"""
Basic, flat networks.

This is basically as re-write of the networks.py file but for tdm.py rather
than sdql.py
"""
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from railrl.pythonplusplus import identity
from railrl.state_distance.util import split_tau, extract_goals, split_flat_obs
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.networks import Mlp
import railrl.torch.pytorch_util as ptu


class StructuredQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            output_size,
            hidden_sizes,
            internal_gcm=True,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + 1,
            output_size=output_size,
            **kwargs
        )
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.internal_gcm = internal_gcm

    def forward(self, flat_obs, actions):
        h = torch.cat((flat_obs, actions), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        if self.internal_gcm:
            _, goals, _ = split_flat_obs(
                flat_obs, self.observation_dim, self.goal_dim
            )
            return - torch.abs(goals - self.last_fc(h))
        return - torch.abs(self.last_fc(h))


class OneHotTauQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a one-hot vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + max_tau + 1,
            output_size=output_size,
            **kwargs
        )
        self.max_tau = max_tau

    def forward(self, flat_obs, action=None):
        obs, taus = split_tau(flat_obs)
        if action is not None:
            h = torch.cat((obs, action), dim=1)
        else:
            h = obs
        batch_size = h.size()[0]
        y_binary = ptu.FloatTensor(batch_size, self.max_tau + 1)
        y_binary.zero_()
        t = taus.data.long()
        t = torch.clamp(t, min=0)
        y_binary.scatter_(1, t, 1)
        if action is not None:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),
                action
            ), dim=1)
        else:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),
            ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))


class BinaryStringTauQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a binary string vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        self.max_tau = np.unpackbits(np.array(max_tau, dtype=np.uint8))
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + len(self.max_tau),
            output_size=output_size,
            **kwargs
        )

    def forward(self, flat_obs, action=None):
        obs, taus = split_tau(flat_obs)
        if action is not None:
            h = torch.cat((obs, action), dim=1)
        else:
            h = obs
        batch_size = taus.size()[0]
        y_binary = make_binary_tensor(taus, len(self.max_tau), batch_size)

        if action is not None:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),
                action
            ), dim=1)
        else:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),

            ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))


def make_binary_tensor(tensor, max_len, batch_size):
    t = tensor.data.numpy().astype(int).reshape(batch_size)
    binary = (((t[:,None] & (1 << np.arange(max_len)))) > 0).astype(int)
    binary = torch.from_numpy(binary)
    binary  = binary.float()
    binary = binary.view(batch_size, max_len)
    return binary


class TauVectorQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a binary string vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            tau_vector_len=0,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + self.tau_vector_len,
            output_size=output_size,
            **kwargs
        )

    def forward(self, flat_obs, action=None):
        obs, taus = split_tau(flat_obs)
        if action is not None:
            h = torch.cat((obs, action), dim=1)
        else:
            h = obs
        batch_size = h.size()[0]
        tau_vector = torch.zeros((batch_size, self.tau_vector_len)) + taus.data
        if action is not None:
            h = torch.cat((
                obs,
                ptu.Variable(tau_vector),
                action
            ), dim=1)
        else:
            h = torch.cat((
                obs,
                ptu.Variable(tau_vector),

            ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))


class SeparateFirstLayerMlp(PyTorchModule):
    def __init__(
            self,
            first_input_size,
            second_input_size,
            hidden_sizes,
            output_size,
            init_w=3e-3,
            first_layer_activation=F.relu,
            first_layer_init=ptu.fanin_init,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []

        self.first_input = nn.Linear(first_input_size, first_input_size)
        hidden_init(self.first_input.weight)
        self.first_input.bias.data.fill_(b_init_value)

        self.second_input = nn.Linear(second_input_size, second_input_size)
        hidden_init(self.second_input.weight)
        self.second_input.bias.data.fill_(b_init_value)

        in_size = first_input_size+second_input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, first_input, second_input):
        h1 = self.hidden_activation(self.first_input(first_input))
        h2 = self.hidden_activation(self.second_input(second_input))
        h = torch.cat((h1, h2), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class TauVectorSeparateFirstLayerQF(SeparateFirstLayerMlp):
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            tau_vector_len=0,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau

        super().__init__(
            hidden_sizes=hidden_sizes,
            first_input_size=observation_dim + action_dim + goal_dim,
            second_input_size=self.tau_vector_len,
            output_size=output_size,
            **kwargs
        )

    def forward(self, flat_obs, action=None):
        obs, taus = split_tau(flat_obs)
        if action is not None:
            h = torch.cat((obs, action), dim=1)
        else:
            h = obs

        batch_size = h.size()[0]
        tau_vector = Variable(torch.zeros((batch_size, self.tau_vector_len)) + taus.data)
        return - torch.abs(super().forward(h, tau_vector))