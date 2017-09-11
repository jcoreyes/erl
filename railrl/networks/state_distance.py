import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from railrl.policies.base import Policy
from railrl.policies.state_distance import UniversalPolicy
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.ddpg import elem_or_tuple_to_variable


class UniversalQfunction(PyTorchModule):
    """
    Represent Q(s, a, s_g, \gamma) with a two-alyer FF network.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_state_dim,
            obs_hidden_size,
            embed_hidden_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            w_weight_generator=ptu.fanin_init_weights_like,
            b_init_value=0.1,
            bn_input=False,
            dropout=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout = dropout
        next_layer_size = observation_dim + goal_state_dim + 1
        if bn_input:
            self.process_input = nn.BatchNorm1d(next_layer_size)
        else:
            self.process_input = identity

        self.obs_fc = nn.Linear(next_layer_size, obs_hidden_size)
        new_weight = w_weight_generator(self.obs_fc.weight.data)
        self.obs_fc.weight.data.copy_(new_weight)
        self.obs_fc.bias.data.fill_(b_init_value)

        self.embed_fc = nn.Linear(
            obs_hidden_size + action_dim,
            embed_hidden_size,
        )
        new_weight = w_weight_generator(self.embed_fc.weight.data)
        self.embed_fc.weight.data.copy_(new_weight)
        self.embed_fc.bias.data.fill_(b_init_value)

        next_layer_size = obs_hidden_size + action_dim

        if dropout:
            self.obs_dropout = nn.Dropout()
            self.embed_dropout = nn.Dropout()

        self.last_fc = nn.Linear(embed_hidden_size, 1)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(b_init_value)

    def forward(self, obs, action, goal_state, discount):
        h = torch.cat((obs, goal_state, discount), dim=1)
        h = self.process_input(h)
        h = self.hidden_activation(self.obs_fc(h))
        if self.dropout:
            h = self.obs_dropout(h)
        h = torch.cat((h, action), dim=1)
        h = self.hidden_activation(self.embed_fc(h))
        if self.dropout:
            h = self.embed_dropout(h)
        return self.output_activation(self.last_fc(h))


class FlatUniversalQfunction(PyTorchModule):
    """
    Represent Q(s, a, s_g, \gamma) with a two-alyer FF network.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_state_dim,
            obs_hidden_size,
            embed_hidden_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            w_weight_generator=ptu.fanin_init_weights_like,
            b_init_value=0.1,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        next_layer_size = observation_dim + goal_state_dim + + action_dim + 1

        self.obs_fc = nn.Linear(next_layer_size, obs_hidden_size)
        new_weight = w_weight_generator(self.obs_fc.weight.data)
        self.obs_fc.weight.data.copy_(new_weight)
        self.obs_fc.bias.data.fill_(b_init_value)

        self.embed_fc = nn.Linear(
            obs_hidden_size,
            embed_hidden_size,
        )
        new_weight = w_weight_generator(self.embed_fc.weight.data)
        self.embed_fc.weight.data.copy_(new_weight)
        self.embed_fc.bias.data.fill_(b_init_value)

        self.last_fc = nn.Linear(embed_hidden_size, 1)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(b_init_value)

    def forward(self, *inputs):
        h = torch.cat(inputs, dim=1)
        h = self.hidden_activation(self.obs_fc(h))
        h = self.hidden_activation(self.embed_fc(h))
        return self.output_activation(self.last_fc(h))


class StructuredUniversalQfunction(PyTorchModule):
    """
    Parameterize QF as

    Q(s, a, s_g) = -||f(s, a) - s_g)||^2

    WARNING: this is only valid for when the reward is l2-norm (as opposed to a
    weighted l2-norm)
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_state_dim,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            bn_input=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []
        in_size = observation_dim + action_dim + 1
        if bn_input:
            self.process_input = nn.BatchNorm1d(in_size)
        else:
            self.process_input = identity

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(0)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, goal_state_dim)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action, goal_state, discount):
        h = torch.cat((obs, action, discount), dim=1)
        h = self.process_input(h)
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        next_state = self.output_activation(self.last_fc(h))
        out = - torch.norm(goal_state - next_state, p=2, dim=1)
        return out


class FFUniversalPolicy(PyTorchModule, UniversalPolicy):
    def __init__(
            self,
            obs_dim,
            action_dim,
            goal_state_dim,
            fc1_size,
            fc2_size,
            init_w=3e-3,
            b_init_value=0.1,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_init = hidden_init

        self.fc1 = nn.Linear(obs_dim + goal_state_dim + 1, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(b_init_value)
        hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(b_init_value)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(b_init_value)

    def forward(self, obs, goal_state, discount):
        h = torch.cat((obs, goal_state, discount), dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs_np):
        obs = elem_or_tuple_to_variable(
            np.expand_dims(obs_np, 0)
        )
        action = self.__call__(
            obs,
            self._goal_expanded_torch,
            self._discount_expanded_torch,
        )
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}
