import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from railrl.policies.state_distance import UniversalPolicy
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu

from railrl.torch.core import PyTorchModule


class HerNetwork(PyTorchModule):
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []
        in_size = input_size

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


class HerQFunction(HerNetwork):
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
    ):
        self.save_init_params(locals())
        super().__init__(
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            output_size=1,
            input_size=observation_dim + goal_dim + action_dim,
            init_w=init_w,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            hidden_init=hidden_init,
            b_init_value=b_init_value,
        )

    def forward(self, obs, action, goal_state):
        h = torch.cat((obs, action, goal_state), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class HerPolicy(HerNetwork, UniversalPolicy):
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
    ):
        self.save_init_params(locals())
        super().__init__(
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            output_size=action_dim,
            input_size=observation_dim + goal_dim,
            init_w=init_w,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            hidden_init=hidden_init,
            b_init_value=b_init_value,
        )

    def forward(self, obs, goal_state):
        h = torch.cat((obs, goal_state), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs_np):
        obs = ptu.np_to_var(
            np.expand_dims(obs_np, 0)
        )
        action = self.__call__(
            obs,
            self._goal_expanded_torch,
        )
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}
