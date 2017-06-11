import torch
from torch import nn as nn
from torch.nn import functional as F

from railrl.pythonplusplus import identity
from railrl.torch.core import PyTorchModule
from railrl.torch.pytorch_util import fanin_init


class FeedForwardQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_size,
            embedded_hidden_size,
            init_w=3e-3,
            output_activation=identity,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size

        self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(observation_hidden_size + action_dim,
                                     embedded_hidden_size)
        self.last_fc = nn.Linear(embedded_hidden_size, 1)
        self.output_activation = output_activation

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.obs_fc.weight.data = fanin_init(self.obs_fc.weight.data.size())
        self.obs_fc.bias.data *= 0
        self.embedded_fc.weight.data = fanin_init(
            self.embedded_fc.weight.data.size()
        )
        self.embedded_fc.bias.data *= 0
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return self.output_activation(self.last_fc(h))


class MemoryQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            observation_hidden_size,
            embedded_hidden_size,
            init_w=3e-3,
            output_activation=identity,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size
        self.init_w = init_w

        self.obs_fc = nn.Linear(obs_dim + memory_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(
            observation_hidden_size + action_dim + memory_dim,
            embedded_hidden_size,
        )
        self.last_fc = nn.Linear(embedded_hidden_size, 1)
        self.output_activation = output_activation

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.obs_fc.weight.data = fanin_init(self.obs_fc.weight.data.size())
        self.obs_fc.bias.data *= 0
        self.embedded_fc.weight.data = fanin_init(
            self.embedded_fc.weight.data.size()
        )
        self.embedded_fc.bias.data *= 0
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, memory, action, write):
        obs_embedded = torch.cat((obs, memory), dim=1)
        obs_embedded = F.relu(self.obs_fc(obs_embedded))
        x = torch.cat((obs_embedded, action, write), dim=1)
        x = F.relu(self.embedded_fc(x))
        return self.output_activation(self.last_fc(x))