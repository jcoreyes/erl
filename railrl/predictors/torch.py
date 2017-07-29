import torch
from torch import nn as nn
from torch.nn import functional as F

from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule


class TwoLayerMlp(PyTorchModule):
    def __init__(
            self,
            input_dim,
            output_dim,
            fc1_size,
            fc2_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, output_dim)

        hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *inputs):
        h = torch.cat(inputs, dim=1)
        h = self.hidden_activation(self.fc1(h))
        h = self.hidden_activation(self.fc2(h))
        return self.output_activation(self.last_fc(h))
