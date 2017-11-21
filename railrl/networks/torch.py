import torch
from torch import nn as nn
from torch.nn import functional as F

from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.modules import SelfOuterProductLinear


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


class Mlp(PyTorchModule):
    def __init__(
            self,
            input_dim,
            output_dim,
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
        in_size = input_dim
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

        self.last_fc = nn.Linear(in_size, output_dim)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *inputs):
        h = torch.cat(inputs, dim=1)
        h = self.process_input(h)
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class OuterProductFF(PyTorchModule):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden1_size,
            hidden2_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.sop1 = SelfOuterProductLinear(input_dim, hidden1_size)
        self.sop2 = SelfOuterProductLinear(hidden1_size, hidden2_size)
        self.last_fc = nn.Linear(hidden2_size, output_dim)
        self.output_activation = output_activation

        hidden_init(self.sop1.fc.weight)
        self.sop1.fc.bias.data.fill_(0)
        hidden_init(self.sop2.fc.weight)
        self.sop2.fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *inputs):
        h = torch.cat(inputs, dim=1)
        h = F.relu(self.sop1(h))
        h = F.relu(self.sop2(h))
        return self.output_activation(self.last_fc(h))
