"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.modules import SelfOuterProductLinear


class Mlp(PyTorchModule):
    def __init__(
            self,
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

        self.input_size = input_size
        self.output_size = output_size
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

    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """
    def forward(self, *inputs):
        h = torch.cat(inputs, dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class OuterProductFF(PyTorchModule):
    """
    An interesting idea that I had where you first take the outer product of
    all inputs, flatten it, and then pass it through a linear layer. I
    haven't really tested this, but I'll leave it here to tempt myself later...
    """
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

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []

        self.first_input = nn.Linear(first_input_size, first_input_size)
        hidden_init(self.first_input.weight)
        self.first_input.bias.data.fill_(b_init_value)
        self.__setattr__("firstinput", self.first_input)

        self.second_input = nn.Linear(second_input_size, second_input_size)
        hidden_init(self.second_input.weight)
        self.second_input.bias.data.fill_(b_init_value)
        self.__setattr__("secondinput", self.second_input)

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
        h = torch.cat((h1, h2))
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))