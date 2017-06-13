import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable

from railrl.pythonplusplus import identity
from railrl.torch.core import PyTorchModule
from railrl.torch import pytorch_util as ptu

from railrl.torch.bnlstm import BNLSTMCell, LSTM


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
        init.kaiming_normal(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        init.kaiming_normal(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
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
        init.kaiming_normal(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        init.kaiming_normal(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        # init.kaiming_normal(self.last_fc.weight)
        # self.last_fc.bias.data.fill_(0)

    def forward(self, obs, memory, action, write):
        obs_embedded = torch.cat((obs, memory), dim=1)
        obs_embedded = F.relu(self.obs_fc(obs_embedded))
        x = torch.cat((obs_embedded, action, write), dim=1)
        x = F.relu(self.embedded_fc(x))
        return self.output_activation(self.last_fc(x))


class RecurrentQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_size,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm = LSTM(
            BNLSTMCell,
            self.obs_dim + self.action_dim,
            self.hidden_size,
            batch_first=True,
        )
        self.last_fc = nn.Linear(self.hidden_size, 1)

    def forward(self, obs, action):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :param action: torch Variable, [batch_size, sequence length, action dim]
        :return: torch Variable, [batch_size, sequence length, 1]
        """
        assert len(obs.size()) == 3
        inputs = torch.cat((obs, action), dim=2)
        batch_size, subsequence_length = obs.size()[:2]
        cx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        cx.data.fill_(0)
        hx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        hx.data.fill_(0)
        rnn_outputs, _ = self.lstm(inputs, (hx, cx))
        rnn_outputs.contiguous()
        rnn_outputs_flat = rnn_outputs.view(-1, self.hidden_size)
        outputs_flat = self.last_fc(rnn_outputs_flat)
        return outputs_flat.view(batch_size, subsequence_length, 1)


class RecurrentMemoryQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            hidden_size,
            fc1_size,
            fc2_size,
            init_w=3e-3,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.lstm = LSTM(
            BNLSTMCell,
            self.obs_dim + self.action_dim + 2 * memory_dim,
            self.hidden_size,
            batch_first=True,
            )
        self.fc1 = nn.Linear(self.hidden_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.last_fc = nn.Linear(self.fc2_size, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        init.kaiming_normal(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        init.kaiming_normal(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, memory, action, write):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :param memory: torch Variable, [batch_size, sequence length, memory dim]
        :param action: torch Variable, [batch_size, sequence length, action dim]
        :param write: torch Variable, [batch_size, sequence length, memory dim]
        :return: torch Variable, [batch_size, sequence length, 1]
        """
        assert len(obs.size()) == 3
        inputs = torch.cat((obs, memory, action, write), dim=2)
        batch_size, subsequence_length = obs.size()[:2]
        cx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        cx.data.fill_(0)
        hx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        hx.data.fill_(0)
        rnn_outputs, _ = self.lstm(inputs, (hx, cx))
        rnn_outputs.contiguous()
        rnn_outputs_flat = rnn_outputs.view(-1, self.hidden_size)
        h1 = F.relu(self.fc1(rnn_outputs_flat))
        h2 = F.relu(self.fc2(h1))
        outputs_flat = self.last_fc(h2)
        return outputs_flat.view(batch_size, subsequence_length, 1)
