import torch
import numpy as np
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

from railrl.torch.bnlstm import BNLSTMCell, LSTM, LSTMCell
from railrl.torch.core import PyTorchModule
from railrl.torch import pytorch_util as ptu


class FlattenLSTMCell(nn.Module):
    def __init__(self, lstm_cell):
        self.lstm_cell = lstm_cell

    def forward(self, input, state):
        hx, cx = torch.split(state, self.memory_dim // 2, dim=1)
        new_hx, new_cx = self.lstm_cell(input, (hx, cx))
        new_state = torch.cat((new_hx, new_cx), dim=1)
        return hx, new_state


class LinearBN(nn.Module):
    """
    Linear then BN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)
        self.weight = self.fc.weight
        if bias:
            self.bias = self.fc.bias

    def forward(self, input):
        return self.bn(self.fc(input))


class RWACell(PyTorchModule):
    def __init__(
            self,
            input_dim,
            num_units,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.input_dim = input_dim
        self.num_units = num_units

        self.fc_u = nn.Linear(input_dim, num_units)
        self.fc_g = nn.Linear(input_dim + num_units, num_units)
        # Bias term factors when from numerate and denominator
        self.fc_a = nn.Linear(input_dim + num_units, num_units, bias=False)
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal(self.fc_u.weight)
        self.fc_u.bias.data.fill_(0)
        init.kaiming_normal(self.fc_g.weight)
        self.fc_g.bias.data.fill_(0)
        init.kaiming_normal(self.fc_a.weight)

    def forward(self, inputs, state):
        # n, d, h, a_max = state
        h, n, d = state

        u = self.fc_u(inputs)
        g = self.fc_g(torch.cat((inputs, h), dim=1))
        z = u * F.tanh(g)
        a = self.fc_a(torch.cat((inputs, h), dim=1))

        # Numerically stable update of numerator and denom
        # a_newmax = ptu.maximum_2d(a_max, a)
        # exp_diff = torch.exp(a_max-a_newmax)
        # weight_scaled = torch.exp(a-a_newmax)
        # n_new = n * exp_diff + z * weight_scaled
        # d_new = d * exp_diff + weight_scaled
        # h_new = F.tanh(n_new / d_new)

        # next_state = (n_new, d_new, h_new, a_max)

        weight = torch.exp(a)
        n_new = n + z * weight
        d_new = d + weight
        h_new = F.tanh(n_new / d_new)

        return h_new, n_new, d_new

    @staticmethod
    def state_num_split():
        return 3


class MemoryPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            fc1_size,
            fc2_size,
            init_w=1e-3,
            cell_class=LSTMCell,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(obs_dim + memory_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)
        self.num_splits_for_rnn_internally = cell_class.state_num_split()
        assert memory_dim % self.num_splits_for_rnn_internally == 0
        self.rnn_cell = cell_class(
            self.obs_dim, self.memory_dim // self.num_splits_for_rnn_internally
        )
        self.init_weights(init_w)

    def init_weights(self, init_w):
        init.kaiming_normal(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        init.kaiming_normal(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        # init.kaiming_normal(self.last_fc.weight)
        # self.last_fc.bias.data.fill_(0)

    def action_parameters(self):
        for fc in [self.fc1, self.fc2, self.last_fc]:
            for param in fc.parameters():
                yield param

    def write_parameters(self):
        return self.rnn_cell.parameters()

    def forward(self, obs, initial_memory):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :param initial_memory: torch Variable, [batch_size, memory dim]
        :return: (actions, writes) tuple
            actions: [batch_size, sequence length, action dim]
            writes: [batch_size, sequence length, memory dim]
        """
        assert len(obs.size()) == 3
        assert len(initial_memory.size()) == 2
        batch_size, subsequence_length = obs.size()[:2]

        """
        Create the new writes.
        """
        state = torch.split(
            initial_memory,
            self.memory_dim // self.num_splits_for_rnn_internally,
            dim=1,
        )
        subtraj_writes = Variable(
            ptu.FloatTensor(batch_size, subsequence_length, self.memory_dim)
        )
        for i in range(subsequence_length):
            state = self.rnn_cell(obs[:, i, :], state)
            subtraj_writes[:, i, :] = torch.cat(state, dim=1)

        # The reason that using a LSTM doesn't work is that this gives you only
        # the FINAL hx and cx, not all of them :(
        # _, (new_hxs, new_cxs) = self.lstm(obs, (hx, cx))
        # subtraj_writes = torch.cat((new_hxs, new_cxs), dim=2)
        # subtraj_writes = subtraj_writes.permute(1, 0, 2)

        """
        Create the new subtrajectory memories with the initial memories and the
        new writes.
        """
        expanded_init_memory = initial_memory.unsqueeze(1)
        if subsequence_length > 1:
            memories = torch.cat(
                (
                    expanded_init_memory,
                    subtraj_writes[:, :-1, :],
                ),
                dim=1,
            )
        else:
            memories = expanded_init_memory

        """
        Use new memories to create env actions.
        """
        all_subtraj_inputs = torch.cat([obs, memories], dim=2)
        # noinspection PyArgumentList
        subtraj_actions = Variable(
            ptu.FloatTensor(batch_size, subsequence_length, self.action_dim)
        )
        for i in range(subsequence_length):
            all_inputs = all_subtraj_inputs[:, i, :]
            h1 = F.tanh(self.fc1(all_inputs))
            h2 = F.tanh(self.fc2(h1))
            action = F.tanh(self.last_fc(h2))
            subtraj_actions[:, i, :] = action

        return subtraj_actions, subtraj_writes

    def get_action(self, augmented_obs):
        """
        :param augmented_obs: (obs, memories) tuple
            obs: np.ndarray, [obs_dim]
            memories: nd.ndarray, [memory_dim]
        :return: (actions, writes) tuple
            actions: np.ndarray, [action_dim]
            writes: np.ndarray, [writes_dim]
        """
        obs, memory = augmented_obs
        obs = np.expand_dims(obs, axis=0)
        memory = np.expand_dims(memory, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        memory = Variable(ptu.from_numpy(memory).float(), requires_grad=False)
        action, write = self.get_flat_output(obs, memory)
        return (
                   np.squeeze(ptu.get_numpy(action), axis=0),
                   np.squeeze(ptu.get_numpy(write), axis=0),
               ), {}

    def get_flat_output(self, obs, initial_memories):
        """
        Each batch element is processed independently. So, there's no recurrency
        used.

        :param obs: torch Variable, [batch_size X obs_dim]
        :param initial_memories: torch Variable, [batch_size X memory_dim]
        :return: (actions, writes) Tuple
            actions: torch Variable, [batch_size X action_dim]
            writes: torch Variable, [batch_size X writes_dim]
        """
        obs = obs.unsqueeze(1)
        actions, writes = self.__call__(obs, initial_memories)
        return torch.squeeze(actions, dim=1), torch.squeeze(writes, dim=1)

    def reset(self):
        pass

    def log_diagnostics(self, paths):
        pass


class FeedForwardPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            fc1_size,
            fc2_size,
            init_w=1e-3,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(obs_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        init.kaiming_normal(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        init.kaiming_normal(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action = self.__call__(obs)
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}

    def reset(self):
        pass

    def log_diagnostics(self, paths):
        pass


class RecurrentPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_size,
            fc1_size,
            fc2_size,
            init_w=3e-3,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.lstm = LSTM(BNLSTMCell, self.obs_dim, self.hidden_size, 1,
                         batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size + self.obs_dim, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.last_fc = nn.Linear(self.fc2_size, self.action_dim)

        self.hx = None
        self.cx = None
        self.init_weights(init_w)
        self.reset()

    def init_weights(self, init_w):
        init.kaiming_normal(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        init.kaiming_normal(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, cx=None, hx=None):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :return: torch Variable, [batch_size, sequence length, action dim]
        """
        assert len(obs.size()) == 3
        batch_size, subsequence_length = obs.size()[:2]
        if hx is None:
            cx = Variable(
                ptu.FloatTensor(1, batch_size, self.hidden_size)
            )
            cx.data.fill_(0)
            hx = Variable(
                ptu.FloatTensor(1, batch_size, self.hidden_size)
            )
            hx.data.fill_(0)
        rnn_outputs, state = self.lstm(obs, (hx, cx))
        if subsequence_length > 1:
            import ipdb; ipdb.set_trace()
        rnn_outputs.contiguous()
        rnn_outputs_flat = rnn_outputs.view(-1, self.hidden_size)
        obs_flat = obs.view(-1, self.obs_dim)
        h = torch.cat((rnn_outputs_flat, obs_flat), dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        outputs_flat = F.tanh(self.last_fc(h))

        out = outputs_flat.view(
            batch_size, subsequence_length, self.action_dim
        ), state
        print(out[0][0, :, :])
        return out

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = np.expand_dims(obs, axis=1)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, (self.hx, self.cx) = self.__call__(
            obs, cx=self.cx, hx=self.hx
        )
        action = action.squeeze(0)
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}

    def reset(self):
        self.hx = Variable(
            ptu.FloatTensor(1, 1, self.hidden_size)
        )
        self.cx = Variable(
            ptu.FloatTensor(1, 1, self.hidden_size)
        )
        self.hx.data.fill_(0)
        self.cx.data.fill_(0)

    def log_diagnostics(self, paths):
        pass
