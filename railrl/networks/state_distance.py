import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch import optim
from scipy import optimize

from railrl.policies.state_distance import UniversalPolicy
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule


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
        self.last_fc.bias.data.uniform_(-init_w, init_w)

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
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            dropout_prob=0,
            output_multiplier=1,
    ):
        if output_activation == F.selu or output_activation == F.relu:
            assert output_multiplier < 0, "Q function should output negative #s"

        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_prob = dropout_prob
        self.output_multiplier = output_multiplier
        self.dropouts = []
        self.fcs = []
        in_size = observation_dim + goal_state_dim + action_dim + 1

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
            if self.dropout_prob > 0:
                dropout = nn.Dropout(p=self.dropout_prob)
                self.__setattr__("dropout{}".format(i), dropout)
                self.dropouts.append(dropout)

        self.last_fc = nn.Linear(in_size, 1)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action, goal_state, discount):
        h = torch.cat((obs, action, goal_state, discount), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.dropout_prob > 0:
                h = self.dropouts[i](h)
        return self.output_activation(self.last_fc(h)) * self.output_multiplier


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
            hidden_init=ptu.fanin_init,
            bn_input=False,
            dropout_prob=0,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.dropout_prob = dropout_prob
        self.dropouts = []
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
            if self.dropout_prob > 0:
                dropout = nn.Dropout(p=self.dropout_prob)
                self.__setattr__("dropout{}".format(i), dropout)
                self.dropouts.append(dropout)

        self.last_fc = nn.Linear(in_size, goal_state_dim)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(
            self,
            obs,
            action,
            goal_state,
            discount,
            only_return_next_state=False,
    ):
        h = torch.cat((obs, action, discount), dim=1)
        h = self.process_input(h)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.dropout_prob > 0:
                h = self.dropouts[i](h)
        next_state = self.last_fc(h)
        if only_return_next_state:
            return next_state
        out = - torch.norm(goal_state - next_state, p=2, dim=1)
        return out.unsqueeze(1)


class ModelExtractor(PyTorchModule):
    def __init__(self, qf, discount=0.):
        super().__init__()
        assert isinstance(qf, StructuredUniversalQfunction)
        self.qf = qf
        self.discount = discount

    def forward(self, state, action):
        batch_size = state.size()[0]
        discount = ptu.np_to_var(self.discount + np.zeros((batch_size, 1)))
        return self.qf(state, action, None, discount, True)


class ModelExtractorGeneral(PyTorchModule):
    def __init__(
            self,
            qf,
            discount=0.,
            sample_size=100,
            learning_rate=1e-1,
            num_gradient_steps=100,
            state_optimizer='adam',
    ):
        super().__init__()
        self._is_structured_qf = isinstance(qf, StructuredUniversalQfunction)
        self.qf = qf
        self.discount = discount
        self.sample_size = sample_size
        self.learning_rate = learning_rate
        self.num_optimization_steps = num_gradient_steps
        self.state_optimizer = state_optimizer

    def expand_to_sample_size(self, torch_array):
        return torch_array.repeat(self.sample_size, 1)

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.sample_size,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=False)

    def forward(self, states, actions):
        batch_size, obs_dim = states.size()
        assert batch_size == 1
        if self._is_structured_qf:
            discount = ptu.np_to_var(self.discount + np.zeros((batch_size, 1)))
            return self.qf(states, actions, None, discount, True)

        if self.state_optimizer == 'adam':
            discount = ptu.np_to_var(
                self.discount * np.ones((self.sample_size, 1))
            )
            states = self.expand_to_sample_size(states)
            actions = self.expand_to_sample_size(actions)
            next_states_np = np.zeros((self.sample_size, obs_dim))
            next_states = ptu.np_to_var(next_states_np, requires_grad=True)
            optimizer = optim.Adam([next_states], self.learning_rate)

            for _ in range(self.num_optimization_steps):
                losses = -self.qf(
                    states,
                    actions,
                    next_states,
                    discount,
                )
                loss = losses.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # return ptu.get_numpy(next_states)
            losses_np = ptu.get_numpy(losses)
            best_action_i = np.argmin(losses_np)
            return next_states[best_action_i:best_action_i+1, :]
        elif self.state_optimizer == 'lbfgs':
            next_states = []
            for i in range(len(states)):
                state = states[i:i+1, :]
                action = actions[i:i+1, :]
                loss_f = self.create_loss(state, action, return_gradient=True)
                results = optimize.fmin_l_bfgs_b(
                    loss_f,
                    np.zeros((1, obs_dim)),
                    maxiter=self.num_optimization_steps,
                )
                next_state = results[0]
                next_states.append(next_state)
            next_states = np.array(next_states)
            return next_states
        elif self.state_optimizer == 'fmin':
            next_states = []
            for i in range(len(states)):
                state = states[i:i+1, :]
                action = actions[i:i+1, :]
                loss_f = self.create_loss(state, action)
                results = optimize.fmin(
                    loss_f,
                    np.zeros((1, obs_dim)),
                    maxiter=self.num_optimization_steps,
                )
                next_state = results[0]
                next_states.append(next_state)
            next_states = np.array(next_states)
            return next_states
        else:
            raise Exception(
                "Unknown state optimizer mode: {}".format(self.state_optimizer)
            )


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
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, goal_state, discount):
        h = torch.cat((obs, goal_state, discount), dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs_np):
        obs = ptu.np_to_var(
            np.expand_dims(obs_np, 0)
        )
        action = self.__call__(
            obs,
            self._goal_expanded_torch,
            self._discount_expanded_torch,
        )
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}


