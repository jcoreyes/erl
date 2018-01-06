"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from railrl.policies.base import Policy
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.modules import SelfOuterProductLinear, LayerNorm


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
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
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


class MlpQf(FlattenMlp):
    def __init__(
        self,
        *args,
        obs_normalizer: TorchFixedNormalizer=None,
        action_normalizer: TorchFixedNormalizer=None,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer

    def forward(self, obs, actions):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        if self.action_normalizer:
            actions = self.action_normalizer.normalize(actions)
        return super().forward(obs, actions)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """
    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer=None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A simpler interface for creating policies.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivations = self.last_fc(h)
        actions = self.output_activation(preactivations)
        if return_preactivations:
            return actions, preactivations
        else:
            return actions


class FeedForwardQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_size,
            embedded_hidden_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            batchnorm_obs=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size
        self.hidden_init = hidden_init
        self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(observation_hidden_size + action_dim,
                                 embedded_hidden_size)

        self.last_fc = nn.Linear(embedded_hidden_size, 1)
        self.output_activation = output_activation

        self.init_weights(init_w)
        self.batchnorm_obs = batchnorm_obs
        if self.batchnorm_obs:
            self.bn_obs = nn.BatchNorm1d(obs_dim)

    def init_weights(self, init_w):
        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        self.hidden_init(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        if self.batchnorm_obs:
            obs = self.bn_obs(obs)
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return self.output_activation(self.last_fc(h))


class FeedForwardPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            fc1_size,
            fc2_size,
            init_w=1e-3,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_init = hidden_init

        self.fc1 = nn.Linear(obs_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)