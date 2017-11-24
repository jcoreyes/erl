import numpy as np
from railrl.torch.core import PyTorchModule
from railrl.torch import pytorch_util as ptu
from torch import nn as nn
import torch


class ExpectableQF(PyTorchModule):
    """
    A Q-function whose expectation w.r.t. actions is computable in closed form.
    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_size,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_fc = nn.Linear(obs_dim, hidden_size)
        self.action_fc = nn.Linear(action_dim, hidden_size)
        hidden_init(self.fc)
        self.fc.bias.data.fill_(b_init_value)

        self.last_fc = nn.Linear(2*hidden_size, 1)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action, convolve=False):
        h_obs = torch.tanh(self.obs_fc(obs))
        if convolve:
            variance = torch.sum(self.action_fc.weight * self.action_fc.weight,
                                 dim=1).unsqueeze(0)
            conv_factor_inv = torch.sqrt(1 + np.pi / 2 * variance)
            h_action = torch.tanh(self.obs_fc(action) / conv_factor_inv)
        else:
            h_action = torch.tanh(self.obs_fc(action))
        h = torch.cat(h_obs, h_action)
        return self.last_fc(h)
