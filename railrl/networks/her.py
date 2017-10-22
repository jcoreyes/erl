import numpy as np
import torch
from torch.nn import functional as F

from railrl.networks.base import Mlp
from railrl.policies.state_distance import UniversalPolicy
from railrl.torch import pytorch_util as ptu


class HerQFunction(Mlp):
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            output_size=1,
            input_size=observation_dim + goal_dim + action_dim,
            **kwargs
        )

    def forward(self, obs, action, goal_state, _ignored_discount=None):
        h = torch.cat((obs, action, goal_state), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class HerPolicy(Mlp, UniversalPolicy):
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            output_size=action_dim,
            input_size=observation_dim + goal_dim,
            **kwargs
        )

    def forward(self, obs, goal_state, _ignored_discount=None):
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
