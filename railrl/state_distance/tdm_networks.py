"""
This is basically as re-write of the networks.py file but for tdm.py rather
than sdql.py
"""
import torch

import numpy as np
from railrl.state_distance.util import split_tau, extract_goals, split_flat_obs
from railrl.torch.networks import Mlp, TanhMlpPolicy, FlattenMlp
import railrl.torch.pytorch_util as ptu


class StructuredQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau) - s_g|

    element-wise

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            output_size,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + 1,
            output_size=output_size,
            **kwargs
        )
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim

    def forward(self, *inputs):
        h = torch.cat(inputs, dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))


class OneHotTauQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a one-hot vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + max_tau + 1,
            output_size=output_size,
            **kwargs
        )
        self.max_tau = max_tau

    def forward(self, flat_obs, action):
        obs, taus = split_tau(flat_obs)
        h = torch.cat((obs, action), dim=1)

        batch_size = h.size()[0]
        y_onehot = ptu.FloatTensor(batch_size, self.max_tau + 1)
        y_onehot.zero_()
        t = taus.data.long()
        t = torch.clamp(t, min=0)
        y_onehot.scatter_(1, t, 1)

        h = torch.cat((
            obs,
            ptu.Variable(y_onehot),
            action
        ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))


class InternalGcmQf(FlattenMlp):
    """
    Parameterize QF as

    Q(s, a, g, tau) = - |g - f(s, a, s_g, tau)}|

    element-wise

    Also, rather than giving `g`, give `g - goalify(s)` as input.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            env,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=(
                self.observation_dim + self.action_dim + self.goal_dim + 1
            ),
            output_size=self.goal_dim,
            **kwargs
        )
        self.env = env

    def forward(self, flat_obs, actions):
        obs, goals, taus = split_flat_obs(
            flat_obs, self.observation_dim, self.goal_dim
        )
        diffs = goals - self.env.convert_obs_to_goals(obs)
        new_flat_obs = torch.cat((obs, diffs, taus), dim=1)
        predictions = super().forward(new_flat_obs, actions)
        return - torch.abs(goals - predictions)


class TdmPolicy(TanhMlpPolicy):
    """
    Rather than giving `g`, give `g - goalify(s)` as input.
    """
    def __init__(
            self,
            env,
            **kwargs
    ):
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size=self.observation_dim + self.goal_dim + 1,
            output_size=self.action_dim,
            **kwargs
        )
        self.env = env

    def forward(self, flat_obs, return_preactivations=False):
        obs, goals, taus = split_flat_obs(
            flat_obs, self.observation_dim, self.goal_dim
        )
        diffs = goals - self.env.convert_obs_to_goals(obs)
        new_flat_obs = torch.cat((obs, diffs, taus), dim=1)
        return super().forward(
            new_flat_obs,
            return_preactivations=return_preactivations,
        )
