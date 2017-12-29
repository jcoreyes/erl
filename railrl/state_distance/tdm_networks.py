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
            internal_gcm=True,
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
        self.internal_gcm = internal_gcm

    def forward(self, flat_obs, actions):
        h = torch.cat((flat_obs, actions), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        if self.internal_gcm:
            _, goals, _ = split_flat_obs(
                flat_obs, self.observation_dim, self.goal_dim
            )
            return - torch.abs(goals - self.last_fc(h))
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


class TdmQf(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            norm_order,
            structure='norm_difference',
            **kwargs
    ):
        """

        :param env:
        :param hidden_sizes:
        :param vectorized: Boolean. Vectorized or not?
        :param norm_order: int, 1 or 2. What L norm to use.
        :param structure: String defining output structure of network:
            - 'norm_difference': Q = -||g - f(inputs)||
            - 'norm': Q = -||f(inputs)||
            - 'norm_distance_difference': Q = -||f(inputs) + current_distance||
            - 'distance_difference': Q = f(inputs) + current_distance
            - 'difference': Q = f(inputs) - g  (vectorized only)
            - 'none': Q = f(inputs)

        :param kwargs:
        """
        assert structure in [
            'norm_difference',
            'norm',
            'norm_distance_difference',
            'distance_difference',
            'difference',
            'none',
        ]
        if structure == 'difference':
            assert vectorized, "difference only makes sense for vectorized"
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size=(
                    self.observation_dim + self.action_dim + self.goal_dim + 1
            ),
            output_size=self.goal_dim if vectorized else 1,
            **kwargs
        )
        self.env = env
        self.vectorized = vectorized
        self.norm_order = norm_order
        self.structure = structure

    def forward(self, flat_obs, actions, return_internal_prediction=False):
        obs, goals, taus = split_flat_obs(
            flat_obs, self.observation_dim, self.goal_dim
        )
        diffs = goals - self.env.convert_obs_to_goals(obs)
        new_flat_obs = torch.cat((obs, diffs, taus), dim=1)
        if self.vectorized:
            predictions = super().forward(new_flat_obs, actions)
            if return_internal_prediction:
                return predictions
            if self.structure == 'norm_difference':
                return - torch.abs(goals - predictions)
            elif self.structure == 'norm':
                return - torch.abs(predictions)
            elif self.structure == 'norm_distance_difference':
                current_features = self.env.convert_obs_to_goals(obs)
                current_distance = torch.abs(goals - current_features)
                return - torch.abs(predictions + current_distance)
            elif self.structure == 'distance_difference':
                current_features = self.env.convert_obs_to_goals(obs)
                current_distance = torch.abs(goals - current_features)
                return predictions + current_distance
            elif self.structure == 'difference':
                return predictions - goals
            elif self.structure == 'none':
                return predictions
            else:
                raise TypeError("Invalid structure: {}".format(self.structure))
        else:
            predictions = super().forward(new_flat_obs, actions)
            if return_internal_prediction:
                return predictions
            if self.structure == 'norm_difference':
                return - torch.norm(
                    goals - predictions,
                    p=self.norm_order,
                    dim=1,
                    keepdim=True,
                )
            elif self.structure == 'norm':
                return - torch.norm(
                    predictions,
                    p=self.norm_order,
                    dim=1,
                    keepdim=True,
                )
            elif self.structure == 'norm_distance_difference':
                current_features = self.env.convert_obs_to_goals(obs)
                current_distance = torch.norm(
                    goals - current_features,
                    p=self.norm_order,
                    dim=1,
                    keepdim=True,
                )
                return - torch.abs(predictions + current_distance)
            elif self.structure == 'distance_difference':
                current_features = self.env.convert_obs_to_goals(obs)
                current_distance = torch.norm(
                    goals - current_features,
                    p=self.norm_order,
                    dim=1,
                    keepdim=True,
                )
                return predictions + current_distance
            elif self.structure == 'none':
                return predictions
            else:
                raise TypeError(
                    "For vectorized={0}, invalid structure: {1}".format(
                        self.vectorized,
                        self.structure,
                    )
                )


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
