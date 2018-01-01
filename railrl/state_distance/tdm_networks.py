"""
This is basically as re-write of the networks.py file but for tdm.py rather
than sdql.py
"""
import torch

import numpy as np
from railrl.state_distance.util import split_tau, extract_goals, split_flat_obs
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
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


class TdmNormalizer(object):
    def __init__(
            self,
            env,
            obs_normalizer: TorchFixedNormalizer=None,
            goal_normalizer: TorchFixedNormalizer=None,
            action_normalizer: TorchFixedNormalizer=None,
            distance_normalizer: TorchFixedNormalizer=None,
            normalize_tau=False,
            max_tau=0,
            log_tau=False,
    ):
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        self.obs_normalizer = obs_normalizer
        self.goal_normalizer = goal_normalizer
        self.action_normalizer = action_normalizer
        self.distance_normalizer = distance_normalizer
        self.log_tau = log_tau
        self.normalize_tau = normalize_tau
        self.max_tau = max_tau

        # Assuming that the taus are sampled uniformly from [0, max_tau]
        if self.log_tau:
            # If max_tau = 1, then
            # mean = \int_2^3 log(x) dx ~ 0.9095...
            # std = sqrt{  \int_2^3 (log(x) - mean)^2 dx    } ~ 0.165...
            # Thanks wolfram!
            self.tau_mean = self.max_tau * 0.90954250488443855
            self.tau_std = self.max_tau * 0.11656876357329767
        else:
            self.tau_mean = self.max_tau / 2
            self.tau_std = self.max_tau / np.sqrt(12)

    def normalize_flat_obs(self, flat_obs):
        obs, goals, taus = split_flat_obs(
            flat_obs, self.observation_dim, self.goal_dim
        )
        obs = self.obs_normalizer.normalize(obs)
        goals = self.goal_normalizer.normalize(goals)

        if self.log_tau:
            # minimum tau is -1 (although the output should be ignored for
            # the `tau == -1` case.
            taus = torch.log(taus + 2)
        if self.normalize_tau:
            taus = (taus - self.tau_mean) / self.tau_std

        return torch.cat((obs, goals, taus), dim=1)

    def copy_stats(self, other):
        self.obs_normalizer.copy_stats(other.obs_normalizer)
        self.goal_normalizer.copy_stats(other.goal_normalizer)
        self.action_normalizer.copy_stats(other.action_normalizer)
        self.distance_normalizer.copy_stats(other.distance_normalizer)


class TdmQf(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            norm_order,
            structure='norm_difference',
            tdm_normalizer: TdmNormalizer=None,
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
        self.tdm_normalizer = tdm_normalizer

    def forward(self, flat_obs, actions, return_internal_prediction=False):
        if self.tdm_normalizer is not None:
            actions = self.tdm_normalizer.action_normalizer.normalize(actions)
            flat_obs = self.tdm_normalizer.normalize_flat_obs(flat_obs)

        predictions = super().forward(flat_obs, actions)
        if return_internal_prediction:
            return predictions

        obs, goals, taus = split_flat_obs(
            flat_obs, self.observation_dim, self.goal_dim
        )
        if self.vectorized:
            if self.structure == 'norm_difference':
                output = - torch.abs(goals - predictions)
            elif self.structure == 'norm':
                output = - torch.abs(predictions)
            elif self.structure == 'norm_distance_difference':
                current_features = self.env.convert_obs_to_goals(obs)
                current_distance = torch.abs(goals - current_features)
                output = - torch.abs(predictions + current_distance)
            elif self.structure == 'distance_difference':
                current_features = self.env.convert_obs_to_goals(obs)
                current_distance = torch.abs(goals - current_features)
                output = predictions + current_distance
            elif self.structure == 'difference':
                output = predictions - goals
            elif self.structure == 'none':
                output = predictions
            else:
                raise TypeError("Invalid structure: {}".format(self.structure))
        else:
            if self.structure == 'norm_difference':
                output = - torch.norm(
                    goals - predictions,
                    p=self.norm_order,
                    dim=1,
                    keepdim=True,
                )
            elif self.structure == 'norm':
                output = - torch.norm(
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
                output = - torch.abs(predictions + current_distance)
            elif self.structure == 'distance_difference':
                current_features = self.env.convert_obs_to_goals(obs)
                current_distance = torch.norm(
                    goals - current_features,
                    p=self.norm_order,
                    dim=1,
                    keepdim=True,
                )
                output = predictions + current_distance
            elif self.structure == 'none':
                output = predictions
            else:
                raise TypeError(
                    "For vectorized={0}, invalid structure: {1}".format(
                        self.vectorized,
                        self.structure,
                    )
                )
        if self.tdm_normalizer is not None:
            output = self.tdm_normalizer.distance_normalizer.denormalize_scale(
                output
            )
        return output


class TdmPolicy(TanhMlpPolicy):
    """
    Rather than giving `g`, give `g - goalify(s)` as input.
    """
    def __init__(
            self,
            env,
            tdm_normalizer: TdmNormalizer=None,
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
        self.tdm_normalizer = tdm_normalizer

    def forward(self, flat_obs, return_preactivations=False):
        if self.tdm_normalizer is not None:
            flat_obs = self.tdm_normalizer.normalize_flat_obs(flat_obs)
        return super().forward(
            flat_obs,
            return_preactivations=return_preactivations,
        )
