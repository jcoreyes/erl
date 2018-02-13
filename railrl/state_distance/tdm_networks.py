"""
This is basically as re-write of the networks.py file but for tdm.py rather
than sdql.py
"""
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from railrl.pythonplusplus import identity
from railrl.state_distance.util import split_tau, extract_goals, split_flat_obs, merge_into_flat_obs
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.networks import Mlp
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
            goal_dim,
            output_size,
            hidden_sizes,
            max_tau=None,
            action_dim=0,
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

    def forward(self, flat_obs, actions=None):
        if actions is not None:
            h = torch.cat((flat_obs, actions), dim=1)
        else:
            h = flat_obs
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
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            action_dim=0,
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

    def forward(self, flat_obs, actions=None):
        obs, taus = split_tau(flat_obs)
        if actions is not None:
            h = torch.cat((obs, actions), dim=1)
        else:
            h = obs
        batch_size = h.size()[0]
        y_binary = ptu.FloatTensor(batch_size, self.max_tau + 1)
        y_binary.zero_()
        t = taus.data.long()
        t = torch.clamp(t, min=0)
        y_binary.scatter_(1, t, 1)
        if actions is not None:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),
                actions
            ), dim=1)
        else:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),
            ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))


class BinaryStringTauQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a binary string vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        self.max_tau = np.unpackbits(np.array(max_tau, dtype=np.uint8))
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + len(self.max_tau),
            output_size=output_size,
            **kwargs
        )

    def forward(self, flat_obs, actions=None):
        obs, taus = split_tau(flat_obs)
        if actions is not None:
            h = torch.cat((obs, actions), dim=1)
        else:
            h = obs
        batch_size = taus.size()[0]
        y_binary = make_binary_tensor(taus, len(self.max_tau), batch_size)

        if actions is not None:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),
                actions
            ), dim=1)
        else:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),

            ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))


def make_binary_tensor(tensor, max_len, batch_size):
    t = tensor.data.numpy().astype(int).reshape(batch_size)
    binary = (((t[:,None] & (1 << np.arange(max_len)))) > 0).astype(int)
    binary = torch.from_numpy(binary)
    binary  = binary.float()
    binary = binary.view(batch_size, max_len)
    return binary


class TauVectorQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a binary string vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            tau_vector_len=0,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + self.tau_vector_len,
            output_size=output_size,
            **kwargs
        )

    def forward(self, flat_obs, actions=None):
        obs, taus = split_tau(flat_obs)
        if actions is not None:
            h = torch.cat((obs, action), dim=1)
        else:
            h = obs
        batch_size = h.size()[0]
        tau_vector = torch.zeros((batch_size, self.tau_vector_len)) + taus.data
        if actions is not None:
            h = torch.cat((
                obs,
                ptu.Variable(tau_vector),
                actions
            ), dim=1)
        else:
            h = torch.cat((
                obs,
                ptu.Variable(tau_vector),

            ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))


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

        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []

        self.first_input = nn.Linear(first_input_size, first_input_size)
        hidden_init(self.first_input.weight)
        self.first_input.bias.data.fill_(b_init_value)

        self.second_input = nn.Linear(second_input_size, second_input_size)
        hidden_init(self.second_input.weight)
        self.second_input.bias.data.fill_(b_init_value)

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
        h = torch.cat((h1, h2), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class TauVectorSeparateFirstLayerQF(SeparateFirstLayerMlp):
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            tau_vector_len=0,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau

        super().__init__(
            hidden_sizes=hidden_sizes,
            first_input_size=observation_dim + action_dim + goal_dim,
            second_input_size=self.tau_vector_len,
            output_size=output_size,
            **kwargs
        )

    def forward(self, flat_obs, actions=None):
        obs, taus = split_tau(flat_obs)
        if actions is not None:
            h = torch.cat((obs, actions), dim=1)
        else:
            h = obs

        batch_size = h.size()[0]
        tau_vector = Variable(torch.zeros((batch_size, self.tau_vector_len)) + taus.data)
        return - torch.abs(super().forward(h, tau_vector))
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


class DebugQf(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            **kwargs
    ):
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size=(
                    self.observation_dim + self.action_dim
            ),
            output_size=self.goal_dim,
            **kwargs
        )
        self.env = env
        self.vectorized = vectorized
        self.tdm_normalizer = None

    def forward(self, flat_obs, actions, return_internal_prediction=False):
        if self.tdm_normalizer is not None:
            actions = self.tdm_normalizer.action_normalizer.normalize(actions)
            flat_obs = self.tdm_normalizer.normalize_flat_obs(flat_obs)

        obs, goals, _ = split_flat_obs(
            flat_obs, self.observation_dim, self.goal_dim
        )
        deltas = super().forward(obs, actions)
        if return_internal_prediction:
            return deltas
        features = self.env.convert_obs_to_goals(obs)
        next_features_predicted = deltas + features
        diff = next_features_predicted - goals
        if self.vectorized:
            output = -diff**2
        else:
            raise NotImplementedError
            output = -(diff**2).sum(1, keepdim=True)
        return output


class DebugQfToModel(nn.Module):
    def __init__(self, debug_qf):
        super().__init__()
        self.debug_qf = debug_qf

    def forward(self, states, actions):
        fake_flat_obs = merge_into_flat_obs(states, states, states)
        obs_delta = self.debug_qf(
            fake_flat_obs, actions, return_internal_prediction=True
        )


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

class StandardTdmPolicy(TanhMlpPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            init_w=1e-3,
            max_tau=None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim+goal_dim + 1,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )

class OneHotTauTdmPolicy(TanhMlpPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            max_tau,
            init_w=1e-3,
            **kwargs
    ):
        self.max_tau = max_tau
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim+max_tau+goal_dim+1,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )

    def forward(
            self,
            flat_obs,
            return_preactivations=False
    ):
        obs, taus = split_tau(flat_obs)
        h = obs
        batch_size = h.size()[0]
        y_binary = ptu.FloatTensor(batch_size, self.max_tau + 1)
        y_binary.zero_()
        t = taus.data.long()
        t = torch.clamp(t, min=0)
        y_binary.scatter_(1, t, 1)

        h = torch.cat((
            obs,
            ptu.Variable(y_binary),
        ), dim=1)

        return super().forward(
            h,
            return_preactivations=return_preactivations,
        )

class BinaryTauTdmPolicy(TanhMlpPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            max_tau,
            init_w=1e-3,
            **kwargs
    ):
        self.max_tau = np.unpackbits(np.array(max_tau, dtype=np.uint8))
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + goal_dim+ len(self.max_tau),
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
    def forward(
            self,
            flat_obs,
            return_preactivations=False,
    ):
        obs, taus = split_tau(flat_obs)
        batch_size = taus.size()[0]
        y_binary = make_binary_tensor(taus, len(self.max_tau), batch_size)
        h = torch.cat((
            obs,
            ptu.Variable(y_binary),
        ), dim=1)

        return super().forward(
            h,
            return_preactivations=return_preactivations
        )

class TauVectorTdmPolicy(TanhMlpPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            max_tau,
            tau_vector_len=0,
            init_w=1e-3,
            **kwargs
    ):
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + goal_dim + self.tau_vector_len,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )

    def forward(
            self,
            flat_obs,
            return_preactivations=False,
        ):
        obs, taus = split_tau(flat_obs)
        h=obs
        batch_size = h.size()[0]
        tau_vector = torch.zeros((batch_size, self.tau_vector_len)) + taus.data
        h = torch.cat((
                obs,
                ptu.Variable(tau_vector),
            ), dim=1)

        return super().forward(
            h,
            return_preactivations=return_preactivations
        )

class TauVectorSeparateFirstLayerTdmPolicy(SeparateFirstLayerMlp):

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            max_tau,
            tau_vector_len=0,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau
        super().__init__(
            hidden_sizes=hidden_sizes,
            first_input_size=obs_dim + goal_dim,
            second_input_size=self.tau_vector_len,
            output_size=action_dim,
            **kwargs
        )

    def forward(
            self,
            flat_obs,
            return_preactivations=False,
    ):
        obs, taus = split_tau(flat_obs)
        batch_size = obs.size()[0]
        tau_vector = Variable(torch.zeros((batch_size, self.tau_vector_len)) + taus.data)
        h = obs
        h1 = self.hidden_activation(self.first_input(h))
        h2 = self.hidden_activation(self.second_input(tau_vector))
        h = torch.cat((h1, h2), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivations = self.last_fc(h)
        actions = self.output_activation(preactivations)
        if return_preactivations:
            return actions, preactivations
        else:
            return actions

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)

class MakeNormalizedTDMQF(PyTorchModule):
    def __init__(self,
                 qf,
                 obs_normalizer: TorchFixedNormalizer = None,
                 action_normalizer: TorchFixedNormalizer = None,
            ):
        self.save_init_params(locals())
        super().__init__()
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.qf = qf

    def forward(self, flat_obs, actions=None, **kwargs):
        observations, taus = split_tau(flat_obs)
        if self.obs_normalizer:
            observations = self.obs_normalizer.normalize(observations)
        if self.action_normalizer and actions is not None:
            actions = self.action_normalizer.normalize(actions)
        flat_obs = torch.cat((
            observations,
            taus
        ))
        return self.qf.forward(flat_obs, actions=actions, **kwargs)

class MakeNormalizedTDMPolicy(PyTorchModule):
    def __init__(self,
                 qf,
                 obs_normalizer: TorchFixedNormalizer = None,
                 action_normalizer: TorchFixedNormalizer = None,
            ):
        self.save_init_params(locals())
        super().__init__()
        self.obs_normalizer = obs_normalizer
        self.qf = qf

    def forward(self, flat_obs, **kwargs):
        observations, taus = split_tau(flat_obs)
        if self.obs_normalizer:
            observations = self.obs_normalizer.normalize(observations)
        flat_obs = torch.cat((
            observations,
            taus
        ))
        return self.qf.forward(flat_obs, **kwargs)
