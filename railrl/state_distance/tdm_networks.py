import numpy as np
import torch

from railrl.state_distance.policies import UniversalPolicy
from railrl.state_distance.util import split_flat_obs, merge_into_flat_obs
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.networks import TanhMlpPolicy, FlattenMlp
from railrl.torch.sac.policies import TanhGaussianPolicy


class TdmNormalizer(object):
    def __init__(
            self,
            env,
            vectorized,
            normalize_tau=False,
            max_tau=0,
            log_tau=False,
    ):
        if normalize_tau:
            assert max_tau > 0, "Max tau must be larger than 0 if normalizing"
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        self.obs_normalizer = TorchFixedNormalizer(self.observation_dim)
        self.goal_normalizer = TorchFixedNormalizer(env.goal_dim)
        self.action_normalizer = TorchFixedNormalizer(self.action_dim)
        self.distance_normalizer = TorchFixedNormalizer(
            env.goal_dim if vectorized else 1
        )
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

    def normalize_num_steps_left(self, num_steps_left):
        if self.log_tau:
            # minimum tau is -1 (although the output should be ignored for
            # the `tau == -1` case.
            num_steps_left = torch.log(num_steps_left + 2)
        if self.normalize_tau:
            num_steps_left = (num_steps_left - self.tau_mean) / self.tau_std
        return num_steps_left

    def normalize_all(
            self,
            obs,
            actions,
            goals,
            num_steps_left
    ):
        if obs is not None:
            obs = self.obs_normalizer.normalize(obs)
        if actions is not None:
            actions = self.action_normalizer.normalize(actions)
        if goals is not None:
            goals = self.goal_normalizer.normalize(goals)
        if num_steps_left is not None:
            num_steps_left = self.normalize_num_steps_left(num_steps_left)
        return obs, actions, goals, num_steps_left

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

    def forward(
            self,
            observations,
            actions,
            goals,
            num_steps_left,
            return_internal_prediction=False,
    ):
        if self.tdm_normalizer is not None:
            observations, actions, goals, num_steps_left = (
                self.tdm_normalizer.normalize_all(
                    observations, actions, goals, num_steps_left
                )
            )

        predictions = super().forward(
            observations, actions, goals, num_steps_left
        )
        if return_internal_prediction:
            return predictions

        if self.vectorized:
            if self.structure == 'norm_difference':
                output = - torch.abs(goals - predictions)
            elif self.structure == 'norm':
                output = - torch.abs(predictions)
            elif self.structure == 'norm_distance_difference':
                current_features = self.env.convert_obs_to_goals(observations)
                current_distance = torch.abs(goals - current_features)
                output = - torch.abs(predictions + current_distance)
            elif self.structure == 'distance_difference':
                current_features = self.env.convert_obs_to_goals(observations)
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
                current_features = self.env.convert_obs_to_goals(observations)
                current_distance = torch.norm(
                    goals - current_features,
                    p=self.norm_order,
                    dim=1,
                    keepdim=True,
                )
                output = - torch.abs(predictions + current_distance)
            elif self.structure == 'distance_difference':
                current_features = self.env.convert_obs_to_goals(observations)
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
            predict_delta=True,
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
        self.predict_delta = predict_delta
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
        if self.predict_delta:
            features = self.env.convert_obs_to_goals(obs)
            next_features_predicted = deltas + features
        else:
            next_features_predicted = deltas
        diff = next_features_predicted - goals
        if self.vectorized:
            output = -diff**2
        else:
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

    def forward(
            self,
            observations,
            goals,
            num_steps_left,
            return_preactivations=False,
    ):
        if self.tdm_normalizer is not None:
            observations, _, goals, num_steps_left = (
                self.tdm_normalizer.normalize_all(
                    observations, None, goals, num_steps_left
                )
            )

        flat_input = torch.cat((observations, goals, num_steps_left), dim=1)
        return super().forward(
            flat_input,
            return_preactivations=return_preactivations,
        )

    def get_action(self, ob_np, goal_np, tau_np):
        actions = self.eval_np(
            ob_np[None],
            goal_np[None],
            tau_np[None],
        )
        return actions[0, :], {}


class TdmVf(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            tdm_normalizer: TdmNormalizer=None,
            **kwargs
    ):
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size= self.observation_dim + self.goal_dim + 1,
            output_size=self.goal_dim if vectorized else 1,
            **kwargs
        )
        self.env = env
        self.vectorized = vectorized
        self.tdm_normalizer = tdm_normalizer

    def forward(
            self,
            observations,
            goals,
            num_steps_left,
    ):
        if self.tdm_normalizer is not None:
            observations, _, goals, num_steps_left = (
                self.tdm_normalizer.normalize_all(
                    observations, None, goals, num_steps_left
                )
            )

        return super().forward(
            observations,
            goals,
            num_steps_left,
        )


class StochasticTdmPolicy(TanhGaussianPolicy, UniversalPolicy):
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
            obs_dim=self.observation_dim + self.goal_dim + 1,
            action_dim=self.action_dim,
            **kwargs
        )
        self.env = env
        self.tdm_normalizer = tdm_normalizer

    def forward(
            self,
            observations,
            goals,
            num_steps_left,
            **kwargs
    ):
        if self.tdm_normalizer is not None:
            observations, _, goals, num_steps_left = (
                self.tdm_normalizer.normalize_all(
                    observations, None, goals, num_steps_left
                )
            )
        flat_input = torch.cat((observations, goals, num_steps_left), dim=1)
        return super().forward(flat_input, **kwargs)

    def get_action(self, ob_np, goal_np, tau_np, deterministic=False):
        actions = self.get_actions(
            ob_np[None],
            goal_np[None],
            tau_np[None],
            deterministic=deterministic
        )
        return actions[0, :], {}

    def get_actions(self, obs_np, goals_np, taus_np, deterministic=False):
        return self.eval_np(
            obs_np, goals_np, taus_np, deterministic=deterministic
        )[0]
