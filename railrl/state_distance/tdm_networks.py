import numpy as np
import torch

from railrl.state_distance.policies import UniversalPolicy
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
            structure='norm_difference',
            tdm_normalizer: TdmNormalizer=None,
            learn_offset=False,
            observation_dim=None,
            action_dim=None,
            goal_dim=None,
            **flatten_mlp_kwargs
    ):
        """

        :param env:
        :param hidden_sizes:
        :param vectorized: Boolean. Vectorized or not?
        :param structure: String defining output structure of network:
            - 'norm_difference': Q = -||g - f(inputs)||
            - 'squared_difference': Q = -(g - f(inputs))^2
            - 'squared_difference_offset': Q = -(goal - f(inputs))^2 + f2(s, goal, tau)
            - 'none': Q = f(inputs)

        :param kwargs:
        """
        assert structure in [
            'norm_difference',
            'squared_difference',
            'none',
        ]
        self.save_init_params(locals())

        if observation_dim is None:
            self.observation_dim = env.observation_space.low.size
        else:
            self.observation_dim = observation_dim

        if action_dim is None:
            self.action_dim = env.action_space.low.size
        else:
            self.action_dim = action_dim

        if goal_dim is None:
            self.goal_dim = env.goal_dim
        else:
            self.goal_dim = goal_dim

        super().__init__(
            input_size=(
                    self.observation_dim + self.action_dim + self.goal_dim + 1
            ),
            output_size=self.goal_dim if vectorized else 1,
            **flatten_mlp_kwargs
        )
        self.env = env
        self.vectorized = vectorized
        self.structure = structure
        self.tdm_normalizer = tdm_normalizer
        self.learn_offset = learn_offset
        if learn_offset:
            self.offset_network = FlattenMlp(
                input_size=(
                    self.observation_dim + self.action_dim + self.goal_dim + 1
                ),
                output_size=self.goal_dim if vectorized else 1,
                **flatten_mlp_kwargs
            )

    def forward(
            self,
            observations,
            actions,
            goals,
            num_steps_left,
            return_predictions=False
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
        if return_predictions:
            return predictions

        if self.structure == 'norm_difference':
            output = - torch.abs(goals - predictions)
        elif self.structure == 'squared_difference':
            output = - (goals - predictions)**2
        elif self.structure == 'none':
            output = predictions
        else:
            raise TypeError("Invalid structure: {}".format(self.structure))
        if not self.vectorized:
            output = torch.sum(output, dim=1, keepdim=True)

        if self.learn_offset:
            offset = self.offset_network(
                observations, actions, goals, num_steps_left
            )
            output = output + offset

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
            observation_dim=None,
            action_dim=None,
            goal_dim=None,
            **kwargs
    ):
        self.save_init_params(locals())

        if observation_dim is None:
            self.observation_dim = env.observation_space.low.size
        else:
            self.observation_dim = observation_dim

        if action_dim is None:
            self.action_dim = env.action_space.low.size
        else:
            self.action_dim = action_dim

        if goal_dim is None:
            self.goal_dim = env.goal_dim
        else:
            self.goal_dim = goal_dim

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
            structure='none',
            tdm_normalizer: TdmNormalizer=None,
            observation_dim=None,
            action_dim=None,
            goal_dim=None,
            **kwargs
    ):
        assert structure in [
            'norm_difference',
            'squared_difference',
            'none',
        ]
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
        self.structure = structure
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
        predictions = super().forward(
            observations, goals, num_steps_left
        )

        if self.structure == 'norm_difference':
            output = - torch.abs(goals - predictions)
        elif self.structure == 'squared_difference':
            output = - (goals - predictions)**2
        elif self.structure == 'none':
            output = predictions
        else:
            raise TypeError("Invalid structure: {}".format(self.structure))
        if not self.vectorized:
            output = torch.sum(output, dim=1, keepdim=True)

        if self.tdm_normalizer is not None:
            output = self.tdm_normalizer.distance_normalizer.denormalize_scale(
                output
            )
        return output


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
