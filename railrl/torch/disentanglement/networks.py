"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn

from railrl.policies.base import Policy
from railrl.torch.core import PyTorchModule
from railrl.torch.networks import ConcatMlp
import railrl.torch.pytorch_util as ptu


class DisentangledMlpQf(PyTorchModule):

    def __init__(
            self,
            encoder,
            qf_kwargs,
            preprocess_obs_dim,
            action_dim,
            encode_state=False,
            vectorized=False,
            architecture='splice',
            detach_encoder_via_goal=False,
            detach_encoder_via_state=False,
    ):
        """

        :param encoder:
        :param qf_kwargs:
        :param preprocess_obs_dim:
        :param action_dim:
        :param encode_state:
        :param vectorized:
        :param architecture:
         - 'splice': give each Q function a single index into the latent goal
         - 'many_heads': give each Q function the entire latent goal
         - 'single_head': have one Q function that takes in entire latent goal
         - 'huge_single_head': single head but that multiplies the hidden sizes
             by the number of heads in `many_heads`
         - 'single_head_match_many_heads': single head but that multiplies the
             first hidden size by the number of heads in `many_heads`
        """
        super().__init__()
        self.encoder = encoder
        self.preprocess_obs_dim = preprocess_obs_dim
        self.preprocess_goal_dim = encoder.input_size
        self.postprocess_goal_dim = encoder.output_size
        self.encode_state = encode_state
        self.vectorized = vectorized
        self._architecture = architecture
        self._detach_encoder_via_goal = detach_encoder_via_goal
        self._detach_encoder_via_state = detach_encoder_via_state

        # We have a qf for each goal dim, described by qf_kwargs.
        self.feature_qfs = nn.ModuleList()
        if architecture == 'splice':
            qf_goal_input_size = 1
        else:
            qf_goal_input_size = self.postprocess_goal_dim
        if self.encode_state:
            qf_input_size = (
                    self.postprocess_goal_dim + action_dim + qf_goal_input_size
            )
        else:
            qf_input_size = preprocess_obs_dim + action_dim + qf_goal_input_size
        if architecture == 'single_head':
            self.feature_qfs.append(ConcatMlp(
                input_size=qf_input_size,
                output_size=1,
                **qf_kwargs
            ))
        elif architecture == 'huge_single_head':
            new_qf_kwargs = qf_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                size * self.postprocess_goal_dim for size in hidden_sizes
            ]
            self.feature_qfs.append(ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            ))
        elif architecture == 'single_head_match_many_heads':
            new_qf_kwargs = qf_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                hidden_sizes[0] * self.postprocess_goal_dim
            ] + hidden_sizes[1:]
            self.feature_qfs.append(ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            ))
        elif architecture in {'many_heads', 'splice'}:
            for _ in range(self.postprocess_goal_dim):
                self.feature_qfs.append(ConcatMlp(
                    input_size=qf_input_size,
                    output_size=1,
                    **qf_kwargs
                ))
        else:
            raise ValueError(architecture)

    def forward(self, obs, actions, return_individual_q_vals=False, **kwargs):
        obs_and_goal = obs
        # TODO: undo hack. probably just get rid of these variables
        if self.preprocess_obs_dim == self.preprocess_goal_dim:
            obs, goal = obs_and_goal.chunk(2, dim=1)
        else:
            assert obs_and_goal.shape[1] == (
                    self.preprocess_obs_dim + self.preprocess_goal_dim)
            obs = obs_and_goal[:, :self.preprocess_obs_dim]
            goal = obs_and_goal[:, self.preprocess_obs_dim:]

        h_obs = self.encoder(obs) if self.encode_state else obs
        h_obs = h_obs.detach() if self._detach_encoder_via_state else h_obs
        h_goal = self.encoder(goal)
        h_goal = h_goal.detach() if self._detach_encoder_via_goal else h_goal

        total_q_value = 0
        individual_q_vals = []
        for goal_dim_idx, feature_qf in enumerate(self.feature_qfs):
            if self._architecture == 'splice':
                flat_inputs = torch.cat((
                    h_obs,
                    h_goal[:, goal_dim_idx:goal_dim_idx+1],
                    actions
                ), dim=1)
            else:
                flat_inputs = torch.cat((
                    h_obs,
                    h_goal,
                    actions
                ), dim=1)
            q_idx_value = feature_qf(flat_inputs)
            total_q_value += q_idx_value
            individual_q_vals.append(q_idx_value)

        if self.vectorized:
            total_q_value = torch.cat(individual_q_vals, dim=1)

        if return_individual_q_vals:
            return total_q_value, individual_q_vals
        else:
            return total_q_value


class QfMaximizingPolicy(Policy):
    def __init__(
        self,
        qf,
        env,
        num_action_samples=300,
    ):
        self.qf = qf
        self.num_action_samples = num_action_samples
        self.action_lows = env.action_space.low
        self.action_highs = env.action_space.high

    def get_action(self, obs):
        opt_actions, info = self.get_actions(obs[None])
        return opt_actions[0], info

    def get_actions(self, obs):
        obs_tiled = np.repeat(obs, self.num_action_samples, axis=0)
        action_tiled = np.random.uniform(
            low=self.action_lows, high=self.action_highs,
            size=(len(obs_tiled), len(self.action_lows))
        )
        obs_tiled = ptu.from_numpy(obs_tiled)
        action_tiled = ptu.from_numpy(action_tiled)
        q_val_torch = self.qf(obs_tiled, action_tiled)

        # In case it's vectorized, we'll take the largest sum
        q_val_torch = q_val_torch.sum(dim=1)

        # q_val_torch[i][j] is the q_val for obs[i] and random action[j] for
        # that obs (specifically, action_tiled[i * self.num_action_samples + j])
        q_val_torch = q_val_torch.view(len(obs), -1)
        opt_action_idxs = q_val_torch.argmax(dim=1)

        # Defined q_val_torch[i] to be the optimal q_val for obs[i],
        # selected by action_tiled
        opt_actions = ptu.get_numpy(action_tiled[opt_action_idxs])
        return opt_actions, {}

    def to(self, device):
        self.qf.to(device)


class EncodeObsAndGoal(PyTorchModule):
    def __init__(
            self,
            encoder,
            state_dim,
            encode_state=True,
            encode_goal=True,
            detach_encoder_via_goal=False,
            detach_encoder_via_state=False,
    ):
        super().__init__()
        encoder_output_dim = encoder.output_size
        output_dim = 0
        output_dim += encoder_output_dim if encode_state else state_dim
        output_dim += encoder_output_dim if encode_goal else state_dim

        self._encoder = encoder
        self._encode_state = encode_state
        self._encode_goal = encode_goal
        self._detach_encoder_via_goal = detach_encoder_via_goal
        self._detach_encoder_via_state = detach_encoder_via_state
        self.output_size = output_dim

    def forward(self, obs_and_goal, *args, **kwargs):
        obs, goal = obs_and_goal.chunk(2, dim=1)

        h_obs = self._encoder(obs) if self._encode_state else obs
        h_obs = h_obs.detach() if self._detach_encoder_via_state else h_obs
        h_goal = self._encoder(goal) if self._encode_goal else goal
        h_goal = h_goal.detach() if self._detach_encoder_via_goal else h_goal

        return h_obs, h_goal


class DDRArchitecture(PyTorchModule):

    def __init__(
            self,
            encoder: EncodeObsAndGoal,
            qf_kwargs,
            preprocess_obs_dim,
            action_dim,
            encode_state=False,
            vectorized=False,
            architecture='splice',
    ):
        """

        :param encoder:
        :param qf_kwargs:
        :param preprocess_obs_dim:
        :param action_dim:
        :param encode_state:
        :param vectorized:
        :param architecture:
         - 'splice': give each Q function a single index into the latent goal
         - 'many_heads': give each Q function the entire latent goal
         - 'single_head': have one Q function that takes in entire latent goal
         - 'huge_single_head': single head but that multiplies the hidden sizes
             by the number of heads in `many_heads`
         - 'single_head_match_many_heads': single head but that multiplies the
             first hidden size by the number of heads in `many_heads`
        """
        super().__init__()
        self.encoder = encoder
        self.preprocess_obs_dim = preprocess_obs_dim
        self.preprocess_goal_dim = encoder.input_size
        self.postprocess_goal_dim = encoder.output_size
        self.encode_state = encode_state
        self.vectorized = vectorized
        self._architecture = architecture

        # We have a qf for each goal dim, described by qf_kwargs.
        self.feature_qfs = nn.ModuleList()
        if architecture == 'splice':
            qf_goal_input_size = 1
        else:
            qf_goal_input_size = self.postprocess_goal_dim
        if self.encode_state:
            qf_input_size = (
                    self.postprocess_goal_dim + action_dim + qf_goal_input_size
            )
        else:
            qf_input_size = preprocess_obs_dim + action_dim + qf_goal_input_size
        if architecture == 'single_head':
            self.feature_qfs.append(ConcatMlp(
                input_size=qf_input_size,
                output_size=1,
                **qf_kwargs
            ))
        elif architecture == 'huge_single_head':
            new_qf_kwargs = qf_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                size * self.postprocess_goal_dim for size in hidden_sizes
            ]
            self.feature_qfs.append(ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            ))
        elif architecture == 'single_head_match_many_heads':
            new_qf_kwargs = qf_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                                   hidden_sizes[0] * self.postprocess_goal_dim
                               ] + hidden_sizes[1:]
            self.feature_qfs.append(ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            ))
        else:
            for _ in range(self.postprocess_goal_dim):
                self.feature_qfs.append(ConcatMlp(
                    input_size=qf_input_size,
                    output_size=1,
                    **qf_kwargs
                ))

    def forward(self, obs, actions, return_individual_q_vals=False, **kwargs):
        obs_and_goal = obs
        h_obs, h_goal = self.encoder(obs)

        total_q_value = 0
        individual_q_vals = []
        for goal_dim_idx, feature_qf in enumerate(self.feature_qfs):
            if self._architecture == 'splice':
                flat_inputs = torch.cat((
                    h_obs,
                    h_goal[:, goal_dim_idx:goal_dim_idx+1],
                    actions
                ), dim=1)
            else:
                flat_inputs = torch.cat((
                    h_obs,
                    h_goal,
                    actions
                ), dim=1)
            q_idx_value = feature_qf(flat_inputs)
            total_q_value += q_idx_value
            individual_q_vals.append(q_idx_value)

        if self.vectorized:
            total_q_value = torch.cat(individual_q_vals, dim=1)

        if return_individual_q_vals:
            return total_q_value, individual_q_vals
        else:
            return total_q_value
