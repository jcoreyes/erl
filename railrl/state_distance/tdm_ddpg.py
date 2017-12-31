from collections import OrderedDict

import numpy as np
import torch

import railrl.torch.pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.state_distance.exploration import MakeUniversal
from railrl.state_distance.tdm import TemporalDifferenceModel
from railrl.torch.algos.ddpg import DDPG


class TdmDdpg(TemporalDifferenceModel, DDPG):
    def __init__(
            self,
            env,
            qf,
            exploration_policy,
            ddpg_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
    ):
        DDPG.__init__(
            self,
            env=env,
            qf=qf,
            policy=policy,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            **ddpg_kwargs,
            **base_kwargs
        )
        super().__init__(**tdm_kwargs)
        # Not supporting these in this implementation
        assert self.qf_weight_decay == 0
        assert self.residual_gradient_weight == 0

    def _do_training(self):
        batch = self.get_batch(training=True)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        policy_actions, pre_tanh_value = self.policy(
            obs, return_preactivations=True,
        )
        pre_activation_policy_loss = (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        q_output = self.qf(obs, policy_actions)
        raw_policy_loss = - q_output.mean()
        policy_loss = (
                raw_policy_loss +
                pre_activation_policy_loss * self.policy_pre_activation_weight
        )

        """
        Critic operations.
        """
        next_actions = self.target_policy(next_obs)
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
        )
        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
        if self.reward_type == 'indicator':
            q_target = torch.clamp(q_target, -self.reward_scale/(1-self.discount), 0)
        q_pred = self.qf(obs, actions)
        if self.reward_type == 'distance' and self.tdm_normalizer:
            q_pred = self.tdm_normalizer.distance_normalizer.normalize(q_pred)
            q_target = self.tdm_normalizer.distance_normalizer.normalize(
                q_target
            )
        bellman_errors = (q_pred - q_target) ** 2
        qf_loss = self.qf_criterion(q_pred, q_target)

        """
        Update Networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self._update_target_networks()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Raw Policy Loss'] = np.mean(ptu.get_numpy(
                raw_policy_loss
            ))
            self.eval_statistics['Preactivation Policy Loss'] = (
                    self.eval_statistics['Policy Loss'] -
                    self.eval_statistics['Raw Policy Loss']
            )
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors',
                ptu.get_numpy(bellman_errors),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def evaluate(self, epoch):
        DDPG.evaluate(self, epoch)

    def pretrain(self):
        super().pretrain()
        if self.qf.tdm_normalizer is not None:
            self.target_qf.tdm_normalizer.copy_stats(
                self.qf.tdm_normalizer
            )
            self.target_policy.tdm_normalizer.copy_stats(
                self.qf.tdm_normalizer
            )
