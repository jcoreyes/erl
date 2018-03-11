from collections import OrderedDict

import torch
import numpy as np
import torch.optim as optim
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.sac.policies import MakeDeterministic
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm


class TwinSAC(TorchRLAlgorithm):
    """
    TD3 + SAC
    """
    def __init__(
            self,
            env,
            policy,
            qf,
            vf1,
            vf2,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            policy_and_target_update_period=2,
            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,

            eval_policy=None,
            exploration_policy=None,
            **kwargs
    ):
        if eval_policy is None:
            if eval_deterministic:
                eval_policy = MakeDeterministic(policy)
            else:
                eval_policy = policy
        super().__init__(
            env=env,
            exploration_policy=exploration_policy or policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.policy = policy
        self.qf = qf
        self.vf1 = vf1
        self.vf2 = vf2
        self.soft_target_tau = soft_target_tau
        self.policy_and_target_update_period = policy_and_target_update_period
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.target_vf1 = vf1.copy()
        self.target_vf2 = vf2.copy()
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )
        self.vf1_optimizer = optimizer_class(
            self.vf1.parameters(),
            lr=vf_lr,
        )
        self.vf2_optimizer = optimizer_class(
            self.vf2.parameters(),
            lr=vf_lr,
        )

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q_pred = self.qf(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        QF Loss
        """
        target_v_values = torch.min(
            self.target_vf1(next_obs),
            self.target_vf2(next_obs),
        )
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """
        VF Loss
        """
        q_new_actions = self.qf(obs, new_actions)
        v_target = q_new_actions - log_pi
        v1_pred = self.vf1(obs)
        v2_pred = self.vf2(obs)

        vf1_loss = self.vf_criterion(v1_pred, v_target.detach())
        vf2_loss = self.vf_criterion(v2_pred, v_target.detach())

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf1_optimizer.zero_grad()
        vf1_loss.backward()
        self.vf1_optimizer.step()

        self.vf2_optimizer.zero_grad()
        vf2_loss.backward()
        self.vf2_optimizer.step()

        policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            """
            Policy Loss
            """
            # paper says to do + but apparently that's a typo. Do Q - V.
            log_policy_target = q_new_actions - v1_pred
            policy_loss = (
                log_pi * (log_pi - log_policy_target).detach()
            ).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(
                self.vf1, self.target_vf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.vf2, self.target_vf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            if policy_loss is None:
                log_policy_target = q_new_actions - v1_pred
                policy_loss = (
                    log_pi * (log_pi - log_policy_target).detach()
                ).mean()
                mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
                std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
                pre_tanh_value = policy_outputs[-1]
                pre_activation_reg_loss = self.policy_pre_activation_weight * (
                    (pre_tanh_value**2).sum(dim=1).mean()
                )
                policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
                policy_loss = policy_loss + policy_reg_loss

            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF1 Loss'] = np.mean(ptu.get_numpy(vf1_loss))
            self.eval_statistics['VF2 Loss'] = np.mean(ptu.get_numpy(vf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V1 Predictions',
                ptu.get_numpy(v1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V2 Predictions',
                ptu.get_numpy(v2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.vf1,
            self.vf2,
            self.target_vf1,
            self.target_vf2,
        ]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot['qf'] = self.qf
        snapshot['policy'] = self.policy
        snapshot['vf1'] = self.vf1
        snapshot['vf2'] = self.vf2
        snapshot['target_vf2'] = self.target_vf2
        snapshot['target_vf1'] = self.target_vf1
        return snapshot
