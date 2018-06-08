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
            qf1,
            qf2,
            vf,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            train_policy_with_reparameterization=False,
            soft_target_tau=1e-2,
            policy_update_period=1,
            target_update_period=1,
            policy_update_minq=True,
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
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.soft_target_tau = soft_target_tau
        self.policy_update_period = policy_update_period
        self.target_update_period = target_update_period
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )

        self.policy_update_minq = policy_update_minq

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.target_vf = vf.copy()
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(obs,
                                     reparameterize=self.train_policy_with_reparameterization,
                                     return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        VF Loss
        """
        q1_new_actions = self.qf1(obs, new_actions)
        q_new_actions = torch.min(
            q1_new_actions,
            self.qf2(obs, new_actions),
        )
        v_target = q_new_actions - log_pi
        v_pred = self.vf(obs)
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        policy_loss = None
        if self._n_train_steps_total % self.policy_update_period == 0:
            """
            Policy Loss
            """
            # paper says to do + but apparently that's a typo. Do Q - V.
            if self.train_policy_with_reparameterization:
                if self.policy_update_minq is True:
                    policy_loss = (log_pi - q_new_actions).mean()
                else:
                    policy_loss = (log_pi - q1_new_actions).mean()
            else:
                if self.policy_update_minq is True:
                    log_policy_target = q_new_actions - v_pred
                else:
                    log_policy_target = q1_new_actions - v_pred
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

        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.vf, self.target_vf, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            if policy_loss is None:
                if self.train_policy_with_reparameterization:
                    if self.policy_update_minq is True:
                        policy_loss = (log_pi - q_new_actions).mean()
                    else:
                        policy_loss = (log_pi - q1_new_actions).mean()
                else:
                    if self.policy_update_minq is True:
                        log_policy_target = q_new_actions - v_pred
                    else:
                        log_policy_target = q1_new_actions - v_pred
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

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
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
            self.qf1,
            self.qf2,
            self.vf,
            self.target_vf,
        ]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot['qf1'] = self.qf1
        snapshot['qf2'] = self.qf2
        snapshot['policy'] = self.policy
        snapshot['vf'] = self.vf
        snapshot['target_vf'] = self.target_vf
        return snapshot
