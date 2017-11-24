from collections import OrderedDict

import torch
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.sac.sac import SoftActorCritic


class ExpectedSAC(SoftActorCritic):
    """
    Compute

    E_{a \sim \pi(. | s)}[Q(s, a) - \log \pi(a | s)]

    in closed form
    """
    def _do_training(self):
        batch = self.get_batch(training=True)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q_pred = self.qf(obs, actions)
        v_pred = self.vf(obs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        (
            new_actions, policy_mean, policy_log_std, log_pi, expected_log_prob,
            policy_stds
        ) = self.policy(
            obs,
            return_log_prob=True,
            return_expected_log_prob=True,
        )

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """
        VF Loss
        """
        # q_new_actions = self.qf(obs, new_actions)
        # v_target = q_new_actions - log_prob
        expected_q = self.qf(obs, torch.zeros_like(new_actions),
                             action_stds=policy_stds)
        v_target = expected_q - expected_log_prob
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Policy Loss
        """
        # paper says to do + but Tuomas said that's a typo. Do Q - V.
        q_new_actions = self.qf(obs, new_actions)
        log_policy_target = q_new_actions - v_pred
        policy_loss = (
            log_pi * (log_pi - log_policy_target).detach()
        ).mean()
        policy_reg_loss = self.policy_reg_weight * (
            (policy_mean**2).mean()
            + (policy_log_std**2).mean()
        )
        policy_loss = policy_loss + policy_reg_loss

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_network()

        """
        Save some statistics for eval
        """
        self.eval_statistics = OrderedDict()
        self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
        self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
        self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
            policy_loss
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Q Predictions',
            ptu.get_numpy(q_pred),
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
