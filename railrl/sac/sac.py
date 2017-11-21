"""
Soft actor-critic

Notes from Tuomas
a = tanh(z)
Q(a, s) = Q(tanh(z), s)
z is output of actor

Policy
pi ~ e^{Q(s, a)}
   ~ e^{Q(s, tanh(z))}

Mean
 - regularize gaussian means of policy towards zeros

Covariance
 - output log of diagonals
 - Clip the log of the diagonals
 - regularize the log towards zero
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.torch.algos import eval
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
from rllab.misc import logger


class SoftActorCritic(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            qf,
            vf,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,

            soft_target_tau=1e-2,
            **kwargs
    ):
        super().__init__(
            env=env,
            exploration_policy=policy,
            **kwargs
        )
        self.policy = policy
        self.qf = qf
        self.vf = vf
        self.soft_target_tau = soft_target_tau

        self.target_vf = vf.copy()
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.policy_criterion = nn.MSELoss()
        self.eval_statistics = None

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optim.Adam(
            self.vf.parameters(),
            lr=vf_lr,
        )

    def _do_training(self):
        batch = self.get_batch(training=True)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q_pred = self.qf(obs, actions)
        v_pred = self.vf(obs)
        log_pi = self.policy(obs, return_log_prob=True)

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        q_target = q_target.detach()
        qf_loss = self.qf_criterion(q_pred, q_target)

        """
        VF Loss
        """
        # Make sure policy accounts for squashing functions like tanh correctly!
        q_target_for_v = self.qf(obs, self.policy(obs))
        v_target = q_target_for_v - log_pi
        v_target = v_target.detach()
        vf_loss = self.vf_criterion(v_pred, v_target)

        """
        Policy Loss
        """
        log_policy_target = q_pred + v_pred
        policy_loss = self.policy_criterion(log_pi, log_policy_target)

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

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        test_paths = self.eval_sampler.obtain_samples()
        statistics.update(eval.get_generic_path_information(
            test_paths, self.discount, stat_prefix="Test",
        ))
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.vf,
            self.target_vf,
        ]

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def _update_target_network(self):
        ptu.soft_update(self.target_vf, self.vf, self.soft_target_tau)
