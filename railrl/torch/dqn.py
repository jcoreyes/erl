from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.torch.algos.eval import get_statistics_from_pytorch_dict, \
    get_generic_path_information
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
from rllab.misc import logger


class DQN(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            exploration_policy,
            qf,
            learning_rate=1e-3,
            tau=0.001,
            **kwargs
    ):
        super().__init__(env, exploration_policy, **kwargs)
        self.qf = qf
        self.target_qf = self.qf.copy()
        # self.target_qf = self.qf
        self.learning_rate = learning_rate
        self.tau = tau
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.qf_criterion = nn.MSELoss()

        self.eval_statistics = None

    def training_mode(self, mode):
        self.qf.train(mode)
        self.target_qf.train(mode)

    def _do_training(self):
        batch = self.get_batch(training=True)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Compute loss
        """

        target_q_values = self.target_qf(next_obs).detach().max(
            1, keepdim=True
        )[0]
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        ptu.soft_update_from_to(self.target_qf, self.qf, self.tau)

        """
        Save some statistics for eval
        """
        self.eval_statistics = OrderedDict()
        self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Y Predictions',
            ptu.get_numpy(y_pred),
        ))

    def cuda(self):
        self.qf.cuda()
        self.target_qf.cuda()

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        test_paths = self.eval_sampler.obtain_samples()
        statistics.update(get_generic_path_information(
            test_paths, self.discount, stat_prefix="Test",
        ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()
