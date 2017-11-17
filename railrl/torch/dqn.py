from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.torch.algos import eval
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.policies.argmax import ArgmaxDiscretePolicy
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
from rllab.misc import logger


class DQN(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            qf,
            learning_rate=1e-3,
            tau=0.001,
            epsilon=0.1,
            **kwargs
    ):
        """

        :param env: Env.
        :param qf: QFunction. Maps from state to action Q-values.
        :param learning_rate: Learning rate for qf. Adam is used.
        :param tau: Soft target tau to update target QF.
        :param epsilon: Probability of taking a random action.
        :param kwargs: kwargs to pass onto TorchRLAlgorithm
        """
        exploration_strategy = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=epsilon,
        )
        self.policy = ArgmaxDiscretePolicy(qf)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=exploration_strategy,
            policy=self.policy,
        )
        super().__init__(
            env, exploration_policy, eval_policy=self.policy, **kwargs
        )
        self.qf = qf
        self.target_qf = self.qf.copy()
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
        statistics.update(eval.get_generic_path_information(
            test_paths, self.discount, stat_prefix="Test",
        ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def get_epoch_snapshot(self, epoch):
        self.training_env.render(close=True)
        return dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
            policy=self.policy,
            env=self.training_env,
        )
