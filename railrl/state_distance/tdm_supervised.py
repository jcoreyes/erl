from collections import OrderedDict

import numpy as np
import torch

import railrl.torch.pytorch_util as ptu
from railrl.core import logger
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.state_distance.tdm import TemporalDifferenceModel
import torch.optim as optim
import torch.nn as nn

from railrl.torch.modules import HuberLoss
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm


class TdmSupervised(TemporalDifferenceModel, TorchRLAlgorithm):
    def __init__(
            self,
            env,
            exploration_policy,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            loss_fn=None,
            policy_learning_rate=1e-3,
            optimizer_class=optim.Adam,
            policy_criterion='MSE',
            replay_buffer=None,
    ):
        TorchRLAlgorithm.__init__(
            self,
            env,
            exploration_policy,
            **base_kwargs
        )
        super().__init__(**tdm_kwargs)
        self.policy = policy
        self.loss_fn=loss_fn
        self.policy_learning_rate = policy_learning_rate
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )
        if policy_criterion=='MSE':
            self.policy_criterion = nn.MSELoss()
        elif policy_criterion=='Huber':
            self.policy_criterion = HuberLoss()
        self.eval_policy = self.policy
        self.replay_buffer = replay_buffer

    @property
    def networks(self):
        return [
            self.policy,
        ]

    def _do_training(self):
        batch = self.get_batch()
        obs = batch['observations']
        actions = batch['actions']
        num_steps_left = batch['num_steps_left']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        # import ipdb; ipdb.set_trace()
        policy_actions = self.policy(
            obs, self.env.convert_obs_to_goals(next_obs), num_steps_left, return_preactivations=False,
        )
        policy_loss = self.policy_criterion(policy_actions, actions)
        """
        Update Networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.eval_statistics is None:
            """
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def evaluate(self, epoch):
        statistics = OrderedDict()
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        super().evaluate(epoch)

    def pretrain(self):
        pass

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self.eval_policy,
            trained_policy=self.policy,
            exploration_policy=self.exploration_policy,
        )
        return snapshot