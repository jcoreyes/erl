from collections import OrderedDict

import numpy as np
import torch

import railrl.torch.pytorch_util as ptu
from railrl.core import logger
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.state_distance.tdm import TemporalDifferenceModel
from railrl.torch.ddpg.ddpg import DDPG
import torch.optim as optim
import torch.nn as nn

class TdmSupervised(TemporalDifferenceModel):
    def __init__(
            self,
            env,
            exploration_policy,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
            loss_fn=None,
            policy_learning_rate=1e-4,
            optimizer_class=optim.Adam,
            policy_criterion=nn.MSELoss,
    ):
        super().__init__(**tdm_kwargs)
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.loss_fn=loss_fn
        self.policy_learning_rate = policy_learning_rate
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )
        self.policy_criterion = policy_criterion()
        self.eval_policy = self.policy

    @property
    def networks(self):
        return [
            self.policy,
        ]

    def _do_training(self):
        batch = self.get_batch()
        obs = batch['observations']
        actions = batch['actions']
        goals = batch['goals']
        num_steps_left = batch['num_steps_left']

        """
        Policy operations.
        """
        policy_actions, pre_tanh_value = self.policy(
            obs, goals, num_steps_left, return_preactivations=True,
        )
        #policy loss!
        policy_loss = -1 * self.policy.criterion(policy_actions, actions)
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