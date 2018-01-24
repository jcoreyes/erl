from collections import OrderedDict

import numpy as np
import torch.optim as optim

import railrl.torch.pytorch_util as ptu
import torch
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.misc.ml_util import (
    StatConditionalSchedule,
    ConstantSchedule,
)
from railrl.policies.simple import RandomPolicy
from railrl.samplers.util import rollout
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from railrl.core import logger
from torch import nn as nn


class VPG:
    """
    Vanilla Policy Gradient IN PROGRESS
    """

    def __init__(
            self,
            env,
            qf,
            policy,
            num_samples_per_train_step,
            policy_learning_rate=1e-4,
            epoch_discount_schedule=None,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        #for vpg - what do I do about not having an exploration policy
        eval_policy = policy
        super().__init__(
            env,
            policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        if epoch_discount_schedule is None:
            epoch_discount_schedule = ConstantSchedule(self.discount)
        self.epoch_discount_schedule = epoch_discount_schedule
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)
        self.eval_statistics = None

    def _start_epoch(self, epoch):
        super()._start_epoch(epoch)
        self.discount = self.epoch_discount_schedule.get_value(epoch)

    def get_batch(self, training=True):
        pass

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
        if self.policy_pre_activation_weight > 0:
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
        else:
            policy_actions = self.policy(obs)
            q_output = self.qf(obs, policy_actions)
            raw_policy_loss = policy_loss = - q_output.mean()

        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Raw Policy Loss'] = np.mean(ptu.get_numpy(
                raw_policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def train_online(self, start_epoch=0):


    def evaluate(self, epoch):
        statistics = OrderedDict()
        if isinstance(self.epoch_discount_schedule, StatConditionalSchedule):
            table_dict = logger.get_table_dict()
            value = float(
                table_dict[self.epoch_discount_schedule.statistic_name]
            )
            self.epoch_discount_schedule.update(value)

        if not isinstance(self.epoch_discount_schedule, ConstantSchedule):
            statistics['Discount Factor'] = self.discount

        for key, value in statistics.items():
            logger.record_tabular(key, value)
        super().evaluate(epoch)

    def offline_evaluate(self, epoch):
        statistics = OrderedDict()
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def get_epoch_snapshot(self, epoch):
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
            policy=self.eval_policy,
            trained_policy=self.policy,
            exploration_policy=self.exploration_policy,
            batch_size=self.batch_size,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    @property
    def networks(self):
        return [
            self.policy,
        ]
