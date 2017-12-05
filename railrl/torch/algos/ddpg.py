from collections import OrderedDict

import numpy as np
import torch.optim as optim

import railrl.torch.pytorch_util as ptu
import torch
from railrl.misc import rllab_util
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.ml_util import (
    StatConditionalSchedule,
    ConstantSchedule,
)
from railrl.torch import eval_util
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from rllab.misc import logger
from torch import nn as nn


class DDPG(TorchRLAlgorithm):
    """
    Deep Deterministic Policy Gradient
    """

    def __init__(
            self,
            env,
            qf,
            policy,
            exploration_policy,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            qf_criterion=None,
            residual_gradient_weight=0,
            epoch_discount_schedule=None,

            plotter=None,
            render_eval_paths=False,

            **kwargs
    ):
        """

        :param env:
        :param qf:
        :param policy:
        :param exploration_policy:
        :param policy_learning_rate:
        :param qf_learning_rate:
        :param qf_weight_decay:
        :param target_hard_update_period:
        :param tau:
        :param use_soft_update:
        :param qf_criterion: Loss function to use for the q function. Should
        be a function that takes in two inputs (y_predicted, y_target).
        :param residual_gradient_weight: c, float between 0 and 1. The gradient
        used for training the Q function is then
            (1-c) * normal td gradient + c * residual gradient
        :param epoch_discount_schedule: A schedule for the discount factor
        that varies with the epoch.
        :param kwargs:
        """
        super().__init__(
            env,
            exploration_policy,
            eval_policy=policy,
            **kwargs
        )
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf = qf
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.residual_gradient_weight = residual_gradient_weight
        self.qf_criterion = qf_criterion
        if epoch_discount_schedule is None:
            epoch_discount_schedule = ConstantSchedule(self.discount)
        self.epoch_discount_schedule = epoch_discount_schedule
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.target_qf = self.qf.copy()
        self.target_policy = self.policy.copy()
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.qf_learning_rate,
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)
        self.eval_statistics = None

    def _start_epoch(self, epoch):
        super()._start_epoch(epoch)
        self.discount = self.epoch_discount_schedule.get_value(epoch)

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
        policy_actions = self.policy(obs)
        q_output = self.qf(obs, policy_actions)
        policy_loss = - q_output.mean()

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
        q_pred = self.qf(obs, actions)
        bellman_errors = (q_pred - q_target) ** 2
        raw_qf_loss = self.qf_criterion(q_pred, q_target)

        if self.residual_gradient_weight > 0:
            residual_next_actions = self.policy(next_obs)
            # speed up computation by not backpropping these gradients
            residual_next_actions.detach()
            residual_target_q_values = self.qf(
                next_obs,
                residual_next_actions,
            )
            residual_q_target = (
                rewards
                + (1. - terminals) * self.discount * residual_target_q_values
            )
            residual_bellman_errors = (q_pred - residual_q_target) ** 2
            # noinspection PyUnresolvedReferences
            residual_qf_loss = residual_bellman_errors.mean()
            raw_qf_loss = (
                self.residual_gradient_weight * residual_qf_loss
                + (1 - self.residual_gradient_weight) * raw_qf_loss
            )

        if self.qf_weight_decay > 0:
            reg_loss = self.qf_weight_decay * sum(
                torch.sum(param ** 2)
                for param in self.qf.regularizable_parameters()
            )
            qf_loss = raw_qf_loss + reg_loss
        else:
            qf_loss = raw_qf_loss

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

    def _update_target_networks(self):
        if self.use_soft_update:
            ptu.soft_update(self.target_policy, self.policy, self.tau)
            ptu.soft_update(self.target_qf, self.qf, self.tau)
        else:
            if self._n_env_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
                ptu.copy_model_params_from_to(self.policy, self.target_policy)

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, self.discount, stat_prefix="Test",
        ))
        statistics.update(eval_util.get_generic_path_information(
            self._exploration_paths, self.discount, stat_prefix="Exploration",
        ))
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
            logger.set_key_prefix('Exploration ')
            self.env.log_diagnostics(self._exploration_paths)
            logger.set_key_prefix('')

        if isinstance(self.epoch_discount_schedule, StatConditionalSchedule):
            table_dict = rllab_util.get_logger_table_dict()
            # rllab converts things to strings for some reason
            value = float(
                table_dict[self.epoch_discount_schedule.statistic_name]
            )
            self.epoch_discount_schedule.update(value)

        if not isinstance(self.epoch_discount_schedule, ConstantSchedule):
            statistics['Discount Factor'] = self.discount

        average_returns = rllab_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        if self.render_eval_paths:
            self.env.render_paths(test_paths)

        if self.plotter:
            self.plotter.draw()

    def offline_evaluate(self, epoch):
        statistics = OrderedDict()
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            policy=self.policy,
            env=self.training_env,
            exploration_policy=self.exploration_policy,
            qf=self.qf,
            batch_size=self.batch_size,
        )

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        return dict(
            epoch=epoch,
            replay_buffer=self.replay_buffer,
            algorithm=self,
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.target_policy,
            self.target_qf,
        ]
