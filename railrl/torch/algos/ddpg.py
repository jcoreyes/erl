from collections import OrderedDict

import time
import torch
import torch.optim as optim
from torch import nn as nn

import railrl.samplers.util
import railrl.torch.pytorch_util as ptu
from railrl.misc import rllab_util
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.ml_util import (
    StatConditionalSchedule,
    ConstantSchedule,
)
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.eval_util import get_statistics_from_pytorch_dict, \
    get_difference_statistics, get_generic_path_information
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from rllab.misc import logger


class DDPG(TorchRLAlgorithm):
    """
    Online learning algorithm.
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
            num_updates_per_env_step=1,
            qf_criterion=None,
            residual_gradient_weight=0,
            optimize_target_policy=None,
            target_policy_learning_rate=None,
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
        :param num_updates_per_env_step: Number of gradient steps per
        environment step.
        :param qf_criterion: Loss function to use for the q function. Should
        be a function that takes in two inputs (y_predicted, y_target).
        :param residual_gradient_weight: c, float between 0 and 1. The gradient
        used for training the Q function is then
            (1-c) * normal td gradient + c * residual gradient
        :param optimize_target_policy: If False, the target policy is
        updated as in the original DDPG paper. Otherwise, the target policy
        is optimizes the target QF.
        :param target_policy_learning_rate: If None, use the policy_learning
        rate. This parameter is ignored if `optimize_target_policy` is False.
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
        if target_policy_learning_rate is None:
            target_policy_learning_rate = policy_learning_rate
        self.qf = qf
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.num_updates_per_env_step = num_updates_per_env_step
        self.residual_gradient_weight = residual_gradient_weight
        self.qf_criterion = qf_criterion
        self.optimize_target_policy = optimize_target_policy
        self.target_policy_learning_rate = target_policy_learning_rate
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
        if self.optimize_target_policy:
            self.target_policy_optimizer = optim.Adam(
                self.target_policy.parameters(),
                lr=self.target_policy_learning_rate,
            )
        else:
            self.target_policy_optimizer = None

    def _start_epoch(self, epoch):
        super()._start_epoch(epoch)
        self.discount = self.epoch_discount_schedule.get_value(epoch)

    def _do_training(self):
        for i in range(self.num_updates_per_env_step):
            batch = self.get_batch(training=True)
            train_dict = self.get_train_dict(batch)

            self.policy_optimizer.zero_grad()
            policy_loss = train_dict['Policy Loss']
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf_optimizer.zero_grad()
            qf_loss = train_dict['QF Loss']
            qf_loss.backward()
            self.qf_optimizer.step()

            if self.optimize_target_policy:
                self.target_policy_optimizer.zero_grad()
                target_policy_loss = train_dict['Target Policy Loss']
                target_policy_loss.backward()
                self.target_policy_optimizer.step()

            if self.use_soft_update:
                if not self.optimize_target_policy:
                    ptu.soft_update(self.target_policy, self.policy,
                                    self.tau)
                ptu.soft_update(self.target_qf, self.qf, self.tau)
            else:
                if self._n_env_steps_total % self.target_hard_update_period == 0:
                    ptu.copy_model_params_from_to(self.qf, self.target_qf)
                    if not self.optimize_target_policy:
                        ptu.copy_model_params_from_to(self.policy,
                                                      self.target_policy)

    def get_train_dict(self, batch):
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
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions)
        bellman_errors = (y_pred - y_target) ** 2
        raw_qf_loss = self.qf_criterion(y_pred, y_target)

        if self.residual_gradient_weight > 0:
            residual_next_actions = self.policy(next_obs)
            # speed up computation by not backpropping these gradients
            residual_next_actions.detach()
            residual_target_q_values = self.qf(
                next_obs,
                residual_next_actions,
            )
            residual_y_target = (
                rewards
                + (1. - terminals) * self.discount * residual_target_q_values
            )
            residual_bellman_errors = (y_pred - residual_y_target) ** 2
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

        if self.optimize_target_policy:
            target_policy_actions = self.target_policy(obs)
            target_q_output = self.target_qf(obs, target_policy_actions)
            target_policy_loss = - target_q_output.mean()
        else:
            # Always include the target policy loss so that different
            # experiments are easily comparable.
            target_policy_loss = ptu.FloatTensor([0])

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('Unregularized QF Loss', raw_qf_loss),
            ('QF Loss', qf_loss),
            ('Target Policy Loss', target_policy_loss),
        ])

    def evaluate(self, epoch):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        logger.log("Collecting samples for evaluation")
        statistics = OrderedDict()
        train_batch = self.get_batch(training=True)
        validation_batch = self.get_batch(training=False)
        test_paths = self.eval_sampler.obtain_samples()

        if not isinstance(self.epoch_discount_schedule, ConstantSchedule):
            statistics['Discount Factor'] = self.discount

        statistics.update(get_generic_path_information(
            self._exploration_paths, self.discount, stat_prefix="Exploration",
        ))
        statistics.update(self._statistics_from_paths(self._exploration_paths,
                                                      "Exploration"))
        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
        )
        statistics.update(get_generic_path_information(
            test_paths, self.discount, stat_prefix="Test",
        ))
        statistics.update(self._statistics_from_paths(test_paths, "Test"))
        statistics.update(
            get_difference_statistics(
                statistics,
                [
                    'QF Loss Mean',
                    'Policy Loss Mean',
                    'Target Policy Loss Mean',
                ],
            )
        )

        average_returns = rllab_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        logger.set_key_prefix('test ')
        self.env.log_diagnostics(test_paths)
        logger.set_key_prefix('expl ')
        self.env.log_diagnostics(self._exploration_paths)
        logger.set_key_prefix('')
        if isinstance(self.epoch_discount_schedule, StatConditionalSchedule):
            table_dict = rllab_util.get_logger_table_dict()
            # rllab converts things to strings for some reason
            value = float(
                table_dict[self.epoch_discount_schedule.statistic_name]
            )
            self.epoch_discount_schedule.update(value)

        if self.render_eval_paths:
            self.env.render_paths(test_paths)

        if self.plotter:
            self.plotter.draw()

    def offline_evaluate(self, epoch):
        logger.log("Collecting samples for evaluation")
        statistics = OrderedDict()
        # train_batch = self.get_batch(training=True)
        # validation_batch = self.get_batch(training=False)
        #
        # if not isinstance(self.epoch_discount_schedule, ConstantSchedule):
        #     statistics['Discount Factor'] = self.discount
        #
        # statistics.update(self._statistics_from_batch(train_batch, "Train"))

        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def _statistics_from_paths(self, paths, stat_prefix):
        batch = self.paths_to_batch(paths)
        statistics = self._statistics_from_batch(batch, stat_prefix)
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics

    @staticmethod
    def paths_to_batch(paths):
        np_batch = railrl.samplers.util.split_paths_to_dict(paths)
        return np_to_pytorch_batch(np_batch)

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = get_statistics_from_pytorch_dict(
            self.get_train_dict(batch),
            [
                'Unregularized QF Loss',
                'QF Loss',
                'Policy Loss',
                'Target Policy Loss',
            ],
            ['Bellman Errors', 'QF Outputs', 'Policy Actions'],
            stat_prefix
        )
        statistics.update(create_stats_ordered_dict(
            "{} Env Actions".format(stat_prefix),
            ptu.get_numpy(batch['actions'])
        ))

        return statistics

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
