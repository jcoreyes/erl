from collections import OrderedDict

import torch.optim as optim
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.ml_util import ConstantSchedule
from railrl.misc.rllab_util import get_average_returns, split_paths_to_dict
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.algos.eval import get_statistics_from_pytorch_dict, \
    get_difference_statistics, get_generic_path_information
from railrl.torch.online_algorithm import OnlineAlgorithm
from rllab.misc import logger


class DDPG(OnlineAlgorithm):
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
            replay_buffer=None,
            number_of_gradient_steps=1,
            qf_criterion=None,
            differentiate_through_target=False,
            **kwargs
    ):
        super().__init__(
            env,
            exploration_policy,
            **kwargs
        )
        self.qf = qf
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.target_qf = self.qf.copy()
        self.target_policy = self.policy.copy()
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.number_of_gradient_steps=number_of_gradient_steps
        self.differentiate_through_target = differentiate_through_target
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf_criterion = qf_criterion
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.qf_learning_rate,
            weight_decay=self.qf_weight_decay,
       )
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)
        if replay_buffer is None:
            self.replay_buffer = SplitReplayBuffer(
                EnvReplayBuffer(
                    self.replay_buffer_size,
                    self.env,
                    flatten=True,
                ),
                EnvReplayBuffer(
                    self.replay_buffer_size,
                    self.env,
                    flatten=True,
                ),
                fraction_paths_in_train=0.8,
            )
        else:
            self.replay_buffer = replay_buffer

    def cuda(self):
        self.policy.cuda()
        self.target_policy.cuda()
        self.qf.cuda()
        self.target_qf.cuda()

    def _do_training(self, n_steps_total):
        for i in range(self.number_of_gradient_steps):
            batch = self.get_batch()
            train_dict = self.get_train_dict(batch)

            self.policy_optimizer.zero_grad()
            policy_loss = train_dict['Policy Loss']
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf_optimizer.zero_grad()
            qf_loss = train_dict['QF Loss']
            qf_loss.backward()
            self.qf_optimizer.step()

            if self.use_soft_update:
                ptu.soft_update_from_to(self.target_policy, self.policy, self.tau)
                ptu.soft_update_from_to(self.target_qf, self.qf, self.tau)
            else:
                if n_steps_total % self.target_hard_update_period == 0:
                    ptu.copy_model_params_from_to(self.qf, self.target_qf)
                    ptu.copy_model_params_from_to(self.policy, self.target_policy)

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

        if self.differentiate_through_target:
            next_actions = self.target_policy(next_obs)
            target_q_values = self.qf(
                next_obs,
                next_actions,
            )
            y_target = rewards + (1. - terminals) * self.discount * target_q_values
            y_pred = self.qf(obs, actions)
            bellman_errors = (y_pred - y_target)**2
            # noinspection PyUnresolvedReferences
            qf_loss = bellman_errors.mean()
        else:
            next_actions = self.target_policy(next_obs)
            target_q_values = self.target_qf(
                next_obs,
                next_actions,
            )
            y_target = rewards + (1. - terminals) * self.discount * target_q_values
            y_target = y_target.detach()
            y_pred = self.qf(obs, actions)
            bellman_errors = (y_pred - y_target)**2
            qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('QF Loss', qf_loss),
        ])

    def training_mode(self, mode):
        self.policy.train(mode)
        self.qf.train(mode)
        self.target_policy.train(mode)
        self.target_qf.train(mode)

    def evaluate(self, epoch, exploration_paths):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        logger.log("Collecting samples for evaluation")
        statistics = OrderedDict()
        train_batch = self.get_batch(training=True)
        validation_batch = self.get_batch(training=False)
        test_paths = self._sample_eval_paths(epoch)

        if not isinstance(self.epoch_discount_schedule, ConstantSchedule):
            statistics['Discount Factor'] = self.discount

        statistics.update(get_generic_path_information(
            exploration_paths, self.discount, stat_prefix="Exploration",
        ))
        statistics.update(self._statistics_from_paths(exploration_paths,
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
                ['QF Loss Mean', 'Policy Loss Mean'],
            )
        )

        average_returns = get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns

        statistics['Epoch'] = epoch

        self.final_score = average_returns

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(test_paths)

    def get_batch(self, training=True):
        replay_buffer = self.replay_buffer.get_replay_buffer(training)
        sample_size = min(
            replay_buffer.num_steps_can_sample(),
            self.batch_size
        )
        batch = replay_buffer.random_batch(sample_size)
        return np_to_pytorch_batch(batch)

    def _statistics_from_paths(self, paths, stat_prefix):
        batch = self.paths_to_batch(paths)
        statistics = self._statistics_from_batch(batch, stat_prefix)
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics

    @staticmethod
    def paths_to_batch(paths):
        np_batch = split_paths_to_dict(paths)
        return np_to_pytorch_batch(np_batch)

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = get_statistics_from_pytorch_dict(
            self.get_train_dict(batch),
            ['QF Loss', 'Policy Loss'],
            ['Bellman Errors', 'QF Outputs', 'Policy Actions'],
            stat_prefix
        )
        statistics.update(create_stats_ordered_dict(
            "{} Env Actions".format(stat_prefix),
            ptu.get_numpy(batch['actions'])
        ))

        return statistics

    def _can_evaluate(self, exploration_paths):
        return (
            len(exploration_paths) > 0
            and self.replay_buffer.num_steps_can_sample() > 0
        )

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            policy=self.policy,
            env=self.training_env,
            exploration_policy=self.exploration_policy,
            qf=self.qf,
            batch_size=self.batch_size,
            algorithm=self,
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
