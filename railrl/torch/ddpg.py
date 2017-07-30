from collections import OrderedDict

import numpy as np
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.ml_util import ConstantSchedule
from railrl.misc.rllab_util import get_average_returns, split_paths
from railrl.torch.online_algorithm import OnlineAlgorithm
import railrl.torch.pytorch_util as ptu
from rllab.misc import logger, special


class DDPG(OnlineAlgorithm):
    """
    Online learning algorithm.
    """

    def __init__(
            self,
            env,
            qf,
            policy,
            exploration_strategy,
            exploration_policy=None,
            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            replay_buffer=None,
            **kwargs
    ):
        if exploration_policy is None:
            exploration_policy = policy
        super().__init__(
            env,
            exploration_policy,
            exploration_strategy,
            **kwargs
        )
        self.qf = qf
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.target_qf = self.qf.copy()
        self.target_policy = self.policy.copy()
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update

        self.qf_criterion = nn.MSELoss()
        self.qf_optimizer = optim.Adam(self.qf.parameters(),
                                       lr=self.qf_learning_rate)
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
        next_actions = self.target_policy(next_obs)
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
        )
        y_target = rewards + (1. - terminals) * self.discount * target_q_values

        # noinspection PyUnresolvedReferences
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
        paths = self._sample_paths(epoch)

        statistics = OrderedDict()
        if not isinstance(self.epoch_discount_schedule, ConstantSchedule):
            statistics['Discount Factor'] = self.discount

        statistics.update(self._statistics_from_paths(exploration_paths,
                                                      "Exploration"))
        statistics.update(self._statistics_from_paths(paths, "Test"))

        train_batch = self.get_batch(training=True)
        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        validation_batch = self.get_batch(training=False)
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
        )

        statistics['QF Loss Validation - Train Gap'] = (
            statistics['Validation QF Loss Mean']
            - statistics['Train QF Loss Mean']
        )
        statistics['Policy Loss Validation - Train Gap'] = (
            statistics['Validation Policy Loss Mean']
            - statistics['Train Policy Loss Mean']
        )
        average_returns = get_average_returns(paths)
        statistics['AverageReturn'] = average_returns

        statistics['Epoch'] = epoch

        self.final_score = average_returns

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(paths)

    def get_batch(self, training=True):
        replay_buffer = self.replay_buffer.get_replay_buffer(training)
        sample_size = min(
            replay_buffer.num_steps_can_sample(),
            self.batch_size
        )
        batch = replay_buffer.random_batch(sample_size)
        return np_to_pytorch_batch(batch)

    def _statistics_from_paths(self, paths, stat_prefix):
        np_batch = self._paths_to_np_batch(paths)
        batch = np_to_pytorch_batch(np_batch)
        statistics = self._statistics_from_batch(batch, stat_prefix)
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics

    def _paths_to_np_batch(self, paths):
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        return dict(
            rewards=rewards,
            terminals=terminals,
            observations=obs,
            actions=actions,
            next_observations=next_obs,
        )

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = OrderedDict()

        train_dict = self.get_train_dict(batch)
        for name in [
            'QF Loss',
            'Policy Loss',
        ]:
            tensor = train_dict[name]
            statistics_name = "{} {} Mean".format(stat_prefix, name)
            statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

        for name in [
            'Bellman Errors',
        ]:
            tensor = train_dict[name]
            statistics.update(create_stats_ordered_dict(
                '{} {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
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
            es=self.exploration_strategy,
            qf=self.qf,
            replay_buffer=self.replay_buffer,
            algorithm=self,
            batch_size=self.batch_size,
        )


def np_to_pytorch_batch(np_batch):
    torch_batch = {
        k: Variable(ptu.from_numpy(elem).float(), requires_grad=False)
        for k, elem in np_batch.items()
    }
    torch_batch['rewards'] = torch_batch['rewards'].unsqueeze(-1)
    torch_batch['terminals'] = torch_batch['terminals'].unsqueeze(-1)
    return torch_batch


def get_generic_path_information(paths, discount, stat_prefix):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    discounted_returns = [
        special.discount_return(path["rewards"], discount)
        for path in paths
    ]
    rewards = np.hstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                discounted_returns,
                                                stat_prefix=stat_prefix))
    actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
