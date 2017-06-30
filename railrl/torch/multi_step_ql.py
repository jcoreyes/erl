from collections import OrderedDict

import torch
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.torch.ddpg import DDPG
from rllab.misc import logger
from railrl.misc import np_util


def flatten_subtraj_batch(subtraj_batch):
    return {
        k: array.view(-1, array.size()[-1])
        for k, array in subtraj_batch.items()
    }


class MultiStepDdpg(DDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtraj_length = 10
        self.gammas = self.discount * torch.ones(self.subtraj_length)
        discount_factors = torch.cumprod(self.gammas, dim=0)
        self.discount_factors = ptu.Variable(
            discount_factors.view(-1, 1),
            requires_grad=False,
        )
        self.pool = SplitReplayBuffer(
            SubtrajReplayBuffer(
                max_pool_size=self.pool_size,
                env=self.env,
                subtraj_length=self.subtraj_length,
            ),
            SubtrajReplayBuffer(
                max_pool_size=self.pool_size,
                env=self.env,
                subtraj_length=self.subtraj_length,
            ),
            fraction_paths_in_train=0.8,
        )

    def get_train_dict(self, subtraj_batch):
        subtraj_rewards = subtraj_batch['rewards']
        subtraj_rewards_np = ptu.get_numpy(subtraj_rewards).squeeze(2)
        returns = np_util.batch_discounted_cumsum(
            subtraj_rewards_np, self.discount
        )
        returns = np.expand_dims(returns, 2)
        returns = np.ascontiguousarray(returns).astype(np.float32)
        returns = ptu.Variable(ptu.from_numpy(returns))
        subtraj_batch['returns'] = returns
        batch = flatten_subtraj_batch(subtraj_batch)
        # rewards = batch['rewards']
        returns = batch['returns']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        q = self.qf(obs, policy_actions)
        policy_loss = - q.mean()

        """
        Critic operations.
        """
        next_actions = self.policy(next_obs)
        # TODO: try to get this to work
        # next_actions = None
        q_target = self.target_qf(
            next_obs,
            next_actions,
        )
        # y_target = rewards + (1. - terminals) * self.discount * v_target
        batch_size = q_target.size()[0]
        discount_factors = self.discount_factors.repeat(
            batch_size // self.subtraj_length, 1,
        )
        y_target = returns + (1. - terminals) * discount_factors * q_target
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions)
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('Policy Q Values', q),
            ('Target Y', y_target),
            ('Predicted Y', y_pred),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('QF Loss', qf_loss),
        ])

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
            'Target Y',
            'Predicted Y',
            'Policy Q Values',
        ]:
            tensor = train_dict[name]
            statistics.update(create_stats_ordered_dict(
                '{} {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
            ))

        return statistics

    def _statistics_from_paths(self, paths, stat_prefix):
        statistics = OrderedDict()
        eval_pool = SubtrajReplayBuffer(
            len(paths) * (self.max_path_length + 1),
            self.env,
            self.subtraj_length,
        )
        for path in paths:
            eval_pool.add_trajectory(path)
        subtraj_batch = eval_pool.get_all_valid_subtrajectories()
        torch_batch = {
            k: ptu.Variable(ptu.from_numpy(array).float(), requires_grad=False)
            for k, array in subtraj_batch.items()
        }
        torch_batch['rewards'] = torch_batch['rewards'].unsqueeze(-1)
        torch_batch['terminals'] = torch_batch['terminals'].unsqueeze(-1)
        statistics.update(self._statistics_from_batch(torch_batch,
                                                      stat_prefix))
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics

    def evaluate(self, epoch, exploration_paths):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        logger.log("Collecting samples for evaluation")
        paths = self._sample_paths(epoch)
        statistics = OrderedDict()

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
        self.final_score = average_returns
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(paths)
