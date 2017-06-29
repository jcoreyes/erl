from collections import OrderedDict

import torch
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.data_management.updatable_subtraj_replay_buffer import \
    UpdatableSubtrajReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.torch.ddpg import DDPG
from rllab.misc import logger


def flatten_subtraj_batch(subtraj_batch):
    return {
        k: array.view(-1, array.size()[-1])
        for k, array in subtraj_batch.items()
    }


class MultiStepDdpg(DDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtraj_length = 10
        self.gammas = self.discount * torch.ones(
            self.batch_size, self.subtraj_length, 1
        )
        self.discount_factors = torch.cumprod(self.gammas, dim=1)
        import ipdb; ipdb.set_trace()
        self.pool = UpdatableSubtrajReplayBuffer(
            self.pool_size,
            self.env,
            self.subtraj_length,
            1,
        )

    def get_train_dict(self, subtraj_batch):
        subtraj_rewards = subtraj_batch['rewards']
        batch = flatten_subtraj_batch(subtraj_batch)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        v, a = self.qf(obs, policy_actions)
        policy_loss = - a.mean()

        """
        Critic operations.
        """
        next_actions = self.policy(next_obs)
        # TODO: try to get this to work
        # next_actions = None
        v_target, a_target = self.target_qf(
            next_obs,
            next_actions,
        )
        y_target = rewards + (1. - terminals) * self.discount * v_target
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        v_pred, a_pred = self.qf(obs, actions)
        y_pred = v_pred + a_pred
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('Policy Action Value', v),
            ('Policy Action Advantage', a),
            ('Target Value', v_target),
            ('Target Advantage', a_target),
            ('Predicted Value', v_pred),
            ('Predicted Advantage', a_pred),
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
            'Target Value',
            'Target Advantage',
            'Predicted Value',
            'Predicted Advantage',
            'Policy Action Value',
            'Policy Action Advantage',
        ]:
            tensor = train_dict[name]
            statistics.update(create_stats_ordered_dict(
                '{} {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
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
