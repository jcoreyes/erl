from collections import OrderedDict

import numpy as np
import torch
# noinspection PyPep8Naming
import torch.optim as optim
from torch.autograd import Variable

from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.policies.torch import MemoryPolicy
from railrl.pythonplusplus import batch, ConditionTimer
from railrl.qfunctions.torch import MemoryQFunction
from railrl.torch.bptt_ddpg import create_torch_subtraj_batch
from railrl.torch.ddpg import DDPG
from railrl.torch.online_algorithm import OnlineAlgorithm
import railrl.torch.pytorch_util as ptu
from rllab.misc import logger, special


# noinspection PyCallingNonCallable
class Rdpg(DDPG):
    """
    Recurrent DPG.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert self.num_steps_per_eval >= self.env.horizon, (
            "Cannot evaluate RDPG with such short trajectories"
        )
        self.subtraj_length = self.env.horizon
        self.num_subtrajs_per_batch = self.batch_size // self.subtraj_length

        self.pool = SplitReplayBuffer(
            SubtrajReplayBuffer(
                self.pool_size,
                self.env,
                self.subtraj_length,
            ),
            SubtrajReplayBuffer(
                self.pool_size,
                self.env,
                self.subtraj_length,
            ),
            fraction_paths_in_train=0.8,
        )

    """
    Training functions
    """
    def get_train_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        critic, including intermediate values that might be useful to log.
        """
        rewards = subtraj_batch['rewards']
        terminals = subtraj_batch['terminals']
        obs = subtraj_batch['observations']
        actions = subtraj_batch['actions']
        next_obs = subtraj_batch['next_observations']

        policy_actions, _ = self.policy(obs)
        q_output = self.qf(obs, policy_actions)
        policy_loss = - q_output.mean()

        next_actions, _ = self.target_policy(next_obs)
        target_q_values = self.target_qf(next_obs, next_actions)
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_predicted = self.qf(obs, actions)
        bellman_errors = (y_predicted - y_target) ** 2

        return OrderedDict([
            ('Policy Loss', policy_loss),
            ('New Env Actions', policy_actions),
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('QF Loss', bellman_errors.mean()),
        ])

    """
    Eval functions
    """
    def _statistics_from_paths(self, paths, stat_prefix):
        eval_pool = SubtrajReplayBuffer(
            len(paths) * self.max_path_length,
            self.env,
            self.subtraj_length,
        )
        for path in paths:
            eval_pool.add_trajectory(path)
        raw_subtraj_batch = eval_pool.get_all_valid_subtrajectories()
        assert raw_subtraj_batch is not None
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)

        statistics = self._statistics_from_batch(
            subtraj_batch, stat_prefix=stat_prefix
        )
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics
