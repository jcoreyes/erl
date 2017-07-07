from collections import OrderedDict

import random
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns, split_paths
from railrl.pythonplusplus import identity
from railrl.torch.core import PyTorchModule
from railrl.torch.online_algorithm import OnlineAlgorithm
import railrl.torch.pytorch_util as ptu
from rllab.misc import logger, special


class StateDistanceQLearning(OnlineAlgorithm):
    """
    Online learning algorithm.
    """
    def __init__(
            self,
            *args,
            qf,
            qf_learning_rate=1e-3,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.qf = qf
        self.policy = ArgmaxPolicy(self.qf, 5)
        self.qf_learning_rate = qf_learning_rate
        self.target_qf = self.qf.copy()
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update

        self.qf_criterion = nn.MSELoss()
        self.qf_optimizer = optim.Adam(self.qf.parameters(),
                                       lr=self.qf_learning_rate)
        self.pool = SplitReplayBuffer(
            EnvReplayBuffer(
                self.pool_size,
                self.env,
                flatten=True,
            ),
            EnvReplayBuffer(
                self.pool_size,
                self.env,
                flatten=True,
            ),
            fraction_paths_in_train=0.8,
        )
        if ptu.gpu_enabled():
            self.policy.cuda()
            self.qf.cuda()
            self.target_qf.cuda()

    def _do_training(self, n_steps_total):
        batch = self.get_batch()
        train_dict = self.get_train_dict(batch)

        self.qf_optimizer.zero_grad()
        qf_loss = train_dict['QF Loss']
        qf_loss.backward()
        self.qf_optimizer.step()

        if self.use_soft_update:
            ptu.soft_update_from_to(self.target_qf, self.qf, self.tau)
        else:
            if n_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Critic operations.
        """
        target_q_values = torch.max(self.target_qf(next_obs), dim=1)[0]
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs).gather(1, actions)
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('QF Loss', qf_loss),
        ])

    def training_mode(self, mode):
        self.policy.train(mode)
        self.qf.train(mode)
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

        statistics.update(self._statistics_from_paths(exploration_paths,
                                                      "Exploration"))
        statistics.update(self._statistics_from_paths(paths, "Test"))

        train_batch = self.get_batch(training=True)
        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        validation_batch = self.get_batch(training=False)
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
        )

        statistics['AverageReturn'] = get_average_returns(paths)
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(paths)

    def get_batch(self, training=True):
        pool = self.pool.get_replay_buffer(training)
        sample_size = min(
            pool.num_steps_can_sample(),
            self.batch_size
        )
        batch = pool.random_batch(sample_size, flatten=True)
        torch_batch = {
            k: Variable(ptu.from_numpy(array).float(), requires_grad=False)
            for k, array in batch.items()
        }
        rewards = torch_batch['rewards']
        terminals = torch_batch['terminals']
        torch_batch['rewards'] = rewards.unsqueeze(-1)
        torch_batch['terminals'] = terminals.unsqueeze(-1)
        return torch_batch

    def _statistics_from_paths(self, paths, stat_prefix):
        statistics = OrderedDict()
        batch = ptu.paths_to_pytorch_batch(paths)
        statistics.update(self._statistics_from_batch(batch, stat_prefix))
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics

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
            and self.pool.num_steps_can_sample() > 0
        )

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            policy=self.policy,
            env=self.training_env,
            qf=self.qf,
        )


class ArgmaxPolicy(PyTorchModule):
    def __init__(
            self,
            qf,
            num_actions,
            prob_random_action=0.1,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.qf = qf
        self.num_actions = num_actions
        self.prob_random_action = prob_random_action

    def get_action(self, obs):
        if random.random() <= self.prob_random_action:
            return random.randint(0, self.num_actions)
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        qvalues = self.qf(obs)
        action = torch.max(qvalues)[1]
        return action, {}


class DQN(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            fc1_size,
            fc2_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_activation = output_activation
        self.hidden_init = hidden_init

        self.fc1 = nn.Linear(obs_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        self.hidden_init(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return self.output_activation(self.last_fc(h))
