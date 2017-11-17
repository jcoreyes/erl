from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch import nn as nn
from torch.autograd import Variable

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns, split_paths_to_dict
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.algos.eval import get_statistics_from_pytorch_dict, \
    get_difference_statistics
from railrl.torch.core import PyTorchModule
from railrl.torch.online_algorithm import RLAlgorithm
from railrl.torch import pytorch_util as ptu
from rllab.misc import logger


# noinspection PyCallingNonCallable
class NAF(RLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            exploration_policy=None,
            policy_learning_rate=1e-3,
            target_hard_update_period=1000,
            tau=0.001,
            use_soft_update=False,
            replay_buffer=None,
            **kwargs
    ):
        if exploration_policy is None:
            exploration_policy = policy
        super().__init__(
            env,
            exploration_policy,
            **kwargs
        )
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.target_policy = self.policy.copy()
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update

        self.policy_criterion = nn.MSELoss()
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )

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

    def _do_training(self, n_steps_total):
        batch = self.get_batch()

        """
        Optimize Critic.
        """
        train_dict = self.get_train_dict(batch)
        policy_loss = train_dict['Policy Loss']

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update Target Networks
        """
        if self.use_soft_update:
            ptu.soft_update_from_to(self.target_policy, self.policy, self.tau)
        else:
            if n_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.policy, self.target_policy)

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        _, _, v_pred = self.target_policy(next_obs, None)
        y_target = rewards + (1. - terminals) * self.discount * v_pred
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        mu, y_pred, v = self.policy(obs, actions)
        policy_loss = self.policy_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy v', v),
            ('Policy mu', mu),
            ('Policy Loss', policy_loss),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
        ])

    def training_mode(self, mode):
        self.policy.train(mode)
        self.target_policy.train(mode)

    def evaluate(self, epoch, exploration_paths):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        logger.log("Collecting samples for evaluation")
        train_batch = self.get_batch(training=True)
        validation_batch = self.get_batch(training=False)
        test_paths = self._sample_eval_paths(epoch)

        statistics = OrderedDict()
        statistics.update(
            self._statistics_from_paths(exploration_paths, "Exploration")
        )
        statistics.update(self._statistics_from_paths(test_paths, "Test"))
        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
        )
        statistics.update(
            get_difference_statistics(statistics, ['Policy Loss Mean'])
        )
        statistics['AverageReturn'] = get_average_returns(test_paths)
        statistics['Epoch'] = epoch

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
        np_batch = split_paths_to_dict(paths)
        batch = np_to_pytorch_batch(np_batch)
        statistics = self._statistics_from_batch(batch, stat_prefix)
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = get_statistics_from_pytorch_dict(
            self.get_train_dict(batch),
            ['Policy Loss'],
            ['Policy v', 'Policy mu', 'Y targets', 'Y predictions'],
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
            env=self.training_env,
            policy=self.policy,
            replay_buffer=self.replay_buffer,
            algorithm=self,
        )


class NafPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_size,
            use_batchnorm=False,
            b_init_value=0.01,
            hidden_init=ptu.fanin_init,
            use_exp_for_diagonal_not_square=True,
    ):
        self.save_init_params(locals())
        super(NafPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_batchnorm = use_batchnorm
        self.use_exp_for_diagonal_not_square = use_exp_for_diagonal_not_square

        if use_batchnorm:
            self.bn_state = nn.BatchNorm1d(obs_dim)
            self.bn_state.weight.data.fill_(1)
            self.bn_state.bias.data.fill_(0)

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.L = nn.Linear(hidden_size, action_dim ** 2)

        self.tril_mask = ptu.Variable(
            torch.tril(
                torch.ones(action_dim, action_dim),
                -1
            ).unsqueeze(0)
        )
        self.diag_mask = ptu.Variable(torch.diag(
            torch.diag(
                torch.ones(action_dim, action_dim)
            )
        ).unsqueeze(0))

        hidden_init(self.linear1.weight)
        self.linear1.bias.data.fill_(b_init_value)
        hidden_init(self.linear2.weight)
        self.linear2.bias.data.fill_(b_init_value)
        hidden_init(self.V.weight)
        self.V.bias.data.fill_(b_init_value)
        hidden_init(self.L.weight)
        self.L.bias.data.fill_(b_init_value)
        hidden_init(self.mu.weight)
        self.mu.bias.data.fill_(b_init_value)

    def forward(self, state, action, return_P=False):
        if self.use_batchnorm:
            state = self.bn_state(state)
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        V = self.V(x)
        mu = torch.tanh(self.mu(x))

        Q = None
        P = None
        if action is not None or return_P:
            num_outputs = mu.size(1)
            raw_L = self.L(x).view(-1, num_outputs, num_outputs)
            L = raw_L * self.tril_mask.expand_as(raw_L)
            if self.use_exp_for_diagonal_not_square:
                L += torch.exp(raw_L) * self.diag_mask.expand_as(raw_L)
            else:
                L += torch.pow(raw_L, 2) * self.diag_mask.expand_as(raw_L)
            P = torch.bmm(L, L.transpose(2, 1))

            if action is not None:
                u_mu = (action - mu).unsqueeze(2)
                A = - 0.5 * torch.bmm(
                    torch.bmm(u_mu.transpose(2, 1), P), u_mu
                ).squeeze(2)

                Q = A + V

        if return_P:
            return mu, Q, V, P
        return mu, Q, V

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, _, _ = self.__call__(obs, None)
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}

    def get_action_and_P_matrix(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, _, _, P = self.__call__(obs, None, return_P=True)
        action = action.squeeze(0)
        P = P.squeeze(0)
        return ptu.get_numpy(action), ptu.get_numpy(P)
