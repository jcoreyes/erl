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
from railrl.torch.core import PyTorchModule
from railrl.torch.ddpg import np_to_pytorch_batch
from railrl.torch.online_algorithm import OnlineAlgorithm
from railrl.torch import pytorch_util as ptu
from rllab.misc import logger, special


# noinspection PyCallingNonCallable
class NAF(OnlineAlgorithm):
    def __init__(
            self,
            env,
            naf_policy,
            exploration_strategy,
            exploration_policy=None,
            naf_policy_learning_rate=1e-3,
            target_hard_update_period=1000,
            tau=0.001,
            use_soft_update=False,
            replay_buffer=None,
            **kwargs
    ):
        if exploration_policy is None:
            exploration_policy = naf_policy
        super().__init__(
            env,
            exploration_policy,
            exploration_strategy,
            **kwargs
        )
        self.naf_policy = naf_policy
        self.naf_policy_learning_rate = naf_policy_learning_rate
        self.target_naf_policy = self.naf_policy.copy()
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update

        self.naf_policy_criterion = nn.MSELoss()
        self.naf_policy_optimizer = optim.Adam(
            self.naf_policy.parameters(),
            lr=self.naf_policy_learning_rate,
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
        self.naf_policy.cuda()
        self.target_naf_policy.cuda()

    def _do_training(self, n_steps_total):
        batch = self.get_batch()

        """
        Optimize Critic.
        """
        train_dict = self.get_train_dict(batch)
        naf_policy_loss = train_dict['NAF Policy Loss']

        self.naf_policy_optimizer.zero_grad()
        naf_policy_loss.backward()
        self.naf_policy_optimizer.step()

        """
        Update Target Networks
        """
        if self.use_soft_update:
            ptu.soft_update_from_to(self.target_naf_policy, self.naf_policy, self.tau)
        else:
            if n_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.naf_policy, self.target_naf_policy)

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        _, _, v_pred = self.target_naf_policy(next_obs, None)
        y_target = rewards + (1. - terminals) * self.discount * v_pred
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        mu, y_pred, v = self.naf_policy(obs, actions)
        naf_policy_loss = self.naf_policy_criterion(y_pred, y_target)

        return OrderedDict([
            ('NAF Policy v', v),
            ('NAF Policy mu', mu),
            ('NAF Policy Loss', naf_policy_loss),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
        ])

    def training_mode(self, mode):
        self.naf_policy.train(mode)
        self.target_naf_policy.train(mode)

    def evaluate(self, epoch, exploration_paths):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        logger.log("Collecting samples for evaluation")
        test_paths = self._sample_eval_paths(epoch)
        train_batch = self.get_batch(training=True)
        validation_batch = self.get_batch(training=False)

        statistics = OrderedDict()
        statistics.update(
            self._statistics_from_paths(exploration_paths, "Exploration")
        )
        statistics.update(self._statistics_from_paths(test_paths, "Test"))
        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
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
        statistics = OrderedDict()

        train_dict = self.get_train_dict(batch)
        for name in [
            'NAF Policy Loss',
        ]:
            tensor = train_dict[name]
            statistics_name = "{} {} Mean".format(stat_prefix, name)
            statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

        for name in [
            'NAF Policy v',
            'NAF Policy mu',
            'Y targets',
            'Y predictions',
        ]:
            tensor = train_dict[name]
            statistics.update(create_stats_ordered_dict(
                '{} {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
            ))

        statistics.update(create_stats_ordered_dict(
            "{} Env Actions".format(stat_prefix),
            ptu.get_numpy(batch['actions'])
        ))

        return statistics

    def _can_evaluate(self, exploration_paths):
        return len(exploration_paths) > 0

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            env=self.training_env,
            naf_policy=self.naf_policy,
            replay_buffer=self.replay_buffer,
            algorithm=self,
        )

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

    statistics.update(create_stats_ordered_dict('Rewards', rewards, stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns, stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('DiscountedReturns', discounted_returns, stat_prefix=stat_prefix))
    actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict('Actions', actions, stat_prefix=stat_prefix))

    return statistics


# class NormalizedAdvantageFunction(PyTorchModule):
#     def __init__(
#             self,
#             obs_dim,
#             action_dim,
#     ):
#         self.save_init_params(locals())
#         super().__init__()
#
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.observation_hidden_size = observation_hidden_size
#         self.embedded_hidden_size = embedded_hidden_size
#
#         self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)
#         self.embedded_fc = nn.Linear(observation_hidden_size + action_dim,
#                                      embedded_hidden_size)
#         self.last_fc = nn.Linear(embedded_hidden_size, 1)
#         self.output_activation = output_activation
#
#         self.init_weights(init_w)
#
#     def init_weights(self, init_w):
#         init.kaiming_normal(self.obs_fc.weight)
#         self.obs_fc.bias.data.fill_(0)
#         init.kaiming_normal(self.embedded_fc.weight)
#         self.embedded_fc.bias.data.fill_(0)
#         self.last_fc.weight.data.uniform_(-init_w, init_w)
#         self.last_fc.bias.data.uniform_(-init_w, init_w)
#
#     def forward(self, obs, action):
#         h = obs
#         h = F.relu(self.obs_fc(h))
#         h = torch.cat((h, action), dim=1)
#         h = F.relu(self.embedded_fc(h))
#         return self.output_activation(self.last_fc(h))
class NafPolicy(PyTorchModule):

    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_size,
            use_batchnorm=False,
            b_init_value=0.01,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super(NafPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_batchnorm = use_batchnorm

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

    def forward(self, state, action):
        if self.use_batchnorm:
            state = self.bn_state(state)
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        V = self.V(x)
        mu = torch.tanh(self.mu(x))

        Q = None
        if action is not None:
            num_outputs = mu.size(1)
            raw_L = self.L(x).view(-1, num_outputs, num_outputs)
            L = (
                raw_L * self.tril_mask.expand_as(raw_L)
                + torch.exp(raw_L) * self.diag_mask.expand_as(raw_L)
            )
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (action - mu).unsqueeze(2)
            A = - 0.5 * torch.bmm(
                torch.bmm(u_mu.transpose(2, 1), P), u_mu
            ).squeeze(2)

            Q = A + V

        return mu, Q, V

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, _, _ = self.__call__(obs, None)
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}
