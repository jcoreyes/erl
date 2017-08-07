from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import torch.functional as F
from torch import nn as nn
from torch.autograd import Variable

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.torch.core import PyTorchModule
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
        self.naf_policy_optimizer = optim.Adam(self.naf_policy.parameters(),
                                       lr=self.naf_policy_learning_rate)
    def cuda(self):
        self.naf_policy.cuda()
        self.target_naf_policy.cuda()

    def _do_training(self, n_steps_total):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Optimize Critic.
        """
        # Generate y target using target policies
        _, _, v_pred = self.target_naf_policy(next_obs, None)
        y_target = rewards + (1. - terminals) * self.discount * v_pred
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        _, y_pred, _ = self.naf_policy(obs, actions)
        naf_policy_loss = self.naf_policy_criterion(y_pred, y_target)

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
        test_paths = self._sample_paths(epoch)
        statistics = OrderedDict()

        statistics.update(self._statistics_from_paths(exploration_paths,
                                                      "Exploration"))
        statistics.update(self._statistics_from_paths(test_paths, "Test"))

        statistics['AverageReturn'] = get_average_returns(test_paths)
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(test_paths)


    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
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
        returns = [sum(path["rewards"]) for path in paths]

        discounted_returns = [
            special.discount_return(path["rewards"], self.discount)
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
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
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
    ):
        self.save_init_params(locals())
        super(NafPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.bn_state = nn.BatchNorm1d(obs_dim)
        self.bn_state.weight.data.fill_(1)
        self.bn_state.bias.data.fill_(0)

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, action_dim)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, action_dim ** 2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

        self.tril_mask = ptu.Variable(torch.tril(torch.ones(
            action_dim, action_dim), k=-1).unsqueeze(0))
        self.diag_mask = ptu.Variable(torch.diag(torch.diag(
            torch.ones(action_dim, action_dim))).unsqueeze(0))

    def forward(self, state, action):
        state = self.bn_state(state)
        x = state
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))

        V = self.V(x)
        mu = torch.tanh(self.mu(x))

        Q = None
        if action is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * \
                self.tril_mask.expand_as(
                    L) + torch.exp(L) * self.diag_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (action - mu).unsqueeze(2)
            A = -0.5 * \
                torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, _, _ = self.__call__(obs, None)
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}
