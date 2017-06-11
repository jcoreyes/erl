from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.torch.core import PyTorchModule
from railrl.torch.online_algorithm import OnlineAlgorithm
from railrl.torch.pytorch_util import soft_update_from_to, copy_model_params_from_to, fanin_init
from rllab.misc import logger, special


# noinspection PyCallingNonCallable
class DDPG(OnlineAlgorithm):
    """
    Online learning algorithm.
    """

    def __init__(
            self,
            *args,
            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            target_hard_update_period=1000,
            tau=0.001,
            use_soft_update=False,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf = QFunction(
            self.obs_dim,
            self.action_dim,
            400,
            300,
        )
        self.policy = FeedForwardPolicy(
            self.obs_dim,
            self.action_dim,
            400,
            300,
        )
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
        self.use_gpu = self.use_gpu and torch.cuda.is_available()

        if self.use_gpu:
            self.policy.cuda()
            self.target_policy.cuda()
            self.qf.cuda()
            self.target_qf.cuda()

    def _do_training(self, n_steps_total):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']


        """
        Optimize Policy.
        """
        policy_actions = self.policy(obs)
        q_output = self.qf(obs, policy_actions)
        policy_loss = - q_output.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Optimize Critic.

        Update the critic second since so that the policy uses the QF from
        this iteration.
        """
        # Generate y target using target policies
        next_actions = self.target_policy(next_obs)
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
        )
        actions = self.policy(next_obs)
        target_q_values_with_normal_policy = self.target_qf(
            next_obs,
            actions,
        )
       # y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = rewards  + (1. - terminals) * self.discount * target_q_values_with_normal_policy
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions)
        qf_loss = self.qf_criterion(y_pred, y_target)

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Update Target Networks
        """
        if self.use_soft_update:
            soft_update_from_to(self.target_policy, self.policy, self.tau)
            soft_update_from_to(self.target_qf, self.qf, self.tau)
        else:
            if n_steps_total % self.target_hard_update_period == 0:
                copy_model_params_from_to(self.qf, self.target_qf)
                copy_model_params_from_to(self.policy, self.target_policy)

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

        statistics.update(self._statistics_from_paths(exploration_paths,
                                                      "Exploration"))
        statistics.update(self._statistics_from_paths(paths, "Test"))

        statistics['AverageReturn'] = get_average_returns(paths)
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(paths)

    def get_batch(self):
        batch = self.pool.random_batch(self.batch_size, flatten=True)
        if self.use_gpu:
            torch_batch = {
                k: Variable(torch.from_numpy(array).float(),
                            requires_grad=False).cuda()
                for k, array in batch.items()
            }
        else:
            torch_batch = {
                k: Variable(torch.from_numpy(array).float(),
                            requires_grad=False)
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
        return statistics


class QFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_size,
            embedded_hidden_size,
            init_w=3e-3,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size

        self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(observation_hidden_size + action_dim,
                                     embedded_hidden_size)
        self.last_fc = nn.Linear(embedded_hidden_size, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.obs_fc.weight.data = fanin_init(self.obs_fc.weight.data.size())
        self.obs_fc.bias.data *= 0
        self.embedded_fc.weight.data = fanin_init(
            self.embedded_fc.weight.data.size()
        )
        self.embedded_fc.bias.data *= 0
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return self.last_fc(h)


class FeedForwardPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            fc1_size,
            fc2_size,
            init_w=1e-3,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(obs_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc1.bias.data *= 0
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc2.bias.data *= 0
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        if self.last_fc.weight.is_cuda:
            obs = Variable(torch.from_numpy(obs).float(),
                           requires_grad=False).cuda()
        else:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
        action = self.__call__(obs)
        action = action.squeeze(0)
        if self.last_fc.weight.is_cuda:
            return action.data.cpu().numpy(), {}
        else:
            return action.data.numpy(), {}

    def reset(self):
        pass

    def log_diagnostics(self, paths):
        pass
