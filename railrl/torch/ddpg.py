from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.torch.online_algorithm import OnlineAlgorithm
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
            [400],
            [300],
        )
        self.policy = Policy(
            self.obs_dim,
            self.action_dim,
            [400, 300],
        )
        self.target_qf = self.qf.clone()
        self.target_policy = self.policy.clone()
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update

        self.qf_criterion = nn.MSELoss()
        self.qf_optimizer = optim.Adam(self.qf.parameters(),
                                       lr=self.qf_learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)

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
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
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
            soft_update(self.target_policy, self.policy, self.tau)
            soft_update(self.target_qf, self.qf, self.tau)
        else:
            if n_steps_total % self.target_hard_update_period == 0:
                copy_model_params(self.qf, self.target_qf)
                copy_model_params(self.policy, self.target_policy)

    def training_mode(self, mode):
        self.policy.train(mode)
        self.qf.train(mode)
        self.target_policy.train(mode)
        self.target_qf.train(mode)

    def evaluate(self, epoch, es_path_returns):
        """
        Perform evaluation for this algorithm.

        It's recommended
        :param epoch: The epoch number.
        :param es_path_returns: List of path returns from explorations strategy
        :return: Dictionary of statistics.
        """
        logger.log("Collecting samples for evaluation")
        paths = self._sample_paths(epoch)
        statistics = OrderedDict()

        statistics.update(self._get_other_statistics())
        statistics.update(self._statistics_from_paths(paths))

        returns = [sum(path["rewards"]) for path in paths]

        discounted_returns = [
            special.discount_return(path["rewards"], self.discount)
            for path in paths
        ]
        rewards = np.hstack([path["rewards"] for path in paths])
        statistics.update(create_stats_ordered_dict('Rewards', rewards))
        statistics.update(create_stats_ordered_dict('Returns', returns))
        statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                    discounted_returns))
        if len(es_path_returns) > 0:
            statistics.update(create_stats_ordered_dict('TrainingReturns',
                                                        es_path_returns))

        average_returns = np.mean(returns)
        statistics['AverageReturn'] = average_returns
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(paths)

    def get_batch(self):
        batch = self.pool.random_batch(self.batch_size, flatten=True)
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


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class QFunction(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_sizes,
            embedded_hidden_sizes,
            init_w=3e-3,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_sizes = observation_hidden_sizes
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.obs_bn = nn.BatchNorm1d(obs_dim)
        self.obs_fc_bn = nn.BatchNorm1d(observation_hidden_sizes[0])

        input_dim = obs_dim
        self.obs_fcs = []
        self.fc_batchnorms = []
        last_size = input_dim
        for size in self.observation_hidden_sizes:
            self.fc_batchnorms.append(nn.BatchNorm1d(size))
            self.obs_fcs.append(nn.Linear(last_size, size))
            last_size = size

        self.embedded_fcs = []
        last_size = last_size + action_dim
        for size in self.embedded_hidden_sizes:
            self.embedded_fcs.append(nn.Linear(last_size, size))
            last_size = size
        self.last_fc = nn.Linear(last_size, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        for fc in self.obs_fcs:
            fc.weight.data = fanin_init(fc.weight.data.size())
        for fc in self.embedded_fcs:
            fc.weight.data = fanin_init(fc.weight.data.size())
        self.last_fc.weight.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        obs = self.obs_bn(obs)
        h = obs
        for fc, bn in zip(self.obs_fcs, self.fc_batchnorms):
            h = F.relu(
                bn(fc(h))
            )

        h = torch.cat((h, action), dim=1)
        for fc in self.embedded_fcs:
            h = F.relu(fc(h))
        return self.last_fc(h)

    def clone(self):
        copy = QFunction(
            self.obs_dim,
            self.action_dim,
            self.observation_hidden_sizes,
            self.embedded_hidden_sizes,
        )
        copy_model_params(self, copy)
        return copy


class Policy(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            init_w=1e-3,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.obs_bn = nn.BatchNorm1d(self.obs_dim)

        self.fcs = []
        self.fc_batchnorms = []
        last_size = obs_dim
        for size in hidden_sizes:
            self.fc_batchnorms.append(nn.BatchNorm1d(size))
            self.fcs.append(nn.Linear(last_size, size))
            last_size = size

        self.last_fc = nn.Linear(last_size, action_dim)
        self.last_batchnorm = nn.BatchNorm1d(action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        for fc in self.fcs:
            fc.weight.data = fanin_init(fc.weight.data.size())
        self.last_fc.weight.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        obs = self.obs_bn(obs)
        last_layer = obs
        for fc, bn in zip(self.fcs, self.fc_batchnorms):
            last_layer = F.relu(
                bn(fc(last_layer))
            )
        return F.tanh(
            self.last_batchnorm(
                self.last_fc(last_layer)
            )
        )

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
        action = self.__call__(obs)
        action = action.squeeze(0)
        return action.data.numpy(), {}

    def get_param_values(self):
        return [param.data for param in self.parameters()]

    def set_param_values(self, param_values):
        for param, value in zip(self.parameters(), param_values):
            param.data = value

    def reset(self):
        pass

    def clone(self):
        copy = Policy(
            self.obs_dim,
            self.action_dim,
            self.hidden_sizes,
        )
        copy_model_params(self, copy)
        return copy
