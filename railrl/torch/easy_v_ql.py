from collections import OrderedDict

import numpy as np
import torch.optim as optim
import torch
from torch import nn as nn
from torch.autograd import Variable

from railrl.torch.core import PyTorchModule
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.torch.online_algorithm import OnlineAlgorithm
import railrl.torch.pytorch_util as ptu
import railrl.torch.modules as M
from rllab.misc import logger, special


class EasyVQLearning(OnlineAlgorithm):
    """
    Continous action Q learning where the V function is easy:

    Q(s, a) = A(s, a) + V(s)

    The main thing is that the following needs to be enforced:

        max_a A(s, a) = 0

    """
    def __init__(
            self,
            *args,
            qf,
            policy,
            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.qf = qf
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate

        self.qf_criterion = nn.MSELoss()
        self.qf_optimizer = optim.Adam(self.qf.parameters(),
                                       lr=self.qf_learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)
        if ptu.gpu_enabled():
            self.policy.cuda()
            self.qf.cuda()

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
        next_actions = self.policy(next_obs)
        next_v_values = self.qf(next_obs, next_actions)
        y_target = rewards + (1. - terminals) * self.discount * next_v_values
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions)
        qf_loss = self.qf_criterion(y_pred, y_target)

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

    def training_mode(self, mode):
        self.policy.train(mode)
        self.qf.train(mode)

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
            policy=self.policy,
            env=self.training_env,
            qf=self.qf,
        )


class EasyVQFunction(PyTorchModule):
    """
    Parameterize Q function as the follows:

        Q(s, a) = A(s, a) + V(s)

    To ensure that max_a A(s, a) = 0, use the following form:

        A(s, a) = - diff(s, a)^T diag(exp(d(s))) diff(s, a)  *  f(s, a)^2

    where

        diff(s, a) = a - z(s)

    so that a = z(s) is at least one zero.

    d(s) and f(s, a) are arbitrary functions
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            diag_fc1_size,
            diag_fc2_size,
            af_fc1_size,
            af_fc2_size,
            zero_fc1_size,
            zero_fc2_size,
            vf_fc1_size,
            vf_fc2_size,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs_batchnorm = nn.BatchNorm1d(obs_dim)

        self.batch_square = M.BatchSquareDiagonal(action_dim)

        self.diag = nn.Sequential(
            nn.Linear(obs_dim, diag_fc1_size),
            nn.ReLU(),
            nn.Linear(diag_fc1_size, diag_fc2_size),
            nn.ReLU(),
            nn.Linear(diag_fc2_size, action_dim),
        )

        self.zero = nn.Sequential(
            nn.Linear(obs_dim, zero_fc1_size),
            nn.ReLU(),
            nn.Linear(zero_fc1_size, zero_fc2_size),
            nn.ReLU(),
            nn.Linear(zero_fc2_size, action_dim),
        )

        self.root = nn.Sequential(
            nn.Linear(obs_dim, zero_fc1_size),
            nn.ReLU(),
            nn.Linear(zero_fc1_size, zero_fc2_size),
            nn.ReLU(),
            nn.Linear(zero_fc2_size, action_dim),
        )

        self.f = nn.Sequential(
            M.Concat(),
            nn.Linear(obs_dim + action_dim, af_fc1_size),
            nn.ReLU(),
            nn.Linear(af_fc1_size, af_fc2_size),
            nn.ReLU(),
            nn.Linear(af_fc2_size, 1),
        )

        self.vf = nn.Sequential(
            nn.Linear(obs_dim, vf_fc1_size),
            nn.ReLU(),
            nn.Linear(vf_fc1_size, vf_fc2_size),
            nn.ReLU(),
            nn.Linear(vf_fc2_size, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.obs_batchnorm.reset_parameters()

    def forward(self, obs, action):
        obs = self.obs_batchnorm(obs)
        V = self.vf(obs)
        if action is None:
            return V

        diag_values = torch.exp(self.diag(obs))
        diff = action - self.zero(obs)
        quadratic = self.batch_square(diff, diag_values)
        f = self.f((obs, action))
        # AF = f
        AF = - quadratic * (f**2)

        return V + AF
