import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import railrl.torch.pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.pythonplusplus import identity
from railrl.torch.core import PyTorchModule
from railrl.torch.ddpg import DDPG, np_to_pytorch_batch
from rllab.misc import logger


class StateDistanceQLearning(DDPG):
    def __init__(
            self,
            *args,
            pool=None,
            num_batches=100,
            num_batches_per_epoch=100,
            sample_goals_from='environment',
            **kwargs
    ):
        super().__init__(*args, exploration_strategy=None, **kwargs)
        self.num_batches = num_batches
        self.num_batches_per_epoch = num_batches_per_epoch
        assert sample_goals_from in ['environment', 'replay_buffer']
        self.sample_goals_from = sample_goals_from
        self.pool = pool

    def train(self):
        epoch = 0
        for n_steps_total in range(self.num_batches):
            self.training_mode(True)
            self._do_training(n_steps_total=n_steps_total)
            if n_steps_total % self.num_batches_per_epoch == 0:
                logger.push_prefix('Iteration #%d | ' % epoch)
                self.training_mode(False)
                self.evaluate(epoch, None)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                logger.log("Done evaluating")
                logger.pop_prefix()
                epoch += 1

    def get_batch(self, training=True):
        pool = self.pool.get_replay_buffer(training)
        batch_size = min(
            pool.num_steps_can_sample(),
            self.batch_size
        )
        batch = pool.random_batch(batch_size)
        goal_states = self.sample_goal_states(batch_size)
        new_rewards = self.env.compute_rewards(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            goal_states,
        )
        batch['observations'] = np.hstack((batch['observations'], goal_states))
        batch['next_observations'] = np.hstack((
            batch['next_observations'], goal_states
        ))
        batch['rewards'] = new_rewards
        torch_batch = np_to_pytorch_batch(batch)
        return torch_batch

    def sample_goal_states(self, batch_size):
        if self.sample_goals_from == 'environment':
            return self.env.sample_goal_states(batch_size)
        elif self.sample_goals_from == 'replay_buffer':
            pool = self.pool.get_replay_buffer(training=True)
            batch = pool.random_batch(batch_size)
            return batch['observations']

    def reset_env(self):
        self.exploration_strategy.reset()
        self.exploration_policy.reset()
        self.policy.reset()
        return self.training_env.reset()

    def _paths_to_np_batch(self, paths):
        batch = super()._paths_to_np_batch(paths)
        batch_size = len(batch['observations'])
        goal_states = self.sample_goal_states(batch_size)
        new_rewards = self.env.compute_rewards(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            goal_states,
        )
        batch['observations'] = np.hstack((batch['observations'], goal_states))
        batch['next_observations'] = np.hstack((
            batch['next_observations'], goal_states
        ))
        batch['rewards'] = new_rewards
        return batch

    def evaluate(self, epoch, _):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        statistics = OrderedDict()
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
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)


class StateDistanceQLearningSimple(StateDistanceQLearning):
    def _do_training(self, n_steps_total):
        batch = self.get_batch()
        train_dict = self.get_train_dict(batch)

        self.qf_optimizer.zero_grad()
        qf_loss = train_dict['QF Loss']
        qf_loss.backward()
        self.qf_optimizer.step()

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        obs = batch['observations']
        actions = batch['actions']

        y_pred = self.qf(obs, actions)
        bellman_errors = (y_pred - rewards)**2
        qf_loss = bellman_errors.mean()

        return OrderedDict([
            ('Bellman Errors', bellman_errors),
            ('Y predictions', y_pred),
            ('QF Loss', qf_loss),
            ('Target Rewards', rewards),
        ])

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = OrderedDict()

        train_dict = self.get_train_dict(batch)
        for name in [
            'QF Loss',
        ]:
            tensor = train_dict[name]
            statistics_name = "{} {} Mean".format(stat_prefix, name)
            statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

        for name in [
            'Bellman Errors',
            'Target Rewards',
        ]:
            tensor = train_dict[name]
            statistics.update(create_stats_ordered_dict(
                '{} {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
            ))

        return statistics

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


class EvalQ(object):
    def __init__(self, qf):
        self.qf = qf
