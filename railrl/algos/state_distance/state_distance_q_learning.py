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
            replay_buffer=None,
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
        self.replay_buffer = replay_buffer

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
        replay_buffer = self.replay_buffer.get_replay_buffer(training)
        batch_size = min(
            replay_buffer.num_steps_can_sample(),
            self.batch_size
        )
        batch = replay_buffer.random_batch(batch_size)
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
            replay_buffer = self.replay_buffer.get_replay_buffer(training=True)
            batch = replay_buffer.random_batch(batch_size)
            obs = batch['observations']
            return self.env.convert_obs_to_goal_state(obs)

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


def rollout_with_goal(env, agent, goal, max_path_length=np.inf, animated=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    o = np.hstack((o, goal))
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        o = np.hstack((o, goal))
        if animated:
            env.render()

    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
    )