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
from railrl.torch.ddpg import DDPG, np_to_pytorch_batch
from railrl.torch.online_algorithm import OnlineAlgorithm
import railrl.torch.pytorch_util as ptu
from rllab.misc import logger, special


class StateDistanceQLearning(DDPG):
    def __init__(
            self,
            *args,
            exploration_policy,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.exploration_policy = exploration_policy

    def get_batch(self, training=True):
        pool = self.pool.get_replay_buffer(training)
        sample_size = min(
            pool.num_steps_can_sample(),
            self.batch_size
        )
        batch = pool.random_batch(sample_size)
        goal_states = self.env.sample_goal_states(len(batch))
        new_rewards = self.env.compute_rewards(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            goal_states,
        )
        batch['observations'] = batch['observations'], goal_states
        batch['next_observations'] = (
            batch['next_observations'], goal_states
        )
        batch['rewards'] = new_rewards
        torch_batch = np_to_pytorch_batch(batch)
        return torch_batch

    def reset_env(self):
        self.exploration_strategy.reset()
        self.exploration_policy.reset()
        self.policy.reset()
        return self.training_env.reset()


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
