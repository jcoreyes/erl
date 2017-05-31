import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.updatable_subtraj_replay_buffer import \
    UpdatableSubtrajReplayBuffer
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.algos.base import RLAlgorithm
from rllab.algos.batch_polopt import BatchSampler
from rllab.misc import logger, special


# noinspection PyCallingNonCallable
class DDPG(RLAlgorithm):
    """
    Online learning algorithm.
    """

    def __init__(
            self,
            env,
            exploration_strategy=None,
            subtraj_length=None,
            num_epochs=100,
            num_steps_epoch=10000,
            batch_size=1024,
            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
    ):
        self.training_env = env
        self.env = env
        self.action_dim = int(env.action_space.flat_dim)
        self.obs_dim = int(env.observation_space.flat_dim)
        self.subtraj_length = subtraj_length

        self.exploration_strategy = exploration_strategy or NoopStrategy()
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_epoch
        self.batch_size = batch_size
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.max_path_length = 1000
        self.n_eval_samples = 1000
        self.render = False
        self.scale_reward = 1
        self.pool = EnvReplayBuffer(
            10000,
            self.env,
        )
        self.qf = QFunction(
            self.obs_dim,
            self.action_dim,
            [100, 100],
        )
        self.policy = Policy(
            self.obs_dim,
            self.action_dim,
            [100, 100],
        )
        self.target_qf = self.qf.clone()
        self.target_policy = self.policy.clone()
        self.discount = 1.

        self.qf_criterion = nn.MSELoss()
        self.qf_optimizer = optim.Adam(self.qf.parameters(),
                                       lr=self.qf_learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)

        self.scope = None  # Necessary for BatchSampler
        self.whole_paths = True  # Also for BatchSampler
        self.eval_sampler = BatchSampler(self)

    def train(self):
        n_steps_total = 0
        observation = self.training_env.reset()
        self.exploration_strategy.reset()
        path_return = 0
        es_path_returns = []
        self._start_worker()
        for epoch in range(self.num_epochs):
            logger.push_prefix('Iteration #%d | ' % epoch)
            start_time = time.time()
            for _ in range(self.num_steps_per_epoch):
                action, agent_info = (
                    self.exploration_strategy.get_action(
                        n_steps_total,
                        observation,
                        self.policy,
                    )
                )
                if self.render:
                    self.training_env.render()

                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                n_steps_total += 1
                reward = raw_reward * self.scale_reward
                path_return += reward

                self.pool.add_sample(
                    observation,
                    action,
                    reward,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal:
                    self.pool.terminate_episode(
                        next_ob,
                        agent_info=agent_info,
                        env_info=env_info,
                    )
                    observation = self.training_env.reset()
                    self.exploration_strategy.reset()
                    es_path_returns.append(path_return)
                    path_return = 0
                else:
                    observation = next_ob

                if self._can_train(n_steps_total):
                    self._do_training(n_steps_total=n_steps_total)

            logger.log(
                "Training Time: {0}".format(time.time() - start_time)
            )
            start_time = time.time()
            self.evaluate(epoch, es_path_returns)
            es_path_returns = []
            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            logger.log("Eval Time: {0}".format(time.time() - start_time))
            logger.pop_prefix()

    def _do_training(self, n_steps_total):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Optimize critic
        """
        # Generate y target using target policies
        next_actions = self.target_policy(next_obs)
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
        )
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions)
        qf_loss = self.qf_criterion(y_pred, y_target)

        # Do training
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Optimize policy
        """
        policy_actions = self.policy(obs)
        q_output = self.qf(obs, policy_actions)
        policy_loss = - q_output.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update Target Networks
        """
        if n_steps_total % 1000 == 0:
            copy_model_params(self.qf, self.target_qf)
            copy_model_params(self.policy, self.target_policy)

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

    def get_epoch_snapshot(self, epoch):
        pass

    def get_batch(self):
        batch = self.pool.random_batch(self.batch_size, flatten=True)
        torch_batch = {
            k: Variable(torch.from_numpy(array).float(), requires_grad=True)
            for k, array in batch.items()
        }
        rewards = torch_batch['rewards']
        terminals = torch_batch['terminals']
        torch_batch['rewards'] = rewards.unsqueeze(-1)
        torch_batch['terminals'] = terminals.unsqueeze(-1)
        return torch_batch

    def _can_train(self, n_steps_total):
        return self.pool.num_can_sample() >= self.batch_size

    def _start_worker(self):
        self.eval_sampler.start_worker()

    def _shutdown_worker(self):
        self.eval_sampler.shutdown_worker()

    def _sample_paths(self, epoch):
        """
        Returns flattened paths.

        :param epoch: Epoch number
        :return: Dictionary with these keys:
            observations: np.ndarray, shape BATCH_SIZE x flat observation dim
            actions: np.ndarray, shape BATCH_SIZE x flat action dim
            rewards: np.ndarray, shape BATCH_SIZE
            terminals: np.ndarray, shape BATCH_SIZE
            agent_infos: unsure
            env_infos: unsure
        """
        # Sampler uses self.batch_size to figure out how many samples to get
        saved_batch_size = self.batch_size
        self.batch_size = self.n_eval_samples
        paths = self.eval_sampler.obtain_samples(
            itr=epoch,
        )
        self.batch_size = saved_batch_size
        return paths

    def _get_other_statistics(self):
        return {}

    def _statistics_from_paths(self, paths):
        return {}

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)


def copy_model_params(source, target):
    for source_param, target_param in zip(
            source.parameters(),
            target.parameters()
    ):
        target_param.data = source_param.data


class QFunction(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        input_dim = obs_dim + action_dim
        self.fcs = []
        last_size = input_dim
        for size in hidden_sizes:
            self.fcs.append(nn.Linear(last_size, size))
            last_size = size
        self.last_fc = nn.Linear(last_size, 1)

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        return self.last_fc(x)

    def clone(self):
        copy = QFunction(
            self.obs_dim,
            self.action_dim,
            self.hidden_sizes,
        )
        copy_model_params(self, copy)
        return copy


class Policy(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        self.fcs = []
        last_size = obs_dim
        for size in hidden_sizes:
            self.fcs.append(nn.Linear(last_size, size))
            last_size = size
        self.last_fc = nn.Linear(last_size, action_dim)


    def forward(self, obs):
        last_layer = obs
        for fc in self.fcs:
            last_layer = F.relu(fc(last_layer))
        return self.last_fc(last_layer)

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
        action = self.__call__(obs)
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