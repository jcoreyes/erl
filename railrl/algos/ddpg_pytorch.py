import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from railrl.data_management.updatable_subtraj_replay_buffer import \
    UpdatableSubtrajReplayBuffer
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.algos.base import RLAlgorithm
from rllab.algos.batch_polopt import BatchSampler
from rllab.misc import logger, special


class QFunction(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            hidden_sizes,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.hidden_sizes = hidden_sizes

        input_dim = obs_dim + action_dim + 2 * memory_dim
        self.fcs = []
        last_size = input_dim
        for size in hidden_sizes:
            self.fcs.append(nn.Linear(last_size, size))
            last_size = size
        self.last_fc = nn.Linear(last_size, 1)

    def forward(self, obs, memory, action, write):
        x = torch.cat((obs, memory, action, write), dim=1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        return self.last_fc(x)

    def clone(self):
        copy = QFunction(
            self.obs_dim,
            self.action_dim,
            self.memory_dim,
            self.hidden_sizes,
        )
        copy_model_params(self, copy)
        return copy


class Policy(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            hidden_sizes,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.hidden_sizes = hidden_sizes

        self.fcs = []
        all_inputs_dim = obs_dim + memory_dim
        last_size = all_inputs_dim
        for size in hidden_sizes:
            self.fcs.append(nn.Linear(last_size, size))
            last_size = size
        self.last_fc = nn.Linear(last_size, action_dim)

        self.lstm_cell = nn.LSTMCell(all_inputs_dim, memory_dim // 2)

    def forward(self, obs, memory):
        all_inputs = torch.cat([obs, memory], dim=1)
        last_layer = all_inputs
        for fc in self.fcs:
            last_layer = F.relu(fc(last_layer))
        action = self.last_fc(last_layer)

        hx, cx = torch.split(memory, self.memory_dim // 2, dim=1)
        new_hx, new_cx = self.lstm_cell(all_inputs, (hx, cx))
        write = torch.cat((new_hx, new_cx), dim=1)
        return action, write

    def get_action(self, augmented_obs):
        obs, memory = augmented_obs
        obs = np.expand_dims(obs, axis=0)
        memory = np.expand_dims(memory, axis=0)
        obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
        memory = Variable(torch.from_numpy(memory).float(), requires_grad=False)
        action, write = self.__call__(obs, memory)
        return (action.data.numpy(), write.data.numpy()), {}

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
            self.memory_dim,
            self.hidden_sizes,
        )
        copy_model_params(self, copy)
        return copy


# noinspection PyCallingNonCallable
class BDP(RLAlgorithm):
    """
    Online learning algorithm.
    """

    def __init__(
            self,
            env,
            subtraj_length=None,
    ):
        self.training_env = env
        self.env = env
        self.action_dim = 1
        self.obs_dim = 1
        self.memory_dim = env.memory_dim
        self.subtraj_length = subtraj_length

        self.exploration_strategy = NoopStrategy()
        self.num_epochs = 100
        self.num_steps_per_epoch = 100
        self.render = False
        self.scale_reward = 1
        self.pool = UpdatableSubtrajReplayBuffer(
            10000,
            env,
            self.subtraj_length,
            self.memory_dim,
        )
        self.qf = QFunction(
            self.obs_dim,
            self.action_dim,
            self.memory_dim,
            [100, 100],
        )
        self.policy = Policy(
            self.obs_dim,
            self.action_dim,
            self.memory_dim,
            [100, 100],
        )
        self.target_qf = self.qf.clone()
        self.target_policy = self.policy.clone()
        self.discount = 1.
        self.batch_size = 32
        self.max_path_length = 1000
        self.n_eval_samples = 100
        self.scope = None  # Necessary for BatchSampler
        self.whole_paths = True  # Also for BatchSampler

        self.qf_criterion = nn.MSELoss()
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=1e-3)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)

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
        obs = batch['env_obs']
        actions = batch['env_actions']
        next_obs = batch['next_env_obs']
        memories = batch['memories']
        writes = batch['writes']
        next_memories = batch['next_memories']

        """
        Optimize critic
        """
        # Generate y target using target policies
        next_actions, next_writes = self.target_policy(next_obs, next_memories)
        target_q_values = self.target_qf(
            next_obs,
            next_memories,
            next_actions,
            next_writes
        )
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        y_pred = self.qf(obs, memories, actions, writes)
        qf_loss = self.qf_criterion(y_pred, y_target)

        # Do training
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Optimize policy
        """
        policy_actions, policy_writes = self.policy(obs, memories)
        q_output = self.qf(obs, memories, policy_actions, policy_writes)
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
        batch = self.pool.random_batch(self.batch_size)
        torch_batch = {
            k: Variable(torch.from_numpy(array).float(), requires_grad=True)
            for k, array in batch.items()
        }
        rewards = torch_batch['rewards']
        terminals = torch_batch['terminals']
        torch_batch['rewards'] = rewards.view(*rewards.size(), 1)
        torch_batch['terminals'] = terminals.view(*terminals.size(), 1)
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
