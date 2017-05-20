import time
from collections import OrderedDict

from railrl.data_management.updatable_subtraj_replay_buffer import (
    UpdatableSubtrajReplayBuffer
)
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.algos.base import RLAlgorithm
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

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


def expand_dims(tensor, axis):
    return tensor.unsqueeze(axis)


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

        self.lstm_cell = nn.LSTMCell(self.obs_dim, self.memory_dim // 2)

    def forward(self, obs, initial_memory):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :param initial_memory: torch Variable, [batch_size, memory dim]
        :return: (actions, writes) tuple
            actions: [batch_size, sequence length, action dim]
            writes: [batch_size, sequence length, memory dim]
        """
        assert len(obs.size()) == 3
        assert len(initial_memory.size()) == 2
        batch_size, subsequence_length = obs.size()[:2]

        """
        Create the new writes.
        """
        hx, cx = torch.split(initial_memory, self.memory_dim // 2, dim=1)
        # noinspection PyArgumentList
        new_hxs = Variable(torch.FloatTensor(batch_size, subsequence_length,
                                             self.memory_dim // 2))
        # noinspection PyArgumentList
        new_cxs = Variable(torch.FloatTensor(batch_size, subsequence_length,
                                             self.memory_dim // 2))
        for i in range(subsequence_length):
            hx, cx = self.lstm_cell(obs[:, i, :], (hx, cx))
            new_hxs[:, i, :] = hx
            new_cxs[:, i, :] = cx
        subtraj_writes = torch.cat((new_hxs, new_cxs), dim=2)

        """
        Create the new subtrajectory memories with the initial memories and the
        new writes.
        """
        expanded_init_memory = expand_dims(initial_memory, 1)
        if subsequence_length > 1:
            memories = torch.cat(
                (
                    expanded_init_memory,
                    subtraj_writes[:, :-1, :],
                ),
                dim=1,
            )
        else:
            memories = expanded_init_memory

        """
        Use new memories to create env actions.
        """
        all_subtraj_inputs = torch.cat([obs, memories], dim=2)
        # noinspection PyArgumentList
        subtraj_actions = Variable(
            torch.FloatTensor(batch_size, subsequence_length, self.action_dim)
        )
        for i in range(subsequence_length):
            all_inputs = all_subtraj_inputs[:, i, :]
            last_layer = all_inputs
            for fc in self.fcs:
                last_layer = F.relu(fc(last_layer))
            action = F.tanh(self.last_fc(last_layer))
            subtraj_actions[:, i, :] =  action.unsqueeze(1)

        return subtraj_actions, subtraj_writes

    def get_action(self, augmented_obs):
        """
        :param augmented_obs: (obs, memories) tuple
            obs: np.ndarray, [obs_dim]
            memories: nd.ndarray, [memory_dim]
        :return: (actions, writes) tuple
            actions: np.ndarray, [action_dim]
            writes: np.ndarray, [writes_dim]
        """
        obs, memory = augmented_obs
        obs = np.expand_dims(obs, axis=0)
        memory = np.expand_dims(memory, axis=0)
        obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
        memory = Variable(torch.from_numpy(memory).float(), requires_grad=False)
        action, write = self.get_flat_output(obs, memory)
        return (
                   np.squeeze(action.data.numpy(), axis=0),
                    np.squeeze(write.data.numpy(), axis=0)
               ), {}

    def get_flat_output(self, obs, initial_memories):
        """
        Each batch element is processed independently. So, there's no recurrency
        used.

        :param obs: torch Variable, [batch_size X obs_dim]
        :param initial_memories: torch Variable, [batch_size X memory_dim]
        :return: (actions, writes) Tuple
            actions: torch Variable, [batch_size X action_dim]
            writes: torch Variable, [batch_size X writes_dim]
        """
        obs = expand_dims(obs, 1)
        actions, writes = self.__call__(obs, initial_memories)
        return torch.squeeze(actions, dim=1), torch.squeeze(writes, dim=1)

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


def flatten_subtraj_batch(subtraj_batch):
    return {
        k: array.view(-1, array.size()[-1])
        for k, array in subtraj_batch.items()
    }


def get_initial_memories(subtraj_batch):
    return subtraj_batch['memories'][:, 0, :]


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

        # noinspection PyTypeChecker
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
        subtraj_batch = self.get_subtraj_batch()
        self.train_critic(subtraj_batch)
        self.train_policy(subtraj_batch)
        if n_steps_total % 1000 == 0:
            copy_model_params(self.qf, self.target_qf)
            copy_model_params(self.policy, self.target_policy)

    def train_critic(self, subtraj_batch):
        # subtraj_batch = self.add_new_memories_and_writes(subtraj_batch)
        flat_batch = flatten_subtraj_batch(subtraj_batch)
        rewards = flat_batch['rewards']
        terminals = flat_batch['terminals']
        obs = flat_batch['env_obs']
        actions = flat_batch['env_actions']
        next_obs = flat_batch['next_env_obs']
        memories = flat_batch['memories']
        writes = flat_batch['writes']
        next_memories = flat_batch['next_memories']
        # new_memories = flat_batch['new_memories']
        # new_writes = flat_batch['new_writes']
        # new_next_memories = flat_batch['new_next_memories']

        # self.minimize_critic_bellman_error(
        #     obs,
        #     new_memories,
        #     actions,
        #     new_writes,
        #     next_obs,
        #     new_next_memories,
        #     rewards,
        #     terminals,
        # )
        self.minimize_critic_bellman_error(
            obs,
            memories,
            actions,
            writes,
            next_obs,
            next_memories,
            rewards,
            terminals,
        )

    def minimize_critic_bellman_error(
            self,
            obs,
            memories,
            actions,
            writes,
            next_obs,
            next_memories,
            rewards,
            terminals,
    ):
        next_actions, next_writes = self.target_policy.get_flat_output(
            next_obs, next_memories
        )
        target_q_values = self.target_qf(
            next_obs,
            next_memories,
            next_actions,
            next_writes
        )
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, memories, actions, writes)
        qf_loss = self.qf_criterion(y_pred, y_target)

        # Do training
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

    def train_policy(self, subtraj_batch):
        subtraj_obs = subtraj_batch['env_obs']
        initial_memories = get_initial_memories(subtraj_batch)
        # TODO(vitchyr): policy_writes should overwrite the # memories...right?
        policy_actions, policy_writes = self.policy(subtraj_obs, initial_memories)
        if self.subtraj_length > 1:
            new_memories = torch.cat(
                (
                    initial_memories.unsqueeze(1),
                    policy_writes[:, :-1, :],
                ),
                dim=1,
            )
            # TODO(vitchyr): should I detach (stop gradients)?
            # new_memories = new_memories.detach()
            subtraj_batch['new_memories'] = new_memories
        subtraj_batch['policy_actions'] = policy_actions
        subtraj_batch['policy_writes'] = policy_writes

        flat_batch = flatten_subtraj_batch(subtraj_batch)
        flat_obs = flat_batch['env_obs']
        flat_new_memories = flat_batch['new_memories']
        flat_policy_actions = flat_batch['policy_actions']
        flat_policy_writes = flat_batch['policy_writes']

        q_output = self.qf(
            flat_obs, flat_new_memories, flat_policy_actions, flat_policy_writes
        )
        policy_loss = - q_output.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

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

    def get_subtraj_batch(self):
        # batch = self.pool.random_batch(self.batch_size)
        batch, _ = self.pool.random_subtrajectories(self.batch_size)
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

    def subtraj_batch_to_list_of_batch(self, subtraj_batch):
        batches = []
        for i in range(self.subtraj_length):
            batch = {
                k: values[:, i, :] for k, values in subtraj_batch.items()
            }
            batches.append(batch)
        return batches

    def add_new_memories_and_writes(self, subtraj_batch):
        initial_memories = get_initial_memories(subtraj_batch)
        subtraj_obs = subtraj_batch['env_obs']
        _, new_writes = self.target_policy(
            subtraj_obs, initial_memories
        )
        new_memories = torch.cat((initial_memories, new_writes), dim=1)
        subtraj_batch['new_memories'] = new_memories
        subtraj_batch['new_writes'] = new_writes
        subtraj_batch['new_next_memories'] = new_writes
        return subtraj_batch


def copy_model_params(source, target):
    for source_param, target_param in zip(
            source.parameters(),
            target.parameters()
    ):
        target_param.data = source_param.data
