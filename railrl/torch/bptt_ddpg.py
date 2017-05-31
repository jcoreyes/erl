import time
import pickle
from collections import OrderedDict

from railrl.data_management.updatable_subtraj_replay_buffer import (
    UpdatableSubtrajReplayBuffer
)
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.pythonplusplus import line_logger
from railrl.torch.online_algorithm import OnlineAlgorithm
from rllab.algos.base import RLAlgorithm
import numpy as np
import torch
# noinspection PyPep8Naming
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
            embed_obs_hidden_sizes,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.hidden_sizes = hidden_sizes
        self.embed_obs_hidden_sizes = embed_obs_hidden_sizes

        self.obs_embed_fcs = []
        last_size = obs_dim + memory_dim
        for size in self.embed_obs_hidden_sizes:
            self.obs_embed_fcs.append(nn.Linear(last_size, size))
            last_size = size

        self.fcs = []
        last_size = last_size + action_dim + memory_dim
        for size in hidden_sizes:
            self.fcs.append(nn.Linear(last_size, size))
            last_size = size
        self.last_fc = nn.Linear(last_size, 1)

    def forward(self, obs, memory, action, write):
        obs_embedded = torch.cat((obs, memory), dim=1)
        for fc in self.obs_embed_fcs:
            obs_embedded = F.relu(fc(obs_embedded))
        x = torch.cat((obs_embedded, action, write), dim=1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        return self.last_fc(x)

    def clone(self):
        copy = QFunction(
            self.obs_dim,
            self.action_dim,
            self.memory_dim,
            self.hidden_sizes,
            self.embed_obs_hidden_sizes,
        )
        copy_module_params_from_to(self, copy)
        return copy


class SumCell(nn.Module):
    def __init__(self, obs_dim, memory_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, memory_dim)

    def forward(self, obs, memory):
        new_memory = self.fc(obs)
        return memory + new_memory


class RecurrentPolicy(nn.Module):
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
        last_size = obs_dim + memory_dim
        for size in hidden_sizes:
            self.fcs.append(nn.Linear(last_size, size))
            last_size = size
        self.last_fc = nn.Linear(last_size, action_dim)

        self.lstm_cell = nn.LSTMCell(self.obs_dim, self.memory_dim // 2)
        self.memory_to_obs_fc = nn.Linear(self.memory_dim, obs_dim)

    def action_parameters(self):
        for fc in [self.last_fc] + self.fcs:
            for param in fc.parameters():
                yield param

    def write_parameters(self):
        return self.lstm_cell.parameters()

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
            # new_memory = self.lstm_cell(obs[:, i, :], initial_memory)
            # hx, cx = torch.split(new_memory, self.memory_dim // 2, dim=1)
            # new_hx, new_cx = self.lstm_cell(obs[:, i, :], (hx, cx))
            # hx = hx + new_hx
            # cx = cx + new_cx
            new_hxs[:, i, :] = hx
            new_cxs[:, i, :] = cx
        subtraj_writes = torch.cat((new_hxs, new_cxs), dim=2)

        """
        Create the new subtrajectory memories with the initial memories and the
        new writes.
        """
        expanded_init_memory = initial_memory.unsqueeze(1)
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
            # action = self.last_fc(last_layer)
            # action += 0.1 * Variable(torch.randn(*action.size()))
            subtraj_actions[:, i, :] = action

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
        obs = obs.unsqueeze(1)
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
        copy = RecurrentPolicy(
            self.obs_dim,
            self.action_dim,
            self.memory_dim,
            self.hidden_sizes,
        )
        copy_module_params_from_to(self, copy)
        return copy


# noinspection PyCallingNonCallable
class BpttDdpg(OnlineAlgorithm):
    """
    BPTT DDPG implemented in pytorch.
    """
    def __init__(
            self,
            *args,
            subtraj_length=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_dim = int(self.env.env_spec.action_space.flat_dim)
        self.obs_dim = int(self.env.env_spec.observation_space.flat_dim)
        self.memory_dim = self.env.memory_dim
        self.subtraj_length = subtraj_length

        self.train_validation_batch_size = 64
        self.copy_target_param_period = 1000
        self.batch_size = 32
        self.train_validation_batch_size = 64
        self.copy_target_param_period = 1000
        self.action_policy_learning_rate = 1e-3
        self.write_policy_learning_rate = 1e-5
        self.qf_learning_rate = 1e-3
        self.pool = UpdatableSubtrajReplayBuffer(
            10000,
            self.env,
            self.subtraj_length,
            self.memory_dim,
        )
        self.qf = QFunction(
            self.obs_dim,
            self.action_dim,
            self.memory_dim,
            [100],
            [100],
        )
        self.policy = RecurrentPolicy(
            self.obs_dim,
            self.action_dim,
            self.memory_dim,
            [100, 64],
        )
        self.target_qf = self.qf.clone()
        self.target_policy = self.policy.clone()

        self.qf_optimizer = optim.Adam(self.qf.parameters(),
                                       lr=self.qf_learning_rate)
        self.action_policy_optimizer = optim.Adam(
            self.policy.action_parameters(), lr=self.action_policy_learning_rate
        )
        self.write_policy_optimizer = optim.Adam(
            self.policy.write_parameters(), lr=self.write_policy_learning_rate
        )
        self.pps = list(self.policy.parameters())
        self.qps = list(self.qf.parameters())

    def train(self):
        n_steps_total = 0
        observation = self.training_env.reset()
        self.exploration_strategy.reset()
        self._start_worker()
        for epoch in range(self.num_epochs):
            logger.push_prefix('Iteration #%d | ' % epoch)
            path_return = 0
            es_path_returns = []
            actions, writes = [], []
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
                actions.append(action[0])
                writes.append(action[1])

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

                if self._can_train():
                    for _ in range(5):
                        self._do_training(n_steps_total=n_steps_total)

            logger.log(
                "Training Time: {0}".format(time.time() - start_time)
            )
            start_time = time.time()
            self.evaluate(epoch, {
                'Returns': es_path_returns,
                'Env Actions': actions,
                'Writes': writes,
            })
            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            logger.log("Eval Time: {0}".format(time.time() - start_time))
            logger.pop_prefix()

    def _do_training(self, n_steps_total):
        # for _ in range(10):
        #     raw_subtraj_batch, _ = self.pool.random_subtrajectories(self.batch_size)
        #     subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
        #
        #     qf_loss = self.train_critic(subtraj_batch)
        #     qf_loss_np = float(qf_loss.data.numpy())
        # line_logger.print_over("QF loss: {}".format(qf_loss_np))
        raw_subtraj_batch, start_indices = self.pool.random_subtrajectories(
            self.batch_size)
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
        self.train_critic(subtraj_batch)
        self.train_policy(subtraj_batch, start_indices)
        if n_steps_total % self.copy_target_param_period == 0:
            copy_module_params_from_to(self.qf, self.target_qf)
            copy_module_params_from_to(self.policy, self.target_policy)

    def train_critic(self, subtraj_batch):
        critic_dict = self.get_critic_output_dict(subtraj_batch)
        qf_loss = critic_dict['Loss']
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        return qf_loss

    def get_critic_output_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        critic, including intermediate values that might be useful to log.
        """
        flat_batch = flatten_subtraj_batch(subtraj_batch)
        rewards = flat_batch['rewards']
        terminals = flat_batch['terminals']
        obs = flat_batch['env_obs']
        actions = flat_batch['env_actions']
        next_obs = flat_batch['next_env_obs']
        memories = flat_batch['memories']
        writes = flat_batch['writes']
        next_memories = flat_batch['next_memories']

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
        y_predicted = self.qf(obs, memories, actions, writes)
        bellman_errors = (y_predicted - y_target)**2
        return OrderedDict([
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('Loss', bellman_errors.mean()),
        ])

    def train_policy(self, subtraj_batch, start_indices):
        policy_dict = self.get_policy_output_dict(subtraj_batch)

        policy_loss = policy_dict['loss']
        bellman_errors = policy_dict['Bellman Errors']

        self.action_policy_optimizer.zero_grad()
        policy_loss.backward(retain_variables=True)
        self.action_policy_optimizer.step()

        self.write_policy_optimizer.zero_grad()
        bellman_errors.mean().backward(retain_variables=True)
        self.write_policy_optimizer.step()

        self.pool.update_write_subtrajectories(
            policy_dict['New Writes'].data.numpy(), start_indices
        )

        # self.qf_optimizer.zero_grad()
        # bellman_errors.mean().backward()
        # self.qf_optimizer.step()

    def get_policy_output_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        policy, including intermediate values that might be useful to log.
        """
        subtraj_obs = subtraj_batch['env_obs']
        initial_memories = get_initial_memories(subtraj_batch)
        policy_actions, policy_writes = self.policy(subtraj_obs,
                                                    initial_memories)
        if self.subtraj_length > 1:
            new_memories = torch.cat(
                (
                    initial_memories.unsqueeze(1),
                    policy_writes[:, :-1, :],
                ),
                dim=1,
            )
        else:
            new_memories = initial_memories.unsqueeze(1)
        # TODO(vitchyr): should I detach (stop gradients)?
        # I don't think so. If we have dQ/dmemory, why not use it?
        # new_memories = new_memories.detach()
        subtraj_batch['policy_new_memories'] = new_memories
        subtraj_batch['policy_new_writes'] = policy_writes
        subtraj_batch['policy_new_actions'] = policy_actions

        flat_batch = flatten_subtraj_batch(subtraj_batch)
        flat_obs = flat_batch['env_obs']
        flat_new_memories = flat_batch['policy_new_memories']
        flat_new_actions = flat_batch['policy_new_actions']
        flat_new_writes = flat_batch['policy_new_writes']
        q_output = self.qf(
            flat_obs,
            flat_new_memories,
            flat_new_actions,
            flat_new_writes
        )
        policy_loss = - q_output.mean()

        """
        Train policy to minimize Bellman error as well.
        """
        flat_next_obs = flat_batch['next_env_obs']
        flat_actions = flat_batch['env_actions']
        flat_rewards = flat_batch['rewards']
        flat_terminals = flat_batch['terminals']
        flat_next_memories = flat_new_writes
        flat_next_actions, flat_next_writes = self.target_policy.get_flat_output(
            flat_next_obs, flat_next_memories
        )
        target_q_values = self.target_qf(
            flat_next_obs,
            flat_next_memories,
            flat_next_actions,
            flat_next_writes
        )
        y_target = (
            flat_rewards
            + (1. - flat_terminals) * self.discount * target_q_values
        )
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_predicted = self.qf(flat_obs, flat_new_memories, flat_actions,
                              flat_new_writes)
        bellman_errors = (y_predicted - y_target)**2
        # TODO(vitchyr): Still use target policies when minimizing Bellman err?
        return OrderedDict([
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('loss', policy_loss),
            ('New Env Actions', flat_batch['policy_new_actions']),
            ('New Writes', policy_writes),
        ])

    def evaluate(self, epoch, exploration_info_dict):
        """
        Perform evaluation for this algorithm.

        It's recommended
        :param epoch: The epoch number.
        :param exploration_info_dict: Dict from name to torch Variable.
        :return: Dictionary of statistics.
        """
        statistics = OrderedDict()

        for k, v in exploration_info_dict.items():
            statistics.update(create_stats_ordered_dict(
                'Exploration {}'.format(k), np.array(v, dtype=np.float32)
            ))
        statistics.update(self._get_other_statistics())

        logger.log("Collecting samples for evaluation")
        paths = self._sample_paths(epoch)
        statistics.update(self._statistics_from_paths(paths))

        rewards = np.hstack([path["rewards"] for path in paths])
        returns = [sum(path["rewards"]) for path in paths]
        discounted_returns = [
            special.discount_return(path["rewards"], self.discount)
            for path in paths
            ]
        statistics.update(create_stats_ordered_dict('Rewards', rewards))
        statistics.update(create_stats_ordered_dict('Returns', returns))
        statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                    discounted_returns))
        average_returns = np.mean(returns)
        statistics['AverageReturn'] = average_returns  # to match rllab
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(paths)

    def _statistics_from_paths(self, paths):
        eval_pool = UpdatableSubtrajReplayBuffer(
            len(paths) * self.max_path_length,
            self.env,
            self.subtraj_length,
            self.memory_dim,
        )
        for path in paths:
            eval_pool.add_trajectory(path)
        raw_subtraj_batch = eval_pool.get_all_valid_subtrajectories()
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
        return self._statistics_from_subtraj_batch(subtraj_batch,
                                                   stat_prefix='Test ')

    def _statistics_from_subtraj_batch(self, subtraj_batch, stat_prefix=''):
        statistics = OrderedDict()

        critic_dict = self.get_critic_output_dict(subtraj_batch)
        for name, tensor in critic_dict.items():
            statistics.update(create_stats_ordered_dict(
                '{}QF {}'.format(stat_prefix, name),
                tensor.data.numpy()
            ))

        policy_dict = self.get_policy_output_dict(subtraj_batch)
        for name, tensor in policy_dict.items():
            statistics.update(create_stats_ordered_dict(
                '{}Policy {}'.format(stat_prefix, name),
                tensor.data.numpy()
            ))
        return statistics

    def _get_other_statistics(self):
        statistics = OrderedDict()
        for stat_prefix, validation in [
            ('Validation ', True),
            ('Train ', False),
        ]:
            if (self.pool.num_can_sample(validation=validation) >=
                    self.train_validation_batch_size):
                raw_subtraj_batch = self.pool.random_subtrajectories(
                    self.train_validation_batch_size,
                    validation=validation
                )[0]
                subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
                statistics.update(self._statistics_from_subtraj_batch(
                    subtraj_batch, stat_prefix=stat_prefix
                ))
        return statistics

    """
    Random small functions.
    """
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

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.training_env,
            epoch=epoch,
            policy=self.policy,
            es=self.exploration_strategy,
            qf=self.qf,
        )


def copy_module_params_from_to(source, target):
    for source_param, target_param in zip(
            source.parameters(),
            target.parameters()
    ):
        target_param.data = source_param.data


def flatten_subtraj_batch(subtraj_batch):
    return {
        k: array.view(-1, array.size()[-1])
        for k, array in subtraj_batch.items()
    }


def get_initial_memories(subtraj_batch):
    return subtraj_batch['memories'][:, 0, :]


def create_torch_subtraj_batch(subtraj_batch):
    torch_batch = {
        k: Variable(torch.from_numpy(array).float(), requires_grad=True)
        for k, array in subtraj_batch.items()
        }
    rewards = torch_batch['rewards']
    terminals = torch_batch['terminals']
    torch_batch['rewards'] = rewards.unsqueeze(-1)
    torch_batch['terminals'] = terminals.unsqueeze(-1)
    return torch_batch
