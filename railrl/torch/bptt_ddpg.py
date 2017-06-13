from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from railrl.data_management.updatable_subtraj_replay_buffer import (
    UpdatableSubtrajReplayBuffer
)
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.torch.core import PyTorchModule
from railrl.torch.online_algorithm import OnlineAlgorithm
from railrl.torch.pytorch_util import fanin_init, copy_model_params_from_to, \
    soft_update_from_to
from rllab.misc import logger, special


# noinspection PyCallingNonCallable
class BpttDdpg(OnlineAlgorithm):
    """
    BPTT DDPG implemented in pytorch.
    """
    def __init__(
            self,
            *args,
            subtraj_length,
            tau=0.01,
            use_soft_update=True,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_dim = int(self.env.env_spec.action_space.flat_dim)
        self.obs_dim = int(self.env.env_spec.observation_space.flat_dim)
        self.memory_dim = self.env.memory_dim
        self.subtraj_length = subtraj_length

        self.train_validation_batch_size = 64
        self.batch_size = 32
        self.train_validation_batch_size = 64
        self.action_policy_learning_rate = 1e-3
        self.write_policy_learning_rate = 1e-5
        self.qf_learning_rate = 1e-3
        self.bellman_error_loss_weight = 10
        self.target_hard_update_period = 1000
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.pool = UpdatableSubtrajReplayBuffer(
            self.pool_size,
            self.env,
            self.subtraj_length,
            self.memory_dim,
        )
        self.qf = MemoryQFunction(
            self.obs_dim,
            self.action_dim,
            self.memory_dim,
            100,
            100,
        )
        self.policy = MemoryPolicy(
            self.obs_dim,
            self.action_dim,
            self.memory_dim,
            100,
            100,
        )
        self.target_qf = self.qf.copy()
        self.target_policy = self.policy.copy()

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
        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.policy.cuda()
            self.target_policy.cuda()
            self.qf.cuda()
            self.target_qf.cuda()

    """
    Training functions
    """

    def _do_training(self, n_steps_total):
        raw_subtraj_batch, start_indices = self.pool.random_subtrajectories(
            self.batch_size)
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch,
                                                   cuda=self.use_gpu)
        self.train_critic(subtraj_batch)
        self.train_policy(subtraj_batch, start_indices)
        if self.use_soft_update:
            soft_update_from_to(self.target_policy, self.policy, self.tau)
            soft_update_from_to(self.target_qf, self.qf, self.tau)
        else:
            if n_steps_total % self.target_hard_update_period == 0:
                copy_model_params_from_to(self.qf, self.target_qf)
                copy_model_params_from_to(self.policy, self.target_policy)

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
        bellman_loss = self.bellman_error_loss_weight * bellman_errors.mean()

        self.action_policy_optimizer.zero_grad()
        policy_loss.backward(retain_variables=True)
        self.write_policy_optimizer.zero_grad()
        bellman_loss.backward(retain_variables=True)
        self.action_policy_optimizer.step()
        self.write_policy_optimizer.step()

        self.pool.update_write_subtrajectories(
            self.get_numpy(policy_dict['New Writes']), start_indices
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

    """
    Eval functions
    """
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
        statistics.update(self._get_other_statistics())

        statistics['AverageReturn'] = get_average_returns(paths)
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(paths)

    def _statistics_from_paths(self, paths, stat_prefix):
        eval_pool = UpdatableSubtrajReplayBuffer(
            len(paths) * self.max_path_length,
            self.env,
            self.subtraj_length,
            self.memory_dim,
        )
        for path in paths:
            eval_pool.add_trajectory(path)
        raw_subtraj_batch = eval_pool.get_all_valid_subtrajectories()
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch,
                                                   cuda=self.use_gpu)
        statistics = self._statistics_from_subtraj_batch(
            subtraj_batch, stat_prefix=stat_prefix
        )
        rewards = np.hstack([path["rewards"] for path in paths])
        returns = [sum(path["rewards"]) for path in paths]
        discounted_returns = [
            special.discount_return(path["rewards"], self.discount)
            for path in paths
        ]
        statistics.update(create_stats_ordered_dict(
            'Rewards', rewards, stat_prefix=stat_prefix
        ))
        statistics.update(create_stats_ordered_dict(
            'Returns', returns, stat_prefix=stat_prefix
        ))
        statistics.update(create_stats_ordered_dict(
            'DiscountedReturns', discounted_returns, stat_prefix=stat_prefix
        ))
        return statistics

    def _statistics_from_subtraj_batch(self, subtraj_batch, stat_prefix=''):
        statistics = OrderedDict()

        critic_dict = self.get_critic_output_dict(subtraj_batch)
        for name, tensor in critic_dict.items():
            statistics.update(create_stats_ordered_dict(
                '{}QF {}'.format(stat_prefix, name),
                self.get_numpy(tensor)
            ))

        policy_dict = self.get_policy_output_dict(subtraj_batch)
        for name, tensor in policy_dict.items():
            statistics.update(create_stats_ordered_dict(
                '{}Policy {}'.format(stat_prefix, name),
                self.get_numpy(tensor)
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
                subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch,
                                                           cuda=self.use_gpu)
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
        self.batch_size = self.num_steps_per_eval
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

    def get_numpy(self, tensor):
        return get_numpy(tensor, self.use_gpu)


def flatten_subtraj_batch(subtraj_batch):
    return {
        k: array.view(-1, array.size()[-1])
        for k, array in subtraj_batch.items()
    }


def get_initial_memories(subtraj_batch):
    return subtraj_batch['memories'][:, 0, :]


def get_numpy(tensor, use_cuda):
    if use_cuda:
        return tensor.data.cpu().numpy()
    return tensor.data.numpy()


def create_torch_subtraj_batch(subtraj_batch, cuda=False):
    torch_batch = {
        k: Variable(torch.from_numpy(array).float(), requires_grad=True)
        for k, array in subtraj_batch.items()
    }
    if cuda:
        torch_batch = {k: v.cuda() for k, v in torch_batch.items()}
    rewards = torch_batch['rewards']
    terminals = torch_batch['terminals']
    torch_batch['rewards'] = rewards.unsqueeze(-1)
    torch_batch['terminals'] = terminals.unsqueeze(-1)
    return torch_batch


class MemoryQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            observation_hidden_size,
            embedded_hidden_size,
            init_w=3e-3,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size
        self.init_w = init_w

        self.obs_fc = nn.Linear(obs_dim + memory_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(
            observation_hidden_size + action_dim + memory_dim,
            embedded_hidden_size,
            )
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

    def forward(self, obs, memory, action, write):
        obs_embedded = torch.cat((obs, memory), dim=1)
        obs_embedded = F.relu(self.obs_fc(obs_embedded))
        x = torch.cat((obs_embedded, action, write), dim=1)
        x = F.relu(self.embedded_fc(x))
        return self.last_fc(x)


class SumCell(nn.Module):
    def __init__(self, obs_dim, memory_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, memory_dim)

    def forward(self, obs, memory):
        new_memory = self.fc(obs)
        return memory + new_memory


class MemoryPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            fc1_size,
            fc2_size,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(obs_dim + memory_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)
        self.lstm_cell = nn.LSTMCell(self.obs_dim, self.memory_dim // 2)

    def action_parameters(self):
        for fc in [self.fc1, self.fc2, self.last_fc]:
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
        if self.is_cuda:
            new_hxs = new_hxs.cuda()
            new_cxs = new_cxs.cuda()
        for i in range(subsequence_length):
            hx, cx = self.lstm_cell(obs[:, i, :], (hx, cx))
            new_hxs[:, i, :] = hx
            new_cxs[:, i, :] = cx
        subtraj_writes = torch.cat((new_hxs, new_cxs), dim=2)

        # The reason that using a LSTM doesn't work is that this gives you only
        # the FINAL hx and cx, not all of them :(
        # _, (new_hxs, new_cxs) = self.lstm(obs, (hx, cx))
        # subtraj_writes = torch.cat((new_hxs, new_cxs), dim=2)
        # subtraj_writes = subtraj_writes.permute(1, 0, 2)

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
        if self.is_cuda:
            subtraj_actions = subtraj_actions.cuda()
        for i in range(subsequence_length):
            all_inputs = all_subtraj_inputs[:, i, :]
            h1 = F.relu(self.fc1(all_inputs))
            h2 = F.relu(self.fc2(h1))
            action = F.tanh(self.last_fc(h2))
            subtraj_actions[:, i, :] = action

        return subtraj_actions, subtraj_writes

    @property
    def is_cuda(self):
        return self.last_fc.weight.is_cuda

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
        if self.is_cuda:
            obs = obs.cuda()
            memory = memory.cuda()
        action, write = self.get_flat_output(obs, memory)
        return (
                   np.squeeze(get_numpy(action, self.is_cuda), axis=0),
                   np.squeeze(get_numpy(write, self.is_cuda), axis=0),
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

    def reset(self):
        pass

    def log_diagnostics(self):
        pass
