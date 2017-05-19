import time

from railrl.exploration_strategies.noop import NoopStrategy
from rllab.algos.base import RLAlgorithm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from rllab.misc import logger


class QFunction(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            hidden_sizes,
    ):
        super().__init__()
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


class Policy(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            hidden_sizes,
    ):
        super().__init__()
        self.fcs = []
        all_inputs_dim = obs_dim + memory_dim
        last_size = all_inputs_dim
        for size in hidden_sizes:
            self.fcs.append(nn.Linear(last_size, size))
            last_size = size
        self.last_fc = nn.Linear(last_size, action_dim)

        self.lstm_cell = nn.LSTMCell(all_inputs_dim, memory_dim)

    def forward(self, obs, memory):
        all_inputs = torch.cat((obs, memory), dim=1)
        last_layer = all_inputs
        for fc in self.fcs:
            last_layer = F.relu(fc(last_layer))
        action = self.last_fc(last_layer)

        hx, cx = torch.split(memory, 2, dim=1)
        write = self.lstm_cell(all_inputs, (hx, cx))
        return action, write


class BDP(RLAlgorithm):
    """
    Online learning algorithm.
    """

    def __init__(self, env):
        self.training_env = env
        self.exploration_strategy = NoopStrategy()
        self.num_epochs = 100
        self.num_steps_per_epoch = 100
        self.policy = Policy()
        self.render = None
        self.scale_reward = 1
        self.pool = None
        self.qf = None
        self.policy = None
        self.target_qf = None
        self.target_policy = None
        self.discount = 1.

        self.qf_criterion = nn.MSELoss()
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=1e-3)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)

    def train(self):
        n_steps_total = 0
        observation = self.training_env.reset()
        self.exploration_strategy.reset()
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
                    self.policy.reset()
                else:
                    observation = next_ob

                self._do_training(n_steps_total=n_steps_total)

            logger.log(
                "Training Time: {0}".format(time.time() - start_time)
            )
            start_time = time.time()
            self.evaluate(epoch)
            self.es_path_returns = []
            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            logger.log("Eval Time: {0}".format(time.time() - start_time))
            logger.pop_prefix()

    def _do_training(self, n_steps_total):

        # Get data
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
        policy_loss = - q_output
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update Target Networks
        """
        if n_steps_total % 1000 == 0:
            copy_model_params(self.qf, self.target_qf)
            copy_model_params(self.policy, self.target_policy)

    def evaluate(self, epoch):
        pass

    def get_epoch_snapshot(self, epoch):
        pass

    def get_batch(self):
        pass

def copy_model_params(source, target):
    for source_param, target_param in zip(
            source.parameters(),
            target.parameters()
    ):
        target_param.data = source_param.data
