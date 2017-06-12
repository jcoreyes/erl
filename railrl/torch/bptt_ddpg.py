from collections import OrderedDict

import numpy as np
import torch
# noinspection PyPep8Naming
import torch.optim as optim
from torch.autograd import Variable

from railrl.data_management.updatable_subtraj_replay_buffer import (
    UpdatableSubtrajReplayBuffer
)
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.policies.torch import MemoryPolicy
from railrl.pythonplusplus import batch, ConditionTimer
from railrl.qfunctions.torch import MemoryQFunction
from railrl.torch.online_algorithm import OnlineAlgorithm
from railrl.torch.pytorch_util import (
    copy_model_params_from_to,
    soft_update_from_to,
    set_gpu_mode,
    from_numpy,
    get_numpy,
)
from rllab.misc import logger, special


# noinspection PyCallingNonCallable
class BpttDdpg(OnlineAlgorithm):
    """
    BPTT DDPG implemented in pytorch.
    """

    def __init__(
            self,
            *args,
            qf,
            policy,
            subtraj_length,
            tau=0.01,
            use_soft_update=True,
            refresh_entire_buffer_period=None,
            policy_optimize_bellman=True,
            action_policy_learning_rate=1e-3,
            write_policy_learning_rate=1e-5,
            qf_learning_rate=1e-3,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_dim = int(self.env.env_spec.action_space.flat_dim)
        self.obs_dim = int(self.env.env_spec.observation_space.flat_dim)
        self.memory_dim = self.env.memory_dim
        self.qf = qf
        self.policy = policy
        self.subtraj_length = subtraj_length
        self.policy_optimize_bellman = policy_optimize_bellman

        self.num_subtrajs_per_batch = self.batch_size // self.subtraj_length
        self.train_validation_num_subtrajs_per_batch = (
            self.num_subtrajs_per_batch
        )
        self.action_policy_learning_rate = action_policy_learning_rate
        self.write_policy_learning_rate = write_policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.bellman_error_loss_weight = 10
        self.target_hard_update_period = 1000
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.max_number_trajectories_loaded_at_once = (
            self.num_subtrajs_per_batch
        )
        self.pool = UpdatableSubtrajReplayBuffer(
            self.pool_size,
            self.env,
            self.subtraj_length,
            self.memory_dim,
        )
        self.target_qf = self.qf.copy()
        self.target_policy = self.policy.copy()

        self.qf_optimizer = optim.Adam(
            self.qf.parameters(), lr=self.qf_learning_rate
        )
        self.action_policy_optimizer = optim.Adam(
            self.policy.action_parameters(), lr=self.action_policy_learning_rate
        )
        self.write_policy_optimizer = optim.Adam(
            self.policy.write_parameters(), lr=self.write_policy_learning_rate
        )

        self.should_refresh_buffer = ConditionTimer(
            refresh_entire_buffer_period
        )

        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        set_gpu_mode(self.use_gpu)
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
            self.num_subtrajs_per_batch
        )
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
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
        bellman_errors = (y_predicted - y_target) ** 2
        return OrderedDict([
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('Loss', bellman_errors.mean()),
        ])

    def train_policy(self, subtraj_batch, start_indices):
        policy_dict = self.get_policy_output_dict(subtraj_batch)

        policy_loss = policy_dict['Loss']

        self.action_policy_optimizer.zero_grad()
        self.write_policy_optimizer.zero_grad()
        if self.policy_optimize_bellman:
            policy_loss.backward(retain_variables=True)
            bellman_errors = policy_dict['Bellman Errors']
            bellman_loss = self.bellman_error_loss_weight * bellman_errors.mean()
            bellman_loss.backward()
        else:
            policy_loss.backward()
        self.action_policy_optimizer.step()
        self.write_policy_optimizer.step()

        self.pool.update_write_subtrajectories(
            get_numpy(policy_dict['New Writes']), start_indices
        )

    def get_policy_output_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        policy, including intermediate values that might be useful to log.
        """
        subtraj_obs = subtraj_batch['env_obs']
        initial_memories = subtraj_batch['memories'][:, 0, :]
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
        bellman_errors = (y_predicted - y_target) ** 2
        # TODO(vitchyr): Still use target policies when minimizing Bellman err?
        return OrderedDict([
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('Loss', policy_loss),
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

        statistics.update(self._statistics_from_paths(paths, "Test"))
        statistics.update(self._get_other_statistics())
        statistics.update(self._statistics_from_paths(exploration_paths,
                                                      "Exploration"))

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
        assert raw_subtraj_batch is not None
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
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
        env_actions = np.vstack([path["actions"][:self.action_dim] for path in
                                 paths])
        writes = np.vstack([path["actions"][self.action_dim:] for path in
                            paths])
        statistics.update(create_stats_ordered_dict(
            'Env Actions', env_actions, stat_prefix=stat_prefix
        ))
        statistics.update(create_stats_ordered_dict(
            'Writes', writes, stat_prefix=stat_prefix
        ))
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics

    def _statistics_from_subtraj_batch(self, subtraj_batch, stat_prefix=''):
        statistics = OrderedDict()

        critic_dict = self.get_critic_output_dict(subtraj_batch)
        for name, tensor in critic_dict.items():
            statistics.update(create_stats_ordered_dict(
                '{} QF {}'.format(stat_prefix, name),
                get_numpy(tensor)
            ))

        policy_dict = self.get_policy_output_dict(subtraj_batch)
        for name, tensor in policy_dict.items():
            statistics.update(create_stats_ordered_dict(
                '{} Policy {}'.format(stat_prefix, name),
                get_numpy(tensor)
            ))
        return statistics

    def _get_other_statistics(self):
        statistics = OrderedDict()
        for stat_prefix, validation in [
            ('Validation', True),
            ('Train', False),
        ]:
            if (self.pool.num_subtrajs_can_sample(validation=validation) >=
                    self.train_validation_num_subtrajs_per_batch):
                raw_subtraj_batch = self.pool.random_subtrajectories(
                    self.train_validation_num_subtrajs_per_batch,
                    validation=validation
                )[0]
                subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
                statistics.update(self._statistics_from_subtraj_batch(
                    subtraj_batch, stat_prefix=stat_prefix
                ))
        return statistics

    def _can_evaluate(self, exploration_paths):
        return (
            self.pool.num_subtrajs_can_sample(validation=True) >=
                self.train_validation_num_subtrajs_per_batch
            and
            self.pool.num_subtrajs_can_sample(validation=False) >=
            self.train_validation_num_subtrajs_per_batch
            and
            len(exploration_paths) > 0
            # Technically, I should also check that the exploration path has
            # enough subtraj batches, but whatever.
        )

    """
    Random small functions.
    """

    def _can_train(self):
        return (
            self.pool.num_subtrajs_can_sample() >= self.num_subtrajs_per_batch
        )

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

    def handle_rollout_ending(self, n_steps_total):
        if not self._can_train():
            return

        if self.should_refresh_buffer.check(n_steps_total):
            all_start_traj_indices = (
                self.pool.get_all_valid_trajectory_start_indices()
            )
            for start_traj_indices in batch(
                    all_start_traj_indices,
                    self.max_number_trajectories_loaded_at_once,
            ):
                raw_subtraj_batch, start_indices = (
                    self.pool.get_trajectory_minimal_covering_subsequences(
                        start_traj_indices, self.training_env.horizon)
                )
                subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
                subtraj_obs = subtraj_batch['env_obs']
                initial_memories = subtraj_batch['memories'][:, 0, :]
                _, policy_writes = self.policy(subtraj_obs, initial_memories)
                self.pool.update_write_subtrajectories(
                    get_numpy(policy_writes), start_indices
                )


def flatten_subtraj_batch(subtraj_batch):
    return {
        k: array.view(-1, array.size()[-1])
        for k, array in subtraj_batch.items()
    }


def create_torch_subtraj_batch(subtraj_batch):
    torch_batch = {
        k: Variable(from_numpy(array).float(), requires_grad=True)
        for k, array in subtraj_batch.items()
    }
    rewards = torch_batch['rewards']
    terminals = torch_batch['terminals']
    torch_batch['rewards'] = rewards.unsqueeze(-1)
    torch_batch['terminals'] = terminals.unsqueeze(-1)
    return torch_batch
