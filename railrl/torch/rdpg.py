from collections import OrderedDict

import numpy as np
import torch
# noinspection PyPep8Naming
import torch.optim as optim
from torch.autograd import Variable

from railrl.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns
from railrl.policies.torch import MemoryPolicy
from railrl.pythonplusplus import batch, ConditionTimer
from railrl.qfunctions.torch import MemoryQFunction
from railrl.torch.bptt_ddpg import create_torch_subtraj_batch
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
class Rdpg(OnlineAlgorithm):
    """
    Recurrent DPG.
    """

    def __init__(
            self,
            *args,
            qf,
            policy,
            tau=0.01,
            policy_learning_rate=1e-3,
            qf_learning_rate=1e-3,
            subtraj_length=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_dim = int(self.env.action_space.flat_dim)
        self.obs_dim = int(self.env.observation_space.flat_dim)
        self.qf = qf
        self.policy = policy
        if subtraj_length is None:
            subtraj_length = self.env.horizon
        self.subtraj_length = subtraj_length

        self.num_subtrajs_per_batch = self.batch_size // self.subtraj_length
        self.train_validation_num_subtrajs_per_batch = (
            self.num_subtrajs_per_batch
        )
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.tau = tau

        self.pool = SubtrajReplayBuffer(
            self.pool_size,
            self.env,
            self.subtraj_length,
        )
        self.target_qf = self.qf.copy()
        self.target_policy = self.policy.copy()

        self.qf_optimizer = optim.Adam(
            self.qf.parameters(), lr=self.qf_learning_rate
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.policy_learning_rate
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
        raw_subtraj_batch = self.pool.random_subtrajectories(
            self.num_subtrajs_per_batch
        )
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
        self.train_critic(subtraj_batch)
        self.train_policy(subtraj_batch)

        soft_update_from_to(self.target_policy, self.policy, self.tau)
        soft_update_from_to(self.target_qf, self.qf, self.tau)

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
        rewards = subtraj_batch['rewards']
        terminals = subtraj_batch['terminals']
        obs = subtraj_batch['observations']
        actions = subtraj_batch['actions']
        next_obs = subtraj_batch['next_observations']

        next_actions, _ = self.target_policy(next_obs)
        target_q_values = self.target_qf(next_obs, next_actions)
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_predicted = self.qf(obs, actions)
        bellman_errors = (y_predicted - y_target) ** 2
        return OrderedDict([
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('Loss', bellman_errors.mean()),
        ])

    def train_policy(self, subtraj_batch):
        policy_dict = self.get_policy_output_dict(subtraj_batch)

        policy_loss = policy_dict['Loss']
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def get_policy_output_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        policy, including intermediate values that might be useful to log.
        """
        observations = subtraj_batch['observations']
        policy_actions, _ = self.policy(observations)
        q_output = self.qf(observations, policy_actions)
        policy_loss = - q_output.mean()

        return OrderedDict([
            ('Loss', policy_loss),
            ('New Env Actions', policy_actions),
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
        eval_pool = SubtrajReplayBuffer(
            len(paths) * self.max_path_length,
            self.env,
            self.subtraj_length,
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
        actions = np.vstack([path["actions"] for path in paths])
        statistics.update(create_stats_ordered_dict(
            'Actions', actions, stat_prefix=stat_prefix
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
            sample_size = min(
                self.pool.num_subtrajs_can_sample(validation=validation),
                self.train_validation_num_subtrajs_per_batch
            )
            raw_subtraj_batch = self.pool.random_subtrajectories(
                sample_size,
                validation=validation
            )
            subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
            statistics.update(self._statistics_from_subtraj_batch(
                subtraj_batch, stat_prefix=stat_prefix
            ))
        return statistics

    def _can_evaluate(self, exploration_paths):
        return (
            self.pool.num_subtrajs_can_sample(validation=True) >= 1
            and self.pool.num_subtrajs_can_sample(validation=False) >= 1
            and len(exploration_paths) > 0
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
