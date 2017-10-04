"""
Use supervised learning to learn

f(s, a) = s'
"""
import time
from collections import OrderedDict

import numpy as np
from torch import optim as optim

from railrl.algos.state_distance.state_distance_q_learning import \
    multitask_rollout
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.samplers.util import rollout
from railrl.torch import pytorch_util as ptu
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.algos.eval import get_statistics_from_pytorch_dict, \
    get_difference_statistics
from rllab.misc import logger


class ModelLearning(object):
    """
    Train a model to learn the difference between the current state and next
    state, given an action. So, predict

    s' - f(s, a)

    where f is the true dynamics model.
    """

    def __init__(
            self,
            env,
            model,
            replay_buffer,
            eval_policy,
            num_epochs=100,
            num_batches_per_epoch=100,
            learning_rate=1e-3,
            weight_decay=0,
            batch_size=100,
            num_rollouts_per_eval=10,
            max_path_length=100,
            replay_buffer_size=100000,
            add_on_policy_data=True,
            model_learns_deltas=True,
            max_num_on_policy_steps_to_add=None,
    ):
        if replay_buffer is None:
            replay_buffer = SplitReplayBuffer(
                EnvReplayBuffer(
                    replay_buffer_size,
                    env,
                    flatten=True,
                ),
                EnvReplayBuffer(
                    replay_buffer_size,
                    env,
                    flatten=True,
                ),
                fraction_paths_in_train=0.8,
            )

        self.model = model
        self.replay_buffer = replay_buffer
        self.env = env
        self.eval_policy = eval_policy
        self.num_epochs = num_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_rollouts_per_eval = num_rollouts_per_eval
        self.max_path_length = max_path_length
        self.add_on_policy_data = add_on_policy_data
        self.model_learns_deltas = model_learns_deltas
        if max_num_on_policy_steps_to_add is None:
            max_num_on_policy_steps_to_add = np.inf
        self.max_num_on_policy_steps_to_add = max_num_on_policy_steps_to_add

        self.num_on_policy_steps_added = 0
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self.discount = ptu.Variable(
            ptu.from_numpy(np.zeros((batch_size, 1))).float()
        )
        self.start_time = time.time()

    def train(self):
        num_batches_total = 0
        for epoch in range(self.num_epochs):
            goal = self.env.sample_goal_state_for_rollout()
            if (self.add_on_policy_data
                    and self.num_on_policy_steps_added <
                    self.max_num_on_policy_steps_to_add):
                path = multitask_rollout(
                    self.env,
                    self.eval_policy,
                    goal,
                    0,
                    max_path_length=self.max_path_length,
                )
                self.num_on_policy_steps_added += len(path['observations'])
                self.replay_buffer.add_path(path)
            if self.replay_buffer.num_steps_can_sample() == 0:
                if self.add_on_policy_data:
                    raise Exception("If you're not going to add on-policy "
                                    "data, make sure your replay buffer is "
                                    "large enough.")
                continue
            for _ in range(self.num_batches_per_epoch):
                self.model.train(True)
                self._do_training()
                num_batches_total += 1
            logger.push_prefix('Iteration #%d | ' % epoch)
            self.model.train(False)
            self.evaluate(epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            logger.log("Done evaluating")
            logger.pop_prefix()

    def _do_training(self):
        batch = self.get_batch()
        train_dict = self.get_train_dict(batch)

        self.optimizer.zero_grad()
        loss = train_dict['Loss']
        loss.backward()
        self.optimizer.step()

    def get_train_dict(self, batch):
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        if self.model_learns_deltas:
            next_obs_delta_pred = self.model(obs, actions)
            next_obs_delta = next_obs - obs
            errors = (next_obs_delta_pred - next_obs_delta) ** 2
        else:
            next_obs_pred = self.model(obs, actions)
            errors = (next_obs_pred - next_obs) ** 2
        loss = errors.mean()

        return OrderedDict([
            ('Errors', errors),
            ('Loss', loss),
        ])

    def get_batch(self, training=True):
        replay_buffer = self.replay_buffer.get_replay_buffer(training)
        batch_size = min(
            replay_buffer.num_steps_can_sample(),
            self.batch_size
        )
        batch = replay_buffer.random_batch(batch_size)
        return np_to_pytorch_batch(batch)

    def evaluate(self, epoch):
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
        statistics.update(
            get_difference_statistics(
                statistics,
                ['Loss Mean', 'Errors Mean', 'Errors Max'],
                include_test_validation_gap=False,
            )
        )

        statistics['Num steps collected'] = self.replay_buffer.num_steps_saved()
        statistics['Total Wallclock Time (s)'] = time.time() - self.start_time
        statistics['Epoch'] = epoch
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        # Eval using policy
        paths = []
        for _ in range(self.num_rollouts_per_eval):
            goal = self.env.sample_goal_state_for_rollout()
            self.env.set_goal(goal)
            self.eval_policy.set_goal(goal)
            path = rollout(
                self.env,
                self.eval_policy,
                max_path_length=self.max_path_length,
            )
            path['goal_states'] = np.tile(goal, (len(path['observations']), 1))
            paths.append(path)
        self.env.log_diagnostics(paths)

    def _statistics_from_batch(self, batch, stat_prefix):
        train_dict = self.get_train_dict(batch)
        return get_statistics_from_pytorch_dict(
            train_dict,
            ['Loss'],
            ['Errors'],
            stat_prefix
        )

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            model=self.model,
            replay_buffer=self.replay_buffer,
            env=self.env,
            policy=self.eval_policy,
        )

    def cuda(self):
        self.model.cuda()
