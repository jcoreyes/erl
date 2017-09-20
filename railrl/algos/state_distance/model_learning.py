"""
Use supervised learning to learn

f(s, a) = s'
"""
from collections import OrderedDict

import numpy as np
from torch import optim as optim

from railrl.samplers.util import rollout
from railrl.misc.data_processing import create_stats_ordered_dict
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
            batch_size=100,
            num_unique_batches=1000,
            num_rollouts_per_eval=10,
            max_path_length_for_eval=100,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.env = env
        self.eval_policy = eval_policy
        self.num_epochs = num_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_unique_batches = num_unique_batches
        self.num_rollouts_per_eval = num_rollouts_per_eval
        self.max_path_length_for_eval = max_path_length_for_eval

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)
        self.discount = ptu.Variable(
            ptu.from_numpy(np.zeros((batch_size, 1))).float()
        )

    def train(self):
        num_batches_total = 0
        for epoch in range(self.num_epochs):
            for _ in range(self.num_batches_per_epoch):
                self.model.train(True)
                self._do_training()
                num_batches_total += 1
            logger.push_prefix('Iteration #%d | ' % epoch)
            self.model.train(False)
            self.evaluate(epoch)
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

        next_obs_pred = self.model(obs, actions)
        errors = (next_obs - next_obs_pred)**2
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
            get_difference_statistics(statistics, ['Loss Mean', 'Errors Max'])
        )
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
                max_path_length=self.max_path_length_for_eval,
            )
            path['goal_states'] = np.tile(goal, (len(path['observations']), 1))
            paths.append(path)
        self.env.log_diagnostics(paths)

        logger.dump_tabular(with_prefix=False, with_timestamp=False)

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
        )

    def cuda(self):
        self.model.cuda()
