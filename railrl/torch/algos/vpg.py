from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import railrl
import railrl.torch.pytorch_util as ptu
from railrl.data_management.env_replay_buffer import VPGEnvReplayBuffer
from railrl.misc import eval_util
from railrl.misc.ml_util import (
    ConstantSchedule,
)
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from railrl.core import logger
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.distributions import TanhNormal


class VPG(TorchRLAlgorithm):
    """
    Vanilla Policy Gradient
    """

    def __init__(
            self,
            env,
            policy,
            policy_learning_rate=1e-4,
            epoch_discount_schedule=None,
            plotter=None,
            render_eval_paths=False,
            replay_buffer_class=VPGEnvReplayBuffer,
            **kwargs
    ):
        eval_policy = policy
        super().__init__(
            env,
            policy,
            eval_policy=eval_policy,
            collection_mode='batch',
            **kwargs
        )
        self.replay_buffer = replay_buffer_class(
            self.replay_buffer_size,
            env,
            self.discount,
        )
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        if epoch_discount_schedule is None:
            epoch_discount_schedule = ConstantSchedule(self.discount)
        self.epoch_discount_schedule = epoch_discount_schedule
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)
        self.eval_statistics = None

    def _start_epoch(self, epoch):
        super()._start_epoch(epoch)
        self.discount = self.epoch_discount_schedule.get_value(epoch)
        self.replay_buffer.empty_buffer()

    def get_batch(self, training=True):
        '''
        Should return everything in the replay buffer and empty it out
        :param training:
        :return:
        '''
        batch = self.replay_buffer.get_training_data()
        return np_to_pytorch_batch(batch)

    def _do_training(self):
        batch = self.get_batch(training=True)
        obs = batch['observations']
        actions = batch['actions']
        returns = batch['returns']
        """
        Policy operations.
        """

        _, means, _, _, _, stds,_, _ = self.policy.forward(obs,)
        log_probs = TanhNormal(means, stds).log_prob(actions)
        log_probs_times_returns = np.multiply(log_probs, returns)
        policy_loss = -1*np.mean(log_probs_times_returns)

        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))

    def _can_train(self):
        return True

    def _can_evaluate(self):
        return True

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)

        average_returns = railrl.misc.eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        if self.render_eval_paths:
            self.env.render_paths(test_paths)

        if self.plotter:
            self.plotter.draw()


    def offline_evaluate(self, epoch):
        statistics = OrderedDict()
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def get_epoch_snapshot(self, epoch):
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
            policy=self.eval_policy,
            trained_policy=self.policy,
            exploration_policy=self.exploration_policy,
            batch_size=self.batch_size,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    @property
    def networks(self):
        return [
            self.policy,
        ]
