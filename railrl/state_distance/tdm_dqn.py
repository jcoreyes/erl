from collections import OrderedDict

import numpy as np
import torch

import railrl.torch.pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.state_distance.exploration import MakeUniversal
from railrl.state_distance.rollout_util import MultigoalSimplePathSampler
from railrl.state_distance.tdm import TemporalDifferenceModel
from railrl.torch.algos.dqn import DQN


class TdmDqn(TemporalDifferenceModel, DQN):
    def __init__(
            self,
            env,
            qf,
            dqn_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
    ):
        super().__init__(env, qf, **tdm_kwargs)
        DQN.__init__(self, env, qf, replay_buffer=replay_buffer,
                     policy=policy,
                     **dqn_kwargs,
                     **base_kwargs)
        self.policy = MakeUniversal(self.policy)
        self.eval_policy = MakeUniversal(self.eval_policy)
        self.exploration_policy = MakeUniversal(self.exploration_policy)
        self.eval_sampler = MultigoalSimplePathSampler(
            env=env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            discount_sampling_function=self._sample_max_tau_for_rollout,
            goal_sampling_function=self._sample_goal_for_rollout,
            cycle_taus_for_rollout=False,
        )

    def _do_training(self):
        if not self.vectorized:
            DQN._do_training(self)
        batch = self.get_batch(training=True)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Compute loss
        """

        target_q_values = self.target_qf(next_obs).detach().max(
            1, keepdim=False
        )[0]
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions.unsqueeze(2), dim=1,
                           keepdim=False)
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))

    def evaluate(self, epoch):
        DQN.evaluate(self, epoch)
