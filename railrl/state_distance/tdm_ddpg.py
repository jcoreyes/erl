from collections import OrderedDict

import numpy as np
import torch

import railrl.torch.pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.state_distance.exploration import MakeUniversal
from railrl.state_distance.rollout_util import MultigoalSimplePathSampler
from railrl.state_distance.tdm import TemporalDifferenceModel
from railrl.torch.algos.ddpg import DDPG


class TdmDdpg(TemporalDifferenceModel, DDPG):
    def __init__(
            self,
            env,
            qf,
            exploration_policy,
            ddpg_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
    ):
        # super().__init__(env, qf, **tdm_kwargs)
        super().__init__(**tdm_kwargs)
        DDPG.__init__(
            self,
            env=env,
            qf=qf,
            policy=policy,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            **ddpg_kwargs,
            **base_kwargs
        )
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
        DDPG._do_training(self)

    def evaluate(self, epoch):
        DDPG.evaluate(self, epoch)
