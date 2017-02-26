"""
:author: Vitchyr Pong
"""
from collections import OrderedDict

import numpy as np

from railrl.algos.ddpg import DDPG
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides

TARGET_PREFIX = "target_"


class DdpgOcm(DDPG):
    """
    Deep Deterministic Policy Gradient for one character memory task.
    """

    @overrides
    def _get_training_ops(self, epoch=None):
        ops = [
            self.train_qf_op,
            self.update_target_qf_op,
        ]
        # if epoch > 50:
        if True:
            ops += [
                self.train_policy_op,
                self.update_target_policy_op,
            ]
        if self._batch_norm:
            ops += self.qf.batch_norm_update_stats_op
            ops += self.policy.batch_norm_update_stats_op
        return ops

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.training_env,
            epoch=epoch,
            policy=self.policy,
            es=self.exploration_strategy,
            qf=self.qf,
        )
