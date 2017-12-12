import abc
from collections import OrderedDict

import numpy as np

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from rllab.misc import logger
from gym.envs.mujoco import Walker2dEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv
from rllab.core.serializable import Serializable


class MultitaskWalker2D(
    Walker2dEnv, MultitaskEnv, Serializable, metaclass=abc.ABCMeta
):
    def __init__(self):
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(self)
        Walker2dEnv.__init__(self)

    def log_diagnostics(self, paths):
        MultitaskEnv.log_diagnostics(self, paths)
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

    def __setstate__(self, state):
        Serializable.__setstate__(self, state)

    def __getstate__(self):
        return Serializable.__getstate__(self)


class Walker2DTargetXPos(MultitaskWalker2D):
    def __init__(self, max_distance=10):
        Serializable.quick_init(self, locals())
        super().__init__()
        self.set_goal(np.array([max_distance]))
        self.max_distance = max_distance

    def sample_goals(self, batch_size):
        return np.random.uniform(
            -self.max_distance,
            self.max_distance,
            (batch_size, 1),
        )

    def convert_obs_to_goals(self, obs):
        return obs[:, -3:-2]

    @property
    def goal_dim(self) -> int:
        return 1

    def _step(self, action):
        ob, _, done, info_dict = super()._step(action)
        xpos = ob[-3]
        xpos_error = np.linalg.norm(xpos - self.multitask_goal)
        reward = - xpos_error
        info_dict['xpos'] = xpos
        info_dict['desired_xpos'] = xpos_error
        info_dict['xpos_error'] = xpos_error
        return ob, reward, done, info_dict

    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        x_positions = get_stat_in_paths(
            paths, 'env_infos', 'xpos'
        )
        desired_x_positions = get_stat_in_paths(
            paths, 'env_infos', 'desired_xpos'
        )
        xpos_errors = get_stat_in_paths(
            paths, 'env_infos', 'xpos_error'
        )

        statistics = OrderedDict()
        for stat, name in [
            (x_positions, 'xpos'),
            (desired_x_positions, 'desired xpos'),
            (xpos_errors, 'xpos errors'),
        ]:
            statistics.update(create_stats_ordered_dict(
                '{}'.format(name),
                np.hstack(stat),
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)
