from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import HopperEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from rllab.core.serializable import Serializable
from rllab.misc import logger as rllab_logger


class GoalXPosHopper(HopperEnv, MultitaskEnv, Serializable):
    def __init__(self, max_distance=5):
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(self)
        super().__init__()
        self.max_distance = max_distance
        self.set_goal(np.array([self.max_distance]))

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10),
            self.get_body_com("torso").flat,
        ])

    def _step(self, action):
        ob, reward, done, info_dict = super()._step(action)
        x_pos = self.convert_ob_to_goal(ob)
        distance_to_goal = np.linalg.norm(x_pos - self.multitask_goal)
        reward = -distance_to_goal
        return ob, reward, done, dict(
            goal=self.multitask_goal,
            distance_to_goal=distance_to_goal,
            position=x_pos,
            **info_dict
        )

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goals(self, batch_size):
        return np.random.uniform(
            -self.max_distance,
            self.max_distance,
            (batch_size, 1),
        )

    def convert_obs_to_goals(self, obs):
        return obs[:, 11:12]

    def log_diagnostics(self, paths, logger=rllab_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)
        distances_to_goal = get_stat_in_paths(
            paths, 'env_infos', 'distance_to_goal'
        )
        goal_positions = get_stat_in_paths(
            paths, 'env_infos', 'goal'
        )
        positions = get_stat_in_paths(
            paths, 'env_infos', 'position'
        )
        statistics = OrderedDict()
        for stat, name in [
            (distances_to_goal, 'Distance to goal'),
            (goal_positions, 'Goal Position'),
            (positions, 'Position'),
        ]:
            statistics.update(create_stats_ordered_dict(
                '{}'.format(name),
                stat,
                always_show_all_stats=True,
            ))
        for stat, name in [
            (distances_to_goal, 'Distance to goal'),
            (positions, 'Position'),
        ]:
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)
