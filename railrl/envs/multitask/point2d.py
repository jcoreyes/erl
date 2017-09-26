from collections import OrderedDict

import numpy as np
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.envs.pygame.water_maze import WaterMaze
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_stat_in_dict
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.spaces import Box


class MultitaskPoint2DEnv(WaterMaze, MultitaskEnv, Serializable):
    def __init__(
            self,
            horizon=200,
            render_dt_msec=30,
            give_time=False,
            action_l2norm_penalty=0,
    ):
        Serializable.quick_init(self, locals())
        super().__init__(
            horizon=horizon,
            render_dt_msec=render_dt_msec,
            give_time=give_time,
            action_l2norm_penalty=action_l2norm_penalty,
        )
        self._multitask_goal = np.random.uniform(
            size=2, low=-self.MAX_TARGET_DISTANCE, high=self.MAX_TARGET_DISTANCE
        )

    def set_goal(self, goal):
        self._multitask_goal = goal

    def sample_states(self, batch_size):
        return np.random.uniform(
            low=-self.BOUNDARY_DIST,
            high=self.BOUNDARY_DIST,
            size=(batch_size, 2)
        )

    @property
    def goal_dim(self):
        return 2

    def sample_actions(self, batch_size):
        return np.random.uniform(
            low=-1,
            high=1,
            size=(batch_size, 2)
        )

    def sample_goal_states(self, batch_size):
        return self.sample_states(batch_size)

    def _reset(self):
        self._target_position = self._multitask_goal
        self._position = np.random.uniform(
            size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
        )
        self._t = 0
        return self._get_observation_and_on_platform()[0]

    def _get_observation_and_on_platform(self):
        return self._position, False

    def _create_observation_space(self):
        low = np.array([-self.BOUNDARY_DIST, -self.BOUNDARY_DIST])
        high = np.array([self.BOUNDARY_DIST, self.BOUNDARY_DIST])
        return Box(low, high)

    def log_diagnostics(self, paths, **kwargs):
        distance_to_target = get_stat_in_dict(
            paths, 'env_infos', 'distance_to_target'
        )
        actions = np.vstack([path['actions'] for path in paths])
        statistics = OrderedDict()
        for name, stat in [
            ('Euclidean distance to goal', distance_to_target),
            ('Actions', actions),
        ]:
            statistics.update(create_stats_ordered_dict(name, stat))
        statistics.update(create_stats_ordered_dict(
            'Final Euclidean distance to goal',
            distance_to_target[:, -1],
            always_show_all_stats=True,
        ))


        for key, value in statistics.items():
            logger.record_tabular(key, value)
