from collections import OrderedDict

import numpy as np
from gym import Env
from gym.spaces import Box
from pygame import Color

from railrl.core import logger as default_logger
from railrl.core.serializable import Serializable
from railrl.envs.pygame.pygame_viewer import PygameViewer
from railrl.misc.eval_util import create_stats_ordered_dict, get_path_lengths
from railrl.samplers.util import get_stat_in_paths


class Point2DEnv(Serializable, Env):
    """
    A little 2D point whose life goal is to reach a target.
    """
    TARGET_RADIUS = 0.5
    BOUNDARY_DIST = 5
    BALL_RADIUS = 0.25

    def __init__(
            self,
            render_dt_msec=30,
            action_l2norm_penalty=0,
    ):
        Serializable.quick_init(self, locals())
        self.MAX_TARGET_DISTANCE = self.BOUNDARY_DIST - self.TARGET_RADIUS

        self.action_l2norm_penalty = action_l2norm_penalty
        self._target_position = None
        self._position = None

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = Box(
            -self.BOUNDARY_DIST * np.ones(4),
            self.BOUNDARY_DIST * np.ones(4),
        )

        self.drawer = None
        self.render_dt_msec = render_dt_msec

    def _step(self, velocities):
        velocities = np.clip(velocities, a_min=-1, a_max=1)
        distance_to_target = np.linalg.norm(
            self._target_position - self._position
        )
        self._position += velocities
        self._position = np.clip(
            self._position,
            a_min=-self.BOUNDARY_DIST,
            a_max=self.BOUNDARY_DIST,
        )
        observation = self._get_observation()
        on_platform = self.is_on_platform()

        reward = float(on_platform)
        reward -= distance_to_target
        reward -= np.linalg.norm(velocities) * self.action_l2norm_penalty
        done = on_platform
        info = {
            'radius': self.TARGET_RADIUS,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
        }
        return observation, reward, done, info

    def is_on_platform(self):
        dist = np.linalg.norm(self._position - self._target_position)
        return dist <= self.TARGET_RADIUS

    def reset(self):
        self._target_position = np.random.uniform(
            size=2, low=-self.MAX_TARGET_DISTANCE, high=self.MAX_TARGET_DISTANCE
        )
        self._position = np.random.uniform(
            size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
        )
        while self.is_on_platform():
            self._target_position = np.random.uniform(
                size=2, low=-self.MAX_TARGET_DISTANCE,
                high=self.MAX_TARGET_DISTANCE
            )
            self._position = np.random.uniform(
                size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
            )
        return self._get_observation()

    def _get_observation(self):
        return np.hstack((self._position, self._target_position))

    def log_diagnostics(self, paths, logger=default_logger):
        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('distance_to_target', 'Distance to Target'),
            ('speed', 'Speed'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stat,
            ))
        statistics.update(create_stats_ordered_dict(
            "Path Lengths",
            get_path_lengths(paths),
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def render(self, mode='human', close=False):
        if close:
            self.drawer = None
            return

        if self.drawer is None or self.drawer.terminated:
            self.drawer = PygameViewer(
                500,
                500,
                x_bounds=(-self.BOUNDARY_DIST, self.BOUNDARY_DIST),
                y_bounds=(-self.BOUNDARY_DIST, self.BOUNDARY_DIST),
            )

        self.drawer.fill(Color('white'))
        self.drawer.draw_solid_circle(
            self._target_position,
            self.TARGET_RADIUS,
            Color('green'),
        )
        self.drawer.draw_solid_circle(
            self._position,
            self.BALL_RADIUS,
            Color('blue'),
        )

        self.drawer.render()
        self.drawer.tick(self.render_dt_msec)
