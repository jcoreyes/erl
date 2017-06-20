from collections import OrderedDict

import numpy as np
from gym import Env
from pygame import Color

from railrl.envs.pygame.pygame_viewer import PygameViewer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.spaces import Box


class WaterMaze(Serializable, Env):
    TARGET_RADIUS = 2
    BOUNDARY_DIST = 5
    BALL_RADIUS = 0.25

    def __init__(
            self,
            horizon=200,
            render_dt_msec=30,
    ):
        Serializable.quick_init(self, locals())
        self.MAX_TARGET_DISTANCE = self.BOUNDARY_DIST - self.TARGET_RADIUS

        self._horizon = horizon
        self._t = 0
        self._target_position = None
        self._position = None

        self._action_space = self._create_action_space()
        self._observation_space = self._create_observation_space()

        self.drawer = None
        self.render_dt_msec = render_dt_msec

    def _create_action_space(self):
        return Box(np.array([-1, -1]), np.array([1, 1]))

    def _create_observation_space(self):
        return Box(
            np.array([-self.BOUNDARY_DIST, -self.BOUNDARY_DIST, 0]),
            np.array([self.BOUNDARY_DIST, self.BOUNDARY_DIST, 1])
        )

    def _step(self, velocities):
        self._t += 1
        velocities = np.clip(velocities, a_min=-1, a_max=1)
        self._position += velocities
        self._position = np.clip(
            self._position,
            a_min=-self.BOUNDARY_DIST,
            a_max=self.BOUNDARY_DIST,
        )
        observation, on_platform = self._get_observation_and_on_platform()

        reward = float(on_platform)
        done = self._t >= self.horizon
        info = {
            'radius': self.TARGET_RADIUS,
            'target_position': self._target_position,
        }
        return observation, reward, done, info

    def _reset(self):
        self._target_position = np.random.uniform(
            size=2, low=-self.MAX_TARGET_DISTANCE, high=self.MAX_TARGET_DISTANCE
        )
        self._position = np.random.uniform(
            size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
        )
        self._t = 0
        return self._get_observation_and_on_platform()[0]

    def _get_observation_and_on_platform(self):
        """
        :return: Tuple
        - Observation vector
        - Flag: on platform or not.
        """
        dist = np.linalg.norm(self._position - self._target_position)
        on_platform = dist <= self.TARGET_RADIUS
        return np.hstack((self._position, [on_platform])), on_platform

    def log_diagnostics(self, paths, **kwargs):
        list_of_rewards, terminals, obs, actions, next_obs = split_paths(paths)

        returns = []
        for rewards in list_of_rewards:
            returns.append(np.sum(rewards))
        last_statistics = OrderedDict()
        last_statistics.update(create_stats_ordered_dict(
            'UndiscountedReturns',
            returns,
        ))
        last_statistics.update(create_stats_ordered_dict(
            'Rewards',
            list_of_rewards,
        ))
        last_statistics.update(create_stats_ordered_dict(
            'Actions',
            actions,
        ))

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)
        return returns

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def horizon(self):
        return self._horizon

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


class WaterMazeEasy(WaterMaze):
    """
    See the target position at all time steps.
    """
    def _create_observation_space(self):
        return Box(
            np.array([-self.BOUNDARY_DIST, -self.BOUNDARY_DIST, 0,
                      -self.BOUNDARY_DIST, -self.BOUNDARY_DIST]),
            np.array([self.BOUNDARY_DIST, self.BOUNDARY_DIST, 1,
                      self.BOUNDARY_DIST, self.BOUNDARY_DIST])
        )

    def _get_observation_and_on_platform(self):
        """
        :return: Tuple
        - Observation vector
        - Flag: on platform or not.
        """
        dist = np.linalg.norm(self._position - self._target_position)
        on_platform = dist <= self.TARGET_RADIUS
        return np.hstack(
            (self._position, [on_platform], self._target_position)
        ), on_platform


class WaterMazeMemory(WaterMazeEasy):
    """
    See the target position at the very first time step.
    """
    def _get_observation_and_on_platform(self):
        """
        :return: Tuple
        - Observation vector
        - Flag: on platform or not.
        """
        dist = np.linalg.norm(self._position - self._target_position)
        on_platform = dist <= self.TARGET_RADIUS
        if self._t == 0:
            hint = self._target_position
        else:
            hint = np.zeros_like(self._target_position)
        return np.hstack((self._position, [on_platform], hint)), on_platform


class WaterMaze1D(WaterMaze):
    def _create_action_space(self):
        return Box(np.array([-1]), np.array([1]))

    def _create_observation_space(self):
        return Box(
            np.array([-self.BOUNDARY_DIST, 0]),
            np.array([self.BOUNDARY_DIST, 1])
        )

    def _step(self, velocity):
        velocities = np.hstack((velocity, 0))
        return super()._step(velocities)

    def _get_observation_and_on_platform(self):
        dist = np.linalg.norm(self._position - self._target_position)
        on_platform = dist <= self.TARGET_RADIUS
        return np.hstack((self._position[0], [on_platform])), on_platform

    def _reset(self):
        self._target_position = np.random.uniform(
            size=2, low=-self.MAX_TARGET_DISTANCE, high=self.MAX_TARGET_DISTANCE
        )
        self._position = np.random.uniform(
            size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
        )
        self._target_position[1] = 0
        self._position[1] = 0
        self._t = 0
        return self._get_observation_and_on_platform()[0]


class WaterMazeEasy1D(WaterMaze1D):
    """
    See the target position at all time steps.
    """
    def _create_observation_space(self):
        return Box(
            np.array([-self.BOUNDARY_DIST, 0, -self.BOUNDARY_DIST]),
            np.array([self.BOUNDARY_DIST, 1, self.BOUNDARY_DIST]),
        )

    def _get_observation_and_on_platform(self):
        """
        :return: Tuple
        - Observation vector
        - Flag: on platform or not.
        """
        dist = np.linalg.norm(self._position - self._target_position)
        on_platform = dist <= self.TARGET_RADIUS
        return np.hstack(
            (self._position[0], [on_platform], self._target_position[0])
        ), on_platform


class WaterMazeMemory1D(WaterMazeEasy1D):
    """
    See the target position at the very first time step.
    """
    def _get_observation_and_on_platform(self):
        """
        :return: Tuple
        - Observation vector
        - Flag: on platform or not.
        """
        dist = np.linalg.norm(self._position - self._target_position)
        on_platform = dist <= self.TARGET_RADIUS
        if self._t == 0:
            print("hit")
            hint = self._target_position[0]
        else:
            hint = np.zeros_like(self._target_position[0])
        return np.hstack((self._position[0], [on_platform], hint)), on_platform
