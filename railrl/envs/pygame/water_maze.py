from collections import OrderedDict

import numpy as np
from gym import Env

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from rllab.core.serializable import Serializable
from rllab.misc import logger
from sandbox.rocky.tf.spaces import Box


class WaterMaze(Serializable, Env):

    def __init__(
            self,
            horizon=200,
    ):
        Serializable.quick_init(self, locals())
        self.TARGET_RADIUS = 1
        self.BOUNDARY_DIST = 5
        self.BALL_RADIUS = 0.01
        self.MAX_TARGET_DISTANCE = self.BOUNDARY_DIST - self.TARGET_RADIUS

        self._horizon = horizon
        self._t = 0
        self._target_position = None
        self._position = None

        self._action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self._observation_space = Box(
            np.array([-self.BOUNDARY_DIST, -self.BOUNDARY_DIST, 0]),
            np.array([self.BOUNDARY_DIST, self.BOUNDARY_DIST, 1])
        )

    def _step(self, velocities):
        self._t += 1
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

    # def get_param_values(self):
    #     return None

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
