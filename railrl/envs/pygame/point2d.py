from collections import OrderedDict

import numpy as np
from gym import Env
from gym.spaces import Box
from pygame import Color

from railrl.core import logger as default_logger
from railrl.core.serializable import Serializable
from railrl.envs.pygame.pygame_viewer import PygameViewer
from railrl.misc.eval_util import create_stats_ordered_dict, get_path_lengths, \
    get_stat_in_paths
import railrl.misc.visualization_util as vu


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
        distance_reward = -distance_to_target
        action_reward = -np.linalg.norm(velocities) * self.action_l2norm_penalty
        reward += distance_reward + action_reward
        done = on_platform
        info = {
            'radius': self.TARGET_RADIUS,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
            'distance_reward': distance_reward,
            'action_reward': action_reward,
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
            ('distance_reward', 'Distance Reward'),
            ('action_reward', 'Action Reward'),
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


def plot_observations_and_actions(observations, actions):
    import matplotlib.pyplot as plt

    x_pos = observations[:, 0]
    y_pos = observations[:, 1]
    H, xedges, yedges = np.histogram2d(x_pos, y_pos)
    heatmap = vu.HeatMap(H, xedges, yedges, {})
    plt.subplot(2, 1, 1)
    plt.title("Observation Distribution")
    plt.xlabel("0-Dimenion")
    plt.ylabel("1-Dimenion")
    vu.plot_heatmap(heatmap)

    x_actions = actions[:, 0]
    y_actions = actions[:, 1]
    H, xedges, yedges = np.histogram2d(x_actions, y_actions)
    heatmap = vu.HeatMap(H, xedges, yedges, {})
    plt.subplot(2, 1, 2)
    plt.title("Action Distribution")
    plt.xlabel("0-Dimenion")
    plt.ylabel("1-Dimenion")
    vu.plot_heatmap(heatmap)

    plt.show()


if __name__ == "__main__":
    import argparse
    import joblib
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=30, help='Horizon for eval')
    args = parser.parse_args()

    data = joblib.load(args.file)
    replay_buffer = data['replay_buffer']
    max_i = replay_buffer._top - 1
    observations = replay_buffer._observations[:max_i, :]
    actions = replay_buffer._actions[:max_i, :]
    plot_observations_and_actions(observations, actions)
