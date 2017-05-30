from collections import deque, OrderedDict

import numpy as np

from railrl.envs.mujoco_env import MujocoEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from rllab.envs.env_spec import EnvSpec
from rllab.misc import logger
from sandbox.rocky.tf.spaces.box import Box


class WaterMaze(MujocoEnv):
    def __init__(self, horizon=200, l2_action_penalty_weight=1e-2,
                 include_velocity=False,
                 use_small_maze=False,
                 **kwargs):
        if use_small_maze:
            self.TARGET_RADIUS = 0.04
            self.BOUNDARY_RADIUS = 0.02
            self.BOUNDARY_DIST = 0.12
            self.BALL_RADIUS = 0.01
            super().__init__('small_water_maze.xml')
        else:
            self.TARGET_RADIUS = 0.1
            self.BOUNDARY_RADIUS = 0.02
            self.BOUNDARY_DIST = 0.3
            self.BALL_RADIUS = 0.02
            super().__init__('water_maze.xml')
        self.BALL_START_DIST = (
            self.BOUNDARY_DIST - self.BOUNDARY_RADIUS - 2 * self.BALL_RADIUS
        )
        self.MAX_GOAL_DIST = self.BOUNDARY_DIST - self.BOUNDARY_RADIUS
        self.l2_action_penalty_weight = l2_action_penalty_weight
        self.horizon = horizon
        self._t = 0
        self._on_platform_history = deque(maxlen=5)
        for _ in range(5):
            self._on_platform_history.append(False)
        self.include_velocity = include_velocity

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = self._create_observation_space()
        self.spec = EnvSpec(
            self.observation_space,
            self.action_space,
        )
        self.reset_model()

    def _create_observation_space(self):
        num_obs = 4 if self.include_velocity else 2
        return Box(
            np.hstack((-np.inf + np.zeros(num_obs), [0])),
            np.hstack((np.inf + np.zeros(num_obs), [1])),
        )

    def _step(self, force_actions):
        self._t += 1
        mujoco_action = np.hstack([force_actions, [0, 0]])
        self.do_simulation(mujoco_action, self.frame_skip)
        observation = self._get_observation()

        on_platform = observation[2]
        self._on_platform_history.append(on_platform)

        if all(self._on_platform_history):
            self.reset_ball_position()

        reward = (
            on_platform
            - self.l2_action_penalty_weight * np.linalg.norm(force_actions)
        )
        done = self._t >= self.horizon
        info = {
            'radius': self.TARGET_RADIUS,
            'target_position': self._get_target_position(),
        }
        return observation, reward, done, info

    def reset_ball_position(self):
        new_ball_position = self.np_random.uniform(
            size=2, low=-self.BALL_START_DIST, high=self.BALL_START_DIST
        )
        target_position = self._get_target_position()
        qvel = np.zeros(self.model.nv)
        new_pos = np.hstack((new_ball_position, target_position))
        self.set_state(new_pos, qvel)

    def reset_model(self):
        qpos = self.np_random.uniform(size=self.model.nq, low=-self.MAX_GOAL_DIST,
                                      high=self.MAX_GOAL_DIST)
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        self.reset_ball_position()
        self._t = 0
        return self._get_observation()

    def _get_observation(self):
        position = np.concatenate([self.model.data.qpos]).ravel()[:2]
        velocity = np.concatenate([self.model.data.qvel]).ravel()[:2]
        target_position = self._get_target_position()
        dist = np.linalg.norm(position - target_position)
        on_platform = dist <= self.TARGET_RADIUS
        if self.include_velocity:
            return np.hstack((position, velocity, [on_platform]))
        else:
            return np.hstack((position, [on_platform]))

    def _get_target_position(self):
        return np.concatenate([self.model.data.qpos]).ravel()[2:]

    def viewer_setup(self):
        v = self.viewer
        # v.cam.trackbodyid=0
        # v.cam.distance = v.model.stat.extent

    def get_tf_loss(self, observations, actions, target_labels, **kwargs):
        """
        Return the supervised-learning loss.
        :param observation: Tensor
        :param action: Tensor
        :return: loss Tensor
        """
        return -(actions + observations - target_labels) ** 2

    def get_param_values(self):
        return None

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

    def terminate(self):
        self.close()

    @staticmethod
    def get_extra_info_dict_from_batch(batch):
        return {}

    @staticmethod
    def get_flattened_extra_info_dict_from_subsequence_batch(batch):
        return {}

    @staticmethod
    def get_last_extra_info_dict_from_subsequence_batch(batch):
        return {}


class WaterMazeEasy(WaterMaze):
    """
    Always see the target position.
    """
    def _create_observation_space(self):
        num_obs = 4 if self.include_velocity else 2
        return Box(
            np.hstack((-np.inf + np.zeros(num_obs), [0], [-self.BOUNDARY_DIST,
                                                          -self.BOUNDARY_DIST])),
            np.hstack((np.inf + np.zeros(num_obs), [1], [self.BOUNDARY_DIST,
                                                         self.BOUNDARY_DIST])),
        )

    def _get_observation(self):
        obs = super()._get_observation()
        target_position = self._get_target_position()
        return np.hstack((obs, target_position))


class WaterMazeMemory(WaterMaze):
    """
    See the target position at the very first time step.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zeros = np.zeros(2)

    def _create_observation_space(self):
        num_obs = 4 if self.include_velocity else 2
        return Box(
            np.hstack((-np.inf + np.zeros(num_obs), [0], [-self.BOUNDARY_DIST,
                                                          -self.BOUNDARY_DIST])),
            np.hstack((np.inf + np.zeros(num_obs), [1], [self.BOUNDARY_DIST,
                                                         self.BOUNDARY_DIST])),
        )

    def _get_observation(self):
        obs = super()._get_observation()
        if self._t == 0:
            target_position = self._get_target_position()
        else:
            target_position = self.zeros
        return np.hstack((obs, target_position))


def make_heat_map(eval_func, resolution=50):
    linspace = np.linspace(-0.3, 0.3, num=resolution)
    map = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            map[i, j] = eval_func(np.array([linspace[i], linspace[j]]))
    return map


def make_density_map(paths, resolution=50):
    linspace = np.linspace(-0.3, 0.3, num=resolution + 1)
    y = paths[:, 0]
    x = paths[:, 1]
    H, xedges, yedges = np.histogram2d(y, x, bins=(linspace, linspace))
    H = H.astype(np.float)
    H = H / np.max(H)
    return H


def plot_maps(old_combined=None, *heatmaps):
    import matplotlib.pyplot as plt
    combined = np.c_[heatmaps]
    if old_combined is not None:
        combined = np.r_[old_combined, combined]
    plt.figure()
    plt.imshow(combined, cmap='afmhot', interpolation='none')
    plt.show()
    return combined


if __name__ == "__main__":
    def evalfn(a):
        return np.linalg.norm(a - np.array([0, 0]))


    hm = make_heat_map(evalfn, resolution=50)
    paths = np.random.randn(5000, 2) * 0.1
    dm = make_density_map(paths, resolution=50)
    a = plot_maps(None, hm, dm)
    plot_maps(a, hm, dm)
