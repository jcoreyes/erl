from collections import OrderedDict

import torch
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
import railrl.torch.pytorch_util as ptu
from railrl.core.serializable import Serializable
from railrl.core import logger as default_logger


class Reacher7DofMultitaskEnv(
    MultitaskEnv, mujoco_env.MujocoEnv, Serializable
):
    def __init__(self, distance_metric_order=None, goal_dim_weights=None):
        self._desired_xyz = np.zeros(3)
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(
            self,
            distance_metric_order=distance_metric_order,
            goal_dim_weights=goal_dim_weights,
        )
        mujoco_env.MujocoEnv.__init__(
            self,
            get_asset_xml('reacher_7dof.xml'),
            5,
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-7:] = 0
        self.set_state(qpos, qvel)
        self._set_goal_xyz(self._desired_xyz)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
        ])

    def _step(self, a):
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self._desired_xyz
        )
        reward = - distance
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            distance=distance,
            multitask_goal=self.multitask_goal,
            desired_xyz=self._desired_xyz,
            goal=self.multitask_goal,
        )

    def _set_goal_xyz(self, xyz_pos):
        current_qpos = self.model.data.qpos.flat
        current_qvel = self.model.data.qvel.flat.copy()
        new_qpos = current_qpos.copy()
        new_qpos[-7:-4] = xyz_pos
        self._desired_xyz = xyz_pos
        self.set_state(new_qpos, current_qvel)

    def log_diagnostics(self, paths, logger=default_logger):
        super().log_diagnostics(paths)
        statistics = OrderedDict()

        euclidean_distances = get_stat_in_paths(
            paths, 'env_infos', 'distance'
        )
        statistics.update(create_stats_ordered_dict(
            'Euclidean distance to goal', euclidean_distances
        ))
        statistics.update(create_stats_ordered_dict(
            'Final Euclidean distance to goal',
            [d[-1] for d in euclidean_distances],
            always_show_all_stats=True,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)


class Reacher7DofXyzGoalState(Reacher7DofMultitaskEnv):
    """
    The goal state is just the XYZ location of the end effector.
    """
    def sample_goals(self, batch_size):
        # Number taken from running a random policy and seeing what XYZ values
        # are reached
        return np.hstack((
            self.np_random.uniform(low=-0.75, high=0.75, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.25, high=0.25, size=(batch_size, 1)),
            self.np_random.uniform(low=-0.2, high=0.6, size=(batch_size, 1)),
        ))

    def set_goal(self, goal):
        super().set_goal(goal)
        self._set_goal_xyz(goal)

    @property
    def goal_dim(self):
        return 3

    def convert_obs_to_goals(self, obs):
        return obs[:, 14:17]


class Reacher7DofXyzPosAndVelGoalState(Reacher7DofMultitaskEnv):
    def __init__(
            self,
            speed_weight=0.9,
            done_threshold=0.05,
            max_speed=0.03,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        # TODO: fix this hack
        if speed_weight is None:
            self.speed_weight = 0.9  # just for init to work
        else:
            self.speed_weight = speed_weight
        self.done_threshold = done_threshold
        self.max_speed = max_speed
        self.initializing = True
        super().__init__(**kwargs)
        self.initializing = False
        if speed_weight is None:
            assert (
                self.goal_dim_weights[0] == self.goal_dim_weights[1] == self.goal_dim_weights[2]
            ) and (
                self.goal_dim_weights[3] == self.goal_dim_weights[4] == self.goal_dim_weights[5]
            )
            self.speed_weight = self.goal_dim_weights[3]
    """
    The goal state is just the XYZ location and velocity of the end effector.
    """
    def sample_goals(self, batch_size):
        # Number taken from running a random policy and seeing what XYZ values
        # are reached
        return np.random.uniform(
            np.array([
                -0.75,
                -1.25,
                -0.2,
                -self.max_speed,
                -self.max_speed,
                -self.max_speed],
            ),
            np.array([
                0.75,
                0.25,
                0.6,
                self.max_speed,
                self.max_speed,
                self.max_speed],
            ),
            (batch_size, 6)
        )

    def set_goal(self, goal):
        super().set_goal(goal)
        self._set_goal_xyz(goal[0:3])

    @property
    def goal_dim(self):
        return 6

    def convert_obs_to_goals(self, obs):
        return obs[:, 14:20]

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-7:] = 0
        self.set_state(qpos, qvel)
        self._set_goal_xyz(self.multitask_goal[0:3])
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            np.zeros(3),
        ])

    def _get_obs(self):
        raise NotImplementedError()

    def _step(self, action):
        old_xyz = self.get_body_com("tips_arm")
        self.do_simulation(action, self.frame_skip)
        new_xyz = self.get_body_com("tips_arm")
        xyz_vel = new_xyz - old_xyz
        ob = np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            xyz_vel,
        ])
        done = False

        # Compute rewards
        error = self.convert_ob_to_goal(ob) - self.multitask_goal
        pos_error = np.linalg.norm(error[:3])
        vel_error = np.linalg.norm(error[3:])
        weighted_vel_error = vel_error * self.speed_weight
        weighted_pos_error = pos_error * (1 - self.speed_weight)
        reward = - (weighted_pos_error + weighted_vel_error)
        if np.abs(reward) < self.done_threshold and not self.initializing:
            done = True
        return ob, reward, done, dict(
            goal=self.multitask_goal,
            vel_error=vel_error,
            pos_error=pos_error,
            distance=pos_error,
            desired_xyz=self._desired_xyz,
            weighted_vel_error=weighted_vel_error,
            weighted_pos_error=weighted_pos_error,
        )

    def log_diagnostics(self, paths, logger=default_logger):
        super().log_diagnostics(paths)

        statistics = OrderedDict()
        for stat_name in [
            'pos_error',
            'vel_error',
            'weighted_pos_error',
            'weighted_vel_error',
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '{}'.format(stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)


class Reacher7DofFullGoal(Reacher7DofMultitaskEnv):

    @property
    def goal_dim(self) -> int:
        return 17

    def sample_goals(self, batch_size):
        raise NotImplementedError()

    def convert_obs_to_goals(self, obs):
        return obs

    def set_goal(self, goal):
        super().set_goal(goal)
        self._set_goal_xyz(goal[14:17])

    def sample_goal_for_rollout(self):
        angles = np.random.uniform(
            np.array([-2.28, -0.52, -1.4, -2.32, -1.5, -1.094, -1.5]),
            np.array([1.71, 1.39, 1.7, 0,   1.5, 0,   1.5, ]),
        )

        saved_qpos = self.init_qpos.copy()
        saved_qvel = self.init_qvel.copy()
        qpos_tmp = saved_qpos.copy()
        qpos_tmp[:7] = angles
        self.set_state(qpos_tmp, saved_qvel)
        ee_pos = self.get_body_com("tips_arm")
        self.set_state(saved_qpos, saved_qvel)
        velocities = np.zeros(7)
        return np.hstack((
            angles,
            velocities,
            ee_pos,
        ))
