import abc
from collections import OrderedDict

import numpy as np

from railrl.envs.mujoco.mujoco_env import MujocoEnv
from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths
from railrl.core import logger as default_logger


class Pusher2DEnv(MujocoEnv, metaclass=abc.ABCMeta):
    FILE = '3link_gripper_push_2d.xml'

    def __init__(self, goal=(-1, 0), randomize_goals=False,
                 use_hand_to_obj_reward=True):
        self.init_serialization(locals())
        if not isinstance(goal, np.ndarray):
            goal = np.array(goal)
        self._target_cylinder_position = goal
        self._target_hand_position = goal
        self.randomize_goals = randomize_goals
        self.use_hand_to_obj_reward = use_hand_to_obj_reward
        super().__init__(
            '3link_gripper_push_2d.xml',
            frame_skip=5,
            automatically_set_obs_and_action_space=True,
        )

    def _step(self, a):
        hand_to_object_distance = np.linalg.norm(
            self.model.data.site_xpos[0][:2] - self.get_body_com("object")[:2]
        )
        object_to_goal_distance = np.linalg.norm(
            self.get_body_com("goal") - self.get_body_com("object")
        )
        hand_to_hand_goal_distance = np.linalg.norm(
            self.model.data.site_xpos[0][:2] - self.get_body_com("hand_goal")[:2]
        )
        reward = - object_to_goal_distance
        if self.use_hand_to_obj_reward:
            reward = reward - hand_to_object_distance

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            hand_to_hand_goal_distance=hand_to_hand_goal_distance,
            hand_to_object_distance=hand_to_object_distance,
            object_to_goal_distance=object_to_goal_distance,
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        rotation_angle = 90
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset_model(self):
        qpos = (
            np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos.squeeze()
        )
        qpos[-3:] = self.init_qpos.squeeze()[-3:]
        # Object position
        obj_pos = np.random.uniform(
            #         x      y
            np.array([0.3, -0.8]),
            np.array([0.8, -0.3]),
        )
        qpos[-6:-4] = obj_pos
        if self.randomize_goals:
            self._target_cylinder_position = np.random.uniform(
                np.array([-1, -1]),
                np.array([0, 0]),
                2
            )
        self._target_hand_position = self._target_cylinder_position
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        qvel = self.init_qvel.copy().squeeze()
        qvel[:] = 0

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:3],
            self.model.data.qvel.flat[:3],
            self.model.data.site_xpos[0][:2],
            self.get_body_com("object")[:2],
        ])

    def log_diagnostics(self, paths, logger=default_logger):
        statistics = OrderedDict()
        for stat_name_in_paths, stat_name_to_print in [
            ('hand_to_object_distance', 'Distance hand to object'),
            ('object_to_goal_distance', 'Distance object to goal'),
            ('hand_to_hand_goal_distance', 'Distance hand to hand goal'),
        ]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_to_print,
                stats,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_to_print,
                final_stats,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)


class RandomGoalPusher2DEnv(Pusher2DEnv):
    def __init__(self, goal=(-1, 0)):
        self.init_serialization(locals())
        super().__init__(goal, randomize_goals=True)
