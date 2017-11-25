from collections import OrderedDict

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from rllab.misc import logger as rllab_logger


def obs_to_goal(obs):
    return obs[6:8]


def get_sparse_reward(obs):
    """-1 if far, 0 if close"""
    cylinder_pos = obs[8:10]
    goal_pos = obs[10:12]
    r = np.linalg.norm(cylinder_pos - goal_pos) < 0.1
    return (r - 1).astype(float)


def pusher2d_cost_fn(states, actions, next_states):
    input_is_flat = len(states.shape) == 1
    if input_is_flat:
        states = np.expand_dims(states, 0)

    hand_pos = states[:, 6:8]
    cylinder_pos = states[:, 8:10]
    target_pos = states[:, 10:12]
    hand_to_puck_dist = np.linalg.norm(
        hand_pos - cylinder_pos,
        axis=1,
        ord=2
    )
    costs = hand_to_puck_dist
    hand_is_close_to_puck = hand_to_puck_dist <= 0.1
    puck_to_goal_dist = np.linalg.norm(
        cylinder_pos - target_pos,
        axis=1,
        ord=2,
    )
    costs += (puck_to_goal_dist - 2) * hand_is_close_to_puck
    if input_is_flat:
        costs = costs[0]
    return costs


class Pusher2DEnv(
    mujoco_env.MujocoEnv, utils.EzPickle

):
    def __init__(self):
        self._target_cylinder_position = np.zeros(2)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            get_asset_xml('3link_gripper_push_2d.xml'),  # path to xml
            5,  # frameskip
        )
        self.obs_to_goal = obs_to_goal
        self.goal_idx = slice(10, 12)

    def get_reward(self, obs):
        return get_sparse_reward(obs)

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
        self._target_cylinder_position = np.random.uniform(
            np.array([-1, -1]),
            np.array([1, 0]),
            2
        )
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = np.zeros(2)   # ignore for now
        qvel = self.init_qvel.copy().squeeze()
        qvel[:] = 0

        self.set_state(qpos, qvel)

        return self._get_obs()

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
        reward = - hand_to_object_distance
        if hand_to_object_distance <= 0.1:
            reward += 2 - object_to_goal_distance

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            hand_to_hand_goal_distance=hand_to_hand_goal_distance,
            hand_to_object_distance=hand_to_object_distance,
            object_to_goal_distance=object_to_goal_distance,
        )

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:3],
            self.model.data.qvel.flat[:3],
            self.model.data.site_xpos[0][:2],
            self.get_body_com("object")[:2],
            self._target_cylinder_position,
        ])

    def log_diagnostics(self, paths, logger=None):
        final_hand_to_object_dist = get_stat_in_paths(
            paths, 'env_infos', 'hand_to_object_distance'
        )[:, -1]
        final_object_to_goal_dist = get_stat_in_paths(
            paths, 'env_infos', 'object_to_goal_distance'
        )[:, -1]
        final_hand_to_hand_goal_dist = get_stat_in_paths(
            paths, 'env_infos', 'hand_to_hand_goal_distance'
        )[:, -1]

        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Final Euclidean distance to goal',
            final_object_to_goal_dist,
            always_show_all_stats=True,
        ))
        statistics.update(create_stats_ordered_dict(
            'Final Euclidean distance hand to object',
            final_hand_to_object_dist,
            always_show_all_stats=True,
        ))
        statistics.update(create_stats_ordered_dict(
            'Final Euclidean distance hand to hand goal',
            final_hand_to_hand_goal_dist,
            always_show_all_stats=True,
        ))
        for key, value in statistics.items():
            if logger is None:
                rllab_logger.record_tabular(key, value)
            else:
                logger.log_tabular(key, value)

