from collections import OrderedDict

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from railrl.core import logger as default_logger


def obs_to_goal(obs):
    return obs[14:17]


def get_sparse_reward(obs):
    """-1 if far, 0 if close"""
    hand_pos = obs[14:17]
    goal_pos = obs[17:20]
    r = np.linalg.norm(hand_pos - goal_pos) < 0.1
    return (r - 1).astype(float)


def reacher7dof_cost_fn(states, actions, next_states):
    input_is_flat = len(states.shape) == 1
    if input_is_flat:
        states = np.expand_dims(states, 0)

    hand_pos = states[:, 14:17]
    target_pos = states[:, 17:20]
    costs = np.linalg.norm(
        hand_pos - target_pos,
        axis=1,
        ord=2
    )
    if input_is_flat:
        costs = costs[0]
    return costs


class Reacher7Dof(
    mujoco_env.MujocoEnv, utils.EzPickle

):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            get_asset_xml('reacher_7dof.xml'),  # path to xml
            5,  # frameskip
        )
        self._desired_xyz = np.zeros(3)
        self.obs_to_goal = obs_to_goal
        self.goal_idx = slice(17, 20)

    def get_reward(self, obs):
        return get_sparse_reward(obs)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        self._desired_xyz = np.random.uniform(
            np.array([-0.75, -1.25, -0.24]),
            np.array([0.75, 0.25, 0.6]),
        )
        qpos[-7:-4] = self._desired_xyz
        qvel[-7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _step(self, a):
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        reward = - distance
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(distance=distance)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("goal"),
        ])

    def log_diagnostics(self, paths, logger=default_logger):
        statistics = OrderedDict()

        euclidean_distances = get_stat_in_paths(
            paths, 'env_infos', 'distance'
        )
        statistics.update(create_stats_ordered_dict(
            'Euclidean distance to goal', euclidean_distances
        ))
        statistics.update(create_stats_ordered_dict(
            'Final Euclidean distance to goal',
            euclidean_distances[:, -1],
            always_show_all_stats=True,
        ))
        for key, value in statistics.items():
            logger.log_tabular(key, value)
