from collections import OrderedDict

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.misc import logger


class SimpleReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    R1 = 0.1  # from reacher.xml
    R2 = 0.11

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        # Make it so that your actions (torque) actually affect the next
        # observation position.
        self.do_simulation(np.zeros_like(a), self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1,
                                      size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005,
                                                       size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
        ])

    def sample_goal_states(self, batch_size):
        return 0.2 * np.ones((batch_size, 2))
        # return self.np_random.uniform(
        #     low=-0.2,
        #     high=0.2,
        #     size=(batch_size, 2)
        # )

    def compute_rewards(self, obs, action, next_obs, goal_states):
        c1 = next_obs[:, 0:1]  # cosine of angle 1
        c2 = next_obs[:, 1:2]
        s1 = next_obs[:, 2:3]
        s2 = next_obs[:, 3:4]
        next_qpos = (  # forward kinematics equation for 2-link robot
            self.R1 * np.hstack([c1, s1])
            + self.R2 * np.hstack([
                c1 * c2 - s1 * s2,
                s1 * c2 + c1 * s2,
            ])
        )
        return -np.linalg.norm(next_qpos - goal_states, axis=1)

    def log_diagnostics(self, paths):
        distance = [
            np.linalg.norm(path["observations"][-1][-3:])
            for path in paths
        ]

        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Distance to target', distance
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    @property
    def goal_dim(self):
        return 2