from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import ReacherEnv

from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.envs.gym_env import convert_gym_space
from rllab.misc import logger


class MultitaskReacherEnv(ReacherEnv):
    R1 = 0.1  # from reacher.xml
    R2 = 0.11

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
