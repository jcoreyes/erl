from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import ReacherEnv

from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.envs.gym_env import convert_gym_space
from rllab.misc import logger


class MultitaskReacherEnv(ReacherEnv):
    def sample_goal_states(self, batch_size):
        return self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=(batch_size, 2)
        ) + self.init_qpos[-2:]

    def compute_rewards(self, obs, action, next_obs, goal_states):
        next_qpos = next_obs[:, 4:6]
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
