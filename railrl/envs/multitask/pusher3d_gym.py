from collections import OrderedDict

import numpy as np

from railrl.envs.mujoco.pusher import PusherEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from rllab.core.serializable import Serializable
from railrl.core import logger as default_logger


class GoalXYGymPusherEnv(PusherEnv, MultitaskEnv, Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(self)
        super().__init__()

    # We'll fix the cylinder pos an vary the goal.
    def reset_model(self):
        qpos = self.init_qpos
        qpos[-4:-2] = np.array([-0.3, 0])
        # y-axis comes first in the xml
        qpos[-2] = self.goal_cylinder_relative_y
        qpos[-1] = self.goal_cylinder_relative_x
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv,
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

    def _step(self, action):
        ob, _, done, info_dict = super()._step(action)
        xy_pos = self.convert_ob_to_goal(ob)
        distance_to_goal = np.linalg.norm(xy_pos - self.multitask_goal)
        reward = - distance_to_goal
        return ob, reward, done, dict(
            goal=self.multitask_goal,
            distance_to_goal=distance_to_goal,
            position=xy_pos,
            **info_dict
        )

    def set_goal(self, goal):
        super().set_goal(goal)
        self._set_goal_xy(goal)

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goals(self, batch_size):
        return np.random.uniform(
            np.array([-0.3, -0.2]),
            np.array([0, 0.2]),
            (batch_size, 2),
        )

    def convert_obs_to_goals(self, obs):
        return obs[:, 17:19]

    def log_diagnostics(self, paths, logger=default_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)
