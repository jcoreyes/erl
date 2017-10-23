import abc
from collections import OrderedDict

import numpy as np

from railrl.envs.mujoco.pusher2d import Pusher2DEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_stat_in_dict
from rllab.misc import logger


class MultitaskPusher2DEnv(Pusher2DEnv, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(self, goal=(0, -1)):
        self.init_serialization(locals())
        super().__init__(goal=goal)
        MultitaskEnv.__init__(self)

    def sample_actions(self, batch_size):
        return np.random.uniform(self.action_space.low, self.action_space.high)

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_cylinder_position = goal[-2:]
        self._target_hand_position = goal[-4:-2]

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)


class FullStatePusher2DEnv(MultitaskPusher2DEnv):
    def sample_goal_states(self, batch_size):
        # Joint angle and xy position won't be consistent, but oh well!
        return np.random.uniform(
            np.array([-2.5, -2.3213, -2.3213, -1, -1, -1, -1, -1, -1, -1]),
            np.array([2.5, 2.3, 2.3, 1, 1, 1, 0, 1, 0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 10

    def convert_obs_to_goal_states(self, obs):
        return obs


class HandCylinderXYPusher2DEnv(MultitaskPusher2DEnv):
    def sample_goal_states(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1, -1., -1]),
            np.array([0, 1, 0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 4

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -4:]


class HandXYPusher2DEnv(MultitaskPusher2DEnv):
    """
    Only care about the hand position! This is really just for debugging.
    """
    def sample_goal_states(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1]),
            np.array([0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 2

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -4:-2]

    def set_goal(self, goal):
        self._target_hand_position = goal

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)


class CylinderXYPusher2DEnv(MultitaskPusher2DEnv):
    def sample_goal_states(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1]),
            np.array([0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 2

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -2:]

    def set_goal(self, goal):
        self._target_cylinder_position = goal

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)
