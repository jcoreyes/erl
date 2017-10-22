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
        self._multitask_goal = np.zeros(self.goal_dim)
        self.init_serialization(locals())
        super().__init__(goal=goal)

    def sample_actions(self, batch_size):
        return np.random.uniform(self.action_space.low, self.action_space.high)

    def set_goal(self, goal):
        self._target_cylinder_position = goal[-2:]
        self._target_hand_position = goal[-4:-2]
        self._multitask_goal = goal

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        statistics = OrderedDict()
        full_state_go_goal_distance = get_stat_in_dict(
            paths, 'env_infos', 'full_state_to_goal_distance'
        )
        statistics.update(create_stats_ordered_dict(
            'Final state to goal state distance',
            full_state_go_goal_distance[:, -1],
            always_show_all_stats=True,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def _step(self, a):
        full_state_to_goal_distance = np.linalg.norm(
            self.convert_obs_to_goal_states(
                np.expand_dims(self._get_obs(), 0)
            )[0]
            - self._multitask_goal
        )
        ob, reward, done, info_dict = super()._step(a)
        info_dict['full_state_to_goal_distance'] = (
            full_state_to_goal_distance
        )
        return ob, reward, done, info_dict


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
        self._target_cylinder_position = goal[-2:]
        self._multitask_goal = goal

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)
