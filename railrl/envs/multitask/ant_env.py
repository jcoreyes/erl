from collections import OrderedDict

import numpy as np
import torch

from railrl.envs.mujoco.ant import LowGearAntEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from rllab.core.serializable import Serializable
from rllab.misc import logger as rllab_logger

MAX_SPEED = 1


class GoalXYVelAnt(LowGearAntEnv, MultitaskEnv, Serializable):
    # WIP
    def __init__(self):
        self.target_xy_vel = np.random.uniform(-MAX_SPEED, MAX_SPEED, 2)
        super().__init__()
        MultitaskEnv.__init__(self)

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goals(self, batch_size):
        return np.random.uniform(-MAX_SPEED, MAX_SPEED, (batch_size, 2))

    def convert_obs_to_goals(self, obs):
        return obs[:, -3:-1]

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.target_xy_vel = goal

    def _get_obs(self):
        raise NotImplementedError()

    def _step(self, action):
        raise NotImplementedError()
        # get_body_comvel doesn't work, so you need to save the last position
        ob, _, done, info_dict = super()._step(action)
        ob = np.hstack((
            ob,
            info_dict['torso_velocity'],
        ))
        print(self.get_body_com("torso"))
        vel = ob[-3:-1]
        vel_error = np.linalg.norm(vel - self.target_xy_vel)
        reward = - vel_error
        info_dict['xvel'] = vel[0]
        info_dict['yvel'] = vel[1]
        info_dict['desired_xvel'] = self.target_xy_vel[0]
        info_dict['desired_yvel'] = self.target_xy_vel[1]
        info_dict['vel_error'] = vel_error
        return ob, reward, done, info_dict

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths, logger=rllab_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)
        xvels = get_stat_in_paths(
            paths, 'env_infos', 'xvel'
        )
        desired_xvels = get_stat_in_paths(
            paths, 'env_infos', 'desired_xvel'
        )
        vel_errors = get_stat_in_paths(
            paths, 'env_infos', 'vel_error'
        )

        statistics = OrderedDict()
        for stat, name in [
            (xvels, 'xvels'),
            (desired_xvels, 'desired xvels'),
            (vel_errors, 'vel errors'),
        ]:
            statistics.update(create_stats_ordered_dict(
                '{}'.format(name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name),
                stat[:, -1],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)


class GoalXYPosAnt(LowGearAntEnv, MultitaskEnv, Serializable):
    def __init__(self, max_distance=2):
        Serializable.quick_init(self, locals())
        self.max_distance = max_distance
        MultitaskEnv.__init__(self)
        super().__init__()
        self.set_goal(np.array([self.max_distance, self.max_distance]))

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goals(self, batch_size):
        return np.random.uniform(
            -self.max_distance,
            self.max_distance,
            (batch_size, 2),
        )

    def convert_obs_to_goals(self, obs):
        return obs[:, 27:29]

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            self.get_body_com("torso"),
        ])

    def _step(self, action):
        ob, _, done, info_dict = super()._step(action)
        xy_pos = self.convert_ob_to_goal(ob)
        pos_error = np.linalg.norm(xy_pos - self.multitask_goal)
        reward = - pos_error
        info_dict['x_pos'] = xy_pos[0]
        info_dict['y_pos'] = xy_pos[1]
        info_dict['dist_from_origin'] = np.linalg.norm(xy_pos)
        info_dict['desired_x_pos'] = self.multitask_goal[0]
        info_dict['desired_y_pos'] = self.multitask_goal[1]
        info_dict['desired_dist_from_origin'] = (
            np.linalg.norm(self.multitask_goal)
        )
        info_dict['pos_error'] = pos_error
        return ob, reward, done, info_dict

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths, logger=rllab_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)

        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('x_pos', 'X Position'),
            ('y_pos', 'Y Position'),
            ('dist_from_origin', 'Distance from Origin'),
            ('desired_x_pos', 'Desired X Position'),
            ('desired_y_pos', 'Desired Y Position'),
            ('desired_dist_from_origin', 'Desired Distance from Origin'),
            ('pos_error', 'Distance to goal'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stat,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        for name_in_env_infos, name_to_log in [
            ('dist_from_origin', 'Distance from Origin'),
            ('desired_dist_from_origin', 'Desired Distance from Origin'),
            ('pos_error', 'Distance to goal'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name_to_log),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)
