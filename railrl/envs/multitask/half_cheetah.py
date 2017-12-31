from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import HalfCheetahEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from rllab.core.serializable import Serializable
from rllab.misc import logger as rllab_logger

MAX_SPEED = 6


class GoalXVelHalfCheetah(HalfCheetahEnv, MultitaskEnv):
    def __init__(self):
        self.target_x_vel = np.random.uniform(-MAX_SPEED, MAX_SPEED)
        super().__init__()
        MultitaskEnv.__init__(self)
        self.set_goal(np.array([5]))

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goals(self, batch_size):
        return np.random.uniform(-MAX_SPEED, MAX_SPEED, (batch_size, 1))

    def convert_obs_to_goals(self, obs):
        return obs[:, 8:9]

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.target_x_vel = goal

    def _step(self, action):
        ob, _, done, info_dict = super()._step(action)
        xvel = ob[8]
        desired_xvel = self.target_x_vel
        xvel_error = np.linalg.norm(xvel - desired_xvel)
        reward = - xvel_error
        info_dict['xvel'] = xvel
        info_dict['desired_xvel'] = desired_xvel
        info_dict['xvel_error'] = xvel_error
        return ob, reward, done, info_dict

    def log_diagnostics(self, paths, logger=rllab_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)
        xvels = get_stat_in_paths(
            paths, 'env_infos', 'xvel'
        )
        desired_xvels = get_stat_in_paths(
            paths, 'env_infos', 'desired_xvel'
        )
        xvel_errors = get_stat_in_paths(
            paths, 'env_infos', 'xvel_error'
        )

        statistics = OrderedDict()
        for stat, name in [
            (xvels, 'xvels'),
            (desired_xvels, 'desired xvels'),
            (xvel_errors, 'xvel errors'),
        ]:
            statistics.update(create_stats_ordered_dict(
                '{}'.format(name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)


# At a score of 5000, the cheetah is moving at "5 meters per dt" but dt = 0.05s,
# so it's really 100 m/s.
# For a horizon of 50, that's 2.5 seconds.
# So 5 m/s * 2.5s = 12.5 meters
GOAL_X_POSITIONS = [12.5, -12.5]


class GoalXPosHalfCheetah(HalfCheetahEnv, MultitaskEnv, Serializable):
    """
    At a score of 5000, the cheetah is moving at "5 meters per dt" but dt=0.05s,
    so it's really 100 m/s. For a horizon of 100, that's 5 seconds.

    So 5 m/s * 5s = 25 meters
    """
    def __init__(self, max_distance=25):
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(self)
        super().__init__()
        self.max_distance = max_distance
        self.set_goal(np.array([self.max_distance]))

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def _step(self, action):
        ob, reward, done, info_dict = super()._step(action)
        x_pos = self.convert_ob_to_goal(ob)
        distance_to_goal = np.linalg.norm(x_pos - self.multitask_goal)
        reward_run = -distance_to_goal
        reward = reward_run
        return ob, reward, done, dict(
            reward_run=reward_run,
            reward_ctrl=info_dict['reward_ctrl'],
            goal_position=self.multitask_goal,
            goal=self.multitask_goal,
            distance_to_goal=distance_to_goal,
            position=x_pos,
        )

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goals(self, batch_size):
        return np.random.uniform(
            -self.max_distance,
            self.max_distance,
            (batch_size, 1),
        )

    def convert_obs_to_goals(self, obs):
        return obs[:, 17:18]

    def log_diagnostics(self, paths, logger=rllab_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)
        distances_to_goal = get_stat_in_paths(
            paths, 'env_infos', 'distance_to_goal'
        )
        goal_positions = get_stat_in_paths(
            paths, 'env_infos', 'goal_position'
        )
        positions = get_stat_in_paths(
            paths, 'env_infos', 'position'
        )
        statistics = OrderedDict()
        for stat, name in [
            (distances_to_goal, 'Distance to goal'),
            (goal_positions, 'Goal Position'),
        ]:
            statistics.update(create_stats_ordered_dict(
                '{}'.format(name),
                stat,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        for stat, name in [
            (distances_to_goal, 'Distance to goal'),
            (positions, 'Position'),
            ([p[-1] for p in positions], 'Final Position'),
        ]:
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)
