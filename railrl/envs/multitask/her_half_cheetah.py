from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import HalfCheetahEnv

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from railrl.core import logger as default_logger


def obs_to_goal(obs):
    return obs[8:9]


def get_sparse_reward(obs):
    """-1 if far, 0 if close"""
    desired_vel = obs[17:18]
    actual_vel = obs[8:9]
    r = np.linalg.norm(desired_vel - actual_vel) < 0.5
    return (r - 1).astype(float)


def half_cheetah_cost_fn(states, actions, next_states):
    input_is_flat = len(states.shape) == 1
    if input_is_flat:
        states = np.expand_dims(states, 0)
    desired_vels = states[:, 17:18]
    actual_vels = states[:, 8:9]
    costs = np.linalg.norm(
        desired_vels - actual_vels,
        axis=1,
        ord=2,
    )
    if input_is_flat:
        costs = costs[0]
    return costs


class HalfCheetah(HalfCheetahEnv):
    def __init__(self):
        self.target_x_vel = np.random.uniform(-5, 5)
        self.obs_to_goal = obs_to_goal
        self.goal_idx = slice(17, 18)
        super().__init__()

    def get_reward(self, obs):
        return get_sparse_reward(obs)

    def _step(self, action):
        ob, _, done, info_dict = super()._step(action)
        xvel = ob[8]
        desired_xvel = self.target_x_vel
        xvel_error = np.linalg.norm(xvel - desired_xvel)
        reward = - xvel_error
        # ASHVIN!!!!! Please uncomment this for your stuff.
        # reward = get_sparse_reward(ob)
        new_ob = np.hstack((ob, self.target_x_vel))
        info_dict['xvel'] = xvel
        info_dict['desired_xvel'] = desired_xvel
        info_dict['xvel_error'] = xvel_error
        return new_ob, reward, done, info_dict

    def reset_model(self):
        ob = super().reset_model()
        self.target_x_vel = np.random.uniform(-10, 10)
        return np.hstack((ob, self.target_x_vel))

    def log_diagnostics(self, paths, logger=default_logger):
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
            logger.log_tabular(key, value)
