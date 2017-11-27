from collections import OrderedDict

import numpy as np
import torch

from railrl.envs.mujoco.ant import LowGearAntEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from rllab.misc import logger

MAX_SPEED = 1


class GoalXYVelAnt(LowGearAntEnv, MultitaskEnv):
    def __init__(self):
        self.target_xy_vel = np.random.uniform(-MAX_SPEED, MAX_SPEED, 2)
        super().__init__()
        MultitaskEnv.__init__(self)

    def sample_actions(self, batch_size):
        raise NotImplementedError()

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goal_states(self, batch_size):
        return np.random.uniform(-MAX_SPEED, MAX_SPEED, (batch_size, 2))

    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        raise NotImplementedError()

    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        return np.repeat(
            np.expand_dims(goal, 0),
            batch_size,
            axis=0
        )

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_comvel("torso")
        ])

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -3:-1]

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.target_xy_vel = goal

    def _step(self, action):
        ob, _, done, info_dict = super()._step(action)
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

    def log_diagnostics(self, paths):
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

    def oc_reward(self, states, goals, current_states):
        return self.oc_reward_on_goals(
            self.convert_obs_to_goal_states(states),
            goals,
            current_states,
        )

    def oc_reward_on_goals(self, goals_predicted, goals, current_states):
        return - torch.norm(goals_predicted - goals, dim=1, p=2, keepdim=True)

    def compute_her_reward_pytorch(
            self,
            observations,
            actions,
            next_observations,
            goal_states,
    ):
        vels = observations[:, -3:-1]
        target_xvels = goal_states
        costs = torch.norm(
            vels - target_xvels,
            dim=1,
            p=2,
            keepdim=True,
        )
        return - costs
