from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import PusherEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_stat_in_dict
from rllab.misc import logger


class MultitaskPusherEnv(PusherEnv, MultitaskEnv):
    def __init__(self):
        super().__init__()
        self.multitask_goal = None

    def reset_model(self):
        qpos = self.init_qpos

        while True:
            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=-0.3, high=0, size=1),
                self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.multitask_goal) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.multitask_goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _step(self, a):
        arm_to_object = (
            self.get_body_com("tips_arm") - self.get_body_com("object")
        )
        object_to_goal = (
            self.get_body_com("object") - self.get_body_com("goal")
        )
        arm_to_goal = (
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        obs, reward, done, info_dict = super()._step(a)
        info_dict['arm to object distance'] = np.linalg.norm(arm_to_object)
        info_dict['object to goal distance'] = np.linalg.norm(object_to_goal)
        info_dict['arm to goal distance'] = np.linalg.norm(arm_to_goal)
        return obs, reward, done, info_dict

    def sample_actions(self, batch_size):
        return np.random.uniform(
            -1, 1, size=(batch_size, self.action_space.low.size)
        )

    def sample_states(self, batch_size):
        raise NotImplementedError("Would need to do forward kinematics...")

    def sample_goal_states_for_rollouts(self, batch_size):
        return self.sample_goal_states(batch_size=batch_size)

    def set_goal(self, goal):
        self.multitask_goal = goal

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])

    def sample_goal_states(self, batch_size):
        return np.concatenate(
            (
                self.np_random.uniform(low=-0.3, high=0, size=(batch_size, 1)),
                self.np_random.uniform(
                    low=-0.2, high=0.2, size=(batch_size, 1),
                ),
            ),
            axis=1
        )

    @property
    def goal_dim(self):
        return 2

    def convert_obs_to_goal_states(self, obs):
        return obs[:, 17:19]

    def log_diagnostics(self, paths):
        statistics = OrderedDict()

        observations = np.vstack([path['observations'] for path in paths])
        goal_states = np.vstack([path['goal_states'] for path in paths])
        distances = np.linalg.norm(
            self.convert_obs_to_goal_states(observations) - goal_states,
            axis=1,
            )
        statistics.update(create_stats_ordered_dict(
            'State distance to target', distances
        ))

        for stat_name in [
            'arm to object distance',
            'object to goal distance',
            'arm to goal distance',
        ]:
            stat = get_stat_in_dict(
                paths, 'env_infos', stat_name
            )
            statistics.update(create_stats_ordered_dict(
                stat_name, stat
            ))

        rewards = self.compute_rewards(
            None,
            None,
            observations,
            goal_states,
        )
        statistics.update(create_stats_ordered_dict(
            'Rewards', rewards,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)
