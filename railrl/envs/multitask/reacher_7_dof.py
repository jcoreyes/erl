from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import PusherEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.misc import logger


class Reacher7DofXyzGoalState(PusherEnv, MultitaskEnv):
    """
    The goal state is just the XYZ location of the end effector.
    """
    def sample_goal_states_for_rollouts(self, batch_size):
        return self.sample_goal_states(batch_size)

    def sample_goal_states(self, batch_size):
        # Number taken from running a random policy and seeing what XYZ values
        # are reached
        return np.hstack((
            self.np_random.uniform(low=-0.75, high=0.75, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.25, high=0.25, size=(batch_size, 1)),
            self.np_random.uniform(low=-0.2, high=0.6, size=(batch_size, 1)),
        ))

    @property
    def goal_dim(self):
        return 3

    def convert_obs_to_goal_states(self, obs):
        return positions_from_observations(obs)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
        ])

    def _step(self, a):
        if hasattr(self, "multitask_goal"):
            goal = self.multitask_goal
        else:
            goal = np.zeros(3)
        self.end_effector_xyz = self.get_body_com("tips_arm")
        ob, _, done, info_dict = super()._step(a)
        reward = - np.linalg.norm(self.end_effector_xyz - goal)
        return ob, reward, done, info_dict

    def set_goal(self, goal):
        self.multitask_goal = goal

    def log_diagnostics(self, paths):
        observations = np.vstack([path['observations'] for path in paths])
        goal_states = np.vstack([path['goal_states'] for path in paths])
        positions = positions_from_observations(observations)
        distances = np.linalg.norm(positions - goal_states, axis=1)

        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Distance to target', distances
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


def positions_from_observations(obs):
    return obs[:, -3:]
