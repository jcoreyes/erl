from collections import OrderedDict

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.misc import logger


class Reacher7DofXyzGoalState(MultitaskEnv, mujoco_env.MujocoEnv, utils.EzPickle):
    """
    The goal state is just the XYZ location of the end effector.
    """
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.multitask_goal = np.zeros(self.goal_dim)
        mujoco_env.MujocoEnv.__init__(
            self,
            get_asset_xml('reacher_7dof.xml'),
            5,
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qpos[-7:-4] = self.multitask_goal
        qvel[-7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

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
        return obs[:, -3:]

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
        ])

    def _step(self, a):
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        reward = - distance
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(distance=distance)

    def set_goal(self, goal):
        self.multitask_goal = goal

    def log_diagnostics(self, paths):
        observations = np.vstack([path['observations'] for path in paths])
        goal_states = np.vstack([path['goal_states'] for path in paths])
        distances = np.linalg.norm(
            self.convert_obs_to_goal_states(observations) - goal_states,
            axis=1,
        )

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


class Reacher7DofFullGloalState(Reacher7DofXyzGoalState):
    def sample_goal_states_for_rollouts(self, batch_size):
        return self.sample_goal_states(batch_size)

    def sample_goal_states(self, batch_size):
        # Number taken from running a random policy and seeing what XYZ values
        # are reached
        return np.hstack((
            # From the xml
            self.np_random.uniform(low=-2.28, high=1.71, size=(batch_size, 1)),
            self.np_random.uniform(low=-0.52, high=1.39, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.4, high=1.7, size=(batch_size, 1)),
            self.np_random.uniform(low=-2.32, high=0, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.5, high=1.5, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.094, high=0, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.5, high=1.5, size=(batch_size, 1)),
            # velocities
            self.np_random.uniform(low=-1, high=1, size=(batch_size, 7)),
        ))

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
        ])

    @property
    def goal_dim(self):
        return 14

    def convert_obs_to_goal_states(self, obs):
        return obs
