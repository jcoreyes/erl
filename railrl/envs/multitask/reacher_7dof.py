from collections import OrderedDict

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_stat_in_dict
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

    def sample_goal_states(self, batch_size):
        # Number taken from running a random policy and seeing what XYZ values
        # are reached
        return np.hstack((
            self.np_random.uniform(low=-0.75, high=0.75, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.25, high=0.25, size=(batch_size, 1)),
            self.np_random.uniform(low=-0.2, high=0.6, size=(batch_size, 1)),
        ))

    def sample_actions(self, batch_size):
        return np.random.uniform(
            -1, 1, size=(batch_size, self.action_space.low.size)
        )

    def sample_states(self, batch_size):
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

        euclidean_distances = get_stat_in_dict(
            paths, 'env_infos', 'distance'
        )
        statistics.update(create_stats_ordered_dict(
            'Euclidean distance to goal', euclidean_distances
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


class Reacher7DofFullGoalState(Reacher7DofXyzGoalState):
    def sample_goal_state_for_rollout(self):
        goal_states = self.sample_goal_states(0)[1]
        goal_states[:, -7:] = 0
        return goal_states

    def sample_goal_states(self, batch_size):
        return self.sample_states(batch_size)

    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        new_goal_states = super().sample_irrelevant_goal_dimensions(
            goal, batch_size
        )
        new_goal_states[:, -7:] = (
            self.np_random.uniform(low=-1, high=1, size=(batch_size, 7))
        )
        return new_goal_states

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

    def reset_model(self):
        """
        I don't want to manually compute the forward dynamics to compute the
        goal state XYZ coordinate.

        Instead, I just put the arm in the desired goal state. Then I measure
        the end-effector XYZ coordinate.
        """
        saved_init_qpos = self.init_qpos.copy()
        qpos_tmp = self.init_qpos
        qpos_tmp[:14] = self.multitask_goal
        qvel_tmp = np.zeros(self.model.nv)
        self.set_state(qpos_tmp, qvel_tmp)
        goal_xyz = self.get_body_com("tips_arm")

        # Now we actually set the goal position
        qpos = saved_init_qpos
        qpos[7:10] = goal_xyz
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv,
        )
        qvel[7:] = 0
        self.set_state(qpos, qvel)

        return self._get_obs()


class Reacher7DofFullGloalState(Reacher7DofFullGoalState):
    """
    Kept so that loading old envs works still. Used to have this type in the
    class name.
    """
    pass



class Reacher7DofCosSinFullGoalState(Reacher7DofFullGoalState):
    def _get_obs(self):
        angles = self.model.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(angles),
            np.sin(angles),
            self.model.data.qvel.flat[:7],
        ])

    def sample_states(self, batch_size):
        full_states = super().sample_states(batch_size)
        angles = full_states[:, 7:]
        velocities = full_states[:, :7]
        return np.hstack((
            np.cos(angles),
            np.sin(angles),
            velocities
        ))

    @property
    def goal_dim(self):
        return 21

    def convert_obs_to_goal_states(self, obs):
        return obs

    def reset_model(self):
        saved_init_qpos = self.init_qpos.copy()
        qpos_tmp = self.init_qpos
        cos = self.multitask_goal[:7]
        sin = self.multitask_goal[7:14]
        angles = np.arctan2(sin, cos)
        qpos_tmp[:7] = angles
        qpos_tmp[7:] = self.multitask_goal[14:]
        qvel_tmp = np.zeros(self.model.nv)
        self.set_state(qpos_tmp, qvel_tmp)
        goal_xyz = self.get_body_com("tips_arm")

        # Now we actually set the goal position
        qpos = saved_init_qpos
        qpos[7:10] = goal_xyz
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv,
        )
        qvel[7:] = 0
        self.set_state(qpos, qvel)

        return self._get_obs()
