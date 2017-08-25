"""
Extensions to the base ReacherEnv. Basically, these classes add the following
functions:
    - sample_goal_states
    - compute_rewards
Typical usage:

```
    batch = get_batch(batch_size)
    goal_states = env.sample_goal_states(batch_size)
    new_rewards = env.compute_rewards(
        batch['observations'],
        batch['actions'],
        batch['next_observations'],
        goal_states,
    )
    # Update batch to use new rewards and maybe add goal states
```

One example of how I do the last step is I do:

```
    batch['observations'] = np.hstack((batch['observations'], goal_states))
    batch['next_observations'] = np.hstack((
        batch['next_observations'], goal_states
    ))
    batch['rewards'] = new_rewards
```

:author: Vitchyr Pong
"""
import math
from collections import OrderedDict

import numpy as np
from gym import utils
from gym.envs.mujoco import ReacherEnv, mujoco_env

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.misc import logger

R1 = 0.1  # from reacher.xml
R2 = 0.11


def position_from_angles(angles):
    """
    :param angles: np.ndarray [batch_size x feature]
    where the first four entries (along dimesion 1) are
        - cosine of angle 1
        - cosine of angle 2
        - sine of angle 1
        - sine of angle 2
    :return: np.ndarray [batch_size x 2]
    """
    c1 = angles[:, 0:1]  # cosine of angle 1
    c2 = angles[:, 1:2]
    s1 = angles[:, 2:3]
    s2 = angles[:, 3:4]
    return (  # forward kinematics equation for 2-link robot
        R1 * np.hstack([c1, s1])
        + R2 * np.hstack([
            c1 * c2 - s1 * s2,
            s1 * c2 + c1 * s2,
        ])
    )


class XyMultitaskReacherEnv(ReacherEnv, MultitaskEnv):
    """
    The goal states are xy-coordinates.
    """

    def sample_goal_states(self, batch_size):
        return self.np_random.uniform(
            low=-0.2,
            high=0.2,
            size=(batch_size, 2)
        )

    def compute_rewards(self, obs, action, next_obs, goal_states):
        next_qpos = position_from_angles(next_obs)
        reward_dist = -np.linalg.norm(next_qpos - goal_states, axis=1)
        reward_ctrl = - np.sum(action * action, axis=1)
        return reward_ctrl + reward_dist

    def log_diagnostics(self, paths):
        distance = [
            np.linalg.norm(path["observations"][-1][-3:])
            for path in paths
        ]

        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Distance to target', distance
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def convert_obs_to_goal_states(self, obs):
        return position_from_angles(obs)

    @property
    def goal_dim(self):
        return 2

    @staticmethod
    def print_goal_state_info(goal):
        print(goal)


class XyMultitaskSimpleStateReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    The goal states are xy-coordinates.

    Furthermore, the actual state space is simplified. ReacherEnv has the
    following state space:
        - cos(angle 1)
        - cos(angle 2)
        - sin(angle 1)
        - sin(angle 2)
        - goal x-coordinate
        - goal y-coordinate
        - angle 1 velocity
        - angle 2 velocity
        - x-coordinate distance from end effector to goal
        - y-coordinate distance from end effector to goal
        - z-coordinate distance from end effector to goal (always zero)

    This environment only has the following:
        - cos(angle 1)
        - cos(angle 2)
        - sin(angle 1)
        - sin(angle 2)
        - angle 1 velocity
        - angle 2 velocity

    since the goal will constantly change.
    """

    def __init__(self, ctrl_penalty_weight=0):
        self.ctrl_penalty_weight = ctrl_penalty_weight
        utils.EzPickle.__init__(
            self,
            ctrl_penalty_weight=ctrl_penalty_weight,
        )
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        self._fixed_goal = None
        self.goal = None

    def set_goal(self, goal):
        """
        Add option to set the goal. Really only used for debugging. If None (
        by default), then the goal is randomly sampled each time the
        environment is reset.
        :param goal:
        :return:
        """
        self._fixed_goal = goal

    def _step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.sum(a * a)
        reward = reward_dist + reward_ctrl * self.ctrl_penalty_weight
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-np.pi, high=np.pi,
                                      size=self.model.nq) + self.init_qpos
        if self._fixed_goal is None:
            while True:
                self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self.goal) < 2:
                    break
        else:
            self.goal = self._fixed_goal
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005,
                                                       size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        obs = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
        ])
        return obs

    def sample_goal_states(self, batch_size):
        return self.np_random.uniform(
            low=-0.2,
            high=0.2,
            size=(batch_size, 2)
        )

    def compute_rewards(self, obs, action, next_obs, goal_states):
        next_endeffector_positions = position_from_angles(next_obs)
        reward_dist = -np.linalg.norm(
            next_endeffector_positions - goal_states, axis=1
        )
        reward_ctrl = - np.sum(action * action, axis=1)
        return self.ctrl_penalty_weight * reward_ctrl + reward_dist

    def log_diagnostics(self, paths):
        observations = np.vstack([path['observations'] for path in paths])
        actions = np.vstack([path['actions'] for path in paths])
        goal_states = np.vstack([path['goal_states'] for path in paths])
        positions = position_from_angles(observations)
        distances = np.linalg.norm(positions - goal_states, axis=1)

        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Distance to target', distances
        ))
        rewards = self.compute_rewards(
            observations[:-1, ...],
            actions[:-1, ...],
            observations[1:, ...],
            goal_states[:-1, ...],
        )
        statistics.update(create_stats_ordered_dict(
            'Rewards', rewards,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def convert_obs_to_goal_states(self, obs):
        return position_from_angles(obs)

    @property
    def goal_dim(self):
        return 2

    @staticmethod
    def print_goal_state_info(goal):
        print("Goal = ", goal)

    def sample_actions(self, sample_size):
        return np.random.uniform(-1, 1, size=(sample_size, 2))

    def sample_states(self, sample_size):
        theta = np.pi * (2 * np.random.rand(sample_size, 2) - 1)
        velocity = 10 * (2 * np.random.rand(sample_size, 2) - 1)
        return np.hstack((
                np.cos(theta),
                np.sin(theta),
                velocity,
        ))

    def sample_goal_partially(self, goal, sample_size):
        """
        :param goal: np.ndarray, shape GOAL_DIM
        :param sample_size:
        :return: ndarray, shape SAMPLE_SIZE x GOAL_DIM
        """
        return np.repeat(
            np.expand_dims(goal, 0),
            sample_size,
            axis=0
        )


class GoalStateSimpleStateReacherEnv(XyMultitaskSimpleStateReacherEnv):
    """
    The goal state is an actual state (6 dimensions--see parent class), rather
    than just the XY-coordinate of the target end effector.
    """

    def __init__(self, ctrl_penalty_weight=0):
        self.ctrl_penalty_weight = ctrl_penalty_weight
        utils.EzPickle.__init__(
            self,
            ctrl_penalty_weight=ctrl_penalty_weight,
        )
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        self._fixed_goal = None
        self.goal = None

    def set_goal(self, goal_state):
        self._fixed_goal = position_from_angles(
            np.expand_dims(goal_state, 0)
        )[0]

    def sample_goal_states(self, batch_size):
        theta = self.np_random.uniform(
            low=-math.pi,
            high=math.pi,
            size=(batch_size, 2)
        )
        velocities = np.random.uniform(-1, 1, (batch_size, 2))
        obs = np.hstack([
            np.cos(theta),
            np.sin(theta),
            velocities
        ])
        return obs

    def sample_goal_states_for_rollouts(self, batch_size):
        theta = self.np_random.uniform(
            low=-math.pi,
            high=math.pi,
            size=(batch_size, 2)
        )
        velocities = np.zeros((batch_size, 2))
        obs = np.hstack([
            np.cos(theta),
            np.sin(theta),
            velocities
        ])
        return obs

    def compute_rewards(self, obs, action, next_obs, goal_states):
        difference = next_obs - goal_states
        reward_dist = -np.linalg.norm(difference, axis=1)
        reward_ctrl = - np.sum(action * action, axis=1)
        return reward_dist + reward_ctrl + self.ctrl_penalty_weight

    def log_diagnostics(self, paths):
        observations = np.vstack([path['observations'] for path in paths])
        actions = np.vstack([path['actions'] for path in paths])
        goal_states = np.vstack([path['goal_states'] for path in paths])
        positions = position_from_angles(observations)
        goal_positions = position_from_angles(goal_states)
        distances = np.linalg.norm(positions - goal_positions, axis=1)

        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Distance to target', distances
        ))

        rewards = self.compute_rewards(
            observations[:-1, ...],
            actions[:-1, ...],
            observations[1:, ...],
            goal_states[:-1, ...],
        )
        statistics.update(create_stats_ordered_dict(
            'Rewards', rewards,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    @staticmethod
    def print_goal_state_info(goal):
        c1 = goal[0:1]
        c2 = goal[1:2]
        s1 = goal[2:3]
        s2 = goal[3:4]
        print("Goal = ", goal)
        print("angle 1 (degrees) = ", np.arctan2(s1, c1) / math.pi * 180)
        print("angle 2 (degrees) = ", np.arctan2(s2, c2) / math.pi * 180)

    def convert_obs_to_goal_states(self, obs):
        return obs

    @property
    def goal_dim(self):
        return 6

    def sample_goal_partially(self, goal, sample_size):
        """
        Sample the goal a bunch of time, but fill in the desired position with
        what you care about.

        :param goal: np.ndarray, shape GOAL_DIM
        :param sample_size:
        :return: ndarray, shape SAMPLE_SIZE x GOAL_DIM
        """
        sampled_velocities = np.random.uniform(
            -1,
            1,
            size=(sample_size, 2),
        )
        goals = np.repeat(
            np.expand_dims(goal, 0),
            sample_size,
            axis=0
        )
        goals[:, 4:6] = sampled_velocities
        return goals


class XYAndGoalStateReacherEnv(GoalStateSimpleStateReacherEnv):
    """"
    The goal state is a concatenation of the XY position and a full state
    that achieves that XY position.

    However, the reward is just the end effector distance to goal.
    """
    def sample_goal_states(self, batch_size):
        theta = self.np_random.uniform(
            low=-math.pi,
            high=math.pi,
            size=(batch_size, 2)
        )
        velocities = 5 * np.random.rand(batch_size, 2)
        obs = np.hstack([
            np.cos(theta),
            np.sin(theta),
            velocities
        ])
        goal_positions = position_from_angles(obs)
        return np.hstack((obs, goal_positions))

    def convert_obs_to_goal_states(self, obs):
        goal_positions = position_from_angles(obs)
        return np.hstack((obs, goal_positions))

    def compute_rewards(self, obs, action, next_obs, goal_states):
        next_endeffector_positions = position_from_angles(next_obs)
        goal_positions = goal_states[:, -2:]
        reward_dist = -np.linalg.norm(
            next_endeffector_positions - goal_positions, axis=1
        )
        reward_ctrl = - np.sum(action * action, axis=1)
        return self.ctrl_penalty_weight * reward_ctrl + reward_dist

    @property
    def goal_dim(self):
        return 8


class GoalStateXYRewardReacherEnv(GoalStateSimpleStateReacherEnv):
    """"
    The goal state is an actual state.
    However, the reward is just the end effector distance to goal.

    Purpose: See if the problem is the reward or processing the goal state.
    """
    def compute_rewards(self, obs, action, next_obs, goal_states):
        next_endeffector_positions = position_from_angles(next_obs)
        goal_positions = position_from_angles(goal_states)
        reward_dist = -np.linalg.norm(
            next_endeffector_positions - goal_positions, axis=1
        )
        reward_ctrl = - np.sum(action * action, axis=1)
        return self.ctrl_penalty_weight * reward_ctrl + reward_dist


class FullStateWithXYStateReacherEnv(GoalStateSimpleStateReacherEnv):
    def sample_goal_states(self, batch_size):
        theta = self.np_random.uniform(
            low=-math.pi,
            high=math.pi,
            size=(batch_size, 2)
        )
        velocities = 5 * np.random.rand(batch_size, 2)
        ee_pos = position_from_angles(np.hstack([np.cos(theta), np.sin(theta)]))
        obs = np.hstack([
            np.cos(theta),
            np.sin(theta),
            velocities,
            ee_pos
        ])
        return obs

    def compute_rewards(self, obs, action, next_obs, goal_states):
        difference = next_obs - goal_states
        difference = difference[:, :6]
        reward_dist = -np.linalg.norm(difference, axis=1)

        reward_ctrl = - np.sum(action * action, axis=1)
        return reward_dist + reward_ctrl + self.ctrl_penalty_weight

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        obs = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
            self.model.data.qpos.flat[:2],
        ])
        return obs

    @property
    def goal_dim(self):
        return 8
