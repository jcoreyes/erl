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
import abc
import math
from collections import OrderedDict

import numpy as np
from gym import utils
from gym.envs.mujoco import ReacherEnv, mujoco_env

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_stat_in_dict
from rllab.misc import logger
import torch
import railrl.torch.pytorch_util as ptu

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


def position_from_angles_pytorch(angles):
    """
    :param angles: torch.FloatTensor [batch_size x feature]
    where the first four entries (along dimesion 1) are
        - cosine of angle 1
        - cosine of angle 2
        - sine of angle 1
        - sine of angle 2
    :return: torch.FloatTensor [batch_size x 2]
    """
    c1 = angles[:, 0:1]  # cosine of angle 1
    c2 = angles[:, 1:2]
    s1 = angles[:, 2:3]
    s2 = angles[:, 3:4]
    return (  # forward kinematics equation for 2-link robot
        R1 * torch.cat([c1, s1], dim=1)
        + R2 * torch.cat([
            c1 * c2 - s1 * s2,
            s1 * c2 + c1 * s2,
            ],
            dim=1,
        )
    )


class MultitaskReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle, MultitaskEnv,
                          metaclass=abc.ABCMeta):
    def __init__(self):
        self._xy_desired_pos = None
        MultitaskEnv.__init__(self)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        distance = np.linalg.norm(vec)
        reward_dist = - distance
        reward_ctrl = - np.sum(a * a)
        reward = reward_dist
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        full_state_to_goal_distance = np.linalg.norm(
            self.convert_ob_to_goal_state(self._get_obs())
            - self.multitask_goal
        )
        return ob, reward, done, dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            distance=distance,
            full_state_to_goal_distance=full_state_to_goal_distance,
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-np.pi, high=np.pi,
                                      size=self.model.nq) + self.init_qpos
        if self._xy_desired_pos is None:
            while True:
                self._xy_desired_pos = self.np_random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self._xy_desired_pos) < 2:
                    break
        qpos[-2:] = self._xy_desired_pos
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

    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        statistics = OrderedDict()

        xy_distance_to_goal = get_stat_in_dict(
            paths, 'env_infos', 'distance'
        )
        statistics.update(create_stats_ordered_dict(
            'Euclidean distance to desired XY', xy_distance_to_goal
        ))
        statistics.update(create_stats_ordered_dict(
            'Final Euclidean distance to desired XY',
            xy_distance_to_goal[:, -1],
            always_show_all_stats=True,
        ))

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

    def sample_actions(self, batch_size):
        return 2 * np.random.beta(5, 5, size=(batch_size, 2)) - 1

    def sample_states(self, batch_size):
        theta = np.pi * (2 * np.random.rand(batch_size, 2) - 1)
        velocity = 10 * (2 * np.random.rand(batch_size, 2) - 1)
        return np.hstack((
            np.cos(theta),
            np.sin(theta),
            velocity,
        ))

    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        """
        :param goal: np.ndarray, shape GOAL_DIM
        :param batch_size:
        :return: ndarray, shape SAMPLE_SIZE x GOAL_DIM
        """
        return np.repeat(
            np.expand_dims(goal, 0),
            batch_size,
            axis=0
        )


class GoalStateSimpleStateReacherEnv(MultitaskReacherEnv):
    """
    The goal state is an actual state (6 dimensions--see parent class), rather
    than just the XY-coordinate of the target end effector.
    """

    def __init__(self):
        super().__init__()

    def set_goal(self, goal):
        super().set_goal(goal)
        self._xy_desired_pos = position_from_angles(
            np.expand_dims(goal, 0)
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

    def modify_goal_state_for_rollout(self, goal_state):
        # set desired velocity to zero
        goal_state[4:6] = 0
        return goal_state

    @property
    def goal_dim(self):
        return 6

    @staticmethod
    def print_goal_state_info(goal):
        c1 = goal[0:1]
        c2 = goal[1:2]
        s1 = goal[2:3]
        s2 = goal[3:4]
        print("Goal = ", goal)
        print("angle 1 (degrees) = ", np.arctan2(s1, c1) / math.pi * 180)
        print("angle 2 (degrees) = ", np.arctan2(s2, c2) / math.pi * 180)


    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        """
        Sample the goal a bunch of time, but fill in the desired position with
        what you care about.

        :param goal: np.ndarray, shape GOAL_DIM
        :param batch_size:
        :return: ndarray, shape SAMPLE_SIZE x GOAL_DIM
        """
        sampled_velocities = np.random.uniform(
            -1,
            1,
            size=(batch_size, 2),
        )
        goals = np.repeat(
            np.expand_dims(goal, 0),
            batch_size,
            axis=0
        )
        goals[:, 4:6] = sampled_velocities
        return goals


class GoalXYStateXYAndCosSinReacher2D(MultitaskReacherEnv):
    def sample_goal_states(self, batch_size):
        theta = self.np_random.uniform(
            low=-math.pi,
            high=math.pi,
            size=(batch_size, 2)
        )
        obs = np.hstack([
            np.cos(theta),
            np.sin(theta),
        ])
        return position_from_angles(obs)

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        obs = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.get_body_com("fingertip"),
            self.model.data.qvel.flat[:2],
        ])
        return obs

    def convert_obs_to_goal_states(self, obs):
        return obs[:, 4:6]

    def set_goal(self, goal):
        super().set_goal(goal)
        self._xy_desired_pos = goal

    @property
    def goal_dim(self):
        return 2


"""
Reward functions for optimal control
"""
theta1 = np.pi / 2
theta2 = np.pi / 2
REACH_A_POINT_GOAL = np.array([
    np.cos(theta1),
    np.cos(theta2),
    np.sin(theta1),
    np.sin(theta2),
    0,
    0,
])
SPEED_GOAL = 5
DESIRED_POSITION = np.array([-R2, R1])


def reach_a_point_reward(states):
    pos = position_from_angles_pytorch(states)
    desired_pos = ptu.np_to_var(DESIRED_POSITION, requires_grad=False)
    return - torch.norm(pos - desired_pos, p=2, dim=1)


def reach_a_joint_config_reward(states):
    goal_pos_pytorch = ptu.np_to_var(REACH_A_POINT_GOAL, requires_grad=False)
    return - torch.norm(states[:, :4] - goal_pos_pytorch[:4], p=2, dim=1)


def reach_a_point_and_move_joints_reward(states):
    pos = position_from_angles_pytorch(states)
    desired_pos = ptu.np_to_var(DESIRED_POSITION, requires_grad=False)
    speed = torch.norm(states[:, 2:], p=2, dim=1)
    return (
        - torch.abs(speed - SPEED_GOAL)
        - torch.norm(pos - desired_pos, p=2, dim=1)
    )


def hold_first_joint_and_move_second_joint_reward(states):
    second_joint_speed = states[:, 5]
    second_theta_is_large = (states[:, 1] > 0).float()
    second_joint_desired_speed = (
        SPEED_GOAL * (1-second_theta_is_large)
        - SPEED_GOAL * second_theta_is_large
    )
    return (
        - torch.abs(states[:, 4])
        - torch.abs(second_joint_speed - second_joint_desired_speed)
    )
