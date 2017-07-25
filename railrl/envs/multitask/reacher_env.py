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

from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.misc import logger


class XyMultitaskReacherEnv(ReacherEnv):
    """
    The goal states are xy-coordinates.
    """
    R1 = 0.1  # from reacher.xml
    R2 = 0.11

    def sample_goal_states(self, batch_size):
        return self.np_random.uniform(
            low=-0.2,
            high=0.2,
            size=(batch_size, 2)
        )

    def compute_rewards(self, obs, action, next_obs, goal_states):
        c1 = next_obs[:, 0:1]  # cosine of angle 1
        c2 = next_obs[:, 1:2]
        s1 = next_obs[:, 2:3]
        s2 = next_obs[:, 3:4]
        next_qpos = (  # forward kinematics equation for 2-link robot
            self.R1 * np.hstack([c1, s1])
            + self.R2 * np.hstack([
                c1 * c2 - s1 * s2,
                s1 * c2 + c1 * s2,
            ])
        )
        return -np.linalg.norm(next_qpos - goal_states, axis=1)

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

    @property
    def goal_dim(self):
        return 2


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
    R1 = 0.1  # from reacher.xml
    R2 = 0.11

    def __init__(self, add_noop_action=True):
        """
        :param add_noop_action: If True, add an extra no-op after every call to
        the simulator. The reason this is done is so that your current action
        (torque) will affect your next position.
        """
        self.add_noop_action = add_noop_action
        utils.EzPickle.__init__(self, add_noop_action=add_noop_action)
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
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        if self.add_noop_action:
            # Make it so that your actions (torque) actually affect the next
            # observation position.
            self.do_simulation(np.zeros_like(a), self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1,
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
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
        ])

    def sample_goal_states(self, batch_size):
        return self.np_random.uniform(
            low=-0.2,
            high=0.2,
            size=(batch_size, 2)
        )

    def compute_rewards(self, obs, action, next_obs, goal_states):
        next_endeffector_positions = self.position(next_obs)
        return -np.linalg.norm(next_endeffector_positions - goal_states, axis=1)

    def position(self, obs):
        c1 = obs[:, 0:1]  # cosine of angle 1
        c2 = obs[:, 1:2]
        s1 = obs[:, 2:3]
        s2 = obs[:, 3:4]
        return (  # forward kinematics equation for 2-link robot
            self.R1 * np.hstack([c1, s1])
            + self.R2 * np.hstack([
                c1 * c2 - s1 * s2,
                s1 * c2 + c1 * s2,
            ])
        )

    def log_diagnostics(self, paths):
        observations = np.vstack([path['observations'][:, :4] for path in
                                  paths])
        positions = self.position(observations)
        goal_positions = np.vstack([path['observations'][:, -2:] for path in
                                   paths])
        distances = np.linalg.norm(positions - goal_positions, axis=1)

        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Distance to target', distances
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    @property
    def goal_dim(self):
        return 2


class GoalStateSimpleStateReacherEnv(XyMultitaskSimpleStateReacherEnv):
    """
    The goal state is an actual state (6 dimensions--see parent class), rather
    than just the XY-coordinate of the target end effector.
    """
    def set_goal(self, goal_state):
        c1 = goal_state[0:1]
        c2 = goal_state[1:2]
        s1 = goal_state[2:3]
        s2 = goal_state[3:4]
        self._fixed_goal = (  # forward kinematics equation for 2-link robot
            self.R1 * np.hstack([c1, s1])
            + self.R2 * np.hstack([
                c1 * c2 - s1 * s2,
                s1 * c2 + c1 * s2,
            ])
        )

    def sample_goal_states(self, batch_size):
        theta = self.np_random.uniform(
            low=-math.pi,
            high=math.pi,
            size=(batch_size, 2)
        )
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            np.zeros((batch_size, 2))
        ])

    def compute_rewards(self, obs, action, next_obs, goal_states):
        return -np.linalg.norm(next_obs - goal_states, axis=1)

    def log_diagnostics(self, paths):
        observations = np.vstack([path['observations'][:, :6] for path in
                                  paths])
        goal_states = np.vstack([path['observations'][:, -6:] for path in
                                    paths])
        rewards = self.compute_rewards(None, None, observations, goal_states)
        positions = self.position(observations)
        goal_positions = self.position(
            np.vstack([path['observations'][:, -6:] for path in paths])
        )
        distances = np.linalg.norm(positions - goal_positions, axis=1)

        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Distance to target', distances
        ))
        statistics.update(create_stats_ordered_dict(
            'Rewards', rewards,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    @property
    def goal_dim(self):
        return 6
