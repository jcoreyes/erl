import abc
from collections import OrderedDict

import numpy as np
import torch

from railrl.envs.mujoco.pusher import PusherEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.misc import logger


class MultitaskPusherEnv(PusherEnv, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        MultitaskEnv.__init__(self)

    """
    Env Functions
    """
    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        statistics = OrderedDict()

        observations = np.vstack([path['observations'] for path in paths])
        goal_states = np.vstack([path['goal_states'] for path in paths])

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

    """
    MultitaskEnv Functions
    """
    @abc.abstractmethod
    def goal_state_to_cylinder_xy(self, goal_state):
        pass

    @property
    def goal_cylinder_xy(self):
        return self.goal_state_to_cylinder_xy(self.multitask_goal)


class HandPuckXYZPusher3DEnv(MultitaskPusherEnv):
    @property
    def goal_dim(self) -> int:
        return 6

    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        desired_cylinder_pos = goal[-3:]
        current_cylinder_pos = obs[-3:]

        hand_pos = obs[-6:-3]
        if np.linalg.norm(hand_pos - current_cylinder_pos) <= 0.1:
            new_goal = np.hstack((
                current_cylinder_pos,
                desired_cylinder_pos,
            ))
        else:
            new_goal = np.hstack((
                current_cylinder_pos,
                current_cylinder_pos,
            ))
        return np.repeat(
            np.expand_dims(new_goal, 0),
            batch_size,
            axis=0
        )

    def oc_reward_on_goals(
            self, predicted_goal_states, goal_states, current_states
    ):
        predicted_hand_pos = predicted_goal_states[:, -6:-3]
        predicted_puck_pos = predicted_goal_states[:, -3:]
        desired_hand_pos = goal_states[:, -6:-3]
        desired_puck_pos = goal_states[:, -3:]
        return -torch.norm(
            predicted_hand_pos - desired_hand_pos,
            p=2,
            dim=1,
            keepdim=True,
        ) - torch.norm(
            predicted_puck_pos - desired_puck_pos,
            p=2,
            dim=1,
            keepdim=True,
        )

    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        raise NotImplementedError()

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -6:]

    def goal_state_to_cylinder_xy(self, goal_state):
        return goal_state[-3:-1]

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])

    def sample_goal_states(self, batch_size):
        return self.np_random.uniform(
            np.array([-0.75, -1.25, -0.2, -0.3, -0.2, -0.275]),
            np.array([0.75, 0.25, 0.6, 0, 0.2, -0.275]),
            (batch_size, 6),
        )


class JointOnlyPusherEnv(MultitaskPusherEnv):
    def goal_state_to_cylinder_xy(self, goal_state):
        return goal_state[14:16]

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
            # cylinder xy location
            self.np_random.uniform(low=-0.3, high=0, size=(batch_size, 1)),
            self.np_random.uniform(low=-0.2, high=0.2, size=(batch_size, 1)),
            # cylinder z location is always fixed. Taken from xml.
            -0.275 * np.ones((batch_size, 1)),
        ))

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("object"),
        ])
