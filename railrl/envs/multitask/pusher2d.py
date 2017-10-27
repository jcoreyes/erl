import abc

import numpy as np
import torch

from railrl.envs.mujoco.pusher2d import Pusher2DEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv


class MultitaskPusher2DEnv(Pusher2DEnv, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(self, goal=(0, -1)):
        self.init_serialization(locals())
        super().__init__(goal=goal)
        MultitaskEnv.__init__(self)

    def sample_actions(self, batch_size):
        return np.random.uniform(
            self.action_space.low,
            self.action_space.high,
            (batch_size, 3),
        )

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)


class FullStatePusher2DEnv(MultitaskPusher2DEnv):
    def __init__(self, goal=(0, -1)):
        super().__init__(goal=goal)
        self.goal_dim_weights = np.array([.1, .1, .1, .1, .1, .1, 1, 1, 1, 1])

    def sample_goal_states(self, batch_size):
        # Joint angle and xy position won't be consistent, but oh well!
        return np.random.uniform(
            np.array([-2.5, -2.3213, -2.3213, -1, -1, -1, -1, -1, -1, -1]),
            np.array([2.5, 2.3, 2.3, 1, 1, 1, 1, 0, 1, 0]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 10

    def convert_obs_to_goal_states(self, obs):
        return obs

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_cylinder_position = goal[-2:]
        self._target_hand_position = goal[-4:-2]

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)

    @staticmethod
    def oc_reward(
            predicted_states, goal_states, current_states
    ):
        predicted_hand_pos = predicted_states[:, 6:8]
        predicted_cylinder_pos = predicted_states[:, 8:10]
        current_cylinder_pos = current_states[:, 8:10]
        desired_cylinder_pos = goal_states[:, 8:10]
        return -torch.norm(
            predicted_hand_pos - current_cylinder_pos,
            p=2,
            dim=1,
            keepdim=True,
        ) - torch.norm(
            predicted_cylinder_pos - desired_cylinder_pos,
            p=2,
            dim=1,
            keepdim=True,
        )

    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        desired_cylinder_pos = goal[8:10]
        current_cylinder_pos = obs[8:10]
        hand_pos = obs[6:8]

        if np.linalg.norm(hand_pos - current_cylinder_pos) <= 0.1:
            new_goal = np.hstack((
                current_cylinder_pos,
                desired_cylinder_pos,
            ))
        else:
            new_goal = np.hstack((
                current_cylinder_pos,
                # desired_cylinder_pos,
                current_cylinder_pos,
            ))
        goal_expanded = np.repeat(
            np.expand_dims(new_goal, 0),
            batch_size,
            axis=0
        )
        return np.hstack((
            # From the xml
            self.np_random.uniform(low=-2.5, high=2.5, size=(batch_size, 1)),
            self.np_random.uniform(low=-2.32, high=2.3, size=(batch_size, 1)),
            self.np_random.uniform(low=-2.32, high=2.3, size=(batch_size, 1)),
            # velocities
            self.np_random.uniform(low=-1, high=1, size=(batch_size, 3)),
            # self.np_random.uniform(low=-1, high=1, size=(batch_size, 2)),
            goal_expanded,
        ))

    @staticmethod
    def oc_reward(
            predicted_states, goal_states, current_states
    ):
        return FullStatePusher2DEnv.oc_reward_on_goals(
            predicted_states, goal_states, current_states
        )

    @staticmethod
    def oc_reward_on_goals(
            predicted_states, goal_states, current_states
    ):
        predicted_hand_pos = predicted_states[:, 6:8]
        predicted_cylinder_pos = predicted_states[:, 8:10]
        current_cylinder_pos = current_states[:, 8:10]
        desired_cylinder_pos = goal_states[:, 8:10]
        return -torch.norm(
            predicted_hand_pos - current_cylinder_pos,
            p=2,
            dim=1,
            keepdim=True,
        ) - torch.norm(
            predicted_cylinder_pos - desired_cylinder_pos,
            p=2,
            dim=1,
            keepdim=True,
        )
        # return -torch.norm(
        #     predicted_states[:, 4:5] - 1,
        #     p=2,
        #     dim=1,
        #     keepdim=True,
        # )


class HandCylinderXYPusher2DEnv(MultitaskPusher2DEnv):
    def __init__(self, goal=(0, -1)):
        super().__init__(goal=goal)
        self.goal_dim_weights = np.array([1, 1, 1, 1])

    def sample_goal_states(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1, -1., -1]),
            np.array([0, 1, 0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 4

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -4:]

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_cylinder_position = goal[-2:]
        self._target_hand_position = goal[-4:-2]

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)

    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        desired_cylinder_pos = goal[2:4]
        current_cylinder_pos = obs[8:10]

        hand_pos = obs[6:8]
        if np.linalg.norm(hand_pos - current_cylinder_pos) <= 0.1:
            new_goal = np.hstack((
                current_cylinder_pos,
                desired_cylinder_pos,
            ))
        else:
            new_goal = np.hstack((
                current_cylinder_pos,
                # desired_cylinder_pos,
                current_cylinder_pos,
            ))
        return np.repeat(
            np.expand_dims(new_goal, 0),
            batch_size,
            axis=0
        )
        # cyl_pos_exp = np.repeat(
        #     np.expand_dims(desired_cylinder_pos, 0),
        #     batch_size,
        #     axis=0
        # )
        # return np.hstack((
        #     np.random.uniform(-1, 1, (batch_size, 2)),
        #     cyl_pos_exp,
        # ))


    @staticmethod
    def oc_reward(
            predicted_states, goal_states, current_states
    ):
        return HandCylinderXYPusher2DEnv.oc_reward_on_goals(
            predicted_states[:, :4],
            goal_states,
            current_states,
        )

    @staticmethod
    def oc_reward_on_goals(
            predicted_goals, goal_states, current_states
    ):
        predicted_hand_pos = predicted_goals[:, :2]
        predicted_cylinder_pos = predicted_goals[:, 2:4]
        current_cylinder_pos = current_states[:, 8:10]
        desired_cylinder_pos = goal_states[:, 2:4]
        return -torch.norm(
            predicted_hand_pos - current_cylinder_pos,
            p=2,
            dim=1,
            keepdim=True,
        ) - torch.norm(
            predicted_cylinder_pos - desired_cylinder_pos,
            p=2,
            dim=1,
            keepdim=True,
        )


class NoShapeHandCylinderXYPusher2DEnv(HandCylinderXYPusher2DEnv):
    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        desired_cylinder_pos = goal[2:4]
        desired_cylinder_pos_expanded = np.repeat(
            np.expand_dims(desired_cylinder_pos, 0),
            batch_size,
            axis=0
        )
        return np.hstack((
            np.random.uniform(-1, 1, (batch_size, 2)),
            desired_cylinder_pos_expanded,
        ))


    @staticmethod
    def oc_reward(
            predicted_states, goal_states, current_states
    ):
        return HandCylinderXYPusher2DEnv.oc_reward_on_goals(
            predicted_states[:, :4],
            goal_states,
            current_states,
        )

    @staticmethod
    def oc_reward_on_goals(
            predicted_goals, goal_states, current_states
    ):
        predicted_cylinder_pos = predicted_goals[:, 2:4]
        desired_cylinder_pos = goal_states[:, 2:4]
        return - torch.norm(
            predicted_cylinder_pos - desired_cylinder_pos,
            p=2,
            dim=1,
            keepdim=True,
        )


class HandXYPusher2DEnv(MultitaskPusher2DEnv):
    """
    Only care about the hand position! This is really just for debugging.
    """
    def sample_goal_states(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1]),
            np.array([0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 2

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -4:-2]

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_hand_position = goal
        self._target_cylinder_position = np.random.uniform(-1, 1, 2)

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)

    @staticmethod
    def oc_reward(states, goals, current_states):
        """
        Reminder:

        ```
        def _get_obs(self):
            return np.concatenate([
                self.model.data.qpos.flat[:3],
                self.model.data.qvel.flat[:3],
                self.get_body_com("distal_4")[:2],
                self.get_body_com("object")[:2],
            ])
        ```

        :param states:
        :param goals:
        :return:
        """
        return HandXYPusher2DEnv.oc_reward_on_goals(
            states[:, 6:8], goals, current_states
        )

    @staticmethod
    def oc_reward_on_goals(goals_predicted, goals, current_states):
        return - torch.norm(goals_predicted - goals)

    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        new_goal = obs[8:10]

        # To cheat (e.g. for debugging):
        # hand_pos = obs[6:8]
        # cylinder_pos = obs[8:10]
        # if np.linalg.norm(hand_pos - cylinder_pos) <= 0.2:
        #     new_goal = self._target_cylinder_position
        # else:
        #     new_goal = obs[8:10]
        return np.repeat(
            np.expand_dims(new_goal, 0),
            batch_size,
            axis=0
        )


class FixedHandXYPusher2DEnv(HandXYPusher2DEnv):
    def sample_goal_state_for_rollout(self):
        return np.array([-1, 0])


class CylinderXYPusher2DEnv(MultitaskPusher2DEnv):
    def sample_goal_states(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1]),
            np.array([0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 2

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -2:]

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_cylinder_position = goal

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        self.set_state(qpos, qvel)
