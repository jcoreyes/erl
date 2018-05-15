import abc

import numpy as np
import torch
from gym.spaces import Box

from railrl.envs.mujoco.pusher2d import Pusher2DEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv

from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths
from railrl.core import logger as default_logger
from collections import OrderedDict

class MultitaskPusher2DEnv(Pusher2DEnv, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)
        MultitaskEnv.__init__(self, **kwargs)

    def sample_actions(self, batch_size):
        return np.random.uniform(
            self.action_space.low,
            self.action_space.high,
            (batch_size, 3),
        )

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths, **kwargs):
        super().log_diagnostics(paths, **kwargs)
        MultitaskEnv.log_diagnostics(self, paths, **kwargs)


class FullStatePusher2DEnv(MultitaskPusher2DEnv):
    def __init__(self, goal=(0, -1)):
        super().__init__(goal=goal)
        self.goal_dim_weights = np.array([.1, .1, .1, .1, .1, .1, 1, 1, 1, 1])

    def sample_goals(self, batch_size):
        # Joint angle and xy position won't be consistent, but oh well!
        return np.random.uniform(
            np.array([-2.5, -2.3213, -2.3213, -1, -1, -1, -1, -1, -1, -1]),
            np.array([2.5, 2.3, 2.3, 1, 1, 1, 1, 0, 1, 0]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 10

    def convert_obs_to_goals(self, obs):
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

    def sample_goals(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1, -1., -1]),
            np.array([0, 1, 0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 4

    def convert_obs_to_goals(self, obs):
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


class LessShapeHandCylinderXYPusher2DEnv(HandCylinderXYPusher2DEnv):
    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        desired_cylinder_pos = goal[2:4]
        current_cylinder_pos = obs[8:10]

        new_goal = np.hstack((
            current_cylinder_pos,
            desired_cylinder_pos,
        ))
        return np.repeat(
            np.expand_dims(new_goal, 0),
            batch_size,
            axis=0
        )

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
    def __init__(self, **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.goal_space = Box(
            low=np.array([-1, -1]),
            high=np.array([0, 0]),
        )

    def sample_goals(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1]),
            np.array([0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 2

    def convert_obs_to_goals(self, obs):
        return obs[:, -4:-2]

    def set_goal(self, goal):
        super().set_goal(goal)
        if self.ignore_multitask_goal:
            self._target_hand_position = np.random.uniform(
                np.array([-1, -1]),
                np.array([0, 1]),
                (self.goal_dim,)
            )
        else:
            self._target_hand_position = goal
        self._target_cylinder_position = np.random.uniform(-1, 1, 2)

        qpos = self.sim.data.qpos.flat.copy()
        qvel = self.sim.data.qvel.flat.copy()
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
    def __init__(self, **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.goal_space = Box(
            low=np.array([-1, -1]),
            high=np.array([0, 0]),
        )

    def sample_goals(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1]),
            np.array([0, 0]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 2

    def convert_obs_to_goals(self, obs):
        return obs[:, -2:]

    def set_goal(self, goal):
        super().set_goal(goal)
        if self.ignore_multitask_goal:
            self._target_hand_position = np.random.uniform(
                np.array([-1, -1]),
                np.array([0, 1]),
                (self.goal_dim,)
            )
            self._target_cylinder_position = self._target_hand_position
        else:
            self._target_hand_position = goal
            self._target_cylinder_position = goal

        qpos = self.sim.data.qpos.flat.copy()
        qvel = self.sim.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)

    def compute_her_reward_np(
            self,
            observation,
            action,
            next_observation,
            goal,
    ):
        hand_pos = next_observation[6:8]
        cylinder_pos = next_observation[8:10]
        target_pos = goal
        hand_to_puck_dist = np.linalg.norm(hand_pos - cylinder_pos)
        puck_to_goal_dist = np.linalg.norm(cylinder_pos - target_pos)
        if self.use_sparse_rewards:
                return float(puck_to_goal_dist < 0.1)
        else:
            if self.use_hand_to_obj_reward:
                return - hand_to_puck_dist - puck_to_goal_dist
            else:
                return - puck_to_goal_dist

    def compute_her_reward_pytorch(
            self,
            observations,
            actions,
            next_observations,
            goal_states,
    ):
        hand_pos = observations[:, 6:8]
        cylinder_pos = observations[:, 8:10]
        target_pos = goal_states
        hand_to_puck_dist = torch.norm(
            hand_pos - cylinder_pos,
            dim=1,
            p=2,
            keepdim=True,
        )
        costs = hand_to_puck_dist
        hand_is_close_to_puck = (hand_to_puck_dist <= 0.1).float()
        puck_to_goal_dist = torch.norm(
            cylinder_pos - target_pos,
            dim=1,
            p=2,
            keepdim=True,
        )
        costs = costs + (puck_to_goal_dist - 2) * hand_is_close_to_puck
        return - costs


class FullPusher2DEnv(MultitaskPusher2DEnv):
    """Randomize joints fully
    qpos:
    3 arm joints
    2 puck joints
    2 hand xy goal
    2 puck xy goal
    """
    def __init__(self, include_puck=True, arm_range=0.1, reward_params={"type": "euclidean", "puck_reward_only": False}, **kwargs):
        self.quick_init(locals())
        self.include_puck = include_puck
        self.arm_range = arm_range
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", "euclidean")
        self.epsilon = self.reward_params.get("epsilon", 0.5 if include_puck else 0.25)
        self.puck_reward_only = self.reward_params.get("puck_reward_only", False)
        if include_puck:
            if self.puck_reward_only:
                self.goal_space = Box(
                    low=np.array([-0.8, -0.8]),
                    high=np.array([0.8, -0.3]),
                    dtype=np.float32,
                )
            else:
                self.goal_space = Box(
                    low=np.array([-arm_range, -arm_range, -arm_range, -1, -1]),
                    high=np.array([arm_range, arm_range, arm_range, 1, 0]),
                    dtype=np.float32,
                )
            self.obs_dim = 5
        else:
            self.goal_space = Box(
                low=np.array([-arm_range, -arm_range, -arm_range]),
                high=np.array([arm_range, arm_range, arm_range]),
                # dtype=np.float32,
            )
            self.obs_dim = 3
        self.goal_slice = slice(0, self.goal_dim)
        if self.puck_reward_only:
            self.goal_slice = slice(3, 5)
        self.current_goal = self.goal_space.sample()
        super().__init__(goal=self.current_goal, **kwargs)
        self.observation_space = Box(
            low=-5 * np.ones((self.obs_dim)),
            high=5 * np.ones((self.obs_dim)),
            dtype=np.float32,
        )

    def get_qpos(self):
        return self.sim.data.qpos.copy()

    def sample_goals(self, batch_size):
        return np.vstack([self.goal_space.sample() for i in range(batch_size)])

    @property
    def goal_dim(self):
        return self.goal_space.low.shape[0]

    def set_goal(self, goal):
        super().set_goal(goal)
        self.current_goal = goal
        if self.include_puck:
            if self.puck_reward_only:
                self._target_hand_position = np.zeros((3))
                self._target_cylinder_position = goal
            else:
                self._target_hand_position = goal[:3]
                self._target_cylinder_position = goal[3:5]
        else:
            self._target_hand_position = goal
            self._target_cylinder_position = np.array((6, 6))

        qpos = self.sim.data.qpos.flat.copy()
        qvel = self.sim.data.qvel.flat.copy()
        qpos[:3] = self._target_hand_position
        qpos[3:5] = self._target_cylinder_position
        # qpos[-4:-2] = self._target_cylinder_position
        # qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)

    def step(self, u):
        ob, _, done, info = super().step(u)
        reached_goal = ob[self.goal_slice]
        dist = np.linalg.norm(reached_goal - self.current_goal)
        info["dist"] = dist
        info["success"] = int(dist < self.epsilon)
        if self.reward_type == "sparse":
            reward = 0 if dist < self.epsilon else -1
        else:
            reward = -dist
        # print(info["dist"], info["success"])
        return ob, reward, done, info

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.qvel[:3] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:5],
            # self.sim.data.qvel.flat[:3],
            self.get_body_com("distal_4")[:2],
            # self.get_body_com("object")[:2],
        ])[:self.obs_dim]

    def reset_model(self):
        # fully random
        # low = np.array([-2.5, -2.32, -2.32, -1, -1, 5, 5, 5, 5])
        # high = np.array([2.5, 2.32, 2.32, 1, 1, 6, 6, 6, 6])
        r = self.arm_range
        if self.include_puck:
            low = np.array([0, 0, 0, 0.3, -0.8, 6, 6, 6, 6])
            high = np.array([0, 0, 0, 0.8, -0.3, 6, 6, 6, 6])
        else:
            low = np.array([-r, -r, -r, 6, 6, 6, 6, 6, 6])
            high = np.array([r, r, r, 6, 6, 6, 6, 6, 6])
        qpos = (
            np.random.uniform(low=low, high=high, size=self.model.nq)
        )
        qvel = qpos * 0

        self.set_state(qpos, qvel)

        return self._get_obs()

    def convert_obs_to_goals(self, obs):
        return obs[:, self.goal_slice]

    def compute_her_reward_np(
            self,
            observation,
            action,
            next_observation,
            goal,
    ):
        reached_goal = next_observation[self.goal_slice]
        dist = np.linalg.norm(reached_goal - goal)
        if self.reward_type == "sparse":
            reward = 0 if dist < self.epsilon else -1
        else:
            reward = -dist
        return reward

    def sample_env_goal(self):
        return self.sample_goal_for_rollout()

    def log_diagnostics(self, paths, logger=default_logger, **kwargs):
        super().log_diagnostics(paths, logger=logger, **kwargs)

        statistics = OrderedDict()

        for stat_name_in_paths in ["dist", "success"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)
