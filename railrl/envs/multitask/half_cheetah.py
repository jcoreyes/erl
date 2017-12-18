from collections import OrderedDict

import numpy as np
import torch
from gym.envs.mujoco import HalfCheetahEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.samplers.util import get_stat_in_paths
from rllab.core.serializable import Serializable
from rllab.misc import logger

MAX_SPEED = 6


class GoalXVelHalfCheetah(HalfCheetahEnv, MultitaskEnv):
    def __init__(self):
        self.target_x_vel = np.random.uniform(-MAX_SPEED, MAX_SPEED)
        super().__init__()
        MultitaskEnv.__init__(self)
        self.set_goal(np.array([5]))

    def sample_actions(self, batch_size):
        return np.random.uniform(-0.5, -0.5, (batch_size, 6))

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goals(self, batch_size):
        return np.random.uniform(-MAX_SPEED, MAX_SPEED, (batch_size, 1))

    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        raise NotImplementedError()

    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        # return np.random.uniform(-10, 10, batch_size)
        return np.repeat(
            np.expand_dims(goal, 0),
            batch_size,
            axis=0
        )

    def convert_obs_to_goals(self, obs):
        return obs[:, 8:9]

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.target_x_vel = goal

    def _step(self, action):
        ob, _, done, info_dict = super()._step(action)
        xvel = ob[8]
        desired_xvel = self.target_x_vel
        xvel_error = np.linalg.norm(xvel - desired_xvel)
        reward = - xvel_error
        info_dict['xvel'] = xvel
        info_dict['desired_xvel'] = desired_xvel
        info_dict['xvel_error'] = xvel_error
        return ob, reward, done, info_dict

    def sample_states(self, batch_size):
        raise NotImplementedError()
        # lows = np.array([
        #     -0.59481074,
        #     - 3.7495437,
        #     - 0.71380705
        #     - 0.96735003,
        #     - 0.57593354,
        #     - 1.15863039,
        #     - 1.24097252,
        #     - 0.6463361,
        #     - 3.66419601,
        #     - 4.39410921,
        #     - 9.04578552,
        #     - 27.18058883,
        #     - 30.22956479,
        #     - 26.8349202,
        #     - 28.4277106,
        #     - 30.47684186,
        #     - 22.79845961,
        #     ])
        # highs = np.array([
        #     0.55775534,
        #     10.39850087,
        #     1.0833258,
        #     0.91681375,
        #     0.89186029,
        #     0.91657275,
        #     1.13528496,
        #     0.69514478,
        #     3.98017764,
        #     5.17706281,
        #     8.30073489,
        #     25.93850538,
        #     27.8804229,
        #     23.84783459,
        #     30.58961975,
        #     36.80954249,
        #     24.14562621,
        # ])
        #
        # return np.random.uniform(lows, highs, batch_size)

    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)
        xvels = get_stat_in_paths(
            paths, 'env_infos', 'xvel'
        )
        desired_xvels = get_stat_in_paths(
            paths, 'env_infos', 'desired_xvel'
        )
        xvel_errors = get_stat_in_paths(
            paths, 'env_infos', 'xvel_error'
        )

        statistics = OrderedDict()
        for stat, name in [
            (xvels, 'xvels'),
            (desired_xvels, 'desired xvels'),
            (xvel_errors, 'xvel errors'),
        ]:
            statistics.update(create_stats_ordered_dict(
                '{}'.format(name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def oc_reward(self, states, goals, current_states):
        return self.oc_reward_on_goals(
            self.convert_obs_to_goals(states),
            goals,
            current_states,
        )

    def oc_reward_on_goals(self, goals_predicted, goals, current_states):
        return - torch.norm(goals_predicted - goals, dim=1, p=2, keepdim=True)

    def compute_her_reward_pytorch(
            self,
            observations,
            actions,
            next_observations,
            goal_states,
    ):
        xvels = observations[:, 8:9]
        target_xvels = goal_states
        costs = torch.norm(
            xvels - target_xvels,
            dim=1,
            p=2,
            keepdim=True,
        )
        return - costs


# At a score of 5000, the cheetah is moving at "5 meters per dt" but dt = 0.05s,
# so it's really 100 m/s.
# For a horizon of 50, that's 2.5 seconds.
# So 5 m/s * 2.5s = 12.5 meters
GOAL_X_POSITIONS = [12.5, -12.5]


class GoalXPosHalfCheetah(HalfCheetahEnv, MultitaskEnv, Serializable):
    def __init__(self, goal_x_positions=GOAL_X_POSITIONS):
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(self)
        super().__init__()
        self.goal_x_positions = np.array(goal_x_positions)
        self.set_goal(self.goal_x_positions[0])

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def _step(self, action):
        ob, reward, done, info_dict = super()._step(action)
        xposafter = self.model.data.qpos[0, 0]
        reward_ctrl = info_dict['reward_ctrl']
        distance_to_goal = float(np.abs(xposafter - self.multitask_goal))
        reward_run = -distance_to_goal
        reward = float(reward_ctrl + reward_run)
        return ob, reward, done, dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            goal_position=self.multitask_goal,
            distance_to_goal=distance_to_goal,
            position=xposafter,
        )

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goals(self, batch_size):
        return np.random.choice(self.goal_x_positions, (batch_size, 1))

    def convert_obs_to_goals(self, obs):
        return obs[:, 17:18]

    def log_diagnostics(self, paths):
        distances_to_goal = get_stat_in_paths(
            paths, 'env_infos', 'distance_to_goal'
        )
        goal_positions = get_stat_in_paths(
            paths, 'env_infos', 'goal_position'
        )
        positions = get_stat_in_paths(
            paths, 'env_infos', 'position'
        )
        statistics = OrderedDict()
        for stat, name in [
            (distances_to_goal, 'Distances to goal'),
            (goal_positions, 'Goal Position'),
            (positions, 'Position'),
            ([p[-1] for p in positions], 'Final Position'),
        ]:
            statistics.update(create_stats_ordered_dict(
                '{}'.format(name),
                stat,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        for stat, name in [
            (distances_to_goal, 'Distances to goal'),
        ]:
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)

    @staticmethod
    def cost_fn(states, actions, next_states):
        """
        This is added for Abhishek's model-based code.
        """
        input_is_flat = len(next_states.shape) == 1
        if input_is_flat:
            next_states = np.expand_dims(next_states, 0)
        actual_xpos = next_states[:, 17:18]
        desired_xpos = GOAL_X_POSITIONS[0] * np.ones_like(actual_xpos)
        costs = np.linalg.norm(
            desired_xpos - actual_xpos,
            axis=1,
            ord=1,
        )
        if input_is_flat:
            costs = costs[0]
        return costs
