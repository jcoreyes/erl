import numpy as np
import torch
from gym.envs.mujoco import HalfCheetahEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv


class GoalXVelHalfCheetah(HalfCheetahEnv, MultitaskEnv):
    def __init__(self):
        super().__init__()
        MultitaskEnv.__init__(self)

    def sample_actions(self, batch_size):
        return np.random.uniform(-0.5, -0.5, (batch_size, 6))

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goal_states(self, batch_size):
        return np.random.uniform(-10, 10, (batch_size, 1))

    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        raise NotImplementedError()

    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        # return np.random.uniform(-10, 10, batch_size)
        return np.repeat(
            np.expand_dims(goal, 0),
            batch_size,
            axis=0
        )

    def convert_obs_to_goal_states(self, obs):
        return obs[:, 8:9]

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

    def oc_reward(self, states, goals, current_states):
        return self.oc_reward_on_goals(
            self.convert_obs_to_goal_states(states),
            goals,
            current_states,
        )

    def oc_reward_on_goals(self, goals_predicted, goals, current_states):
        return - torch.norm(goals_predicted - goals, dim=1, p=2, keepdim=True)
