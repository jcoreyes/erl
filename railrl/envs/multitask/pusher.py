import numpy as np
from gym.envs.mujoco import PusherEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv


class MultitaskPusherEnv(PusherEnv, MultitaskEnv):
    def sample_goal_states(self, batch_size):
        raise NotImplementedError("Sample from replay buffer for now.")

    @property
    def goal_dim(self):
        return 23

    def convert_obs_to_goal_states(self, obs):
        return obs

    def compute_rewards(self, obs, action, next_obs, goal_states):
        return - np.linalg.norm(next_obs - goal_states, axis=1)
