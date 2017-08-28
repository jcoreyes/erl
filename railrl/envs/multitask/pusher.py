import numpy as np
from gym.envs.mujoco import PusherEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv


class MultitaskPusherEnv(PusherEnv, MultitaskEnv):
    def sample_goal_states(self, batch_size):
        if batch_size != 1:
            raise NotImplementedError("Sample from replay buffer for now.")
        goal = np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            # self.get_body_com("tips_arm"),
            # self.get_body_com("goal"),  # try to move the arm to the goal
            self.get_body_com("object"),  # try to move the arm to the object
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
        return np.array([goal])

    @property
    def goal_dim(self):
        return 23

    def convert_obs_to_goal_states(self, obs):
        return obs

    def compute_rewards(self, obs, action, next_obs, goal_states):
        return - np.linalg.norm(next_obs - goal_states, axis=1)
