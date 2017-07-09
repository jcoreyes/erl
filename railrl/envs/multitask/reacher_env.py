import numpy as np
from gym.envs.mujoco import ReacherEnv


class MultitaskReacherEnv(ReacherEnv):
    def sample_goal_states(self, batch_size):
        return self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=(batch_size, self.model.nq)
        ) + self.init_qpos

    def compute_rewards(self, obs, action, next_obs, goal_states):
        next_qpos = next_obs[:, 4:6]
        return -(next_qpos - goal_states)**2
