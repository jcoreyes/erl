import numpy as np

from railrl.envs.mujoco.pusher2d import Pusher2DEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv


class MultitaskPusher2DEnv(Pusher2DEnv, MultitaskEnv):
    def __init__(self, goal=(0, -1)):
        self.init_serialization(locals())
        super().__init__(goal=goal)
        self._multitask_goal = goal

    def sample_actions(self, batch_size):
        return np.random.uniform(self.action_space.low, self.action_space.high)

    def sample_goal_states(self, batch_size):
        return self.sample_states(batch_size)

    def set_goal(self, goal):
        self._goal = goal[-2:]

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-2:] = self._goal
        self.set_state(qpos, qvel)

    @property
    def goal_dim(self):
        return 8

    def sample_states(self, batch_size):
        """
        From XML. Also setting the goal to always be on bottom half.
        dimension meanings:
        1. joint 1 angle
        2. joint 2 angle
        3. joint 3 angle
        4. joint 1 angular velocity
        5. joint 2 angular velocity
        6. joint 3 angular velocity
        7. cyclinder y position (not x)
        8. cyclinder x position

        :param batch_size:
        :return:
        """
        return np.random.uniform(
            np.array([-2.5, -2.3213, -2.3213, -1, -1, -1, -1., -1]),
            np.array([2.5, 2.3, 2.3, 1, 1, 1, 0, 1]),
            (batch_size, 8)
        )

    def modify_goal_state_for_rollout(self, goal_state):
        goal_state[3:6] = 0
        return goal_state
