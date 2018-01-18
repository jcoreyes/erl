import numpy as np
from gym.envs.classic_control import CartPoleEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.core.serializable import Serializable


class CartPole(CartPoleEnv, MultitaskEnv, Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())
        super().__init__()
        MultitaskEnv.__init__(self)
        self.multitask_goal = np.zeros(4)

    def sample_goals(self, batch_size):
        return np.random.uniform(
            low=[-self.x_threshold, -1, -self.theta_threshold_radians, -1],
            high=[self.x_threshold, -1, self.theta_threshold_radians, 1],
            size=(batch_size, 4),
        )

    def sample_goal_for_rollout(self):
        return np.zeros(4)

    @property
    def goal_dim(self) -> int:
        return 4

    def convert_obs_to_goals(self, obs):
        return obs


class CartPoleAngleOnly(CartPoleEnv, MultitaskEnv, Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())
        super().__init__()
        MultitaskEnv.__init__(self)
        self.multitask_goal = np.zeros(1)

    def sample_goals(self, batch_size):
        return np.random.uniform(
            low=-self.theta_threshold_radians,
            high=self.theta_threshold_radians,
            size=(batch_size, 1),
        )

    def sample_goal_for_rollout(self):
        return np.zeros(1)

    @property
    def goal_dim(self) -> int:
        return 1

    def convert_obs_to_goals(self, obs):
        return obs[:, 2:3]
