import numpy as np
from gym.envs.classic_control import CartPoleEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv
from rllab.core.serializable import Serializable


class CartPole(CartPoleEnv, MultitaskEnv, Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())
        super().__init__()

    def sample_goals(self, batch_size):
        return np.random.uniform(
            low=-self.theta_threshold_radians,
            high=self.theta_threshold_radians,
            size=(batch_size, 1),
        )

    def sample_goal_for_rollout(self):
        return np.array([0.])

    @property
    def goal_dim(self) -> int:
        return 1

    def convert_obs_to_goals(self, obs):
        return obs[:, 2:3]
