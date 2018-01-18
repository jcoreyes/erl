import numpy as np
from gym.envs.classic_control import MountainCarEnv

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.core.serializable import Serializable


class MountainCar(MountainCarEnv, MultitaskEnv, Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())
        super().__init__()
        MultitaskEnv.__init__(self)
        self.multitask_goal = np.array([0.5])

    def sample_goals(self, batch_size):
        return np.random.uniform(
            low=-1.2,
            high=0.6,
            size=(batch_size, 1),
        )

    def sample_goal_for_rollout(self):
        return np.array([0.5])

    @property
    def goal_dim(self) -> int:
        return 1

    def convert_obs_to_goals(self, obs):
        return obs[:, 0:1]
