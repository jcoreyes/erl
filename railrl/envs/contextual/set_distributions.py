from railrl.core.distribution import DictDistribution
import numpy as np
from gym.spaces import Box

class GoalDictDistributionFromSet(DictDistribution):
    def __init__(
            self,
            set,
            desired_goal_keys=('desired_goal',),
    ):
        self.set = set
        self._desired_goal_keys = desired_goal_keys

        set_space = Box(
            -10 * np.ones(set.shape[1:]),
            10 * np.ones(set.shape[1:]),
            dtype=np.float32,
        )
        self._spaces = {
            k: set_space
            for k in self._desired_goal_keys
        }

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.set), batch_size)
        sampled_data = self.set[indices]
        return {
            k: sampled_data
            for k in self._desired_goal_keys
        }

    @property
    def spaces(self):
        return self._spaces