import numpy as np

from railrl.core.serializable import Serializable
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.envs.pygame.point2d_uwall import Point2dUWall


class MultitaskPoint2dUWall(Point2dUWall, MultitaskEnv, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())
        Point2dUWall.__init__(self, **kwargs)
        MultitaskEnv.__init__(self)

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_position = goal

    def reset(self):
        self._target_position = self.multitask_goal
        return super().reset()

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goals(self, batch_size):
        return np.random.uniform(
            -self.OUTER_WALL_MAX_DIST,
            self.OUTER_WALL_MAX_DIST,
            (batch_size, 2)
        )

    def convert_obs_to_goals(self, obs):
        return obs
