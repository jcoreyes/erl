import numpy as np

from railrl.core.serializable import Serializable
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.envs.pygame.point2d_wall import Point2dWall


class MultitaskPoint2dWall(Point2dWall, MultitaskEnv, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())
        Point2dWall.__init__(self, **kwargs)
        MultitaskEnv.__init__(self)
        self.ob_to_goal_slice = slice(0, 2)

    def set_goal(self, goal):
        goal = np.clip(
            goal,
            a_min=-self.OUTER_WALL_MAX_DIST,
            a_max=self.OUTER_WALL_MAX_DIST,
        )
        super().set_goal(goal)
        self._target_position = goal

    def reset(self):
        self._target_position = self.multitask_goal
        return super().reset()

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goals(self, batch_size):
        # goal = np.array([[0, self.OUTER_WALL_MAX_DIST]])
        # return goal.repeat(batch_size, 1)
        return np.random.uniform(
            -self.OUTER_WALL_MAX_DIST,
            self.OUTER_WALL_MAX_DIST,
            (batch_size, 2)
        )

    def convert_obs_to_goals(self, obs):
        return obs
