import numpy as np
import torch

from railrl.core.serializable import Serializable
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.envs.pygame.point2d import Point2DEnv
from railrl.torch.core import PyTorchModule


class MultitaskPoint2DEnv(Point2DEnv, MultitaskEnv):
    def __init__(
            self,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        Point2DEnv.__init__(self, **kwargs)
        MultitaskEnv.__init__(self)

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_position = goal

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goals(self, batch_size):
        return np.random.uniform(
            -self.BOUNDARY_DIST,
            self.BOUNDARY_DIST,
            (batch_size, 2)
        )

    def convert_obs_to_goals(self, obs):
        return obs[:, 2:]


class PerfectPoint2DQF(PyTorchModule):
    """
    Give the perfect QF for MultitaskPoint2D for discount/tau = 0.

    state = [x, y]
    action = [dx, dy]
    next_state = clip(state + action, -boundary, boundary)
    """
    def __init__(self, boundary_dist):
        super().__init__()
        self.boundary_dist = boundary_dist

    def forward(self, obs, action, goal_state, discount):
        next_state = torch.clamp(
            obs + action,
            -self.boundary_dist,
            self.boundary_dist,
        )
        return - torch.norm(next_state - goal_state, p=2, dim=1)