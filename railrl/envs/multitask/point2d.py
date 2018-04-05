import numpy as np
import torch
from gym.spaces import Box

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
        self.ob_to_goal_slice = slice(0, 2)
        self.observation_space = Box(
            -self.BOUNDARY_DIST * np.ones(2),
            self.BOUNDARY_DIST * np.ones(2),
        )

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_position = goal

    def reset(self):
        self._target_position = self.multitask_goal
        self._position = np.random.uniform(
            size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
        )
        while self.is_on_platform():
            self._position = np.random.uniform(
                size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
            )
        return self._get_observation()

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
        return obs

    def _get_observation(self):
        return self._position

    def log_diagnostics(self, paths, **kwargs):
        Point2DEnv.log_diagnostics(self, paths, **kwargs)
        MultitaskEnv.log_diagnostics(self, paths, **kwargs)


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


class CustomBeta(PyTorchModule):
    def __init__(self, env):
        super().__init__()
        self.quick_init(locals())
        self.env = env

    def forward(self, observations, actions, goals, num_steps_left):
        squared_distance = ((observations + actions - goals)**2).sum(dim=1,
                                                             keepdim=True)
        return torch.exp(-squared_distance)