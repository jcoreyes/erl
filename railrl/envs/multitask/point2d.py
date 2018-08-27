import numpy as np
import torch
from gym.spaces import Box

from railrl.core.serializable import Serializable
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.envs.pygame.point2d import Point2DEnv
from railrl.torch.core import PyTorchModule
from torch.autograd import Variable

import joblib
import railrl.torch.pytorch_util as ptu

class MultitaskPoint2DEnv(Point2DEnv, MultitaskEnv):
    def __init__(
            self,
            use_sparse_rewards=False,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        Point2DEnv.__init__(self, **kwargs)
        MultitaskEnv.__init__(self)
        self.use_sparse_rewards = use_sparse_rewards
        self.ob_to_goal_slice = slice(0, 2)
        self.observation_space = Box(
            -self.BOUNDARY_DIST * np.ones(2),
            self.BOUNDARY_DIST * np.ones(2)
            #dtype=np.float32,
        )
        self.goal_space = Box(
            -self.BOUNDARY_DIST * np.ones(2),
            self.BOUNDARY_DIST * np.ones(2)
            #dtype=np.float32,
        )

    def step(self, u):
        observation, reward, done, info = self._step(u)
        done = False # no early termination
        return observation, reward, done, info

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_position = goal

    def reset(self):
        self._target_position = self.multitask_goal
        self._position = np.random.uniform(
            size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
        )
        return self._get_observation()

    def compute_her_reward_np(
            self,
            observation,
            action,
            next_observation,
            goal,
            env_info=None
    ):
        dist = np.linalg.norm(next_observation - goal)
        if self.use_sparse_rewards:
            return -float(dist > 0.1)
        else:
            return -dist

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

    def get_qpos(self):
        return self._position.copy()

class MultitaskImagePoint2DEnv(MultitaskPoint2DEnv, MultitaskEnv):
    def _get_observation(self):
        return self.get_image()

    def set_goal(self, goal):
        super().set_goal(goal)
        self._position = goal
        self._target_position = goal

    # def reset(self):
    #     goal = self.sample_goals(1)
    #     self.set_goal(goal[0, :])
    #     return super().reset()

class MultitaskVAEPoint2DEnv(MultitaskImagePoint2DEnv, MultitaskEnv):
    def __init__(
            self,
            render_size=84,
            ball_radius=1,
            representation_size=2,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.vae = joblib.load("/home/ashvin/data/s3doodad/ashvin/vae/point2d-conv/run2/id0/params.pkl")
        self.vae.to(ptu.device)
        super().__init__(render_size=render_size, ball_radius=ball_radius, **kwargs)

        self.representation_size = representation_size
        self.observation_space = Box(
            -self.BOUNDARY_DIST * np.ones(representation_size),
            self.BOUNDARY_DIST * np.ones(representation_size),
            dtype=np.float32,
        )

    def _get_observation(self):
        img = Variable(ptu.from_numpy(self.get_image()))
        # import pdb; pdb.set_trace()
        if ptu.gpu_enabled():
            self.vae.to(ptu.device)
        e = self.vae.encode(img)[0]
        return ptu.get_numpy(e).flatten()

    def reset(self):
        goal = self.sample_goals(1)
        self.set_goal(goal[0, :])
        return super().reset()

class MultitaskFullVAEPoint2DEnv(MultitaskVAEPoint2DEnv, MultitaskEnv):
    def sample_goals(self, batch_size):
        return np.random.randn(batch_size, self.representation_size)

    def _reward(self):
        distance_to_target = np.linalg.norm(
            self._get_observation() - self._target_position
        )
        return -distance_to_target

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
