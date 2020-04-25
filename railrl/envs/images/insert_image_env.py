import gym
import numpy as np
from gym.spaces import Box, Dict

from railrl.envs.images import Renderer


class InsertImageEnv(gym.Wrapper):
    def __init__(
            self,
            wrapped_env: gym.Env,
            renderer: Renderer,
            image_key='image_observation',
    ):
        """
        Appends images to the observation.
        """
        super().__init__(wrapped_env)
        self._renderer = renderer
        if self._renderer.normalize:
            img_space = Box(0, 1, renderer.image_shape, dtype=np.float32)
        else:
            img_space = Box(0, 255, renderer.image_shape, dtype=np.uint8)
        spaces = self.env.observation_space.spaces.copy()
        spaces[image_key] = img_space
        self._image_key = image_key
        self.observation_space = Dict(spaces)
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._update_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._update_obs(obs)
        return obs

    def _update_obs(self, obs):
        img_obs = self._get_image()
        obs[self._image_key] = img_obs

    def _get_image(self):
        return self._renderer.create_image(self.env)
