import os
from gym import spaces
import numpy as np
import gym


class Point(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self):
        self.goal = np.zeros((2,))
        self.state = np.zeros((2,))

    @property
    def action_space(self):
        return spaces.Box(low=-0.2*np.ones((2,)), high=0.2*np.ones((2,)))

    @property
    def observation_space(self):
        return spaces.Box(low=-5*np.ones((4,)), high=5*np.ones((4,)))


    def reset(self):
        self.state = np.zeros((2,))
        self.goal = np.random.uniform(-5, 5, size=(2,))
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -0.2, 0.2)
        new_state = self.state + action
        new_state = np.clip(new_state, -5, 5)
        self.state = new_state
        reward = -np.linalg.norm(new_state - self.goal)

        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        return np.concatenate([self.state, self.goal])
