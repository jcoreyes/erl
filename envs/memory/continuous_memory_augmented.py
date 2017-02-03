import numpy as np
from rllab.envs.base import Env
from rllab.spaces.product import Product
from rllab.spaces.box import Box


class ContinuousMemoryAugmented(Env):
    """
    An environment that wraps another environments and adds continuous memory
    states/actions.
    """
    def __init__(self, env, num_memory_states=10):
        self._env = env
        self._num_memory_states = num_memory_states
        self._memory_state = np.zeros(self._num_memory_states)
        self._action_space = Product(
            env.action_space,
            self._memory_state_space()
        )
        self._observation_space = Product(
            env.observation_space,
            self._memory_state_space()
        )

    def _memory_state_space(self):
        return Box(-np.ones(self._num_memory_states),
                   np.ones(self._num_memory_states))

    def reset(self):
        self._memory_state = np.zeros(self._num_memory_states)
        env_obs = self._env.reset()
        return self.observation_space.flatten([env_obs, self._memory_state])

    @property
    def horizon(self):
        return self._env.horizon

    def step(self, action):
        env_action, memory_state = self.action_space.unflatten(action)
        observation, reward, done, info = self._env.step(env_action)
        return (
            self.observation_space.flatten([observation, memory_state]),
            reward,
            done,
            info
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
