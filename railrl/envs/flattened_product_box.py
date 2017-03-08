import numpy as np
from rllab.envs.base import Env
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.overrides import overrides
from rllab.spaces.product import Product
from rllab.spaces.box import Box
from sandbox.rocky.tf.spaces.box import Box as TfBox


class FlattenedProductBox(ProxyEnv):
    """
    Converts a Product space of box envs into a flat tf Box env.
    """

    def __init__(self, env):
        assert isinstance(env.action_space, Product)
        assert isinstance(env.observation_space, Product)
        for c in env.action_space.components + env.observation_space.components:
            assert isinstance(c, Box) or isinstance(c, TfBox)

        super().__init__(env)

        self._action_space = self._create_flat_box(
            self.wrapped_env.action_space
        )
        self._observation_space = self._create_flat_box(
            self.wrapped_env.observation_space
        )

    @staticmethod
    def _create_flat_box(product_space):
        dim = product_space.flat_dim
        lows = np.vstack([c.low for c in product_space.components])
        highs = np.vstack([c.high for c in product_space.components])
        return TfBox(
            np.ones(dim) * np.min(lows),
            np.ones(dim) * np.max(highs),
        )

    def reset(self):
        unflat_obs = self._wrapped_env.reset()
        return self._flatten_obs(unflat_obs)

    def _flatten_action(self, action):
        return self._wrapped_env.action_space.flatten(action)

    def _unflatten_action(self, action):
        return self._wrapped_env.action_space.unflatten(action)

    def _flatten_obs(self, obs):
        return self._wrapped_env.observation_space.flatten(obs)

    def _unflatten_obs(self, obs):
        return self._wrapped_env.observation_space.unflatten(obs)

    def step(self, action):
        """
        :param action: A flat action
        :return: A flat observation.
        """
        unflat_action = self._unflatten_action(action)
        unflat_observation, reward, done, info = self._wrapped_env.step(
            unflat_action)
        return (
            self._flatten_obs(unflat_observation),
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
