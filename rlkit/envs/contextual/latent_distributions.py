import numpy as np
from gym.spaces import Box, Dict

from rlkit.core.distribution import DictDistribution


class AddLatentDistribution(DictDistribution):
    def __init__(
            self,
            dist,
            input_key,
            output_key,
            model,
    ):
        self.dist = dist
        self._spaces = dist.spaces
        self.input_key = input_key
        self.output_key = output_key
        self.model = model
        self.representation_size = self.model.representation_size
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[output_key] = latent_space

    def sample(self, batch_size: int):
        s = self.dist.sample(batch_size)
        s[self.output_key] = self.model.encode_np(s[self.input_key])
        return s

    @property
    def spaces(self):
        return self._spaces


class PriorDistribution(DictDistribution):
    def __init__(
            self,
            representation_size,
            key,
            dist=None,
    ):
        self._spaces = dist.spaces if dist else {}
        self.key = key
        self.representation_size = representation_size
        self.dist = dist
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[key] = latent_space

    def sample(self, batch_size: int):
        s = self.dist.sample(batch_size) if self.dist else {}
        mu, sigma = 0, 1 # sample from prior
        n = np.random.randn(batch_size, self.representation_size)
        s[self.key] = sigma * n + mu
        return s

    @property
    def spaces(self):
        return self._spaces
