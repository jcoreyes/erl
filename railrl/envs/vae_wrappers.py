import mujoco_py
import numpy as np
import gym.spaces
import itertools
from gym import Env
from gym.spaces import Box
from scipy.misc import imresize

from railrl.core.serializable import Serializable
from gym.spaces import Discrete

from gym import Env
from railrl.envs.wrappers import ProxyEnv
import railrl.torch.pytorch_util as ptu
from torch.autograd import Variable

class VAEWrappedEnv(ProxyEnv, Env):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    """
    def __init__(self, wrapped_env, vae, use_vae_obs=True, use_vae_reward=True, use_vae_goals=True):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.vae = vae
        self.representation_size = self.vae.representation_size
        self.input_channels = self.vae.input_channels
        self.use_vae_goals = use_vae_goals
        self.use_vae_reward = use_vae_reward
        self.use_vae_obs = use_vae_obs
        if ptu.gpu_enabled():
            self.vae.cuda()

        self.observation_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self.goal_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )

    def step(self, action):
        observation, reward, done, info = self._wrapped_env.step(action)
        if self.use_vae_obs:
            img = Variable(ptu.from_numpy(observation))
            if ptu.gpu_enabled():
                self.vae.cuda()
            e = self.vae.encode(img)[0]
            observation = ptu.get_numpy(e).flatten()
        if self.use_vae_reward:
            # replace reward with Euclidean distance in VAE latent space
            # currently assumes obs and goals are also from VAE
            reward = -np.linalg.norm(self.multitask_goal - observation)
        return observation, reward, done, info

    def reset(self):
        observation = self._wrapped_env.reset()
        if self.use_vae_obs:
            img = Variable(ptu.from_numpy(observation))
            if ptu.gpu_enabled():
                self.vae.cuda()
            e = self.vae.encode(img)[0]
            observation = ptu.get_numpy(e).flatten()
        return observation

    def sample_goals(self, batch_size):
        if self.use_vae_goals:
            # sample from prior
            return np.random.randn(batch_size, self.representation_size)
        else:
            return self._wrapped_env.sample_goals(batch_size)

    def sample_goal_for_rollout(self):
        """
        These goals are fed to a policy when the policy wants to actually
        do rollouts.
        :return:
        """
        goal = self.sample_goals(1)[0]
        return self.modify_goal_for_rollout(goal)
