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

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.envs.wrappers import ProxyEnv
import railrl.torch.pytorch_util as ptu
from torch.autograd import Variable

from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths
from railrl.core import logger as default_logger
from collections import OrderedDict

from railrl.misc.asset_loader import sync_down

import cv2
import torch

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = torch.load(local_path, map_location=lambda storage, loc: storage)
    print("loaded", local_path)
    return vae

class VAEWrappedEnv(ProxyEnv, Env):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    """
    def __init__(self, wrapped_env, vae,
        use_vae_obs=True,
        use_vae_reward=True,
        use_vae_goals=True, # whether you use goals from VAE or rendered from environment state

        decode_goals=False,
        render_goals=False,
        render_rollouts=False,
        render_decoded=False,

        do_reset = True,

        reward_params=dict(),
        track_qpos_goal=0, # UNUSED
        mode="train",
    ):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        if type(vae) is str:
            self.vae = load_vae(vae)
        else:
            self.vae = vae
        self.representation_size = self.vae.representation_size
        self.input_channels = self.vae.input_channels
        self.use_vae_goals = use_vae_goals
        self.use_vae_reward = use_vae_reward
        self.use_vae_obs = use_vae_obs

        self.decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.render_decoded = render_decoded

        self.do_reset = do_reset

        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", None)
        self.epsilon = self.reward_params.get("epsilon", 20)
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

        self.mode(mode)

    def mode(self, name):
        self.current_mode = name
        if name == "train":
            self.use_vae_goals = True
        elif name == "train_env_goals":
            self.use_vae_goals = False
        elif name == "test":
            self.use_vae_goals = False
        elif name == "video_vae":
            self.use_vae_goals = True
            self.decode_goals = True
            self.render_goals = False
            self.render_rollouts = False
            self.render_decoded = False
        elif name == "video_env":
            self.use_vae_goals = False
            self.decode_goals = False
            self.render_goals = False
            self.render_rollouts = False
            self.render_decoded = False
        else:
            error

    @property
    def goal_dim(self):
        return self.representation_size

    def step(self, action):
        observation, reward, done, info = self._wrapped_env.step(action)
        done = False # no early termination
        self.cur_obs = observation.reshape(self.input_channels, 84, 84).transpose()
        if self.render_rollouts:
            cv2.imshow('env', self.cur_obs)
            cv2.waitKey(1)
        if self.use_vae_obs:
            img = Variable(ptu.from_numpy(observation))
            if ptu.gpu_enabled():
                self.vae.cuda()
            mu, logvar = self.vae.encode(img)
            observation = ptu.get_numpy(mu).flatten()
        if self.use_vae_reward:
            # replace reward with Euclidean distance in VAE latent space
            # currently assumes obs and goals are also from VAE
            dist = self.vae_goal - observation
            var = np.exp(ptu.get_numpy(logvar).flatten())
            err = dist * dist / 2 / var
            mdist = np.sum(err) # mahalanobis distance
            if self.reward_type is None:
                reward = -mdist
            elif self.reward_type == "sparse":
                reward = 0 if mdist < self.epsilon else -1
            info["vae_mdist"] = mdist
            info["vae_success"] = 1 if mdist < self.epsilon else 0

        return observation, reward, done, info

    def reset(self):
        self.vae_goal = self.sample_goal_for_rollout()

        observation = self._wrapped_env.reset()
        self.cur_obs = observation.reshape(self.input_channels, 84, 84).transpose()
        if self.render_rollouts:
            cv2.imshow('env', self.cur_obs)
            cv2.waitKey(1)
        if self.use_vae_obs:
            img = Variable(ptu.from_numpy(observation))
            if ptu.gpu_enabled():
                self.vae.cuda()
            e = self.vae.encode(img)[0]
            observation = ptu.get_numpy(e).flatten()
        return observation

    def enable_render(self):
        self.use_vae_goals = True
        self.decode_goals = True
        self.render_goals = True
        self.render_rollouts = True
        self.render_decoded = True

    """
    Multitask functions
    """

    def convert_obs_to_goals(self, obs):
        return obs
        # return ptu.get_numpy(
            # self.vae.encode(ptu.np_to_var(obs))
        # )

    def sample_goals(self, batch_size):
        goals = np.zeros((batch_size, self.representation_size))
        for i in range(batch_size):
            goals[i, :] = self.sample_goal_for_rollout()
        return goals

    def sample_goal_for_rollout(self):
        """
        These goals are fed to a policy when the policy wants to actually
        do rollouts.
        :return:
        """
        if self.use_vae_goals:
            mu, sigma = self.vae.dist_mu, self.vae.dist_std
            # mu, sigma = 0, 1 # sample from prior
            n = np.random.randn(self.representation_size)
            goal = sigma * n + mu
        else:
            self._wrapped_env.set_goal(self._wrapped_env.sample_goal_for_rollout())
            observation = self._wrapped_env.get_image()

            self.true_goal_obs = observation.reshape(self.input_channels, 84, 84).transpose()
            img = Variable(ptu.from_numpy(observation))
            e = self.vae.encode(img)[0]
            goal = ptu.get_numpy(e).flatten()

        if self.do_reset:
            self._wrapped_env.reset()

        if self.decode_goals:
            observation = self.vae.decode(Variable(ptu.from_numpy(goal))).data.view(1, self.input_channels, 84, 84)
            observation = ptu.get_numpy(observation)
            self.goal_decoded = observation.reshape(self.input_channels, 84, 84).transpose()

        if self.use_vae_goals:
            if self.decode_goals:
                self.goal_obs = self.goal_decoded
        else:
            self.goal_obs = self.true_goal_obs

        if self.render_goals and not self.use_vae_goals:
            cv2.imshow('goal', self.goal_obs)
            cv2.waitKey(1)

        if self.render_goals:
            cv2.imshow('decoded', self.goal_decoded)
            cv2.waitKey(1)

        return goal

    def log_diagnostics(self, paths, logger=default_logger, **kwargs):
        super().log_diagnostics(paths, logger=logger, **kwargs)

        statistics = OrderedDict()
        for stat_name_in_paths in ["vae_mdist", "vae_success"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def get_goal(self):
        return self.vae_goal.copy()

    def compute_her_reward_np(
            self,
            observation,
            action,
            next_observation,
            goal,
    ):
        reached_goal = next_observation
        dist = np.linalg.norm(reached_goal - goal)
        if self.reward_type == "sparse":
            reward = 0 if dist < self.epsilon else -1
        else:
            reward = -dist
        return reward
