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
import joblib
import pickle

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pickle.load(open(local_path, "rb"))
    # vae = torch.load(local_path, map_location=lambda storage, loc: storage)
    print("loaded", local_path)
    return vae

class VAEWrappedEnv(ProxyEnv, Env):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.

    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(self, wrapped_env, vae,
        use_vae_obs=True,
        use_vae_reward=True,
        use_vae_goals=True, # whether you use goals from VAE or rendered from environment state
        sample_from_true_prior=False,

        decode_goals=False,
        render_goals=False,
        render_rollouts=False,

        reset_on_sample_goal_for_rollout = True,

        reward_params=None,
        mode="train",
    ):
        if reward_params is None:
            reward_params = dict()
        self.quick_init(locals())
        super().__init__(wrapped_env)
        if type(vae) is str:
            self.vae = load_vae(vae)
        else:
            self.vae = vae
        self.vae.eval()
        self.representation_size = self.vae.representation_size
        self.input_channels = self.vae.input_channels
        self.use_vae_goals = use_vae_goals
        self.use_vae_reward = use_vae_reward
        self.use_vae_obs = use_vae_obs
        self.sample_from_true_prior = sample_from_true_prior

        self.decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts

        self.reset_on_sample_goal_for_rollout = reset_on_sample_goal_for_rollout

        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.epsilon = self.reward_params.get("epsilon", 20)
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
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
        elif name == "video_env":
            self.use_vae_goals = False
            self.decode_goals = False
            self.render_goals = False
            self.render_rollouts = False
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
        else:
            raise NotImplementedError
        if self.use_vae_reward:
            # replace reward with Euclidean distance in VAE latent space
            # currently assumes obs and goals are also from VAE
            dist = self.vae_goal - observation
            var = np.exp(ptu.get_numpy(logvar).flatten())
            var = np.maximum(var, self.reward_min_variance)
            err = dist * dist / 2 / var
            mdist = np.sum(err) # mahalanobis distance
            info["vae_mdist"] = mdist
            info["vae_success"] = 1 if mdist < self.epsilon else 0
            info["var"] = var
            reward = self.compute_her_reward_np(
                None,
                action,
                observation,
                self.vae_goal,
                env_info=info,
            )
        else:
            raise NotImplementedError()

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
        self.decode_goals = True
        self.render_goals = True
        self.render_rollouts = True

    def disable_render(self):
        self.decode_goals = False
        self.render_goals = False
        self.render_rollouts = False

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
            if self.sample_from_true_prior:
                mu, sigma = 0, 1 # sample from prior
            else:
                mu, sigma = self.vae.dist_mu, self.vae.dist_std
            n = np.random.randn(self.representation_size)
            goal = sigma * n + mu
        else:
            goal = self._wrapped_env.sample_goal_for_rollout()
            self._wrapped_env.set_goal(goal)
            state = self._wrapped_env.get_nongoal_state()
            self._wrapped_env.set_to_goal(goal)
            observation = self._wrapped_env.get_image()
            self._wrapped_env.set_nongoal_state(state)

            self.true_goal_obs = observation.reshape(self.input_channels, 84, 84).transpose()
            img = Variable(ptu.from_numpy(observation))
            e = self.vae.encode(img)[0]
            goal = ptu.get_numpy(e).flatten()

        if self.reset_on_sample_goal_for_rollout:
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
            env_info=None,
    ):
        if self.reward_type == 'latent_distance':
            reached_goal = next_observation
            dist = np.linalg.norm(reached_goal - goal)
            reward = -dist
            return reward
        elif self.reward_type == 'latent_sparse':
            reached_goal = next_observation
            dist = np.linalg.norm(reached_goal - goal)
            reward = 0 if dist < self.epsilon else -1
            return reward
        if not self.use_vae_obs:
            raise NotImplementedError
        if not self.use_vae_reward:
            raise NotImplementedError
        var = env_info['var']
        dist = goal - next_observation
        var = np.maximum(var, self.reward_min_variance)
        err = dist * dist / 2 / var
        mdist = np.sum(err) # mahalanobis distance
        if self.reward_type == "log_prob":
            reward = -mdist
        elif self.reward_type == "mahalanobis_distance":
            reward = -np.sqrt(mdist)
        elif self.reward_type == "sparse":
            reward = 0 if mdist < self.epsilon else -1
        else:
            raise NotImplementedError
        return reward
