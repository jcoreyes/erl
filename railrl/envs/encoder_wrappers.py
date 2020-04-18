import copy
import random
import warnings

import torch

# import cv2
import numpy as np
from gym import Env
from gym.spaces import Box, Dict
import railrl.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from railrl.envs.wrappers import ProxyEnv
from railrl.misc.asset_loader import load_local_or_remote_file
import time

from railrl.envs.vae_wrappers import VAEWrappedEnv

class VQVAEWrappedEnv(VAEWrappedEnv):

    def __init__(
        self,
        wrapped_env,
        vae,
        pixel_cnn=None,
        vae_input_key_prefix='image',
        sample_from_true_prior=False,
        decode_goals=False,
        decode_goals_on_reset=True,
        render_goals=False,
        render_rollouts=False,
        reward_params=None,
        goal_sampling_mode="vae_prior",
        num_goals_to_presample=0,
        imsize=84,
        obs_size=None,
        norm_order=2,
        epsilon=20,
        presampled_goals=None,
    ):
        if reward_params is None:
            reward_params = dict()
        super().__init__(
            wrapped_env,
            vae,
            vae_input_key_prefix,
            sample_from_true_prior,
            decode_goals,
            decode_goals_on_reset,
            render_goals,
            render_rollouts,
            reward_params,
            goal_sampling_mode,
            num_goals_to_presample,
            imsize,
            obs_size,
            norm_order,
            epsilon,
            presampled_goals,
            )

        if type(pixel_cnn) is str:
            self.pixel_cnn = load_local_or_remote_file(pixel_cnn)
        self.num_keys = self.vae.num_embeddings
        self.representation_size = 144 * self.vae.representation_size

        latent_space = Box(
            -10 * np.ones(obs_size or self.representation_size),
            10 * np.ones(obs_size or self.representation_size),
            dtype=np.float32,
        )

        spaces = self.wrapped_env.observation_space.spaces
        spaces['observation'] = latent_space
        spaces['desired_goal'] = latent_space
        spaces['achieved_goal'] = latent_space
        spaces['latent_observation'] = latent_space
        spaces['latent_desired_goal'] = latent_space
        spaces['latent_achieved_goal'] = latent_space
        self.observation_space = Dict(spaces)


    def get_latent_distance(self, latent1, latent2):
        latent1 = ptu.from_numpy(latent1 * self.num_keys).long()
        latent2 = ptu.from_numpy(latent2 * self.num_keys).long()
        return self.vae.get_distance(latent1, latent2)        


    def _update_info(self, info, obs):
        self.vae.eval()
        latent_obs = self._encode_one(obs[self.vae_input_observation_key])[None]
        latent_goal = self.desired_goal['latent_desired_goal'][None]
        dist = np.linalg.norm(latent_obs - latent_goal)
        info["vae_success"] = 1 if dist < self.epsilon else 0
        info["vae_dist"] = dist
        info["vae_mdist"] = 0
        info["vae_dist_l1"] = 0 #np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = 0 #np.linalg.norm(dist, ord=2)

    def compute_rewards(self, actions, obs):
        self.vae.eval()
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, axis=1)
            return -dist
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, axis=1)
            success = dist < self.epsilon
            reward = success - 1
            return reward

        elif self.reward_type == 'latent_clamp':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, axis=1)
            reward = - np.minimum(dist, self.epsilon)
            return reward
        #WARNING: BELOW ARE HARD CODED FOR SIM PUSHER ENV (IN DIMENSION SIZES)
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal'].reshape(-1, 4)
            desired_goals = obs['state_desired_goal'].reshape(-1, 4)
            return - np.linalg.norm(desired_goals - achieved_goals, axis=1)
        elif self.reward_type == 'state_sparse':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)
            max_dist = np.linalg.norm(distance, axis=1, ord=np.inf)
            success = max_dist < self.epsilon
            reward = success - 1
            return reward
        elif self.reward_type == 'state_hand_distance':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)[:, :1]
            return - distance
        elif self.reward_type == 'state_puck_distance':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)[:, 1:]
            return - distance
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    # def _decode(self, latents):
    #     #MAKE INTEGER
    #     self.vae.eval()
    #     latents = ptu.from_numpy(latents * self.num_keys).long()
    #     reconstructions = self.vae.decode(latents)
    #     decoded = ptu.get_numpy(reconstructions)
    #     decoded = np.clip(decoded, 0, 1)
    #     return decoded

    def _decode(self, latents):
        #MAKE INTEGER
        self.vae.eval()
        latents = ptu.from_numpy(latents)
        reconstructions = self.vae.decode(latents,cont=True)
        decoded = ptu.get_numpy(reconstructions)
        decoded = np.clip(decoded, 0, 1)
        return decoded

    def _encode(self, imgs):
        #MAKE FLOAT
        self.vae.eval()
        latents = self.vae.encode(ptu.from_numpy(imgs), cont=True)
        latents = np.array(ptu.get_numpy(latents))
        return latents

    # def _encode(self, imgs):
    #     #MAKE FLOAT
    #     self.vae.eval()
    #     latents = self.vae.encode(ptu.from_numpy(imgs))
    #     latents = np.array(ptu.get_numpy(latents)) / self.num_keys
    #     return latents

    def _reconstruct_img(self, flat_img):
        self.vae.eval()
        img = flat_img.reshape(1, self.input_channels, self.imsize, self.imsize)
        latents = self._encode_one(img)[None]
        imgs = self._decode(latents)
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]

    def _sample_vae_prior(self, batch_size, cont=True):
        self.vae.eval()
        samples = self.vae.sample_prior(batch_size, cont)
        return samples
