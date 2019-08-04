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

class EncoderWrappedEnv(ProxyEnv):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(
        self,
        wrapped_env,
        vae,
        reward_params=None,
        imsize=84,
        obs_size=None,
        vae_input_observation_key="image_observation",
    ):
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        if type(vae) is str:
            self.vae = load_local_or_remote_file(vae)
        else:
            self.vae = vae
        self.representation_size = self.vae.representation_size
        self.input_channels = self.vae.input_channels
        self.imsize = imsize
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.vae_input_observation_key = vae_input_observation_key

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

    def reset(self):
        self.vae.eval()
        obs = self.wrapped_env.reset()
        self._initial_obs = obs
        goal = self.sample_goal()
        self.set_goal(goal)
        obs = self._update_obs(obs)
        return obs

    def step(self, action):
        self.vae.eval()
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        self._update_info(info, new_obs)
        reward = self.compute_reward(
            action,
            new_obs,
            # {'latent_achieved_goal': new_obs['latent_achieved_goal'],
            #  'latent_desired_goal': new_obs['latent_desired_goal']}
        )
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        self.vae.eval()
        latent_obs = self._encode_one(obs[self.vae_input_observation_key])
        obs['latent_observation'] = latent_obs
        obs['latent_achieved_goal'] = np.array([])
        obs['observation'] = latent_obs
        obs['achieved_goal'] = np.array([])
        # obs = {**obs, **self.desired_goal}
        return obs

    def _update_info(self, info, obs):
        pass
        # self.vae.eval()
        # latent_distribution_params = self.vae.encode(
        #     ptu.from_numpy(obs[self.vae_input_observation_key].reshape(1,-1))
        # )
        # latent_obs, logvar = ptu.get_numpy(latent_distribution_params[0])[0], ptu.get_numpy(latent_distribution_params[1])[0]
        # # assert (latent_obs == obs['latent_observation']).all()
        # latent_goal = self.desired_goal['latent_desired_goal']
        # dist = latent_goal - latent_obs
        # var = np.exp(logvar.flatten())
        # var = np.maximum(var, self.reward_min_variance)
        # err = dist * dist / 2 / var
        # mdist = np.sum(err)  # mahalanobis distance
        # info["vae_mdist"] = mdist
        # info["vae_success"] = 1 if mdist < self.epsilon else 0
        # info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        # info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        # info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    def compute_reward(self, action, obs):
        actions = action[None]
        return np.linalg.norm(obs["latent_observation"])
        # next_obs = {
        #     k: v[None] for k, v in obs.items()
        # }
        # reward = self.compute_rewards(actions, next_obs)
        # return reward[0]

    def compute_rewards(self, actions, obs):
        self.vae.eval()
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return -dist
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["vae_mdist", "vae_success", "vae_dist"]:
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
        return statistics

    def _encode_one(self, img):
        im = img.reshape(300, 500, 3, 1).transpose()
        return self._encode(im)[0]

    def _encode(self, imgs):
        self.vae.eval()
        latent_distribution_params = self.vae.encoder(ptu.from_numpy(imgs))
        return ptu.get_numpy(latent_distribution_params)

    def _image_and_proprio_from_decoded(self, decoded):
        if decoded is None:
            return None, None
        if self.vae_input_key_prefix == 'image_proprio':
            images = decoded[:, :self.image_length]
            proprio = decoded[:, self.image_length:]
            return images, proprio
        elif self.vae_input_key_prefix == 'image':
            return decoded, None
        else:
            raise AssertionError("Bad prefix for the vae input key.")

    def __getstate__(self):
        state = super().__getstate__()
        state = copy.copy(state)
        state['_custom_goal_sampler'] = None
        warnings.warn('VAEWrapperEnv.custom_goal_sampler is not saved.')
        return state

    def __setstate__(self, state):
        warnings.warn('VAEWrapperEnv.custom_goal_sampler was not loaded.')
        super().__setstate__(state)
