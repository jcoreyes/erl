import random
import torch

import cv2
import numpy as np
from gym import Env
from gym.spaces import Box, Dict
import railrl.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from railrl.envs.wrappers import ProxyEnv
from railrl.misc.asset_loader import load_local_or_remote_file


class VAEWrappedEnv(ProxyEnv, MultitaskEnv):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(
        self,
        wrapped_env,
        vae,
        vae_input_key_prefix='image',
        use_vae_goals=True,
        sample_from_true_prior=False,
        decode_goals=False,
        render_goals=False,
        render_rollouts=False,
        reward_params=None,
        mode="train",
        imsize=84,
        obs_size = None,
        norm_order=2,
        epsilon=20,
        presampled_goals=None,
    ):
        self.quick_init(locals())
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        if type(vae) is str:
            self.vae = load_local_or_remote_file(vae)
        else:
            self.vae = vae
        self.representation_size = self.vae.representation_size
        self.input_channels = self.vae.input_channels
        self._use_vae_goals = use_vae_goals
        self.sample_from_true_prior = sample_from_true_prior
        self.decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.default_kwargs=dict(
            decode_goals=decode_goals,
            render_goals=render_goals,
            render_rollouts=render_rollouts,
        )
        self.imsize = imsize
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        self.epsilon = self.reward_params.get("epsilon", epsilon)
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
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
        self.mode(mode)
        self._presampled_goals = presampled_goals
        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = presampled_goals[random.choice(list(presampled_goals))].shape[0]
        self.use_replay_buffer_goals = False

        self.vae_input_key_prefix = vae_input_key_prefix
        assert vae_input_key_prefix in set(['image', 'image_proprio'])
        self.vae_input_observation_key = vae_input_key_prefix + '_observation'
        self.vae_input_achieved_goal_key = vae_input_key_prefix + '_achieved_goal'
        self.vae_input_desired_goal_key = vae_input_key_prefix + '_desired_goal'
        self._mode_map = {}
        self.desired_goal = {}
        self.desired_goal['latent_desired_goal'] = latent_space.sample()
        self._initial_obs = None

    def reset(self):
        obs = self.wrapped_env.reset()
        goal = {}

        if self.use_vae_goals:
            if self.use_replay_buffer_goals:
                latent_goals = self.sample_replay_buffer_latent_goals(1)
            else:
                latent_goals = self._sample_vae_prior(1)
            latent_goal = latent_goals[0]
        else:
            if self.num_goals_presampled > 0:
                # TODO: hack for now. There's no documentation on set_goal
                goal = self.sample_goal()
                latent_goal = goal['latent_desired_goal']
                self.wrapped_env.set_goal(goal)
            else:
                latent_goal = self._encode_one(obs[self.vae_input_desired_goal_key])

        if self.decode_goals:
            decoded_goal = self._decode(latent_goal)[0]
            image_goal, proprio_goal = self._image_and_proprio_from_decoded_one(
                decoded_goal
            )
        elif self.num_goals_presampled>0:
            decoded_goal = goal.get(self.vae_input_desired_goal_key, None)
            image_goal = goal.get('image_desired_goal', None)
            proprio_goal = goal.get('proprio_desired_goal', None)
        else:
            image_goal = obs.get('image_desired_goal', None)
            decoded_goal=obs.get('image_desired_goal', None)
            proprio_goal=None

        goal['desired_goal'] = latent_goal
        goal['latent_desired_goal'] = latent_goal
        goal['image_desired_goal'] = image_goal
        goal['proprio_desired_goal'] = proprio_goal
        goal[self.vae_input_desired_goal_key] = decoded_goal
        self.desired_goal = goal
        self._initial_obs = obs
        return self._update_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        self._update_info(info, new_obs)
        reward = self.compute_reward(
            action,
            {'latent_achieved_goal': new_obs['latent_achieved_goal'],
             'latent_desired_goal': new_obs['latent_desired_goal']}
        )
        self.try_render(new_obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        latent_obs = self._encode_one(obs[self.vae_input_observation_key])
        obs['latent_observation'] = latent_obs
        obs['latent_achieved_goal'] = latent_obs
        obs['observation'] = latent_obs
        obs['achieved_goal'] = latent_obs
        obs = {**obs, **self.desired_goal}
        return obs

    def _update_info(self, info, obs):
        latent_obs, logvar = self.vae.encode(
            ptu.from_numpy(obs[self.vae_input_observation_key])
        )
        latent_obs, logvar = ptu.get_numpy(latent_obs)[0], ptu.get_numpy(logvar)[0]
        # assert (latent_obs == obs['latent_observation']).all()
        latent_goal = self.desired_goal['latent_desired_goal']
        dist = latent_goal - latent_obs
        var = np.exp(logvar.flatten())
        var = np.maximum(var, self.reward_min_variance)
        err = dist * dist / 2 / var
        mdist = np.sum(err)  # mahalanobis distance
        info["vae_mdist"] = mdist
        info["vae_success"] = 1 if mdist < self.epsilon else 0
        info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    @property
    def use_vae_goals(self):
        return self._use_vae_goals and not self.reward_type.startswith('state')

    """
    Multitask functions
    """
    def sample_goals(self, batch_size):
        if self.num_goals_presampled > 0 and not self.use_vae_goals:
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            #ensures goals are encoded using latest vae
            if 'image_desired_goal' in sampled_goals:
                sampled_goals['latent_desired_goal'] = self._encode(sampled_goals['image_desired_goal'])
            return sampled_goals

        if self.use_vae_goals:
            goals = {}
            if self.use_replay_buffer_goals:
                latent_goals = self.sample_replay_buffer_latent_goals(batch_size)
            else:
                latent_goals = self._sample_vae_prior(batch_size)
            goals['state_desired_goal'] = None
        else:
            goals = self.wrapped_env.sample_goals(batch_size)
            latent_goals = self._encode(goals[self.vae_input_desired_goal_key])

        if self.decode_goals:
            decoded_goals = self._decode(latent_goals)
        else:
            decoded_goals = None
        image_goals, proprio_goals = self._image_and_proprio_from_decoded(
            decoded_goals
        )

        goals['desired_goal'] = latent_goals
        goals['latent_desired_goal'] = latent_goals
        goals['proprio_desired_goal'] = proprio_goals
        goals['image_desired_goal'] = image_goals
        goals[self.vae_input_desired_goal_key] = decoded_goals
        return goals

    def sample_goal(self):
        goals = self.sample_goals(1)
        return self.unbatchify_dict(goals, 0)

    def get_goal(self):
        return self.desired_goal

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return -dist
        elif self.reward_type == 'vectorized_latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            reward = 0 if dist < self.epsilon else -1
            return reward
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            return - np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    @property
    def goal_dim(self):
        return self.representation_size

    def set_goal(self, goal):
        """
        Assume goal contains both image_desired_goal and any goals required for wrapped envs

        :param goal:
        :return:
        """
        self.desired_goal = goal
        self.wrapped_env.set_goal(goal)

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

    """
    Other functions
    """
    def mode(self, name):
        if name == "train":
            self._use_vae_goals = True
            self.decode_goals = self.default_kwargs['decode_goals']
            self.render_goals = self.default_kwargs['render_goals']
            self.render_rollouts = self.default_kwargs['render_rollouts']
        elif name == "train_env_goals":
            self._use_vae_goals = False
            self.decode_goals = self.default_kwargs['decode_goals']
            self.render_goals = self.default_kwargs['render_goals']
            self.render_rollouts = self.default_kwargs['render_rollouts']
        elif name == "test":
            self._use_vae_goals = False
            self.decode_goals = self.default_kwargs['decode_goals']
            self.render_goals = self.default_kwargs['render_goals']
            self.render_rollouts = self.default_kwargs['render_rollouts']
        elif name == "video_vae":
            self._use_vae_goals = True
            self.decode_goals = True
            self.render_goals = False
            self.render_rollouts = False
        elif name == "video_env":
            self._use_vae_goals = False
            self.decode_goals = False
            self.render_goals = False
            self.render_rollouts = False
            self.render_decoded = False
        else:
            raise ValueError("Invalid mode: {}".format(name))
        if hasattr(self.wrapped_env, "mode"):
            self.wrapped_env.mode(name)
        self.cur_mode = name

    def add_mode(self, env_type, mode):
        assert env_type in ['train',
                        'eval',
                        'video_vae',
                        'video_env',
                        'relabeling']
        assert mode in ['train',
                        'train_env_goals',
                        'test',
                        'video_vae',
                        'video_env']
        assert env_type not in self._mode_map
        self._mode_map[env_type] = mode

    def train(self):
        self.mode(self._mode_map['train'])

    def eval(self):
        self.mode(self._mode_map['eval'])

    def get_env_update(self):
        """
        For online-parallel. Gets updates to the environment since the last time
        the env was serialized.

        subprocess_env.update_env(**env.get_env_update())
        """
        return dict(
            mode_map=self._mode_map,
            gpu_info=dict(
                use_gpu=ptu._use_gpu,
                gpu_id=ptu._gpu_id,
            ),
            vae_state=self.vae.__getstate__(),
        )

    def update_env(self, mode_map, vae_state, gpu_info):
        self._mode_map = mode_map
        self.vae.__setstate__(vae_state)
        gpu_id = gpu_info['gpu_id']
        use_gpu = gpu_info['use_gpu']
        ptu.device = torch.device("cuda:" + str(gpu_id) if use_gpu else "cpu")
        self.vae.to(ptu.device)

    def enable_render(self):
        self._use_vae_goals = False
        self.decode_goals = True
        self.render_goals = True
        self.render_rollouts = True

    def disable_render(self):
        self.decode_goals = False
        self.render_goals = False
        self.render_rollouts = False

    def try_render(self, obs):
        if self.render_rollouts:
            img = obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('env', img)
            cv2.waitKey(1)
            reconstruction = self._reconstruct_img(obs['image_observation']).transpose()
            cv2.imshow('env_reconstruction', reconstruction)
            cv2.waitKey(1)
            init_img = self._initial_obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('initial_state', init_img)
            cv2.waitKey(1)
            init_reconstruction = self._reconstruct_img(
                self._initial_obs['image_observation']
            ).transpose()
            cv2.imshow('init_reconstruction', init_reconstruction)
            cv2.waitKey(1)

        if self.render_goals:
            goal = obs['image_desired_goal'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('goal', goal)
            cv2.waitKey(1)

    def _sample_vae_prior(self, batch_size):
        if self.sample_from_true_prior:
            mu, sigma = 0, 1  # sample from prior
        else:
            mu, sigma = self.vae.dist_mu, self.vae.dist_std
        n = np.random.randn(batch_size, self.representation_size)
        return sigma * n + mu

    def sample_replay_buffer_latent_goals(self, batch_size):
        if self.replay_buffer._size > 0:
            idxs = self.replay_buffer._sample_indices(batch_size)
            latent_goals = self.replay_buffer._next_obs['latent_achieved_goal'][idxs]
            return latent_goals
        return self._sample_vae_prior(batch_size)

    def _decode(self, latents):
        batch_size = latents.shape[0]
        decoded = ptu.get_numpy(self.vae.decode(ptu.from_numpy(latents)))
        return decoded

    def _encode_one(self, img):
        return self._encode(img[None])[0]

    def _encode(self, imgs):
        return ptu.get_numpy(self.vae.encode(ptu.from_numpy(imgs))[0])

    def _reconstruct_img(self, flat_img):
        zs = self.vae.encode(ptu.from_numpy(flat_img[None]))[0]
        imgs = ptu.get_numpy(self.vae.decode(zs))
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]

    def _image_and_proprio_from_decoded_one(self, decoded):
        if len(decoded.shape) == 1:
            decoded = np.array([decoded])
        images, proprios = self._image_and_proprio_from_decoded(decoded)
        image = None
        proprio = None
        if images is not None:
            image = images[0]
        if proprios is not None:
            proprio = proprios[0]
        return image, proprio

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


class StateVAEWrappedEnv(ProxyEnv, Env):
    def __init__(
        self,
        wrapped_env,
        vae,
        observation_key='latent_observation',
        use_vae_goals=True,
        sample_from_true_prior=False,
        decode_goals=False,
        render_goals=False,
        render_rollouts=False,
        reward_params=None,
        mode="train",
        imsize=84,
        use_goal_caching=False,
        # cached_goal_generation_function=generate_goal_data_set,
        num_cached_goals=100,
        cached_goal_keys=None,
        goal_sizes=None,
        obs_to_goal_fctns=None,
        observation_keys=None,
        use_cached_dataset=False,
        num_goals_presampled=0,
    ):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        if reward_params is None:
            reward_params = dict()
        if type(vae) is str:
            self.vae = load_local_or_remote_file(vae)
        else:
            self.vae = vae
        self.representation_size = self.vae.representation_size
        self._use_vae_goals = use_vae_goals
        self.sample_from_true_prior = sample_from_true_prior
        self.decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.default_kwargs=dict(
            decode_goals=decode_goals,
            render_goals=render_goals,
            render_rollouts=render_rollouts,
        )
        self.imsize = imsize
        self.num_goals_presampled = num_goals_presampled
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.norm_order = self.reward_params.get("norm_order", 1)
        self.epsilon = self.reward_params.get("epsilon", 20)
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
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
        self._latent_goal = np.random.uniform(0, 1, self.representation_size)
        self.mode(mode)
        self._presampled_goals = None
        self._mode_map = {}

    @property
    def goal_dim(self):
        return self.representation_size

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        self._update_info(info, obs)
        reward = self.compute_reward(
            action,
            {'latent_achieved_goal': new_obs['latent_achieved_goal'],
             'latent_desired_goal': new_obs['latent_desired_goal']}
        )
        return new_obs, reward, done, info

    def _update_info(self, info, obs):
        latent_obs, logvar = self.vae.encode(ptu.from_numpy(obs['state_observation']))
        latent_obs, logvar = ptu.get_numpy(latent_obs), ptu.get_numpy(logvar)
        # assert (latent_obs == obs['latent_observation']).all()
        latent_goal = self._latent_goal
        dist = latent_goal - latent_obs
        var = np.exp(logvar.flatten())
        var = np.maximum(var, self.reward_min_variance)
        err = dist * dist / 2 / var
        mdist = np.sum(err)  # mahalanobis distance
        info["vae_mdist"] = mdist
        info["vae_success"] = 1 if mdist < self.epsilon else 0
        info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    def reset(self):
        obs = self.wrapped_env.reset()
        if self.use_vae_goals:
            latent_goals = self._sample_vae_prior(1)
            self._latent_goal = latent_goals[0]
        else:
            self._latent_goal = self._encode_one(obs['state_desired_goal'])
        obs['desired_goal'] = self._latent_goal
        obs['latent_desired_goal'] = self._latent_goal
        return self._update_obs(obs)

    def _update_obs(self, obs):
        latent_obs = self._encode_one(obs['state_observation'])
        obs['latent_observation'] = latent_obs
        obs['latent_desired_goal'] = self._latent_goal
        obs['latent_achieved_goal'] = latent_obs
        obs['observation'] = latent_obs
        obs['desired_goal'] = self._latent_goal
        obs['achieved_goal'] = latent_obs
        return obs

    def _decode(self, latents):
        states = ptu.get_numpy(self.vae.decode(ptu.from_numpy(latents)))
        return states

    def sample_goals(self, batch_size):
        if self.use_vae_goals:
            goals = {}
            latent_goals = self._sample_vae_prior(batch_size)
        else:
            goals = self.wrapped_env.sample_goals(batch_size)
            latent_goals = self._encode(goals['state_desired_goal'])
        goals['desired_goal'] = latent_goals
        goals['latent_desired_goal'] = latent_goals
        return goals

    @property
    def use_vae_goals(self):
        return self._use_vae_goals and not self.reward_type.startswith('state')

    def mode(self, name):
        if name == "train":
            self._use_vae_goals = True
            self.decode_goals = self.default_kwargs['decode_goals']
            self.render_goals = self.default_kwargs['render_goals']
            self.render_rollouts = self.default_kwargs['render_rollouts']
        elif name == "train_env_goals":
            self._use_vae_goals = False
            self.decode_goals = self.default_kwargs['decode_goals']
            self.render_goals = self.default_kwargs['render_goals']
            self.render_rollouts = self.default_kwargs['render_rollouts']
        elif name == "test":
            self._use_vae_goals = False
            self.decode_goals = self.default_kwargs['decode_goals']
            self.render_goals = self.default_kwargs['render_goals']
            self.render_rollouts = self.default_kwargs['render_rollouts']
        elif name == "video_vae":
            self._use_vae_goals = True
            self.decode_goals = True
            self.render_goals = False
            self.render_rollouts = False
        elif name == "video_env":
            self._use_vae_goals = False
            self.decode_goals = False
            self.render_goals = False
            self.render_rollouts = False
            self.render_decoded = False
        else:
            raise ValueError("Invalid mode: {}".format(name))
        self.cur_mode = name

    def add_mode(self, env_type, mode):
        assert env_type in ['train',
                            'eval',
                            'video_vae',
                            'video_env',
                            'relabeling']
        assert mode in ['train',
                        'train_env_goals',
                        'test',
                        'video_vae',
                        'video_env']
        assert env_type not in self._mode_map
        self._mode_map[env_type] = mode

    def train(self):
        self.mode(self._mode_map['train'])

    def eval(self):
        self.mode(self._mode_map['eval'])

    def enable_render(self):
        self._use_vae_goals = False
        self.decode_goals = True
        self.render_goals = True
        self.render_rollouts = True

    def disable_render(self):
        self.decode_goals = False
        self.render_goals = False
        self.render_rollouts = False

    def _sample_vae_prior(self, batch_size):
        if self.sample_from_true_prior:
            mu, sigma = 0, 1  # sample from prior
        else:
            mu, sigma = self.vae.dist_mu, self.vae.dist_std
        n = np.random.randn(batch_size, self.representation_size)
        return sigma * n + mu

    def _encode_one(self, img):
        return self._encode(img[None])[0]

    def _encode(self, imgs):
        return ptu.get_numpy(self.vae.encode(ptu.from_numpy(imgs))[0])


    """
    Multitask functions
    """

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

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return -dist
        elif self.reward_type == 'vectorized_latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            reward = 0 if dist < self.epsilon else -1
            return reward
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            return - np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

def temporary_mode(env, mode, func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    cur_mode = env.cur_mode
    env.mode(env._mode_map[mode])
    return_val = func(*args, **kwargs)
    env.mode(cur_mode)
    return return_val
