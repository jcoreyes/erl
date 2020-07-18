from functools import partial

import numpy as np

from railrl.core.distribution import DictDistribution
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)

from railrl.envs.contextual import ContextualRewardFn

from gym.spaces import Box
from railrl.samplers.rollout_functions import contextual_rollout
from railrl import pythonplusplus as ppp
from collections import OrderedDict

from typing import Any, Callable, Dict

Observation = Dict
Goal = Any

class MaskDictDistribution(DictDistribution):
    def __init__(
            self,
            env,
            desired_goal_keys=('desired_goal',),
            mask_format='vector',
            masks=None,
            mask_distr=None,
            max_subtasks_to_focus_on=None,
            prev_subtask_weight=None,
    ):
        self._env = env
        self._desired_goal_keys = desired_goal_keys
        self.mask_keys = list(masks.keys())
        self.mask_dims = []
        for key in self.mask_keys:
            self.mask_dims.append(masks[key].shape[1:])

        env_spaces = self._env.observation_space.spaces
        self._spaces = {
            k: env_spaces[k]
            for k in self._desired_goal_keys
        }
        for mask_key, mask_dim in zip(self.mask_keys, self.mask_dims):
            self._spaces[mask_key] = Box(
                low=np.zeros(mask_dim),
                high=np.ones(mask_dim),
                dtype=np.float32,
            )

        self.mask_format = mask_format
        self.masks = masks
        self._num_atomic_masks = next(iter(masks.values())).shape[0]

        self._max_subtasks_to_focus_on = max_subtasks_to_focus_on
        if self._max_subtasks_to_focus_on is not None:
            assert isinstance(self._max_subtasks_to_focus_on, int)
        self._prev_subtask_weight = prev_subtask_weight
        if self._prev_subtask_weight is not None:
            assert isinstance(self._prev_subtask_weight, float)

        for key in mask_distr:
            assert key in ['atomic', 'cumul', 'subset', 'full']
            assert mask_distr[key] >= 0
        for key in ['atomic', 'cumul', 'subset', 'full']:
            if key not in mask_distr:
                mask_distr[key] = 0.0
        if np.sum(list(mask_distr.values())) > 1:
            raise ValueError("Invalid distribution sum: {}".format(
                np.sum(list(mask_distr.values()))
            ))
        self.mask_distr = mask_distr

        self.cumul_masks = None
        self.subset_masks = None

    @property
    def spaces(self):
        return self._spaces

    def sample(self, batch_size: int):
        goals = self.sample_masks(batch_size)

        ### sample the desired_goal ###
        if self.mask_format == 'distribution':
            ### the desired goal is exactly the same as mu ###
            goals.update({
                k: goals['mask_mu']
                for k in self._desired_goal_keys
            })
        else:
            env_samples = self._env.sample_goals(batch_size)
            goals.update({
                k: env_samples[k]
                for k in self._desired_goal_keys
            })

        return goals

    def sample_masks(self, batch_size):
        num_atomic_masks = int(batch_size * self.mask_distr['atomic'])
        num_cumul_masks = int(batch_size * self.mask_distr['cumul'])
        num_subset_masks = int(batch_size * self.mask_distr['subset'])
        num_full_masks = batch_size - num_atomic_masks - num_cumul_masks - num_subset_masks

        mask_goals = []
        if num_atomic_masks > 0:
            mask_goals.append(self.sample_atomic_masks(num_atomic_masks))

        if num_full_masks > 0:
            mask_goals.append(self.sample_full_masks(num_full_masks))

        if num_cumul_masks > 0:
            mask_goals.append(self.sample_cumul_masks(num_cumul_masks))

        if num_subset_masks > 0:
            mask_goals.append(self.sample_subset_masks(num_subset_masks))

        def concat(*x):
            return np.concatenate(x, axis=0)
        mask_goals = ppp.treemap(concat, *tuple(mask_goals),
                                   atomic_type=np.ndarray)

        return mask_goals

    def sample_atomic_masks(self, batch_size):
        sampled_masks = {}
        mask_ids = np.random.choice(self._num_atomic_masks, batch_size)
        for mask_key in self.mask_keys:
            sampled_masks[mask_key] = self.masks[mask_key][mask_ids]
        return sampled_masks

    def sample_full_masks(self, batch_size):
        assert self.mask_format in ['vector', 'matrix']
        sampled_masks = {}
        mask_ids = np.arange(self._num_atomic_masks)
        for mask_key in self.mask_keys:
            sampled_masks[mask_key] = np.repeat(
                np.sum(
                    self.masks[mask_key][mask_ids],
                    axis=0
                )[np.newaxis, ...],
                batch_size,
                axis=0
            )
        return sampled_masks

    def sample_cumul_masks(self, batch_size):
        assert self.mask_format in ['vector', 'matrix']

        if self.cumul_masks is None:
            self.create_cumul_masks()

        sampled_masks = {}
        mask_ids = np.random.choice(self._num_cumul_masks, batch_size)
        for mask_key in self.mask_keys:
            sampled_masks[mask_key] = self.cumul_masks[mask_key][mask_ids]
        return sampled_masks

    def sample_subset_masks(self, batch_size):
        assert self.mask_format in ['vector', 'matrix']

        if self.subset_masks is None:
            self.create_subset_masks()

        sampled_masks = {}
        mask_ids = np.random.choice(self._num_subset_masks, batch_size)
        for mask_key in self.mask_keys:
            sampled_masks[mask_key] = self.subset_masks[mask_key][mask_ids]
        return sampled_masks

    def create_cumul_masks(self):
        assert self.mask_format in ['vector', 'matrix']
        num_atomic_masks = len(self.masks[list(self.masks.keys())[0]])

        self.cumul_masks = {}
        for mask_key, mask_dim in zip(self.mask_keys, self.mask_dims):
            self.cumul_masks[mask_key] = np.zeros([num_atomic_masks] + list(mask_dim))

        for i in range(1, num_atomic_masks + 1):
            mask_idx_bitmap = np.zeros(num_atomic_masks)
            if self._max_subtasks_to_focus_on is None:
                start_idx = 0
                end_idx = i
            else:
                start_idx = max(0, i - self._max_subtasks_to_focus_on)
                end_idx = i
            mask_idx_bitmap[start_idx:end_idx] = 1
            if self._prev_subtask_weight is not None:
                mask_idx_bitmap[start_idx:end_idx - 1] = self._prev_subtask_weight
            for mask_key in self.mask_keys:
                self.cumul_masks[mask_key][i - 1] = (
                        mask_idx_bitmap @ (self.masks[mask_key].reshape((num_atomic_masks, -1)))
                ).reshape(list(self._spaces[mask_key].shape))

        self._num_cumul_masks = next(iter(self.cumul_masks.values())).shape[0]

    def create_subset_masks(self):
        assert self.mask_format in ['vector', 'matrix']
        num_atomic_masks = len(self.masks[list(self.masks.keys())[0]])

        self.subset_masks = {}
        for mask_key, mask_dim in zip(self.mask_keys, self.mask_dims):
            self.subset_masks[mask_key] = np.zeros([2 ** num_atomic_masks - 1] + list(mask_dim))

        def bin_array(num, m):
            """https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding"""
            """Convert a positive integer num into an m-bit bit vector"""
            return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

        for i in range(1, 2 ** num_atomic_masks):
            mask_idx_bitmap = bin_array(i, num_atomic_masks)
            for mask_key in self.mask_keys:
                self.subset_masks[mask_key][i - 1] = (
                        mask_idx_bitmap @ (self.masks[mask_key].reshape((num_atomic_masks, -1)))
                ).reshape(list(self._spaces[mask_key].shape))

        self._num_subset_masks = next(iter(self.subset_masks.values())).shape[0]

    def get_cumul_mask_to_indices(self, masks):
        assert self.mask_format in ['vector']
        if self.cumul_masks is None:
            self.create_cumul_masks()
        cumul_masks_to_indices = OrderedDict()
        for mask in self.cumul_masks['mask']:
            cumul_masks_to_indices[tuple(mask)] = np.where(np.all(masks == mask, axis=1))[0]
        return cumul_masks_to_indices

    def get_atomic_mask_to_indices(self, masks):
        assert self.mask_format in ['vector']
        atomic_masks_to_indices = OrderedDict()
        for mask in self.masks['mask']:
            atomic_masks_to_indices[tuple(mask)] = np.where(np.all(masks == mask, axis=1))[0]
        return atomic_masks_to_indices

class MaskPathCollector(ContextualPathCollector):
    def __init__(
            self,
            *args,
            mask_sampler=None,
            mask_distr=None,
            mask_groups=None,
            max_path_length=100,
            rollout_mask_order='fixed',
            concat_context_to_obs_fn=None,
            prev_subtask_weight=False,
            max_subtasks_to_focus_on=None,
            max_subtasks_per_rollout=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_sampler = mask_sampler

        for key in mask_distr:
            assert key in ['atomic', 'full', 'atomic_seq', 'cumul_seq']
            assert mask_distr[key] >= 0
        for key in ['atomic', 'full', 'atomic_seq', 'cumul_seq']:
            if key not in mask_distr:
                mask_distr[key] = 0.0
        if np.sum(list(mask_distr.values())) > 1:
            raise ValueError("Invalid distribution sum: {}".format(
                np.sum(list(mask_distr.values()))
            ))
        self.mask_distr = mask_distr
        self.mask_groups = mask_groups
        if self.mask_groups is None:
            atomic_masks = self.mask_sampler.masks
            self.mask_groups = np.arange(len(atomic_masks[list(atomic_masks.keys())[0]])).reshape(-1, 1)
        self.mask_groups = np.array(self.mask_groups)

        self.rollout_mask_order = rollout_mask_order
        self.max_path_length = max_path_length
        self.rollout_masks = []
        self._concat_context_to_obs_fn = concat_context_to_obs_fn

        self._prev_subtask_weight = prev_subtask_weight
        self._max_subtasks_to_focus_on = max_subtasks_to_focus_on
        self._max_subtasks_per_rollout = max_subtasks_per_rollout

        def obs_processor(o):
            if len(self.rollout_masks) > 0:
                mask_dict = self.rollout_masks[0]
                self.rollout_masks = self.rollout_masks[1:]
                for k in mask_dict:
                    o[k] = mask_dict[k]
                    self._env._rollout_context_batch[k] = mask_dict[k][None]

            if self._concat_context_to_obs_fn is None:
                combined_obs = [o[self._observation_key]]
                for k in self._context_keys_for_policy:
                    combined_obs.append(o[k])
                return np.concatenate(combined_obs, axis=0)
            else:
                batch = {}
                batch['observations'] = o[self._observation_key][None]
                batch['next_observations'] = o[self._observation_key][None]
                for k in self._context_keys_for_policy:
                    batch[k] = o[k][None]
                return self._concat_context_to_obs_fn(batch)['observations'][0]

        def unbatchify(d):
            for k in d:
                d[k] = d[k][0]
            return d

        def reset_callback(env, agent, o):
            self.rollout_masks = []

            rollout_types = list(self.mask_distr.keys())
            probs = list(self.mask_distr.values())
            rollout_type = np.random.choice(rollout_types, 1, replace=True, p=probs)[0]

            if rollout_type == 'full':
                mask = unbatchify(self.mask_sampler.sample_full_masks(1))
                for _ in range(self.max_path_length):
                    self.rollout_masks.append(mask)
            else:
                atomic_masks = self.mask_sampler.masks
                mask_groups = self.mask_groups.copy()

                if self.rollout_mask_order == 'fixed':
                    pass
                elif self.rollout_mask_order == 'random':
                    np.random.shuffle(mask_groups)
                elif isinstance(self.rollout_mask_order, list):
                    mask_groups = mask_groups[self.rollout_mask_order]
                else:
                    raise NotImplementedError

                if self._max_subtasks_per_rollout is not None:
                    mask_groups = mask_groups[:self._max_subtasks_per_rollout]

                if rollout_type == 'atomic':
                    mask_groups = mask_groups[0:1]

                mask_ids = mask_groups.reshape(-1)

                num_steps_per_mask = self.max_path_length // len(mask_ids)

                for i in range(len(mask_ids)):
                    mask = {}
                    for k in atomic_masks.keys():
                        if rollout_type in ['atomic_seq', 'atomic']:
                            mask[k] = atomic_masks[k][mask_ids[i]]
                        elif rollout_type == 'cumul_seq':
                            if self._max_subtasks_to_focus_on is not None:
                                start_idx = max(0, i + 1 - self._max_subtasks_to_focus_on)
                                end_idx = i + 1
                                atomic_mask_ids_for_rollout_mask = mask_ids[start_idx:end_idx]
                            else:
                                atomic_mask_ids_for_rollout_mask = mask_ids[0:i + 1]

                            atomic_mask_weights = np.ones(len(atomic_mask_ids_for_rollout_mask))
                            if self._prev_subtask_weight is not None:
                                assert isinstance(self._prev_subtask_weight, float)
                                atomic_mask_weights[:-1] = self._prev_subtask_weight
                            mask[k] = np.sum(
                                atomic_masks[k][atomic_mask_ids_for_rollout_mask] * atomic_mask_weights[:, np.newaxis],
                                axis=0
                            )
                        else:
                            raise NotImplementedError
                    num_steps = num_steps_per_mask
                    if i == len(mask_ids) - 1:
                        num_steps = self.max_path_length - len(self.rollout_masks)
                    self.rollout_masks += num_steps*[mask]

        self._rollout_fn = partial(
            contextual_rollout,
            context_keys_for_policy=self._context_keys_for_policy,
            observation_key=self._observation_key,
            obs_processor=obs_processor,
            reset_callback=reset_callback,
        )

class ContextualMaskingRewardFn(ContextualRewardFn):
    def __init__(
            self,
            achieved_goal_from_observation: Callable[[Observation], Goal],
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            mask_keys=None,
            mask_format=None,
            use_g_for_mean=True,
    ):
        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key
        self._achieved_goal_from_observation = achieved_goal_from_observation

        self._mask_keys = mask_keys
        self._mask_format = mask_format
        self._use_g_for_mean = use_g_for_mean

    def __call__(self, states, actions, next_states, contexts):
        del states
        achieved = self._achieved_goal_from_observation(next_states)
        obs = {
            self._achieved_goal_key: achieved,
            self._desired_goal_key: contexts[self._desired_goal_key],
        }
        for key in self._mask_keys:
            obs[key] = contexts[key]

        return default_masked_reward_fn(
            actions, obs,
            mask_format=self._mask_format,
            use_g_for_mean=self._use_g_for_mean,
        )

def default_masked_reward_fn(actions, obs, mask_format, use_g_for_mean):
    achieved_goals = obs['state_achieved_goal']

    if mask_format == 'vector':
        desired_goals = obs['state_desired_goal']
        mask = obs['mask']
        prod = (achieved_goals - desired_goals) * mask
        return -np.linalg.norm(prod, axis=-1)
    elif mask_format in ['matrix', 'distribution', 'cond_distribution']:
        mu = obs['state_desired_goal']
        if mask_format == 'matrix':
            mask = obs['mask']
        elif mask_format == 'distribution':
            mask = obs['mask_sigma_inv']
        elif mask_format == 'cond_distribution':
            mask = obs['mask_sigma_inv']
            if not use_g_for_mean:
                mu_w = obs['mask_mu_w']
                mu_g = obs['mask_mu_g']
                mu_A = obs['mask_mu_mat']
                mu = mu_w + np.squeeze(
                    mu_A @ np.expand_dims(obs['state_desired_goal'] - mu_g, axis=-1),
                    axis=-1
                )
        else:
            raise TypeError
        batch_size, state_dim = achieved_goals.shape
        diff = (achieved_goals - mu).reshape((batch_size, state_dim, 1))
        prod = (diff.transpose(0, 2, 1) @ mask @ diff).reshape(batch_size)
        return -np.sqrt(prod)
    else:
        raise TypeError