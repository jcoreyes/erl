from functools import partial

import numpy as np
from scipy import linalg

from railrl.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
)
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)

from gym.spaces import Box
from railrl.samplers.rollout_functions import contextual_rollout
from railrl import pythonplusplus as ppp
from collections import OrderedDict
import time

class MaskedGoalDictDistributionFromMultitaskEnv(
        GoalDictDistributionFromMultitaskEnv):
    def __init__(
            self,
            *args,
            mask_dims=[(1,)],
            mask_keys=['mask'],
            mask_format='vector',
            masks=None,
            idx_masks=None,
            matrix_masks=None,
            mask_distr=None,
            max_subtasks_to_focus_on=None,
            prev_subtask_weight=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_keys = mask_keys
        self.mask_dims = mask_dims
        for mask_key, mask_dim in zip(self.mask_keys, self.mask_dims):
            self._spaces[mask_key] = Box(
                low=np.zeros(mask_dim),
                high=np.ones(mask_dim),
                dtype=np.float32,
            )
        self.mask_format = mask_format

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

        if masks is not None:
            self.masks = masks
        else:
            assert ((idx_masks is not None) + (matrix_masks is not None)) == 1

            self.masks = {}

            if idx_masks is not None:
                num_masks = len(idx_masks)
            elif matrix_masks is not None:
                num_masks = len(matrix_masks)
            else:
                raise NotImplementedError

            for mask_key, mask_dim in zip(self.mask_keys, self.mask_dims):
                self.masks[mask_key] = np.zeros([num_masks] + list(mask_dim))

            if self.mask_format in ['vector', 'matrix']:
                assert len(self.mask_keys) == 1
                mask_key = self.mask_keys[0]
                if idx_masks is not None:
                    for (i, idx_dict) in enumerate(idx_masks):
                        for (k, v) in idx_dict.items():
                            assert k == v
                            if self.mask_format == 'vector':
                                self.masks[mask_key][i][k] = 1
                            elif self.mask_format == 'matrix':
                                self.masks[mask_key][i][k, k] = 1
                elif matrix_masks is not None:
                    self.masks[mask_key] = np.array(matrix_masks)
            elif self.mask_format == 'distribution':
                if idx_masks is not None:
                    self.masks['mask_mu_mat'][:] = np.identity(self.masks['mask_mu_mat'].shape[-1])
                    idx_masks = np.array(idx_masks)

                    for (i, idx_dict) in enumerate(idx_masks):
                        for (k, v) in idx_dict.items():
                            assert k == v
                            self.masks['mask_sigma_inv'][i][k, k] = 1
                elif matrix_masks is not None:
                    self.masks['mask_mu_mat'][:] = np.identity(self.masks['mask_mu_mat'].shape[-1])
                    self.masks['mask_sigma_inv'] = np.array(matrix_masks)
            else:
                raise NotImplementedError

        self.cumul_masks = None
        self.subset_masks = None

    def sample(self, batch_size: int, use_env_goal=False):
        goals = super().sample(batch_size, use_env_goal)
        mask_goals = self.sample_masks(batch_size)
        goals.update(mask_goals)
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
        num_masks = len(self.masks[list(self.masks.keys())[0]])
        mask_ids = np.random.choice(num_masks, batch_size)
        for mask_key in self.mask_keys:
            sampled_masks[mask_key] = self.masks[mask_key][mask_ids]
        return sampled_masks

    def sample_full_masks(self, batch_size):
        assert self.mask_format in ['vector', 'matrix']
        sampled_masks = {}
        num_masks = len(self.masks[list(self.masks.keys())[0]])
        mask_ids = np.arange(num_masks)
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
        num_masks = len(self.cumul_masks[list(self.cumul_masks.keys())[0]])
        mask_ids = np.random.choice(num_masks, batch_size)
        for mask_key in self.mask_keys:
            sampled_masks[mask_key] = self.cumul_masks[mask_key][mask_ids]
        return sampled_masks

    def sample_subset_masks(self, batch_size):
        assert self.mask_format in ['vector', 'matrix']

        if self.subset_masks is None:
            self.create_subset_masks()

        sampled_masks = {}
        num_masks = len(self.subset_masks[list(self.subset_masks.keys())[0]])
        mask_ids = np.random.choice(num_masks, batch_size)
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
            prev_subtasks_solved=True,
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
        self._prev_subtasks_solved = prev_subtasks_solved
        self._max_subtasks_to_focus_on = max_subtasks_to_focus_on
        self._max_subtasks_per_rollout = max_subtasks_per_rollout

        def obs_processor(o):
            if len(self.rollout_masks) > 0:
                mask_dict = self.rollout_masks[0]
                self.rollout_masks = self.rollout_masks[1:]
                for k in mask_dict:
                    o[k] = mask_dict[k]
                    self._env._rollout_context_batch[k] = mask_dict[k][None]

                # hack: set previous objects goals to states
                if self._prev_subtasks_solved:
                    indices = np.argwhere(mask_dict['mask'] == 1)[:-2].reshape(-1)
                    if len(indices) > 0:
                        self._env._rollout_context_batch['state_desired_goal'][0][indices] = o[self._observation_key][indices]
                        new_goal = {
                            'state_desired_goal': self._env._rollout_context_batch['state_desired_goal'][0]
                        }
                        self._env.env.set_goal(new_goal)

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

        def reset_postprocess_func():
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
            reset_postprocess_func=reset_postprocess_func,
        )

def default_masked_reward_fn(actions, obs, mask_format='vector'):
    achieved_goals = obs['state_achieved_goal']
    desired_goals = obs['state_desired_goal']

    if mask_format == 'vector':
        # vector mask
        mask = obs['mask']
        prod = (achieved_goals - desired_goals) * mask
        return -np.linalg.norm(prod, axis=-1)
    elif mask_format == 'matrix':
        # matrix mask
        mask = obs['mask']

        # ### hack for testing H->A ###
        # if -1 in mask:
        #     desired_goals = desired_goals.copy()
        #     desired_goals[:,0:4] = 0

        batch_size, state_dim = achieved_goals.shape
        diff = (achieved_goals - desired_goals).reshape((batch_size, state_dim, 1))
        prod = (diff.transpose(0, 2, 1) @ mask @ diff).reshape(batch_size)
        return -np.sqrt(prod)
    elif mask_format == 'distribution':
        # matrix mask
        g = desired_goals
        mu_w = obs['mask_mu_w']
        mu_g = obs['mask_mu_g']
        mu_A = obs['mask_mu_mat']
        sigma_inv = obs['mask_sigma_inv']
        mu_w_given_g = mu_w + np.squeeze(mu_A @ np.expand_dims(g - mu_g, axis=-1), axis=-1)
        sigma_w_given_g_inv = sigma_inv

        batch_size, state_dim = achieved_goals.shape
        diff = (achieved_goals - mu_w_given_g).reshape((batch_size, state_dim, 1))
        prod = (diff.transpose(0, 2, 1) @ sigma_w_given_g_inv @ diff).reshape(batch_size)
        return -np.sqrt(prod)
    else:
        raise NotImplementedError

def action_penalty_masked_reward_fn(actions, obs, mask_format='vector'):
    orig_reward = default_masked_reward_fn(actions, obs, mask_format=mask_format)
    action_reward = -np.linalg.norm(actions[:,:2], axis=1) * 0.15
    reward = orig_reward + action_reward
    return reward

def get_cond_distr_params(mu, sigma, x_dim):
    mu_x = mu[:x_dim]
    mu_y = mu[x_dim:]

    sigma_xx = sigma[:x_dim, :x_dim]
    sigma_yy = sigma[x_dim:, x_dim:]
    sigma_xy = sigma[:x_dim, x_dim:]
    sigma_yx = sigma[x_dim:, :x_dim]

    sigma_yy_inv = linalg.inv(sigma_yy)

    mu_mat = sigma_xy @ sigma_yy_inv
    sigma_xgy = sigma_xx - sigma_xy @ sigma_yy_inv @ sigma_yx

    w, v = np.linalg.eig(sigma_xgy)
    if np.min(sorted(w)) < 1e-6:
        eps = 1e-6
    else:
        eps = 0
    sigma_xgy_inv = linalg.inv(sigma_xgy + eps * np.identity(sigma_xgy.shape[0]))

    return mu_x, mu_y, mu_mat, sigma_xgy_inv

def print_matrix(matrix, format="raw", threshold=0.1, normalize=True):
    if normalize:
        matrix = matrix.copy() / np.max(np.abs(matrix))

    assert format in ["signed", "raw"]

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if format == "raw":
                value = matrix[i][j]
            elif format == "signed":
                if np.abs(matrix[i][j]) > threshold:
                    value = 1 * np.sign(matrix[i][j])
                else:
                    value = 0
            if format == "signed":
                print(int(value), end=", ")
            else:
                if value > 0:
                    print("", end=" ")
                print("{:.2f}".format(value), end=" ")
        print()
    print()

def infer_masks(env, mask_variant):
    from tqdm import tqdm

    mask_inference_variant = mask_variant['mask_inference_variant']
    n = int(mask_inference_variant['n'])
    obs_noise = mask_inference_variant['noise']
    normalize_sigma_inv = mask_inference_variant['normalize_sigma_inv']

    assert mask_variant['mask_format'] == 'distribution'
    assert 'idx_masks' in mask_variant
    idx_masks = mask_variant['idx_masks']
    num_masks = len(idx_masks)
    mask_keys = mask_variant['mask_keys']
    mask_dims = mask_variant['mask_dims']

    masks = {}
    for (key, dim) in zip(mask_keys, mask_dims):
        masks[key] = np.zeros([num_masks] + list(dim))

    goal_dim = env.observation_space.spaces['state_desired_goal'].low.size

    goals = np.zeros((n, goal_dim))
    list_of_waypoints = np.zeros((num_masks, n, goal_dim))

    t1 = time.time()
    print("Generating dataset...")

    # data collection
    for i in tqdm(range(n)):
        obs_dict = env.reset()
        obs = obs_dict['state_achieved_goal']
        goal = obs_dict['state_desired_goal']

        for (mask_id, idx_dict) in enumerate(idx_masks):
            wp = obs.copy()
            for (k, v) in idx_dict.items():
                if v >= 0:
                    wp[k] = goal[v]
            for (k, v) in idx_dict.items():
                if v < 0:
                    wp[k] = wp[-v - 10]
            list_of_waypoints[mask_id][i] = wp

        goals[i] = goal

    print("Done. Time:", time.time() - t1)

    # add noise to all of the data
    list_of_waypoints += np.random.normal(0, obs_noise, list_of_waypoints.shape)
    goals += np.random.normal(0, obs_noise, goals.shape)

    for (mask_id, waypoints) in enumerate(list_of_waypoints):
        mu = np.mean(np.concatenate((waypoints, goals), axis=1), axis=0)
        sigma = np.cov(np.concatenate((waypoints, goals), axis=1).T)
        mu_w, mu_g, mu_mat, sigma_inv = get_cond_distr_params(mu, sigma, x_dim=goal_dim)
        if normalize_sigma_inv:
            sigma_inv = sigma_inv / np.max(np.abs(sigma_inv))
        masks['mask_mu_w'][mask_id] = mu_w
        masks['mask_mu_g'][mask_id] = mu_g
        masks['mask_mu_mat'][mask_id] = mu_mat
        masks['mask_sigma_inv'][mask_id] = sigma_inv

    # for mask_id in range(num_masks):
    #     # print('mask_mu_mat')
    #     # print_matrix(masks['mask_mu_mat'][mask_id])
    #     print('mask_sigma_inv')
    #     print_matrix(masks['mask_sigma_inv'][mask_id])
    # exit()

    return masks