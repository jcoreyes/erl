from functools import partial

import numpy as np
from scipy import linalg

from railrl.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
)
from railrl.envs.contextual import ContextualEnv, delete_info

from railrl.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
    IndexIntoAchievedGoal,
)
from railrl.envs.images import Renderer, InsertImageEnv
from railrl.launchers.contextual.util import (
    get_save_video_function,
)
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)

from railrl.launchers.rl_exp_launcher_util import (
    preprocess_rl_variant,
    get_envs,
    get_exploration_strategy,
)

from gym.spaces import Box
from railrl.samplers.rollout_functions import contextual_rollout

class TaskGoalDictDistributionFromMultitaskEnv(
        GoalDictDistributionFromMultitaskEnv):
    def __init__(
            self,
            *args,
            task_key='task_id',
            task_ids=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.task_key = task_key
        self._spaces[task_key] = Box(
            low=np.zeros(1),
            high=np.ones(1))
        self.task_ids = np.array(task_ids)

    def sample(self, batch_size: int, use_env_goal=False):
        goals = super().sample(batch_size, use_env_goal)
        idxs = np.random.choice(len(self.task_ids), batch_size)
        goals[self.task_key] = self.task_ids[idxs].reshape(-1, 1)
        return goals

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

    def sample(self, batch_size: int, use_env_goal=False):
        goals = super().sample(batch_size, use_env_goal)
        num_masks = len(self.masks[list(self.masks.keys())[0]])
        mask_ids = np.random.choice(num_masks, batch_size)
        for mask_key in self.mask_keys:
            goals[mask_key] = self.masks[mask_key][mask_ids]
        return goals

class TaskPathCollector(ContextualPathCollector):
    def __init__(
            self,
            *args,
            task_key=None,
            max_path_length=100,
            task_ids=None,
            rotate_freq=0.0,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rotate_freq = rotate_freq
        self.rollout_tasks = []

        def obs_processor(o):
            if len(self.rollout_tasks) > 0:
                task = self.rollout_tasks[0]
                self.rollout_tasks = self.rollout_tasks[1:]
                o[task_key] = task
                self._env._rollout_context_batch[task_key] = task[None]

            combined_obs = [o[self._observation_key]]
            for k in self._context_keys_for_policy:
                combined_obs.append(o[k])
            return np.concatenate(combined_obs, axis=0)

        def reset_postprocess_func():
            rotate = (np.random.uniform() < self.rotate_freq)
            self.rollout_tasks = []
            if rotate:
                num_steps_per_task = max_path_length // len(task_ids)
                self.rollout_tasks = np.ones((max_path_length, 1)) * (len(task_ids) - 1)
                for (idx, id) in enumerate(task_ids):
                    start = idx * num_steps_per_task
                    end = start + num_steps_per_task
                    self.rollout_tasks[start:end] = id

        self._rollout_fn = partial(
            contextual_rollout,
            context_keys_for_policy=self._context_keys_for_policy,
            observation_key=self._observation_key,
            obs_processor=obs_processor,
            reset_postprocess_func=reset_postprocess_func,
        )

class MaskPathCollector(ContextualPathCollector):
    def __init__(
            self,
            *args,
            max_path_length=100,
            masks=None,
            rotate_freq=0.0,
            concat_context_to_obs_fn=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.masks = masks
        self.rotate_freq = rotate_freq
        self.rollout_mask_ids = []
        self._concat_context_to_obs_fn = concat_context_to_obs_fn

        def obs_processor(o):
            if len(self.rollout_mask_ids) > 0:
                mask_id = self.rollout_mask_ids[0]
                self.rollout_mask_ids = self.rollout_mask_ids[1:]
                for mask_key in self.masks.keys():
                    mask = self.masks[mask_key][mask_id]
                    o[mask_key] = mask
                    self._env._rollout_context_batch[mask_key] = mask[None]

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

        def reset_postprocess_func():
            rotate = (np.random.uniform() < self.rotate_freq)
            self.rollout_mask_ids = []
            if rotate:
                num_mask_ids = len(self.masks[list(self.masks.keys())[0]])
                num_steps_per_mask = max_path_length // num_mask_ids
                self.rollout_mask_ids = np.zeros(max_path_length, dtype=np.int32)
                for id in range(num_mask_ids):
                    start = id * num_steps_per_mask
                    end = start + num_steps_per_mask
                    self.rollout_mask_ids[start:end] = id

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
    mask_inference_variant = mask_variant['mask_inference_variant']
    n = mask_inference_variant['n']
    n = int(n)
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

    # data collection
    for i in range(n):
        obs_dict = env.reset()
        obs = obs_dict['state_observation']
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

def rl_context_experiment(variant):
    import railrl.torch.pytorch_util as ptu
    from railrl.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from railrl.torch.td3.td3 import TD3 as TD3Trainer
    from railrl.torch.sac.sac import SACTrainer
    from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
    from railrl.torch.sac.policies import TanhGaussianPolicy
    from railrl.torch.sac.policies import MakeDeterministic

    preprocess_rl_variant(variant)
    max_path_length = variant['max_path_length']
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_achieved_goal')
    context_key = desired_goal_key

    task_variant = variant.get('task_variant', {})
    task_conditioned = task_variant.get('task_conditioned', False)

    mask_variant = variant.get('mask_variant', {})
    mask_conditioned = mask_variant.get('mask_conditioned', False)

    if 'sac' in variant['algorithm'].lower():
        rl_algo = 'sac'
    elif 'td3' in variant['algorithm'].lower():
        rl_algo = 'td3'
    else:
        raise NotImplementedError
    print("RL algorithm:", rl_algo)

    assert not (task_conditioned and mask_conditioned)

    if task_conditioned:
        task_key = 'task_id'
        context_keys = [context_key, task_key]
    elif mask_conditioned:
        env = get_envs(variant)
        mask_format = mask_variant.get('mask_format', 'vector')
        assert mask_format in ['vector', 'matrix', 'distribution']
        goal_dim = env.observation_space.spaces[context_key].low.size
        if mask_format == 'vector':
            mask_keys = ['mask']
            mask_dims = [(goal_dim,)]
            context_dim = goal_dim + goal_dim
        elif mask_format == 'matrix':
            mask_keys = ['mask']
            mask_dims = [(goal_dim, goal_dim)]
            context_dim = goal_dim + (goal_dim * goal_dim)
        elif mask_format == 'distribution':
            mask_keys = ['mask_mu_w', 'mask_mu_g', 'mask_mu_mat', 'mask_sigma_inv']
            mask_dims = [(goal_dim,), (goal_dim,), (goal_dim, goal_dim), (goal_dim, goal_dim)]
            context_dim = goal_dim + (goal_dim * goal_dim)  # mu and sigma_inv
        else:
            raise NotImplementedError

        if mask_variant.get('infer_masks', False):
            mask_variant['mask_keys'] = mask_keys
            mask_variant['mask_dims'] = mask_dims
            masks = infer_masks(env, mask_variant)
            mask_variant['masks'] = masks

        context_keys = [context_key] + mask_keys
    else:
        context_keys = [context_key]

    def contextual_env_distrib_and_reward(goal_sampling_mode):
        env = get_envs(variant)
        env.goal_sampling_mode = goal_sampling_mode
        if task_conditioned:
            goal_distribution = TaskGoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
                task_key=task_key,
                task_ids=task_variant['task_ids']
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key), # achieved_goal_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
                additional_context_keys=[task_key],
            )
        elif mask_conditioned:
            goal_distribution = MaskedGoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
                mask_keys=mask_keys,
                mask_dims=mask_dims,
                mask_format=mask_format,
                masks=mask_variant.get('masks', None),
                idx_masks=mask_variant.get('idx_masks', None),
                matrix_masks=mask_variant.get('matrix_masks', None),
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key), # achieved_goal_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
                additional_context_keys=mask_keys,
                reward_fn=partial(default_masked_reward_fn, mask_format=mask_format),
            )
        else:
            goal_distribution = GoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key), # achieved_goal_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
            )
        diag_fn = GoalConditionedDiagnosticsToContextualDiagnostics(
            env.goal_conditioned_diagnostics,
            desired_goal_key=desired_goal_key,
            observation_key=observation_key,
        )
        env = ContextualEnv(
            env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[diag_fn],
            # update_env_info_fn=delete_info,
        )
        return env, goal_distribution, reward_fn

    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        variant.get("exploration_goal_sampling_mode", None)
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        variant.get("evaluation_goal_sampling_mode", None)
    )

    if task_conditioned:
        obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[context_key].low.size
            + 1
        )
    elif mask_conditioned:
        obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + context_dim
        )
    else:
        obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[context_key].low.size
        )

    action_dim = expl_env.action_space.low.size

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    if rl_algo == 'td3':
        policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        target_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        es = get_exploration_strategy(variant, expl_env)
        expl_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=policy,
        )
        eval_policy = policy
    elif rl_algo == 'sac':
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant['policy_kwargs']
        )
        expl_policy = policy
        eval_policy = MakeDeterministic(policy)

    def context_from_obs_dict_fn(obs_dict):
        context_dict = {
            context_key: obs_dict[observation_key],
        }
        if task_conditioned:
            context_dict[task_key] = obs_dict[task_key]
        elif mask_conditioned:
            for mask_key in mask_keys:
                context_dict[mask_key] = obs_dict[mask_key]
        return context_dict

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        if task_conditioned:
            task = batch[task_key]
            batch['observations'] = np.concatenate([obs, context, task], axis=1)
            batch['next_observations'] = np.concatenate([next_obs, context, task], axis=1)
        elif mask_conditioned:
            if mask_format in ['vector', 'matrix']:
                assert len(mask_keys) == 1
                mask = batch[mask_keys[0]].reshape((len(context), -1))
                batch['observations'] = np.concatenate([obs, context, mask], axis=1)
                batch['next_observations'] = np.concatenate([next_obs, context, mask], axis=1)
            elif mask_format == 'distribution':
                g = context
                mu_w = batch['mask_mu_w']
                mu_g = batch['mask_mu_g']
                mu_A = batch['mask_mu_mat']
                sigma_inv = batch['mask_sigma_inv']
                mu_w_given_g = mu_w + np.squeeze(mu_A @ np.expand_dims(g - mu_g, axis=-1), axis=-1)
                sigma_w_given_g_inv = sigma_inv.reshape((len(context), -1))
                batch['observations'] = np.concatenate([obs, mu_w_given_g, sigma_w_given_g_inv], axis=1)
                batch['next_observations'] = np.concatenate([next_obs, mu_w_given_g, sigma_w_given_g_inv], axis=1)
            else:
                raise NotImplementedError
        else:
            batch['observations'] = np.concatenate([obs, context], axis=1)
            batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch

    if 'observation_keys' not in variant['contextual_replay_buffer_kwargs']:
        variant['contextual_replay_buffer_kwargs']['observation_keys'] = []
    observation_keys = variant['contextual_replay_buffer_kwargs']['observation_keys']
    if observation_key not in observation_keys:
        observation_keys.append(observation_key)

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=context_keys,
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=context_from_obs_dict_fn,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **variant['contextual_replay_buffer_kwargs']
    )

    if rl_algo == 'td3':
        trainer = TD3Trainer(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['td3_trainer_kwargs']
        )
    elif rl_algo == 'sac':
        trainer = SACTrainer(
            env=env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['sac_trainer_kwargs']
        )

    def create_path_collector(
            env,
            policy,
            mode='expl',
            mask_kwargs={},
            task_kwargs={},
    ):
        assert mode in ['expl', 'eval']

        if task_conditioned:
            rotate_freq = task_variant['rotate_task_freq_for_expl'] if mode == 'expl' \
                else task_variant['rotate_task_freq_for_eval']
            return TaskPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                task_key=task_key,
                max_path_length=max_path_length,
                task_ids=task_variant['task_ids'],
                rotate_freq=rotate_freq,
            )
        elif mask_conditioned:
            if 'rotate_freq' in mask_kwargs:
                rotate_freq = mask_kwargs['rotate_freq']
            else:
                rotate_freq = mask_variant['rotate_mask_freq_for_expl'] if mode == 'expl' \
                    else mask_variant['rotate_mask_freq_for_eval']
            if 'masks' in mask_kwargs:
                masks = mask_kwargs['masks']
            else:
                mask_distribution = eval_context_distrib if mode == 'eval' else expl_context_distrib
                masks = mask_distribution.masks.copy()
            return MaskPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                max_path_length=100,
                masks=masks,
                rotate_freq=rotate_freq,
                concat_context_to_obs_fn=concat_context_to_obs,
            )
        else:
            return ContextualPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
            )

    expl_path_collector = create_path_collector(expl_env, expl_policy, mode='expl')
    eval_path_collector = create_path_collector(eval_env, eval_policy, mode='eval')

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    # if not variant.get("do_state_exp", False):
    #     env.vae.to(ptu.device)

    if variant.get("save_video", True):
        save_period = variant.get('save_video_period', 50)
        dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        dump_video_kwargs['horizon'] = max_path_length

        # rollout_function = partial(
        #     rf.contextual_rollout,
        #     max_path_length=max_path_length,
        #     observation_key=observation_key,
        #     context_keys_for_policy=context_keys,
        # )
        renderer = Renderer(**variant.get('renderer_kwargs', {}))

        def add_images(env, state_distribution):
            state_env = env.env
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            img_env = InsertImageEnv(state_env, renderer=renderer)
            return ContextualEnv(
                img_env,
                context_distribution=image_goal_distribution,
                reward_fn=eval_reward,
                observation_key=observation_key,
                update_env_info_fn=delete_info,
            )

        img_eval_env = add_images(eval_env, eval_context_distrib)
        video_path_collector = create_path_collector(img_eval_env, eval_policy, mode='eval')
        rollout_function = video_path_collector._rollout_fn
        eval_video_func = get_save_video_function(
            rollout_function,
            img_eval_env,
            eval_policy,
            tag="eval",
            imsize=renderer.image_shape[0],
            image_format='HWC',
            save_video_period=save_period,
            **dump_video_kwargs
        )
        algorithm.post_train_funcs.append(eval_video_func)

        log_expl_video = variant.get('log_expl_video', True)
        if log_expl_video:
            img_expl_env = add_images(expl_env, expl_context_distrib)
            video_path_collector = create_path_collector(img_expl_env, expl_policy, mode='expl')
            rollout_function = video_path_collector._rollout_fn
            expl_video_func = get_save_video_function(
                rollout_function,
                img_expl_env,
                expl_policy,
                tag="expl",
                imsize=renderer.image_shape[0],
                image_format='HWC',
                save_video_period=save_period,
                **dump_video_kwargs
            )
            algorithm.post_train_funcs.append(expl_video_func)

    if mask_conditioned and mask_variant.get('log_mask_diagnostics', True):
        masks = eval_context_distrib.masks.copy()
        collectors = []

        num_masks = len(masks[list(masks.keys())[0]])
        for mask_id in range(num_masks):
            masks_for_eval = {}
            for mask_key in masks.keys():
                masks_for_eval[mask_key] = np.array([masks[mask_key][mask_id]])
            mask_kwargs=dict(
                rotate_freq=1.0,
                masks=masks_for_eval,
            )
            collector = create_path_collector(eval_env, eval_policy, mode='eval', mask_kwargs=mask_kwargs)
            collectors.append(collector)
        log_prefixes = [
            'mask_{}/'.format(''.join(str(mask_id)))
            for mask_id in range(num_masks)
        ]

        def get_mask_diagnostics(unused):
            from railrl.core.logging import append_log, add_prefix, OrderedDict
            from railrl.misc import eval_util
            log = OrderedDict()
            for prefix, collector in zip(log_prefixes, collectors):
                paths = collector.collect_new_paths(
                    max_path_length,
                    max_path_length, #masking_eval_steps,
                    discard_incomplete_paths=True,
                )
                old_path_info = eval_util.get_generic_path_information(paths)

                keys_to_keep = []
                for key in old_path_info.keys():
                    if ('env_infos' in key) and ('final' in key) and ('Mean' in key):
                        keys_to_keep.append(key)
                path_info = OrderedDict()
                for key in keys_to_keep:
                    path_info[key] = old_path_info[key]

                generic_info = add_prefix(
                    path_info,
                    prefix,
                )
                append_log(log, generic_info)

            for collector in collectors:
                collector.end_epoch(0)
            return log

        algorithm._eval_get_diag_fns.append(get_mask_diagnostics)

    algorithm.train()
