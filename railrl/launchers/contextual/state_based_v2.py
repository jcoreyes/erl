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
from railrl.envs.contextual.mask_conditioned import (
    MaskedGoalDictDistributionFromMultitaskEnv,
    MaskPathCollector,
    default_masked_reward_fn,
)
from railrl.envs.contextual.mask_inference import infer_masks

from railrl.envs.contextual.task_conditioned import (
    TaskGoalDictDistributionFromMultitaskEnv,
    TaskPathCollector,
)
from railrl.envs.images import EnvRenderer, InsertImagesEnv
from railrl.launchers.contextual.util import (
    get_save_video_function,
)
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)

from railrl.launchers.rl_exp_launcher_util import (
    preprocess_rl_variant,
    get_envs,
    create_exploration_policy,
)

import copy
import torch
from functools import partial
import numpy as np

def rl_context_experiment(variant):
    import railrl.torch.pytorch_util as ptu
    from railrl.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from railrl.torch.td3.td3 import TD3 as TD3Trainer
    from railrl.torch.sac.sac import SACTrainer
    from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from railrl.torch.networks import ConcatMlp, TanhMlpPolicy
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
    print("mask_conditioned:", mask_conditioned)

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
            assert mask_format == 'distribution'
            env_kwargs = copy.deepcopy(variant['env_kwargs'])
            env_kwargs['lite_reset'] = True
            infer_masks_env = variant["env_class"](**env_kwargs)

            masks = infer_masks(
                infer_masks_env,
                mask_variant['idx_masks'],
                mask_variant['mask_inference_variant'],
            )
            mask_variant['masks'] = masks

        # relabel_context_key_blacklist = variant['contextual_replay_buffer_kwargs'].get('relabel_context_key_blacklist',
        #                                                                                [])
        # if not mask_variant.get('relabel_goals', True):
        #     relabel_context_key_blacklist += [context_key]
        # if not mask_variant.get('relabel_masks', True):
        #     relabel_context_key_blacklist += mask_keys
        # variant['contextual_replay_buffer_kwargs']['relabel_context_key_blacklist'] = relabel_context_key_blacklist

        context_keys = [context_key] + mask_keys
    else:
        context_keys = [context_key]

    def contextual_env_distrib_and_reward(mode='expl'):
        assert mode in ['expl', 'eval']
        env = get_envs(variant)

        if mode == 'expl':
            goal_sampling_mode = variant.get("expl_goal_sampling_mode", None)
        elif mode == 'eval':
            goal_sampling_mode = variant.get("eval_goal_sampling_mode", None)
        if goal_sampling_mode is not None:
            env.goal_sampling_mode = goal_sampling_mode

        if task_conditioned:
            context_distrib = TaskGoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
                task_key=task_key,
                task_ids=task_variant['task_ids']
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(achieved_goal_key), # observation_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
                additional_context_keys=[task_key],
            )
        elif mask_conditioned:
            context_distrib = MaskedGoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
                mask_keys=mask_keys,
                mask_dims=mask_dims,
                mask_format=mask_format,
                max_subtasks_to_focus_on=mask_variant.get('max_subtasks_to_focus_on', None),
                prev_subtask_weight=mask_variant.get('prev_subtask_weight', None),
                masks=mask_variant.get('masks', None),
                idx_masks=mask_variant.get('idx_masks', None),
                matrix_masks=mask_variant.get('matrix_masks', None),
                mask_distr=mask_variant.get('train_mask_distr', None),
            )
            reward_fn = mask_variant.get('reward_fn', default_masked_reward_fn)
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(achieved_goal_key), # observation_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
                additional_context_keys=mask_keys,
                reward_fn=partial(reward_fn, mask_format=mask_format, use_g_for_mean=mask_variant['use_g_for_mean']),
            )
        else:
            context_distrib = GoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(achieved_goal_key), # observation_key
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
            context_distribution=context_distrib,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[diag_fn],
            update_env_info_fn=delete_info,
        )
        return env, context_distrib, reward_fn

    env, context_distrib, reward_fn = contextual_env_distrib_and_reward(mode='expl')
    eval_env, eval_context_distrib, _ = contextual_env_distrib_and_reward(mode='eval')

    if task_conditioned:
        obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[context_key].low.size
            + 1
        )
    elif mask_conditioned:
        obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + context_dim
        )
    else:
        obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[context_key].low.size
        )

    action_dim = env.action_space.low.size

    if 'ckpt' in variant:
        from railrl.misc.asset_loader import local_path_from_s3_or_local_path
        import joblib
        import os.path as osp

        ckpt_epoch = variant.get('ckpt_epoch', None)
        if ckpt_epoch is not None:
            epoch = variant['ckpt_epoch']
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'itr_%d.pkl' % epoch))
        else:
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'params.pkl'))
        print("Loading ckpt from", filename)
        # data = joblib.load(filename)
        data = torch.load(filename, map_location='cuda:1')
        qf1 = data['trainer/qf1']
        qf2 = data['trainer/qf2']
        target_qf1 = data['trainer/target_qf1']
        target_qf2 = data['trainer/target_qf2']
        policy = data['trainer/policy']
        eval_policy = data['evaluation/policy']
        expl_policy = data['exploration/policy']
    else:
        qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **variant['qf_kwargs']
        )
        qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **variant['qf_kwargs']
        )
        target_qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **variant['qf_kwargs']
        )
        target_qf2 = ConcatMlp(
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
            expl_policy = create_exploration_policy(
                env, policy,
                exploration_version=variant['exploration_type'],
                exploration_noise=variant['exploration_noise'],
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
            context_key: obs_dict[achieved_goal_key], #observation_key
        }
        if task_conditioned:
            context_dict[task_key] = obs_dict[task_key]
        elif mask_conditioned:
            sample_masks_for_relabeling = mask_variant.get('sample_masks_for_relabeling', True)
            if sample_masks_for_relabeling:
                batch_size = obs_dict[list(obs_dict.keys())[0]].shape[0]
                sampled_contexts = context_distrib.sample(batch_size)
                for mask_key in mask_keys:
                    context_dict[mask_key] = sampled_contexts[mask_key]
            else:
                for mask_key in mask_keys:
                    context_dict[mask_key] = obs_dict[mask_key]
        return context_dict

    def post_process_mask_fn(obs_dict, context_dict):
        assert mask_conditioned
        pp_context_dict = copy.deepcopy(context_dict)

        mode = mask_variant.get('context_post_process_mode', None)
        assert mode in [
            'prev_subtasks_solved',
            'dilute_prev_subtasks_uniform',
            'dilute_prev_subtasks_fixed',
            'atomic_to_corresp_cumul',
            None
        ]

        if mode in [
            'prev_subtasks_solved',
            'dilute_prev_subtasks_uniform',
            'dilute_prev_subtasks_fixed',
            'atomic_to_corresp_cumul'
        ]:
            frac = mask_variant.get('context_post_process_frac', 0.50)
            cumul_mask_to_indices = context_distrib.get_cumul_mask_to_indices(context_dict['mask'])
            for k in cumul_mask_to_indices:
                indices = cumul_mask_to_indices[k]
                subset = np.random.choice(len(indices), int(len(indices)*frac), replace=False)
                cumul_mask_to_indices[k] = indices[subset]
        else:
            cumul_mask_to_indices = None

        if mode in ['prev_subtasks_solved', 'dilute_prev_subtasks_uniform', 'dilute_prev_subtasks_fixed']:
            cumul_masks = list(cumul_mask_to_indices.keys())
            for i in range(1, len(cumul_masks)):
                curr_mask = cumul_masks[i]
                prev_mask = cumul_masks[i-1]
                prev_obj_indices = np.where(np.array(prev_mask) > 0)[0]
                indices = cumul_mask_to_indices[curr_mask]
                if mode == 'prev_subtasks_solved':
                    pp_context_dict[context_key][indices][:,prev_obj_indices] = \
                        obs_dict[achieved_goal_key][indices][:,prev_obj_indices]
                elif mode == 'dilute_prev_subtasks_uniform':
                    pp_context_dict['mask'][indices][:, prev_obj_indices] = \
                        np.random.uniform(size=(len(indices), len(prev_obj_indices)))
                elif mode == 'dilute_prev_subtasks_fixed':
                    pp_context_dict['mask'][indices][:, prev_obj_indices] = 0.5
            indices_to_relabel = np.concatenate(list(cumul_mask_to_indices.values()))
            orig_masks = obs_dict['mask'][indices_to_relabel]
            atomic_mask_to_subindices = context_distrib.get_atomic_mask_to_indices(orig_masks)
            atomic_masks = list(atomic_mask_to_subindices.keys())
            cumul_masks = list(cumul_mask_to_indices.keys())
            for i in range(1, len(atomic_masks)):
                orig_atomic_mask = atomic_masks[i]
                relabeled_cumul_mask = cumul_masks[i]
                subindices = atomic_mask_to_subindices[orig_atomic_mask]
                pp_context_dict['mask'][indices_to_relabel][subindices] = relabeled_cumul_mask

        return pp_context_dict

    # if mask_conditioned:
    #     variant['contextual_replay_buffer_kwargs']['post_process_batch_fn'] = post_process_mask_fn

    def concat_context_to_obs(batch, replay_buffer=None, obs_dict=None, next_obs_dict=None, new_contexts=None):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        if task_conditioned:
            task = batch[task_key]
            batch['observations'] = np.concatenate([obs, context, task], axis=1)
            batch['next_observations'] = np.concatenate([next_obs, context, task], axis=1)
        elif mask_conditioned:
            if obs_dict is not None and new_contexts is not None:
                updated_contexts = post_process_mask_fn(obs_dict, new_contexts)
                batch.update(updated_contexts)

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
                if mask_variant['use_g_for_mean']:
                    mu_w_given_g = g
                else:
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
    if achieved_goal_key not in observation_keys:
        observation_keys.append(achieved_goal_key)

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=env,
        context_keys=context_keys,
        context_distribution=context_distrib,
        sample_context_from_obs_dict_fn=context_from_obs_dict_fn,
        reward_fn=reward_fn,
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
    ):
        assert mode in ['expl', 'eval']

        save_env_in_snapshot = variant.get('save_env_in_snapshot', True)

        if task_conditioned:
            rotate_freq = task_variant['rotate_task_freq_for_expl'] if mode == 'expl' \
                else task_variant['rotate_task_freq_for_eval']
            return TaskPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                save_env_in_snapshot=save_env_in_snapshot,
                task_key=task_key,
                max_path_length=max_path_length,
                task_ids=task_variant['task_ids'],
                rotate_freq=rotate_freq,
            )
        elif mask_conditioned:
            if 'rollout_mask_order' in mask_kwargs:
                rollout_mask_order = mask_kwargs['rollout_mask_order']
            else:
                if mode == 'expl':
                    rollout_mask_order = mask_variant.get('rollout_mask_order_for_expl', 'fixed')
                elif mode == 'eval':
                    rollout_mask_order = mask_variant.get('rollout_mask_order_for_eval', 'fixed')
                else:
                    raise NotImplementedError

            if 'mask_distr' in mask_kwargs:
                mask_distr = mask_kwargs['mask_distr']
            else:
                if mode == 'expl':
                    mask_distr = mask_variant['expl_mask_distr']
                elif mode == 'eval':
                    mask_distr = mask_variant['eval_mask_distr']
                else:
                    raise NotImplementedError

            prev_subtask_weight = mask_variant.get('prev_subtask_weight', None)
            prev_subtasks_solved = mask_variant.get('prev_subtasks_solved', False)
            max_subtasks_to_focus_on = mask_variant.get('max_subtasks_to_focus_on', None)
            max_subtasks_per_rollout = mask_variant.get('max_subtasks_per_rollout', None)
            mask_groups = mask_variant.get('mask_groups', None)

            mode = mask_variant.get('context_post_process_mode', None)
            if mode in ['dilute_prev_subtasks_uniform', 'dilute_prev_subtasks_fixed']:
                prev_subtask_weight = 0.5

            return MaskPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                concat_context_to_obs_fn=concat_context_to_obs,
                save_env_in_snapshot=save_env_in_snapshot,
                mask_sampler=(context_distrib if mode=='expl' else eval_context_distrib),
                mask_distr=mask_distr.copy(),
                mask_groups=mask_groups,
                max_path_length=max_path_length,
                rollout_mask_order=rollout_mask_order,
                prev_subtask_weight=prev_subtask_weight,
                prev_subtasks_solved=prev_subtasks_solved,
                max_subtasks_to_focus_on=max_subtasks_to_focus_on,
                max_subtasks_per_rollout=max_subtasks_per_rollout,
            )
        else:
            return ContextualPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                save_env_in_snapshot=save_env_in_snapshot,
            )

    expl_path_collector = create_path_collector(env, expl_policy, mode='expl')
    eval_path_collector = create_path_collector(eval_env, eval_policy, mode='eval')

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)

    if variant.get("save_video", True):
        save_period = variant.get('save_video_period', 50)
        dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        dump_video_kwargs['horizon'] = max_path_length

        renderer = EnvRenderer(**variant.get('renderer_kwargs', {}))

        def add_images(env, state_distribution):
            state_env = env.env
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            img_env = InsertImagesEnv(state_env, renderers={
                'image_observation' : renderer,
            })
            context_env = ContextualEnv(
                img_env,
                context_distribution=image_goal_distribution,
                reward_fn=reward_fn,
                observation_key=observation_key,
                update_env_info_fn=delete_info,
            )
            return context_env

        img_eval_env = add_images(eval_env, eval_context_distrib)

        if variant.get('log_eval_video', True):
            video_path_collector = create_path_collector(img_eval_env, eval_policy, mode='eval')
            rollout_function = video_path_collector._rollout_fn
            eval_video_func = get_save_video_function(
                rollout_function,
                img_eval_env,
                eval_policy,
                tag="eval",
                imsize=variant['renderer_kwargs']['width'],
                image_format='CHW',
                save_video_period=save_period,
                **dump_video_kwargs
            )
            algorithm.post_train_funcs.append(eval_video_func)

        # additional eval videos for mask conditioned case
        if mask_conditioned:
            default_list = [
                'atomic',
                'atomic_seq',
                'cumul_seq',
                'full',
            ]
            eval_rollouts_for_videos = mask_variant.get('eval_rollouts_for_videos', default_list)
            for key in eval_rollouts_for_videos:
                assert key in default_list

            if 'cumul_seq' in eval_rollouts_for_videos:
                video_path_collector = create_path_collector(
                    img_eval_env,
                    eval_policy,
                    mode='eval',
                    mask_kwargs=dict(
                        mask_distr=dict(
                            cumul_seq=1.0
                        ),
                    ),
                )
                rollout_function = video_path_collector._rollout_fn
                eval_video_func = get_save_video_function(
                    rollout_function,
                    img_eval_env,
                    eval_policy,
                    tag="eval_cumul" if mask_conditioned else "eval",
                    imsize=variant['renderer_kwargs']['width'],
                    image_format='HWC',
                    save_video_period=save_period,
                    **dump_video_kwargs
                )
                algorithm.post_train_funcs.append(eval_video_func)

            if 'full' in eval_rollouts_for_videos:
                video_path_collector = create_path_collector(
                    img_eval_env,
                    eval_policy,
                    mode='eval',
                    mask_kwargs=dict(
                        mask_distr=dict(
                            full=1.0
                        ),
                    ),
                )
                rollout_function = video_path_collector._rollout_fn
                eval_video_func = get_save_video_function(
                    rollout_function,
                    img_eval_env,
                    eval_policy,
                    tag="eval_full",
                    imsize=variant['renderer_kwargs']['width'],
                    image_format='HWC',
                    save_video_period=save_period,
                    **dump_video_kwargs
                )
                algorithm.post_train_funcs.append(eval_video_func)

            if 'atomic_seq' in eval_rollouts_for_videos:
                video_path_collector = create_path_collector(
                    img_eval_env,
                    eval_policy,
                    mode='eval',
                    mask_kwargs=dict(
                        mask_distr=dict(
                            atomic_seq=1.0
                        ),
                    ),
                )
                rollout_function = video_path_collector._rollout_fn
                eval_video_func = get_save_video_function(
                    rollout_function,
                    img_eval_env,
                    eval_policy,
                    tag="eval_atomic",
                    imsize=variant['renderer_kwargs']['width'],
                    image_format='HWC',
                    save_video_period=save_period,
                    **dump_video_kwargs
                )
                algorithm.post_train_funcs.append(eval_video_func)

        if variant.get('log_expl_video', True):
            img_expl_env = add_images(env, context_distrib)
            video_path_collector = create_path_collector(img_expl_env, expl_policy, mode='expl')
            rollout_function = video_path_collector._rollout_fn
            expl_video_func = get_save_video_function(
                rollout_function,
                img_expl_env,
                expl_policy,
                tag="expl",
                imsize=variant['renderer_kwargs']['width'],
                image_format='CHW',
                save_video_period=save_period,
                **dump_video_kwargs
            )
            algorithm.post_train_funcs.append(expl_video_func)

    if mask_conditioned and mask_variant.get('log_mask_diagnostics', True):
        collectors = []
        log_prefixes = []

        default_list = [
            'atomic',
            'atomic_seq',
            'cumul_seq',
            'full',
        ]
        eval_rollouts_to_log = mask_variant.get('eval_rollouts_to_log', default_list)
        for key in eval_rollouts_to_log:
            assert key in default_list

        # atomic masks
        if 'atomic' in eval_rollouts_to_log:
            # masks = eval_context_distrib.masks.copy()
            # num_masks = len(masks[list(masks.keys())[0]])
            num_masks = len(eval_path_collector.mask_groups)
            for mask_id in range(num_masks):
                mask_kwargs=dict(
                    rollout_mask_order=[mask_id],
                    mask_distr=dict(
                        atomic_seq=1.0,
                    ),
                )
                collector = create_path_collector(eval_env, eval_policy, mode='eval', mask_kwargs=mask_kwargs)
                collectors.append(collector)
            log_prefixes += [
                'mask_{}/'.format(''.join(str(mask_id)))
                for mask_id in range(num_masks)
            ]

        # full mask
        if 'full' in eval_rollouts_to_log:
            mask_kwargs=dict(
                mask_distr=dict(
                    full=1.0,
                ),
            )
            collector = create_path_collector(eval_env, eval_policy, mode='eval', mask_kwargs=mask_kwargs)
            collectors.append(collector)
            log_prefixes.append('mask_full/')

        # cumulative, sequential mask
        if 'cumul_seq' in eval_rollouts_to_log:
            mask_kwargs=dict(
                rollout_mask_order='fixed',
                mask_distr=dict(
                    cumul_seq=1.0,
                ),
            )
            collector = create_path_collector(eval_env, eval_policy, mode='eval', mask_kwargs=mask_kwargs)
            collectors.append(collector)
            log_prefixes.append('mask_cumul_seq/')

        # atomic, sequential mask
        if 'atomic_seq' in eval_rollouts_to_log:
            mask_kwargs=dict(
                rollout_mask_order='fixed',
                mask_distr=dict(
                    atomic_seq=1.0,
                ),
            )
            collector = create_path_collector(eval_env, eval_policy, mode='eval', mask_kwargs=mask_kwargs)
            collectors.append(collector)
            log_prefixes.append('mask_atomic_seq/')

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
                # old_path_info = eval_util.get_generic_path_information(paths)
                old_path_info = eval_env.get_diagnostics(paths)

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