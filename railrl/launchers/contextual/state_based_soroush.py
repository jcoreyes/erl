from functools import partial

import numpy as np

from railrl.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
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

def td3_experiment(variant):
    import railrl.samplers.rollout_functions as rf
    import railrl.torch.pytorch_util as ptu
    from railrl.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from railrl.torch.td3.td3 import TD3 as TD3Trainer
    from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from railrl.torch.networks import FlattenMlp, TanhMlpPolicy

    preprocess_rl_variant(variant)
    max_path_length = variant['max_path_length']
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_achieved_goal')
    context_key = desired_goal_key
    sample_context_from_obs_dict_fn = RemapKeyFn({context_key: observation_key})

    def contextual_env_distrib_and_reward(goal_sampling_mode):
        env = get_envs(variant)
        env.goal_sampling_mode = goal_sampling_mode
        goal_distribution = GoalDictDistributionFromMultitaskEnv(
            env,
            desired_goal_keys=[desired_goal_key],
        )
        reward_fn = ContextualRewardFnFromMultitaskEnv(
            env=env,
            achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key),
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
            additional_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
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
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
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

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
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
        context_keys=[context_key],
        # observation_keys=observation_keys,
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=sample_context_from_obs_dict_fn,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **variant['contextual_replay_buffer_kwargs']
    )

    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )

    eval_path_collector = ContextualPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        context_keys_for_policy=[context_key],
    )
    expl_path_collector = ContextualPathCollector(
        expl_env,
        expl_policy,
        observation_key=observation_key,
        context_keys_for_policy=[context_key],
    )

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

        rollout_function = partial(
            rf.contextual_rollout,
            max_path_length=max_path_length,
            observation_key=observation_key,
            context_keys_for_policy=[context_key],
        )
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
        eval_video_func = get_save_video_function(
            rollout_function,
            img_eval_env,
            policy,
            tag="",
            imsize=renderer.image_shape[0],
            image_format='CWH',
            save_video_period=save_period,
            **dump_video_kwargs
        )

        algorithm.post_train_funcs.append(eval_video_func)


    algorithm.train()
