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
    PixelDistance,
    PresampledDistribution,
)
from railrl.envs.images import Renderer, InsertImageEnv
from railrl.launchers.contextual.util import (
    get_gym_env,
    get_save_video_function,
)
from railrl.launchers.rl_exp_launcher_util import create_exploration_policy
from railrl.samplers import rollout_functions as rf
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from railrl.torch import pytorch_util as ptu
from railrl.torch.networks import FlattenMlp
from railrl.torch.networks.basic import (
    MultiInputSequential,
    FlattenEachParallel,
    Flatten,
)
from railrl.torch.sac.policies import (
    MakeDeterministic,
    TanhGaussianWithBasicObsProcessorPolicy,
)
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def image_based_goal_conditioned_sac_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        reward_type='state_distance',
        env_renderer_kwargs=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        video_renderer_kwargs=None,
):
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not env_renderer_kwargs:
        env_renderer_kwargs = {}
    if not video_renderer_kwargs:
        video_renderer_kwargs = {}
    img_observation_key = 'image_observation'
    img_desired_goal_key = 'image_desired_goal'
    state_observation_key = 'state_observation'
    state_desired_goal_key = 'state_desired_goal'
    state_achieved_goal_key = 'state_achieved_goal'
    sample_context_from_obs_dict_fn = RemapKeyFn({
        'image_desired_goal': 'image_observation',
        'state_desired_goal': 'state_observation',
    })

    def setup_contextual_env(
            env_id, env_class, env_kwargs, goal_sampling_mode, renderer
    ):
        state_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        state_env.goal_sampling_mode = goal_sampling_mode
        state_goal_distribution = GoalDictDistributionFromMultitaskEnv(
            state_env,
            desired_goal_keys=[state_desired_goal_key],
        )
        state_diag_fn = GoalConditionedDiagnosticsToContextualDiagnostics(
            state_env.goal_conditioned_diagnostics,
            desired_goal_key=state_desired_goal_key,
            observation_key=state_observation_key,
        )
        image_goal_distribution = AddImageDistribution(
            env=state_env,
            base_distribution=state_goal_distribution,
            image_goal_key=img_desired_goal_key,
            renderer=renderer,
        )
        goal_distribution = PresampledDistribution(image_goal_distribution, 256)
        img_env = InsertImageEnv(state_env, renderer=renderer)
        if reward_type == 'state_distance':
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=state_env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(
                    'state_observation'
                ),
                desired_goal_key=state_desired_goal_key,
                achieved_goal_key=state_achieved_goal_key,
            )
        elif reward_type == 'pixel_distance':
            reward_fn = PixelDistance(
                env=state_env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(
                    img_observation_key
                ),
                desired_goal_key=img_desired_goal_key,
            )
        else:
            raise ValueError(reward_type)
        env = ContextualEnv(
            img_env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            observation_key=img_observation_key,
            contextual_diagnostics_fns=[state_diag_fn],
            update_env_info_fn=delete_info,
        )
        return env, goal_distribution, reward_fn

    env_renderer = Renderer(**env_renderer_kwargs)
    expl_env, expl_context_distrib, expl_reward = setup_contextual_env(
        env_id, env_class, env_kwargs, exploration_goal_sampling_mode,
        env_renderer
    )
    eval_env, eval_context_distrib, eval_reward = setup_contextual_env(
        env_id, env_class, env_kwargs, evaluation_goal_sampling_mode,
        env_renderer
    )

    obs_dim = (
            expl_env.observation_space.spaces[img_observation_key].low.size
            + expl_env.observation_space.spaces[img_desired_goal_key].low.size
    )
    action_dim = expl_env.action_space.low.size

    def create_qf():
        return MultiInputSequential(
            FlattenEachParallel(),
            FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                **qf_kwargs
            )
        )

    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    policy = TanhGaussianWithBasicObsProcessorPolicy(
        obs_processor=Flatten(),
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs
    )

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[img_desired_goal_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=[img_desired_goal_key, state_desired_goal_key],
        observation_key=img_observation_key,
        observation_keys=[img_observation_key, state_observation_key],
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=sample_context_from_obs_dict_fn,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs
    )
    trainer = SACTrainer(
        env=expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs
    )

    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
        observation_key=img_observation_key,
        context_keys_for_policy=[img_desired_goal_key],
    )
    exploration_policy = create_exploration_policy(
        policy, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=img_observation_key,
        context_keys_for_policy=[img_desired_goal_key],
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs
    )
    algorithm.to(ptu.device)

    if save_video:
        rollout_function = partial(
            rf.contextual_rollout,
            max_path_length=max_path_length,
            observation_key=img_observation_key,
            context_keys_for_policy=[img_desired_goal_key],
        )
        video_renderer = Renderer(**video_renderer_kwargs)
        video_eval_env = InsertImageEnv(
            eval_env, renderer=video_renderer, image_key='video_observation')
        video_expl_env = InsertImageEnv(
            expl_env, renderer=video_renderer, image_key='video_observation')
        video_eval_env = ContextualEnv(
            video_eval_env,
            context_distribution=eval_env.context_distribution,
            reward_fn=lambda *_: np.array([0]),
            observation_key=img_observation_key,
        )
        video_expl_env = ContextualEnv(
            video_expl_env,
            context_distribution=expl_env.context_distribution,
            reward_fn=lambda *_: np.array([0]),
            observation_key=img_observation_key,
        )
        eval_video_func = get_save_video_function(
            rollout_function,
            video_eval_env,
            MakeDeterministic(policy),
            tag="eval",
            imsize=video_renderer.image_shape[1],
            image_format='HWC',
            keys_to_show=[
                'image_desired_goal', 'image_observation', 'video_observation'
            ],
            **save_video_kwargs
        )
        expl_video_func = get_save_video_function(
            rollout_function,
            video_expl_env,
            exploration_policy,
            tag="train",
            imsize=video_renderer.image_shape[1],
            image_format='HWC',
            keys_to_show=[
                'image_desired_goal', 'image_observation', 'video_observation'
            ],
            **save_video_kwargs
        )

        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(expl_video_func)

    algorithm.train()