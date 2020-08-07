import os.path as osp
import typing
from collections import OrderedDict
from functools import partial

import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.distribution import DictDistribution
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
)
from rlkit.envs.contextual import (
    ContextualEnv,
    delete_info,
    ContextualRewardFn,
)
from rlkit.envs.contextual.set_distributions import (
    LatentGoalDictDistributionFromSet,
    SetDiagnostics,
    OracleRIGMeanSetter,
    SetReward,
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector,
)
from rlkit.samplers.rollout_functions import contextual_rollout
from rlkit.torch.distributions import MultivariateDiagonalNormal
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sets import set_vae_trainer
from rlkit.torch.sets.models import create_dummy_image_vae
from rlkit.torch.sets.rl_launcher import (
    InitStateConditionedContextualEnv,
    NormalLikelihoodRewardFn,
    FilterKeys,
    DisCoVideoSaveFunction,
)
from rlkit.torch.sets.set_creation import create_sets
from rlkit.torch.sets.set_projection import Set
from rlkit.torch.sets.vae_launcher import train_set_vae
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOfflineBatchRLAlgorithm,
)
from rlkit.visualization import video
from rlkit.torch.vae.vae_torch_trainer import VAE



def offline_disco_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        generate_set_for_rl_kwargs,
        # VAE parameters
        create_vae_kwargs,
        vae_trainer_kwargs,
        vae_algo_kwargs,
        data_loader_kwargs,
        generate_set_for_vae_pretraining_kwargs,
        num_ungrouped_images,
        beta_schedule_kwargs=None,
        # Oracle settings
        use_ground_truth_reward=False,
        use_onehot_set_embedding=False,
        use_dummy_model=False,
        observation_key="latent_observation",
        # RIG comparison
        rig_goal_setter_kwargs=None,
        rig=False,
        # Miscellaneous
        reward_fn_kwargs=None,
        # None-VAE Params
        env_id=None,
        env_class=None,
        env_kwargs=None,
        latent_observation_key="latent_observation",
        state_observation_key="state_observation",
        image_observation_key="image_observation",
        set_description_key="set_description",
        example_state_key="example_state",
        example_image_key="example_image",
        # Exploration
        no_exploration=False,
        replay_buffer_path=None,
        exploration_policy_kwargs=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
):
    if rig_goal_setter_kwargs is None:
        rig_goal_setter_kwargs = {}
    if reward_fn_kwargs is None:
        reward_fn_kwargs = {}
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}

    renderer = EnvRenderer(**renderer_kwargs)
    state_env = get_gym_env(
        env_id, env_class=env_class, env_kwargs=env_kwargs
    )

    sets = create_sets(
        env_id,
        env_class,
        env_kwargs,
        renderer,
        example_state_key=example_state_key,
        example_image_key=example_image_key,
        **generate_set_for_rl_kwargs,
    )
    if use_dummy_model:
        model = create_dummy_image_vae(
            img_chw=renderer.image_chw,
            **create_vae_kwargs)
    else:
        model = train_set_vae(
            create_vae_kwargs,
            vae_trainer_kwargs,
            vae_algo_kwargs,
            data_loader_kwargs,
            generate_set_for_vae_pretraining_kwargs,
            num_ungrouped_images,
            env_id=env_id,
            env_class=env_class,
            env_kwargs=env_kwargs,
            beta_schedule_kwargs=beta_schedule_kwargs,
            sets=sets,
            renderer=renderer,
        )
    expl_env, expl_context_distrib, expl_reward = (
        contextual_env_distrib_and_reward(
            model,
            sets,
            state_env,
            renderer,
            reward_fn_kwargs,
            use_ground_truth_reward,
            state_observation_key,
            latent_observation_key,
            example_image_key,
            set_description_key,
            observation_key,
            image_observation_key,
            rig_goal_setter_kwargs,
        )
    )
    eval_env, eval_context_distrib, eval_reward = (
        contextual_env_distrib_and_reward(
            model,
            sets,
            state_env,
            renderer,
            reward_fn_kwargs,
            use_ground_truth_reward,
            state_observation_key,
            latent_observation_key,
            example_image_key,
            set_description_key,
            observation_key,
            image_observation_key,
            rig_goal_setter_kwargs,
            oracle_rig_goal=rig,
        )
    )
    context_keys = [
        expl_context_distrib.mean_key,
        expl_context_distrib.covariance_key,
        expl_context_distrib.set_index_key,
        expl_context_distrib.set_embedding_key,
    ]
    if rig:
        context_keys_for_rl = [
            expl_context_distrib.mean_key,
        ]
    else:
        if use_onehot_set_embedding:
            context_keys_for_rl = [
                expl_context_distrib.set_embedding_key,
            ]
        else:
            context_keys_for_rl = [
                expl_context_distrib.mean_key,
                expl_context_distrib.covariance_key,
            ]

    obs_dim = np.prod(expl_env.observation_space.spaces[observation_key].shape)
    obs_dim += sum(
        [np.prod(expl_env.observation_space.spaces[k].shape)
         for k in context_keys_for_rl]
    )
    action_dim = np.prod(expl_env.action_space.shape)

    def create_qf():
        return ConcatMlp(
            input_size=obs_dim + action_dim, output_size=1, **qf_kwargs
        )

    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim, action_dim=action_dim, **policy_kwargs
    )

    def concat_context_to_obs(batch, *args, **kwargs):
        obs = batch["observations"]
        next_obs = batch["next_observations"]
        contexts = [batch[k] for k in context_keys_for_rl]
        batch["observations"] = np.concatenate((obs, *contexts), axis=1)
        batch["next_observations"] = np.concatenate(
            (next_obs, *contexts), axis=1,
        )
        return batch

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=context_keys,
        observation_keys=list({
            observation_key,
            state_observation_key,
            latent_observation_key
        }),
        observation_key=observation_key,
        context_distribution=FilterKeys(expl_context_distrib, context_keys,),
        sample_context_from_obs_dict_fn=None,
        # RemapKeyFn({context_key: observation_key}),
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs,
    )
    trainer = SACTrainer(
        env=expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs,
    )

    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
        observation_key=observation_key,
        context_keys_for_policy=context_keys_for_rl,
    )
    exploration_policy = create_exploration_policy(
        expl_env, policy, **exploration_policy_kwargs
    )
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        context_keys_for_policy=context_keys_for_rl,
    )

    if no_exploration:
        algorithm = TorchOfflineBatchRLAlgorithm(
            trainer=trainer,
            evaluation_env=eval_env,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=max_path_length,
            **algo_kwargs,
        )
    else:
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=max_path_length,
            **algo_kwargs,
        )
    algorithm.to(ptu.device)

    if save_video:
        set_index_key = eval_context_distrib.set_index_key
        eval_video_func = DisCoVideoSaveFunction(
            model,
            sets,
            eval_path_collector,
            tag="eval",
            reconstruction_key="image_reconstruction",
            decode_set_image_key="decoded_set_prior",
            set_visualization_key="set_visualization",
            example_image_key=example_image_key,
            set_index_key=set_index_key,
            columns=len(sets),
            unnormalize=True,
            imsize=48,
            image_format=renderer.output_image_format,
            **save_video_kwargs,
        )
        algorithm.post_train_funcs.append(eval_video_func)

    algorithm.train()


def contextual_env_distrib_and_reward(
        vae,
        sets: typing.List[Set],
        state_env,
        renderer,
        reward_fn_kwargs,
        use_ground_truth_reward,
        state_observation_key,
        latent_observation_key,
        example_image_key,
        set_description_key,
        observation_key,
        image_observation_key,
        rig_goal_setter_kwargs,
        oracle_rig_goal=False,
):
    img_env = InsertImageEnv(state_env, renderer=renderer)
    encoded_env = EncoderWrappedEnv(
        img_env,
        vae,
        step_keys_map={image_observation_key: latent_observation_key},
    )
    if oracle_rig_goal:
        context_env_class = InitStateConditionedContextualEnv
        goal_distribution_params_distribution = (
            OracleRIGMeanSetter(
                sets, vae, example_image_key,
                env=state_env,
                renderer=renderer,
                cycle_for_batch_size_1=True,
                **rig_goal_setter_kwargs
            )
        )
    else:
        context_env_class = ContextualEnv
        goal_distribution_params_distribution = (
            LatentGoalDictDistributionFromSet(
                sets, vae, example_image_key, cycle_for_batch_size_1=True,
            )
        )
    set_diagnostics = SetDiagnostics(
        set_description_key=set_description_key,
        set_index_key=goal_distribution_params_distribution.set_index_key,
        observation_key=state_observation_key,
    )
    reward_fn, unbatched_reward_fn = create_reward_fn(
        sets,
        use_ground_truth_reward,
        goal_distribution_params_distribution,
        state_observation_key,
        latent_observation_key,
        reward_fn_kwargs,
    )
    env = context_env_class(
        encoded_env,
        context_distribution=goal_distribution_params_distribution,
        reward_fn=reward_fn,
        unbatched_reward_fn=unbatched_reward_fn,
        observation_key=observation_key,
        contextual_diagnostics_fns=[
            # goal_diagnostics,
            set_diagnostics,
        ],
        update_env_info_fn=delete_info,
    )
    return env, goal_distribution_params_distribution, reward_fn


def create_reward_fn(
        sets,
        use_ground_truth_reward,
        goal_distribution_params_distribution,
        state_observation_key,
        latent_observation_key,
        reward_fn_kwargs,
):
    if use_ground_truth_reward:
        reward_fn = SetReward(
            sets=sets,
            set_index_key=goal_distribution_params_distribution.set_index_key,
            observation_key=state_observation_key,
        )
        unbatched_reward_fn = SetReward(
            sets=sets,
            set_index_key=goal_distribution_params_distribution.set_index_key,
            observation_key=state_observation_key,
            batched=False,
        )
    else:
        reward_fn = NormalLikelihoodRewardFn(
            observation_key=latent_observation_key,
            mean_key=goal_distribution_params_distribution.mean_key,
            covariance_key=goal_distribution_params_distribution.covariance_key,
            **reward_fn_kwargs
        )
        unbatched_reward_fn = NormalLikelihoodRewardFn(
            observation_key=latent_observation_key,
            mean_key=goal_distribution_params_distribution.mean_key,
            covariance_key=goal_distribution_params_distribution.covariance_key,
            batched=False,
            **reward_fn_kwargs
        )
    return reward_fn, unbatched_reward_fn
