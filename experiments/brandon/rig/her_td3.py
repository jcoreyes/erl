import gym
import numpy as np

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.launchers.launcher_util import run_experiment
from rlkit.samplers.data_collector.path_collector import ObsDictPathCollector
from rlkit.samplers.data_collector.step_collector import ObsDictStepCollector
from rlkit.torch.networks import (
    ConcatMlp, MergedCNN, PretrainedCNN, Flatten,
    MlpQfWithObsProcessor,
)
from rlkit.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicy,
    TanhCNNGaussianPolicy,
    TanhGaussianPolicyAdapter,
)
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)
import torchvision.models as models
import torch.nn as nn
from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0


def experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.her.her import HERTrainer
    from rlkit.torch.td3.td3 import TD3 as TD3Trainer
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.samplers.data_collector import GoalConditionedPathCollector
    from rlkit.torch.grill.launcher import (
        grill_preprocess_variant, get_envs, get_exploration_strategy,
        full_experiment_variant_preprocess,
        train_vae_and_update_variant,
        get_video_save_func,
    )

    full_experiment_variant_preprocess(variant)
    if not variant['grill_variant'].get('do_state_exp', False):
        train_vae_and_update_variant(variant)
    variant = variant['grill_variant']

    grill_preprocess_variant(variant)
    eval_env = get_envs(variant)
    expl_env = get_envs(variant)
    es = get_exploration_strategy(variant, expl_env)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = expl_env.action_space.low.size
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
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
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
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True): # Does not work
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            eval_env,
            policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        eval_env.vae.to(ptu.device)
        expl_env.vae.to(ptu.device)

    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        imsize=84,
        init_camera=sawyer_xyz_reacher_camera_v0,
        env_id='SawyerReachXYEnv-v1',
        grill_variant=dict(
            save_video=False,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                num_epochs=505,
                num_eval_steps_per_epoch=1000,
                num_expl_steps_per_train_loop=10,
                num_trains_per_train_loop=10,
                min_num_steps_before_training=4000,
                batch_size=128,
                max_path_length=100,
            ),
            her_kwargs=dict(),
            td3_kwargs=dict(
                tau=1e-2,
                reward_scale=1,
                discount=0.99,
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_rollout_goals=0.1,
                fraction_goals_env_goals=0.5,
            ),
            algorithm='OFFLINE-VAE-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.3,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=16,
            beta=1,
            num_epochs=1,
            dump_skew_debug_plots=False,
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=50,
                oracle_dataset=False,
                use_cached=True,
                vae_dataset_specific_kwargs=dict(),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
            ),
            save_period=50,
            decoder_activation='sigmoid',
        ),
    )

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 3
    # mode = 'gcp'
    # exp_prefix = 'sawyer_xy_reacher_replicate'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
          )