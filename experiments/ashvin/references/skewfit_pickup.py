import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from rlkit.envs.goal_generation.pickup_goal_dataset import \
        generate_vae_dataset, get_image_presampled_goals_from_vae_env
from multiworld.envs.mujoco.cameras import (
        sawyer_pick_and_place_camera,
        sawyer_pick_and_place_camera_slanted_angle,
        # sawyer_pick_and_place_camera_zoomed,
)

from rlkit.torch.vae.conv_vae import imsize48_default_architecture

from rlkit.launchers.arglauncher import run_variants

if __name__ == "__main__":
    num_images = 1
    variant = dict(
        imsize=48,
        double_algo=False,
        env_id="SawyerPickupEnvYZEasy-v0",
        grill_variant=dict(
            sample_goals_from_buffer=True,
            save_video=True,
            save_video_period=50,
            presample_goals=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=30,
                lr=1e-3,
            ),
            generate_goal_dataset_fctn=get_image_presampled_goals_from_vae_env,
            goal_generation_kwargs=dict(
                num_presampled_goals=500,
            ),
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=50,
            algo_kwargs=dict(
                batch_size=1024,
                num_epochs=750,
                num_eval_steps_per_epoch=500,
                num_expl_steps_per_train_loop=500,
                num_trains_per_train_loop=1000,
                min_num_steps_before_training=10000,
                vae_training_schedule=vae_schedules.custom_schedule,
                oracle_data=False,
                vae_save_period=50,
                parallel_vae_train=False,
            ),
            twin_sac_trainer_kwargs=dict(
                reward_scale=1,
                discount=0.99,
                soft_target_tau=1e-3,
                target_update_period=1,
                use_automatic_entropy_tuning=True,
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    # decoder_distribution='bernoulli',
                    num_latents_to_sample=10,
                ),
                power=-1,
                relabeling_goal_sampling_mode='custom_goal_sampler',
            ),
            exploration_goal_sampling_mode='custom_goal_sampler',
            evaluation_goal_sampling_mode='env',
            algorithm='GRILL-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,  # True does a bit better
            ),
        ),
        train_vae_variant=dict(
            representation_size=16,
            beta=5,
            num_epochs=0,
            dump_skew_debug_plots=True,
            decoder_activation='gaussian',
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            ),
            generate_vae_data_fctn=generate_vae_dataset,
            generate_vae_dataset_kwargs=dict(
                N=10,
                oracle_dataset=True,
                use_cached=False,
                num_channels=3*num_images,
            ),


            algo_kwargs=dict(
                start_skew_epoch=12000,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                skew_dataset=True,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    # sampling_method='importance_sampling',
                    sampling_method='true_prior_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),
            save_period=10,
        ),
        init_camera=sawyer_pick_and_place_camera,
    )

    search_space = {
        'seedid': range(5),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(grill_her_twin_sac_online_vae_full_experiment, variants)
