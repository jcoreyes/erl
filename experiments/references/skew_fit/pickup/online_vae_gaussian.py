import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
import railrl.torch.vae.vae_schedules as vae_schedules
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from railrl.envs.goal_generation.pickup_goal_dataset import \
        generate_vae_dataset, get_image_presampled_goals_from_vae_env
from multiworld.envs.mujoco.cameras import (
        sawyer_pick_and_place_camera,
        sawyer_pick_and_place_camera_slanted_angle,
        # sawyer_pick_and_place_camera_zoomed,
)

from railrl.torch.vae.conv_vae import imsize48_default_architecture

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
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=750,
                    num_steps_per_epoch=500,
                    num_steps_per_eval=500,
                    min_num_steps_before_training=10000,
                    batch_size=1024,
                    max_path_length=50,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    # collection_mode='online-parallel',
                    parallel_env_params=dict(
                        num_workers=2,
                    ),
                    reward_scale=1,
                ),

                her_kwargs=dict(
                ),
                twin_sac_kwargs=dict(
                    train_policy_with_reparameterization=True,
                    soft_target_tau=1e-3,  # 1e-2
                    policy_update_period=1,
                    target_update_period=1,  # 1
                    use_automatic_entropy_tuning=True,
                ),
                online_vae_kwargs=dict(
                    vae_training_schedule=vae_schedules.custom_schedule_2,
                    oracle_data=False,
                    vae_save_period=5,
                    parallel_vae_train=False,
                ),
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
                power=.1,

            ),

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

    )

    search_space = {
        # 'train_vae_variant.algo_kwargs.skew_config.power': [-.1],
        'grill_variant.online_vae_beta': [30],
        'grill_variant.replay_buffer_kwargs.power': [-1],
        'init_camera': [
            sawyer_pick_and_place_camera,
        ],
        'grill_variant.algo_kwargs.online_vae_kwargs.vae_training_schedule':
            [vae_schedules.custom_schedule],
        'grill_variant.vae_wrapped_env_kwargs.goal_sampler_for_exploration': [True],
        'grill_variant.vae_wrapped_env_kwargs.goal_sampler_for_relabeling': [True],
        'train_vae_variant.beta': [30],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 3
    mode = 'local'
    exp_prefix = 'pickup-online-vae-paper-master'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                snapshot_gap=200,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-west1-b',
                    # preemptible=False,
                    # instance_type="n1-standard-4"
                ),

            )
