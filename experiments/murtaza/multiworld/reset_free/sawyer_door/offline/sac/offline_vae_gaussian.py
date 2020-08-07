import rlkit.misc.hyperparameter as hyp
from rlkit.torch.vae.dataset.generate_goal_dataset import generate_goal_dataset_using_policy
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v3
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_twin_sac_full_experiment

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        init_camera=sawyer_door_env_camera_v3,
        env_id='SawyerDoorHookResetFreeEnv-v5',
        grill_variant=dict(
            save_video=True,
            save_video_period=50,
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
                    num_epochs=1005,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=500,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    collection_mode='online-parallel',
                    parallel_env_params=dict(
                        num_workers=1,
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
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='OFFLINE-INV-GAUSSIAN-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            generate_goal_dataset_fctn=generate_goal_dataset_using_policy,
            goal_generation_kwargs=dict(
                num_goals=1000,
                use_cached_dataset=False,
                policy_file='10-06-her-twin-sac-door-auto-tune/10-06-her-twin-sac-door-auto_tune_2018_10_07_01_38_40_id001--s1850/params.pkl',
                path_length=100,
                show=False,
                tag='_twin_sac'
            ),
            presampled_goals_path='goals/SawyerDoorHookResetFreeEnv-v5_N1000_imsize48goals_twin_sac.npy',
            presample_goals=True,
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=16,
            beta=1.0,
            num_epochs=1000,
            dump_skew_debug_plots=False,
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=5000,
                oracle_dataset=False,
                use_cached=False,
                oracle_dataset_from_policy=False,
                random_and_oracle_policy_data=True,
                random_and_oracle_policy_data_split=.99,
                non_presampled_goal_img_is_garbage=True,
                vae_dataset_specific_kwargs=dict(),
                policy_file='10-06-her-twin-sac-door-auto-tune/10-06-her-twin-sac-door-auto_tune_2018_10_07_01_38_40_id001--s1850/params.pkl',
                n_random_steps=100,
                show=False,
                tag='_twin_sac'
            ),
            vae_kwargs=dict(
                input_channels=3,
                unit_variance=False,
                decoder_activation='sigmoid',
                variance_scaling=1,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
                skew_config=dict(
                    method='inv_gaussian_p_x',
                ),
                full_gaussian_decoder=True,
                skew_dataset=True,
                normalize_log_probs=True,
                normalize_mean=True,
                normalize_std=True,
                normalize_max=False,
            ),
            save_period=50,
        ),
    )

    search_space = {
        'train_vae_variant.beta':[1, 2.5],
        'train_vae_variant.algo_kwargs.skew_dataset':[True, False],
        'train_vae_variant.generate_vae_dataset_kwargs.dataset_path': [
            'datasets/SawyerDoorHookResetFreeEnv-v5_N5000_sawyer_door_env_camera_v3_imsize48_random_oracle_split_0.9_twin_sac.npy',
            'datasets/SawyerDoorHookResetFreeEnv-v5_N5000_sawyer_door_env_camera_v3_imsize48_random_oracle_split_0.99_twin_sac.npy',
            'datasets/SawyerDoorHookResetFreeEnv-v5_N5000_sawyer_door_env_camera_v3_imsize48_random_oracle_split_1_twin_sac.npy',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'sawyer_door_offline_vae_inv_gaussian_priority_sweep'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
          )
