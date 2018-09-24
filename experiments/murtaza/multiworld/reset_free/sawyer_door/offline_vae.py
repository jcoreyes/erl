import railrl.misc.hyperparameter as hyp
from railrl.torch.vae.generate_goal_dataset import generate_goal_dataset_using_policy
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v3
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_td3_full_experiment

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        init_camera=sawyer_door_env_camera_v3,
        env_class=SawyerDoorHookEnv,
        env_kwargs=dict(
            goal_low=(-0.1, 0.45, 0.15, 0),
            goal_high=(0.0, 0.65, .225, 1.0472),
            hand_low=(-0.1, 0.45, 0.15),
            hand_high=(0., 0.65, .225),
            max_angle=1.0472,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reset_free=True,
        ),
        grill_variant=dict(
            save_video=True,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=500,
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
                her_kwargs=dict(),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='OFFLINE-VAE-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.8,
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
                policy_file='09-22-sawyer-door-new-door-60-reset-free-space-fix/09-22-sawyer_door_new_door_60_reset_free_space_fix_2018_09_23_04_05_41_id000--s34898/params.pkl',
                path_length=100,
                show=False,
            ),
            presampled_goals_path=None,
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
                use_cached=True,
                oracle_dataset_from_policy=False,
                random_and_oracle_policy_data=True,
                random_and_oracle_policy_data_split=.5,
                non_presampled_goal_img_is_garbage=True,
                vae_dataset_specific_kwargs=dict(),
                policy_file='09-22-sawyer-door-new-door-60-reset-free-space-fix/09-22-sawyer_door_new_door_60_reset_free_space_fix_2018_09_23_04_05_41_id000--s34898/params.pkl',
                n_random_steps=100,
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                lr=1e-3,
                # full_gaussian_decoder=False,
                skew_config=dict(
                    method='squared_error',
                    power=1,
                ),
                skew_dataset=True,
            ),
            save_period=100,
        ),
    )

    search_space = {
        'train_vae_variant.beta':[2.5],
        'train_vae_variant.algo_kwargs.skew_config.power':[0, 1, 3, 5],
        'train_vae_variant.generate_vae_dataset_kwargs.random_and_oracle_policy_data_split':[.5, .75, .9],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'sawyer_hook_door_offline_vae_reconstruction_prioritization'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
          )
