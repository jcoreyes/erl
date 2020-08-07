import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import HER_baseline_twin_sac_full_experiment

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        env_id='SawyerDoorHookResetFreeEnv-v0',
        init_camera=sawyer_door_env_camera_v0,
        grill_variant=dict(
            save_video=False,
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=1010,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=10000,
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
                max_size=int(1e4),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0,
            ),
            algorithm='PIX-REWARD-BASELINE-HER-TWIN-SAC',
            normalize=False,
            render=False,
            exploration_noise=0,
            exploration_type='ou',
            training_mode='test',
            testing_mode='test',
            observation_key='image_observation',
            desired_goal_key='image_desired_goal',
            cnn_params=dict(
                kernel_sizes=[5, 3, 3],
                n_channels=[16, 32, 64],
                strides=[3, 2, 2],
                hidden_sizes=[32, 32],
                paddings=[0, 0, 0],
            ),
            presampled_goals_path='goals/SawyerDoorHookResetFreeEnv-v0_N1000_imsize48goals_twin_sac.npy',
            presample_goals=True,
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=16,
            beta=.5,
            num_epochs=1000,
            dump_skew_debug_plots=False,
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=5000,
                oracle_dataset=False,
                use_cached=False,
                oracle_dataset_from_policy=True,
                non_presampled_goal_img_is_garbage=True,
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
            save_period=10,
        ),
    )

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 5
    mode = 'gcp'
    exp_prefix = 'sawyer_door_pix_reward_baseline'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                HER_baseline_twin_sac_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
                gcp_kwargs=dict(
                    zone='us-west2-b',
                    preemptible=False,
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-p4',
                        num_gpu=1,
                    )
                )
          )
