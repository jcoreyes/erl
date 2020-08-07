import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v0
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        imsize=84,
        init_camera=sawyer_pusher_camera_upright_v0,
        env_id='SawyerPushAndReachEnvEasy-v0',
        grill_variant=dict(
            save_video=True,
            save_video_period=500,
            do_state_exp=True,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=5005,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=1,
                    collection_mode='online-parallel',
                    parallel_env_params=dict(
                        num_workers=1,
                    ),
                    reward_scale=100,
                ),
                her_kwargs=dict(),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0.5,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='STATE-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.5,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            observation_key='state_observation',
            desired_goal_key='state_desired_goal',
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
                oracle_dataset=True,
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
            save_period=10,
        ),
    )

    search_space = {
        'env_id':['SawyerPushAndReachEnvEasy-v0', 'SawyerPushAndReachSmallArenaEnv-v0', 'SawyerPushAndReachSmallArenaResetFreeEnv-v0'],
        'grill_variant.exploration_noise':[0.5, .8]
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )


    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_pusher_state_her_td3'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                snapshot_mode='gap_and_last',
                snapshot_gap=500,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-west1-a',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-p100',
                        num_gpu=1,
                    )
                )
            )
