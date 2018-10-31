import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_twin_sac_full_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        imsize=84,
        init_camera=sawyer_pusher_camera_upright_v2,
        env_class=SawyerPushAndReachXYEnv,
        env_kwargs=dict(
            norm_order=2,
        ),
        grill_variant=dict(
            save_video=True,
            save_video_period=50,
            do_state_exp=True,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=2005,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=1,
                    collection_mode='online',
                    parallel_env_params=dict(
                        num_workers=1,
                    ),
                    reward_scale=1,
                ),
                her_kwargs=dict(),
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
            algorithm='STATE-HER-TWIN-SAC',
            normalize=False,
            render=False,
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
        'env_kwargs.reward_type':['puck_distance', 'state_distance', 'hand_and_puck_distance'],
        'env_kwargs.reset_free':[True, False],
        'grill_variant.replay_buffer_kwargs.fraction_goals_are_rollout_goals':[0, .5],
        'grill_variant.algo_kwargs.base_kwargs.max_path_length':[100, 500],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )


    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_pusher_state_her_twin_sac_finalized_lengthy_sweep'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                snapshot_mode='gap_and_last',
                snapshot_gap=50,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=5,
            )
