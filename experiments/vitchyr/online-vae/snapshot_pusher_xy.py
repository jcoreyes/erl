"""
Snapshot of experiments giving results of:

/home/vitchyr/git/railrl/data/doodads3/08-21-recreate-online-vae-pushing-results-online-parallel-collection-one-seed-per-instance/
"""
import rlkit.misc.hyperparameter as hyp
import rlkit.torch.vae.vae_schedules as vae_schedules
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_top_down
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_online_vae_full_experiment

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        double_algo=False,
        env_class=SawyerPushAndReachXYEnv,
        env_kwargs=dict(
            hide_goal_markers=True,
            action_scale=.02,
            puck_low=[-0.25, .4],
            puck_high=[0.25, .8],
            mocap_low=[-0.2, 0.45, 0.],
            mocap_high=[0.2, 0.75, 0.5],
            goal_low=[-0.2, 0.45, 0.02, -0.25, 0.4],
            goal_high=[0.2, 0.75, 0.02, 0.25, 0.8],
        ),
        init_camera=sawyer_pusher_camera_top_down,
        grill_variant=dict(
            save_video=True,
            save_video_period=25,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=1000,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=500,
                    discount=0.99,
                    num_updates_per_env_step=4,
                    collection_mode='online-parallel',
                ),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
                online_vae_kwargs=dict(
                    vae_training_schedule=vae_schedules.every_six,
                ),
            ),
            replay_kwargs=dict(
                max_size=int(80000),
                fraction_goals_are_rollout_goals=0.,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='GRILL-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.5,
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
            beta=1.0,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=101,
                test_p=.9,
                oracle_dataset=True,
                use_cached=False,
                num_channels=3,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                lr=1e-3,
            ),
            save_period=5,
        ),
        version='new-pnp-gripper-open-58aee2f',
        num_exps_per_instance=2,
    )

    search_space = {
        'grill_variant.online_vae_beta': [2.5],
        'grill_variant.use_replay_buffer_goals': [False],
        'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [.5],
        'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.0],
        'grill_variant.replay_kwargs.exploration_rewards_type': [
            'reconstruction_error',
            # 'None',
        ],
        'grill_variant.replay_kwargs.power': [3],
        'grill_variant.exploration_noise': [.8],
        'grill_variant.algo_kwargs.vae_training_schedule':
            [
                vae_schedules.every_six,
            ],
        'grill_variant.algo_kwargs.base_kwargs.num_updates_per_env_step': [2],
        'grill_variant.algo_kwargs.base_kwargs.max_path_length': [100],
        'grill_variant.algo_kwargs.online_vae_kwargs.oracle_data': [False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'pusher-test-pnp-merge-2'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_online_vae_full_experiment,
                exp_id=exp_id,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                snapshot_gap=200,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=2,
            )
