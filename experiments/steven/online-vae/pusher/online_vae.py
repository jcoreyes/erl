import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1, init_sawyer_camera_v4, sawyer_top_down
from rlkit.envs.mujoco.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEasyEnv
)
from rlkit.images.camera import (
    sawyer_init_camera_zoomed_in_fixed,
    sawyer_init_camera_zoomed_in,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_online_vae_full_experiment
from rlkit.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset
import rlkit.torch.vae.vae_schedules as vae_schedules

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        double_algo=False,
        # env_class=SawyerReachXYEnv,
        env_class=SawyerPushAndReachXYEnv,
        # env_class=SawyerPickAndPlaceEnv,
        env_kwargs=dict(
            hide_goal_markers=True,
            action_scale=.02,
            puck_low=[-0.25, .4],
            puck_high=[0.25, .8],
            mocap_low=[-0.2, 0.45, 0.],
            mocap_high=[0.2, 0.75, 0.5],
            goal_low=[-0.2, 0.45, 0.02, -0.25, 0.4],
            goal_high=[0.2, 0.75, 0.02, 0.25, 0.8],

            # puck_low=[-0.2, .5],
            # puck_high=[0.2, .7],
            # mocap_low=[-0.1, 0.5, 0.],
            # mocap_high=[0.1, 0.7, 0.5],
            # goal_low=[-0.05, 0.55, 0.02, -0.2, 0.5],
            # goal_high=[0.05, 0.65, 0.02, 0.2, 0.7],

            # puck_low=[-0.15, .5],
            # puck_high=[0.15, .7],
            # mocap_low=[-0.1, 0.5, 0.],
            # mocap_high=[0.1, 0.7, 0.5],
            # goal_low=[-0.05, 0.55, 0.02, -0.15, 0.5],
            # goal_high=[0.05, 0.65, 0.02, 0.15, 0.7],
        ),
        # init_camera=sawyer_init_camera_zoomed_in,
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
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=4,
                    # collection_mode='online-parallel',
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
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
                # exploration_rewards_scale=0.0,
                # exploration_rewards_type='reconstruction_error',
                # exploration_schedule_kwargs=dict(
                    # x_values=[0, 100, 200, 500, 1000],
                    # y_values=[.1, .1, .1, 0, 0],
                # ),

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
                # vae_dataset_specific_env_kwargs=dict(
                    # goal_low=[-0.1, 0.5, 0.02, -0.2, 0.5],
                    # goal_high=[0.1, 0.7, 0.02, 0.2, 0.7],
                # ),

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
    )

    search_space = {
        'init_camera': [sawyer_top_down],
        'grill_variant.online_vae_beta': [2.5],
        'grill_variant.use_replay_buffer_goals': [False],
        'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [.5],
        'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.0],
        # 'grill_variant.replay_kwargs.exploration_rewards_scale': [.1, .01, .001],
        'grill_variant.replay_kwargs.exploration_rewards_type': ['None'],
        'grill_variant.replay_kwargs.power': [0],
        'grill_variant.exploration_noise': [.8],
        # 'grill_variant.exploration_kwargs': [
            # dict(
                # min_sigma=.2,
                # decay_period=200000,
            # ),
            # dict(
                # min_sigma=.2,
                # decay_period=500000,
            # ),
            # dict(
                # min_sigma=.1,
                # decay_period=500000,
            # )
        # ],
        'grill_variant.algo_kwargs.vae_training_schedule':
                [
                 vae_schedules.every_six,
                ],
        'grill_variant.algo_kwargs.base_kwargs.num_updates_per_env_step': [2],
        'grill_variant.algo_kwargs.online_vae_kwargs.oracle_data': [False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'pusher-large-range-48x48-ou-huge-buffer'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'multiworld-goalenv-full-grill-her-td3'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                snapshot_gap=200,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=2,
            )
