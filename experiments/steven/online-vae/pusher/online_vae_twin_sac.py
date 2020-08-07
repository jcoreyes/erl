import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1
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
        double_algo=False,
        # env_class=SawyerReachXYEnv,
        env_class=SawyerPushAndReachXYEnv,
        # env_class=SawyerPickAndPlaceEnv,
        env_kwargs=dict(
            hide_goal_markers=True,
            action_scale=.02,
            puck_low=[-.15, .5],
            puck_high=[.15, .7],
            mocap_low=[-0.1, 0.5, 0.],
            mocap_high=[0.1, 0.7, 0.5],
            goal_low=[-0.05, 0.55, 0.02, -0.15, 0.5],
            goal_high=[0.05, 0.65, 0.02, 0.15, 0.7],

        ),
        init_camera=sawyer_init_camera_zoomed_in,
        grill_variant=dict(
            save_video=True,
            save_video_period=25,
            online_vae_beta=2.5,
            algo_kwargs=dict(
                save_networks=True,
                num_epochs=1000,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                min_num_steps_before_training=4000,
                batch_size=128,
                max_path_length=100,
                discount=0.99,
                num_updates_per_env_step=2,
            ),
            online_vae_algo_kwargs=dict(
                vae_training_schedule=vae_schedules.every_three,
            ),
            replay_kwargs=dict(
                max_size=int(30000),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
                exploration_rewards_scale=0.0,
                exploration_rewards_type='reconstruction_error',

            ),
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
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=5.0,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=5000,
                oracle_dataset=True,
                use_cached=False,
                num_channels=3,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-3,
            ),
            #beta_schedule_kwargs=dict(
            #    x_values=[0, 100, 200, 500],
            #    y_values=[0, 0, 5, 5],
            #),
            save_period=5,
        ),
    )

    search_space = {
        # 'grill_variant.training_mode': ['test'],
        # 'grill_variant.observation_key': ['latent_observation'],
        # 'grill_variant.desired_goal_key': ['state_desired_goal'],
        # 'grill_variant.observation_key': ['state_observation'],
        # 'grill_variant.desired_goal_key': ['latent_desired_goal'],
        # 'grill_variant.vae_paths': [
        #     {"16": "/home/vitchyr/git/rlkit/data/doodads3/06-12-dev/06-12"
        #            "-dev_2018_06_12_18_57_14_0000--s-28051/vae.pkl",
        #      }
        # ],
        # 'grill_variant.vae_path': [
        #     "/home/vitchyr/git/rlkit/data/doodads3/06-14-dev/06-14-dev_2018_06_14_15_21_20_0000--s-69980/vae.pkl",
        # ]
        'grill_variant.training_mode': ['train'],
        'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [1.0],
        'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.0],
        'grill_variant.algo_kwargs.soft_target_tau': [1e-2],
        'grill_variant.online_vae_algo_kwargs.oracle_data': [True],
        'grill_variant.replay_kwargs.power': [0],
        'grill_variant.replay_kwargs.max_size': [30000, 40000, 50000, 60000],


    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'pusher-memory-limit-test-2'

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
