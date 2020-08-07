import rlkit.misc.hyperparameter as hyp
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
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerPushAndPullDoorEnv
from rlkit.images.camera import (
    sawyer_door_env_camera
)


if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        env_class=SawyerPushAndPullDoorEnv,
        env_kwargs=dict(
        ),
        # init_camera=sawyer_init_camera_zoomed_in,
        grill_variant=dict(
            use_replay_buffer_goals=True,
            save_video=True,
            save_video_period=25,
            online_vae_beta=2.5,
            algo_kwargs=dict(
                save_environment=False,
                num_epochs=2000,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                min_num_steps_before_training=4000,
                tau=1e-2,
                batch_size=128,
                max_path_length=200,
                discount=0.99,
                num_updates_per_env_step=2,
                vae_training_schedule=vae_schedules.every_six,
                # collection_mode='online-parallel',
            ),
            replay_kwargs=dict(
                max_size=int(30000),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
                exploration_rewards_scale=0.0,
                exploration_rewards_type='reconstruction_error',

            ),
            algorithm='GRILL-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.4,
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
            beta=5.0,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=500,
                test_p=.9,
                oracle_dataset=True,
                use_cached=False,
                num_channels=3,

            ),
            vae_kwargs=dict(
                action_dim=4,
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=True,
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
        'grill_variant.exploration_noise': [0.4],
        'grill_variant.use_replay_buffer_goals': [False],
        'grill_variant.training_mode': ['train'],
        # 'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [.5, 1],
        'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.2],

        'grill_variant.replay_kwargs.exploration_rewards_scale': [0],
        'grill_variant.replay_kwargs.power': [0, 1, 2],
        'grill_variant.algo_kwargs.num_updates_per_env_step': [4],
        'grill_variant.replay_kwargs.exploration_rewards_type':
                ['reconstruction_error'],
        'grill_variant.algo_kwargs.vae_training_schedule':
                [vae_schedules.every_three],
        'init_camera': [sawyer_door_env_camera],
        # 'grill_variant.exploration_noise': [.1, .3, .4],
        # 'grill_variant.exploration_type': ['ou', 'gaussian', 'epsilon'],
        'grill_variant.algo_kwargs.oracle_data': [False],
        'train_vae_variant.algo_kwargs.linearity_weight': [0]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'online-sawyer-push-pull-not-parallel'

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
