import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import \
        init_sawyer_camera_v1, init_sawyer_camera_v3, push_camera
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    # SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from railrl.envs.mujoco.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEasyEnv
)
from railrl.images.camera import (
    sawyer_init_camera_zoomed_in_fixed,
    sawyer_init_camera_zoomed_in,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import (
    SawyerReachXYEnv, SawyerReachXYZEnv
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_td3_online_vae_full_experiment
from railrl.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset
import railrl.torch.vae.vae_schedules as vae_schedules

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        # env_class=SawyerReachXYEnv,
        env_class=SawyerPushAndReachXYEnv,
        # env_class=SawyerPickAndPlaceEnvYZ,
        env_kwargs=dict(
            hide_goal_markers=True,
            # hide_arm=True,
            action_scale=.02,
            puck_low=[-.15, .5],
            puck_high=[.15, .7],
            mocap_low=[-0.1, 0.5, 0.],
            mocap_high=[0.1, 0.7, 0.5],
            goal_low=[-0.05, 0.55, 0.02, -0.2, 0.5],
            goal_high=[0.05, 0.65, 0.02, 0.2, 0.7],

        ),
        init_camera=push_camera,
        grill_variant=dict(
            # do_state_exp=True,
            # vae_path="07-14-vae-old-pusher-env-test/07-14-vae-old-pusher-env-test_2018_07_14_18_27_56_0000--s-76316/vae.pkl",
            # save_video=False,
            online_vae_beta=2.0,
            algo_kwargs=dict(
                num_epochs=2000,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                tau=1e-2,
                batch_size=128,
                max_path_length=100,
                discount=0.99,
                reward_scale=1e-4,
                num_updates_per_env_step=1,
                vae_training_schedule=vae_schedules.never_train,
                # collection_mode='online-parallel',
                # parallel_env_params=dict(
                    # num_workers=2,
                # )
            ),
            replay_kwargs=dict(
                max_size=20000,
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
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
            representation_size=8,
            beta=.01,
            num_epochs=500,
            generate_vae_dataset_kwargs=dict(
                N=3500,
                oracle_dataset=True,
                use_cached=False,
                num_channels=3,
                show=False,
                random_oracle=False,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-2,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            beta_schedule_kwargs=dict(
               x_values=[0, 300, 500, 1000],
               y_values=[0, 0, 1, 1],
            ),
            save_period=5,
        ),
    )

    search_space = {
        # 'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [.5, 1],
        # 'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.0, .2],
        'init_camera': [push_camera, init_sawyer_camera_v3],
        'grill_variant.algo_kwargs.num_updates_per_env_step': [4],
        'grill_variant.exploration_type': ['ou'],
        'grill_variant.exploration_noise': [.2, .4],
        'env_kwargs.action_scale': [.02],
        # 'train_vae_variant.algo_kwargs.lr': [1e-2, 1e-3],
        'train_vae_variant.representation_size': [4],
        # 'train_vae_variant.beta': [.5, 1, 2.5],
        'grill_variant.algo_kwargs.vae_training_schedule': [
            # vae_schedules.every_three,
            # vae_schedules.every_six,
            # vae_schedules.every_ten
            vae_schedules.never_train,
        ]

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 2
    mode = 'local'
    exp_prefix = 'vae-pusher-v4-30k-buffer-original'

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
                num_exps_per_instance=2,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                # snapshot_gap=50,
                # snapshot_mode='gap_and_last',
            )
