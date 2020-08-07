import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1, init_sawyer_camera_v3 
from rlkit.images.camera import sawyer_init_camera_zoomed_in
from rlkit.envs.mujoco.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEasyEnv
)
from rlkit.images.camera import (
    sawyer_init_camera_zoomed_in_fixed,
    sawyer_init_camera_zoomed_in,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import (
    SawyerReachXYEnv, SawyerReachXYZEnv
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment
from rlkit.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset

if __name__ == "__main__":
    variant = dict(
        # env_class=SawyerReachXYEnv,
        env_class=SawyerPushAndReachXYEnv,
       # env_class=SawyerPickAndPlaceEnv,
        env_kwargs=dict(
            # action_scale=.02,
            hide_goal_markers=True,
            action_scale=.02,
            puck_low=[-.2, .5],
            puck_high=[.2, .7],
            mocap_low=[-0.1, 0.5, 0.],
            mocap_high=[0.1, 0.7, 0.5],
            goal_low=[-0.05, 0.5, 0.02, -0.2, 0.5],
            goal_high=[0.05, 0.7, 0.02, 0.2, 0.7],
            # fix_goal=True,
            # hide_arm=True,
            # reward_type="hand_distance",
        ),
        init_camera=sawyer_init_camera_zoomed_in,
        grill_variant=dict(
            # vae_path='vae.pkl',
            save_video_period=25,
            algo_kwargs=dict(
                num_epochs=2000,
                # num_steps_per_epoch=100,
                # num_steps_per_eval=100,
                # num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                tau=1e-2,
                batch_size=128,
                max_path_length=100,
                discount=0.99,
                # num_updates_per_env_step=2,
                # collection_mode='online-parallel',
                # sim_throttle=True,
                # parallel_env_params=dict(
                    # num_workers=2,
                # )
            ),
            replay_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='GRILL-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.3,
            exploration_type='ou',
            # observation_key='state_observation',
            # desired_goal_key='state_desired_goal',
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=2.5,
            num_epochs=1000,
            generate_vae_dataset_kwargs=dict(
                num_channels=3,
                N=10000,
                oracle_dataset=True,
                show=False,
                use_cached=False,
                vae_dataset_specific_env_kwargs=dict(
                    goal_low=[-0.06, 0.5, 0.02, -0.2, 0.5],
                    goal_high=[0.06, 0.7, 0.02, 0.2, 0.7],
                ),
            ),
             beta_schedule_kwargs=dict(
                 x_values=[0, 100, 200, 500, 1000],
                 y_values=[0, 0, 0, 1, 2],
             ),

            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-3,
            ),
            save_period=5,
        ),
    )

    search_space = {
        'env_kwargs.action_scale': [.02],
        'grill_variant.num_updates_per_env_step': [ 4],
        'grill_variant.exploration_type': ['ou'],

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
        # 'grill_variant.rdim': ["16"],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 2
    mode = 'local'
    exp_prefix = 'pusher-offline-with-online-vae'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'multiworld-goalenv-full-grill-her-td3'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                snapshot_gap=200,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=3,
            )
