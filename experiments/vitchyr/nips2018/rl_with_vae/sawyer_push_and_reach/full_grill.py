import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1
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
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_td3_full_experiment
from railrl.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'full-grill-her-td3-sawyer-push-camera-toggle'

    init_camera = sawyer_init_camera_zoomed_in_fixed
    # init_camera = sawyer_init_camera_zoomed_in
    variant = dict(
        env_class=SawyerReachXYEnv,
        env_kwargs=dict(
            hide_goal_markers=True,
        ),
        grill_variant=dict(
            algo_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                # num_epochs=50,
                # num_steps_per_epoch=100,
                # num_steps_per_eval=100,
                tau=1e-2,
                batch_size=128,
                max_path_length=100,
                discount=0.99,
                min_num_steps_before_training=1000,
                num_updates_per_env_step=4,
            ),
            replay_kwargs=dict(
                max_size=int(1e6),
                # fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='GRILL-HER-TD3',
            normalize=False,
            render=False,
            use_env_goals=True,
            wrap_mujoco_env=True,
            do_state_based_exp=False,
            exploration_noise=0.2,
            init_camera=init_camera,
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
            # num_epochs=1000,
            num_epochs=1,
            generate_vae_dataset_kwargs=dict(
                N=50,
                # init_camera=init_camera,
                init_camera=init_sawyer_camera_v1,
                # show=True,
                # use_cached=False,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-3,
            ),
            beta_schedule_kwargs=dict(
                x_values=[0, 30, 100],
                y_values=[0, 5, 5],
            ),
            save_period=5,
        ),
    )

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                snapshot_gap=50,
                snapshot_mode='gap_and_last',
            )
