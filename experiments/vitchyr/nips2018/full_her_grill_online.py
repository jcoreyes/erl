import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1, \
    init_sawyer_camera_v2, init_sawyer_camera_v3, init_sawyer_camera_v4
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    SawyerPickAndPlaceEnv
from multiworld.envs.pygame.point2d import Point2DEnv
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
from rlkit.torch.grill.launcher import grill_her_td3_online_vae_full_experiment
from rlkit.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset
import rlkit.torch.vae.vae_schedules as vae_schedules

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        env_class=SawyerReachXYEnv,
        # env_class=SawyerPushAndReachXYEnv,
        # env_class=SawyerPickAndPlaceEnv,
        # env_class=Point2DEnv,
        env_kwargs=dict(
            hide_goal_markers=True,
            # puck_low=(-0.2, 0.5),
            # puck_high=(0.2, 0.7),
            # hand_low=(-0.2, 0.5, 0.),
            # hand_high=(0.2, 0.7, 0.5),
            # mocap_low=(-0.1, 0.5, 0.),
            # mocap_high=(0.1, 0.7, 0.5),
            # goal_low=(-0.05, 0.55, 0.02, -0.2, 0.5),
            # goal_high=(0.05, 0.65, 0.02, 0.2, 0.7),
        ),
        # init_camera=init_sawyer_camera_v4,
        init_camera=init_sawyer_camera_v1,
        grill_variant=dict(
            online_vae_beta=5,
            algo_kwargs=dict(
                num_epochs=250,
                # num_steps_per_epoch=100,
                # num_steps_per_eval=100,
                # num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                tau=1e-2,
                batch_size=128,
                max_path_length=100,
                discount=0.99,
                num_updates_per_env_step=4,
            ),
            replay_kwargs=dict(
                max_size=int(1e4),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            vae_wrapped_env_kwargs=dict(
                num_goals_presampled=100,
            ),
            algorithm='GRILL-HER-TD3',
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
            representation_size=16,
            beta=5.0,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=1000,
                oracle_dataset=True,
                num_channels=3,
                # show=True,
                # use_cached=False,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-3,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            # beta_schedule_kwargs=dict(
                # x_values=[0, 100, 200, 500],
                # y_values=[0, 0, 1, 1],
            # ),
            save_period=5,
        ),
    )

    search_space = {
        'hand-goal-space': ['easy'],
        'mocap-x-range': ['0.1'],
        'grill_variant.algo_kwargs.vae_training_schedule': [
            vae_schedules.always_train,
            vae_schedules.every_ten,
            vae_schedules.never_train,
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    mode = 'local'
    exp_prefix = 'dev'

    # mode = 'ec2'
    # exp_prefix = 'full-her-grill-online-test'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['hand-goal-space'] == 'easy':
            variant['env_kwargs']['goal_low'] = (-0.05, 0.55, 0.02, -0.2, 0.5)
            variant['env_kwargs']['goal_high'] = (0.05, 0.65, 0.02, 0.2, 0.7)
        else:
            variant['env_kwargs']['goal_low'] = (-0.2, 0.5, 0.02, -0.2, 0.5)
            variant['env_kwargs']['goal_high'] = (0.2, 0.7, 0.02, 0.2, 0.7)
        if variant['mocap-x-range'] == '0.1':
            variant['env_kwargs']['mocap_low'] = (-0.1, 0.5, 0.)
            variant['env_kwargs']['mocap_high'] = (0.1, 0.7, 0.5)
        else:
            variant['env_kwargs']['mocap_low'] = (-0.2, 0.5, 0.)
            variant['env_kwargs']['mocap_high'] = (0.2, 0.7, 0.5)
        run_experiment(
            grill_her_td3_online_vae_full_experiment,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            use_gpu=True,
            # snapshot_gap=50,
            snapshot_mode='last',
            exp_id=exp_id,
            num_exps_per_instance=3,
        )
