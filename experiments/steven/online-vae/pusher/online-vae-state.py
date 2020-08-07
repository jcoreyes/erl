import rlkit.misc.hyperparameter as hyp
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    # SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from rlkit.envs.mujoco.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEasyEnv
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import (
    SawyerReachXYEnv, SawyerReachXYZEnv
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_true_state_online_vae_full_experiment
from rlkit.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset
import rlkit.torch.vae.vae_schedules as vae_schedules

if __name__ == "__main__":
    variant = dict(
        normalize=False,
        double_algo=False,
        # env_class=SawyerReachXYEnv,
        env_class=SawyerPushAndReachXYEnv,
        # env_class=SawyerPickAndPlaceEnvYZ,
        env_kwargs=dict(
            hide_goal_markers=True,
            # hide_arm=True,
            action_scale=.02,
        ),
        grill_variant=dict(
            save_video=False,
            online_vae_beta=2.0,
            algo_kwargs=dict(
                num_epochs=1000,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                tau=1e-2,
                batch_size=128,
                max_path_length=100,
                discount=0.99,
                num_updates_per_env_step=1,
                reward_scale=1000,
                vae_training_schedule=vae_schedules.never_train,
                # collection_mode='online-parallel',
                # parallel_env_params=dict(
                    # num_workers=4,
                # )
            ),
            replay_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            render=False,
            exploration_noise=0.3,
            exploration_type='ou',
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=.0,
            num_epochs=300,
            generate_vae_dataset_kwargs=dict(
                N=5000,
                oracle_dataset=True,
                use_cached=True,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-2,
            ),
            vae_kwargs=dict(
                hidden_sizes=[32, 32]
            ),
            # beta_schedule_kwargs=dict(
               # x_values=[0, 100, 500, 1000],
               # y_values=[0, 1, 3, 3],
            # ),
            save_period=5,
        ),
    )

    search_space = {
        # 'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [.5, 1],
        # 'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.0, .2],
        # 'grill_variant.algo_kwargs.num_updates_per_env_step': [1, 4],
        'train_vae_variant.representation_size':  [4],
        'grill_variant.algo_kwargs.num_updates_per_env_step': [1, 4],
        'grill_variant.algo_kwargs.reward_scale': [1, 10, 100, 1000],
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

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'vae-pusher-state-vae-test-2'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'multiworld-goalenv-full-grill-her-td3'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_true_state_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                # snapshot_gap=50,
                # snapshot_mode='gap_and_last',
            )
