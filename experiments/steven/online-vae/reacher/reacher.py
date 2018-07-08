import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    SawyerPickAndPlaceEnvYZ, SawyerPickAndPlaceEnv
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
from railrl.torch.grill.launcher import grill_her_td3_full_experiment
from railrl.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset

if __name__ == "__main__":
    variant = dict(
        env_class=SawyerReachXYEnv,
        # env_class=SawyerPushAndReachXYEnv,
       # env_class=SawyerPickAndPlaceEnv,
        env_kwargs=dict(
            hide_goal_markers=True,
            # fix_goal=True,
            # hide_arm=True,
            # reward_type="hand_distance",
        ),
        init_camera=init_sawyer_camera_v1,
        grill_variant=dict(
            # do_state_exp=True,
            vae_path="07-05-multi-exp-single-instance-multiworld-2/07-05-multi-exp-single-instance-multiworld-2_2018_07_05_23_51_08_0000--s-20902/vae.pkl",
            # vae_path="07-06-push-and-reach-parallel/07-06-push_and_reach_parallel_2018_07_06_17_09_04_0000--s-35032/vae.pkl",
            algo_kwargs=dict(
                num_epochs=2000,
                # num_steps_per_epoch=100,
                # num_steps_per_eval=100,
                # num_epochs=500,
                num_steps_per_epoch=500,
                num_steps_per_eval=500,
                tau=1e-2,
                batch_size=128,
                max_path_length=50,
                discount=0.99,
                min_num_steps_before_training=0,
                num_updates_per_env_step=1,
                # collection_mode='online-parallel',
                collection_mode='online-parallel',
                sim_throttle=True,
                parallel_eval=True,
                parallel_env_params=dict(
                    num_workers=4,
                    dedicated_train=2,
                )
            ),
            replay_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.0,
            ),
            algorithm='GRILL-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.2,
            exploration_type='ou',
            # observation_key='state_observation',
            # desired_goal_key='state_desired_goal',
        ),
        train_vae_variant=dict(
            representation_size=8,
            beta=3.0,
            num_epochs=200,
            generate_vae_dataset_kwargs=dict(
                num_channels=3,
                N=3000,
                oracle_dataset=True,
                show=False,
                use_cached=False,
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
        # 'grill_variant.training_mode': ['test'],
        # 'grill_variant.observation_key': ['latent_observation'],
        # 'grill_variant.desired_goal_key': ['state_desired_goal'],
        # 'grill_variant.observation_key': ['state_observation'],
        # 'grill_variant.desired_goal_key': ['latent_desired_goal'],
        # 'grill_variant.vae_paths': [
        #     {"16": "/home/vitchyr/git/railrl/data/doodads3/06-12-dev/06-12"
        #            "-dev_2018_06_12_18_57_14_0000--s-28051/vae.pkl",
        #      }
        # ],
        # 'grill_variant.rdim': ["16"],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'here_no_doodad'
    exp_prefix = 'reacher-pusher-quality-tests'

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
