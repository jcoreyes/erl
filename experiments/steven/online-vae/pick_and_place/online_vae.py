import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from rlkit.envs.goal_generation.pickup_goal_dataset import \
        generate_vae_dataset, get_image_presampled_goals_from_vae_env
from multiworld.envs.mujoco.cameras import \
        sawyer_pick_and_place_camera, sawyer_pick_and_place_camera_slanted_angle

if __name__ == "__main__":
    num_images = 1
    variant = dict(
        imsize=48,
        double_algo=False,
        env_id="SawyerPickupEnv-v0",
        init_camera=sawyer_pick_and_place_camera,
        grill_variant=dict(
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            save_video=True,
            save_video_period=25,
            online_vae_beta=1.0,
            presample_goals=True,
            generate_goal_dataset_fctn=get_image_presampled_goals_from_vae_env,
            goal_generation_kwargs=dict(
                num_presampled_goals=1000,
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    save_environment=False,
                    num_epochs=600,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=50,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    parallel_env_params=dict(),
                ),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
                her_kwargs=dict(),
                online_vae_kwargs=dict(),
            ),
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            replay_kwargs=dict(
                max_size=int(40000),
                fraction_goals_are_rollout_goals=0.0,
                fraction_resampled_goals_are_env_goals=0.5,
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
            dump_video_kwargs=dict(
                num_images=num_images,
            )
        ),
        train_vae_variant=dict(
            dump_skew_debug_plots=False,
            generate_vae_data_fctn=generate_vae_dataset,
            representation_size=8,
            beta=5.0,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=50,
                test_p=.9,
                oracle_dataset=True,
                show=False,
                use_cached=False,
                num_channels=3*num_images,
            ),
            vae_kwargs=dict(
                input_channels=3*num_images,
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
        'grill_variant.training_mode': ['train'],
        'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'grill_variant.online_vae_beta': [0.5],
        'grill_variant.replay_kwargs.alpha': [0],
        'grill_variant.algo_kwargs.base_kwargs.num_updates_per_env_step': [2],
        'grill_variant.algo_kwargs.base_kwargs.collection_mode': ['online'],
        'grill_variant.algo_kwargs.online_vae_kwargs.oracle_data': [False],
        # 'grill_variant.algo_kwargs.online_vae_kwargs.parallel_vae_train': [True],
        'grill_variant.algo_kwargs.online_vae_kwargs.vae_training_schedule':
            [vae_schedules.every_six],

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 2
    mode = 'local'
    exp_prefix = 'pick-and-place'
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
