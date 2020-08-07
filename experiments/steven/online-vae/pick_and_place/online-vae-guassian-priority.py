import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from multiworld.envs.mujoco.cameras import \
        sawyer_pick_and_place_camera, sawyer_pick_and_place_camera_slanted_angle
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
        get_image_presampled_goals
from rlkit.envs.goal_generation.pickup_goal_dataset import \
        generate_vae_dataset
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ


if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        env_kwargs=dict(
            reset_free=True,
            random_init=False,
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=.02,
            hide_goal_markers=True,
        ),

        env_class=SawyerPickAndPlaceEnv,
        init_camera=sawyer_pick_and_place_camera,
        grill_variant=dict(
            save_video=True,
            online_vae_beta=2.5,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=1005,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=50,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    reward_scale=1,
                    collection_mode='online-parallel',
                    parallel_env_params=dict(
                        num_workers=2,
                    )
                ),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
                online_vae_kwargs=dict(
                   vae_training_schedule=vae_schedules.every_six,
                    oracle_data=False,
                    parallel_vae_train=True,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(50000),
                fraction_goals_are_rollout_goals=0,
                fraction_resampled_goals_are_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='image_gaussian_inv_prob',
                power=1,
            ),
            normalize=False,
            render=False,
            exploration_noise=0.8,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            goal_generation_kwargs=dict(
                num_presampled_goals=1000,
            ),
            presample_image_goals_only=True,
            generate_goal_dataset_fctn=get_image_presampled_goals,
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            algorithm='ONLINE-VAE-INV-GAUSS-HER-TD3',
        ),
        train_vae_variant=dict(
            representation_size=8,
            beta=1.0,
            num_epochs=0,
            dump_skew_debug_plots=False,
            generate_vae_data_fctn=generate_vae_dataset,
            generate_vae_dataset_kwargs=dict(
                N=100,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=True,
                num_channels=3,
            ),
            vae_kwargs=dict(
                input_channels=3,
                unit_variance=False,
                decoder_activation='sigmoid',
                variance_scaling=1,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                lr=1e-3,
                full_gaussian_decoder=True,
            ),
            save_period=5,
        ),
    )

    search_space = {
        'grill_variant.online_vae_beta': [.25, .5, 1],
        'grill_variant.replay_buffer_kwargs.vae_priority_type':['image_gaussian_inv_prob', 'None']
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'pick-and-place-online-vae-inv-gaussian-priority-final'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
          )
