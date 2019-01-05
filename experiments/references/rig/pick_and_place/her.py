import railrl.misc.hyperparameter as hyp
from railrl.torch.vae.generate_goal_dataset import generate_goal_dataset_using_policy
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v3
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import HER_baseline_her_td3_full_experiment
from multiworld.envs.mujoco.cameras import \
        sawyer_pick_and_place_camera, sawyer_pick_and_place_camera_slanted_angle
from railrl.envs.goal_generation.pickup_goal_dataset import \
        generate_vae_dataset, get_image_presampled_goals_from_image_env


if __name__ == "__main__":
    variant = dict(
        imsize=84,
        env_id="SawyerPickupEnv-v0",
        init_camera=sawyer_pick_and_place_camera,
        grill_variant=dict(
            presample_goals=True,
            generate_goal_dataset_fctn=get_image_presampled_goals_from_image_env,
            goal_generation_kwargs=dict(
                num_presampled_goals=1000,
            ),

            save_video=False,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=505,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=1,
                    collection_mode='online',
                    reward_scale=1,
                ),
                her_kwargs=dict(),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e4),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.0,
            ),
            algorithm='PIX-REWARD-BASELINE-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.3,
            exploration_type='ou',
            training_mode='test',
            testing_mode='test',
            observation_key='image_observation',
            desired_goal_key='image_desired_goal',
            cnn_params=dict(
                kernel_sizes=[5, 5, 5],
                n_channels=[16, 32, 32],
                strides=[3, 3, 3],
                pool_sizes=[1, 1, 1],
                hidden_sizes=[32, 32],
                paddings=[0, 0, 0],
                use_batch_norm=False,
            ),
        ),
        train_vae_variant=dict(
            generate_vae_data_fctn=generate_vae_dataset,
            vae_path=None,
            representation_size=16,
            beta=.5,
            num_epochs=1000,
            dump_skew_debug_plots=False,
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=50,
                oracle_dataset=True,
                use_cached=False,
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
            ),
            save_period=10,
        ),
    )

    search_space = {
        'grill_variant.exploration_noise':[.3, .5],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 7
    mode = 'ec2'
    exp_prefix = 'pick-and-place-her-sparse-.2-.0'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                HER_baseline_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=1,
          )
