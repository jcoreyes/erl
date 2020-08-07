import railrl.misc.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.sets.rl_launcher import disco_experiment

if __name__ == "__main__":
    variant = dict(
        env_class=PickAndPlaceEnv,
        env_kwargs=dict(
            # Environment dynamics
            action_scale=1.0,
            boundary_dist=4,
            ball_radius=0.75,
            object_radius=0.50,
            cursor_visual_radius=1.,
            object_visual_radius=1.,
            min_grab_distance=0.5,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense",
            success_threshold=0.60,
            # Reset settings
            fixed_goal=None,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
            object_reward_only=False,

            init_position_strategy='random',
            num_objects=2,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        max_path_length=100,
        # max_path_length=2,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=0,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
            # batch_size=5,
            # num_epochs=2,
            # num_eval_steps_per_epoch=10,
            # num_expl_steps_per_train_loop=10,
            # num_trains_per_train_loop=10,
            # min_num_steps_before_training=10,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.3,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            pad_color=50,
            subpad_length=1,
            pad_length=1,
            num_columns_per_rollout=2,
        ),
        renderer_kwargs=dict(
            # create_image_format='HWC',
            # output_image_format='CWH',
            output_image_format='CHW',
            # flatten_image=True,
            # normalize_image=False,
        ),
        evaluation_goal_sampling_mode="reset_of_env",
        exploration_goal_sampling_mode="vae_prior",
        create_vae_kwargs=dict(
            latent_dim=128,
            encoder_cnn_kwargs=dict(
                kernel_sizes=[5, 3, 3],
                n_channels=[16, 32, 64],
                strides=[3, 2, 2],
                paddings=[0, 0, 0],
                pool_type='none',
                hidden_activation='relu',
                normalization_type='layer',
            ),
            encoder_mlp_kwargs=dict(
                hidden_sizes=[],
            ),
            decoder_dcnn_kwargs=dict(
                kernel_sizes=[3, 3, 6],
                n_channels=[32, 16, 3],
                strides=[2, 2, 3],
                paddings=[0, 0, 0],
            ),
            decoder_mlp_kwargs=dict(
                hidden_sizes=[256, 256],
            ),
            decoder_distribution='bernoulli',
            use_mlp_decoder=True,
        ),
        vae_trainer_kwargs=dict(
            vae_lr=1e-3,
            vae_visualization_config=dict(
                num_recons=5,
                num_samples=20,
                # debug_period=50,
                debug_period=5,
                unnormalize_images=True,
            ),
            beta=1,
            # set_loss_weight=1,
            # beta=0.001,
            set_loss_weight=0,
        ),
        data_loader_kwargs=dict(
            batch_size=32,
        ),
        vae_algo_kwargs=dict(
            num_iters=101,
            num_epochs_per_iter=100,
            progress_csv_file_name='vae_progress.csv',
        ),
        generate_set_kwargs=dict(
            num_sets=16,
            num_samples_per_set=32,
        ),
        num_ungrouped_images=1024,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 2
    # mode = 'sss'
    exp_prefix = 'exp4-just-vae-new-env'

    search_space = {
        'create_vae_kwargs.latent_dim': [
            64, 32, 16
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['exp_id'] = exp_id
        for _ in range(n_seeds):
            run_experiment(
                disco_experiment,
                exp_name=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                gpu_id=1,
            )
