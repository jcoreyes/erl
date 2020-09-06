import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.contextual.rig.rig_launcher import rig_experiment
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sets.rl_launcher import disco_experiment

if __name__ == "__main__":
    variant = dict(
        env_class=PickAndPlaceEnv,
        env_kwargs=dict(
            # Environment dynamics
            action_scale=1.0,
            boundary_dist=4,
            ball_radius=1.5,
            object_radius=1.,
            ball_visual_radius=1.5,
            object_visual_radius=1.,
            min_grab_distance=1.,
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
            num_objects=1,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        sac_trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            reward_scale='auto_normalize_by_max_magnitude',
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=501,
            num_eval_steps_per_epoch=3000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        # max_path_length=2,
        # algo_kwargs=dict(
        #     batch_size=5,
        #     num_epochs=1,
        #     num_eval_steps_per_epoch=2*20,
        #     num_expl_steps_per_train_loop=2*20,
        #     num_trains_per_train_loop=10,
        #     min_num_steps_before_training=10,
        # ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.3,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            pad_color=50,
            subpad_length=1,
            pad_length=1,
            num_columns_per_rollout=2,
            num_imgs=8,
            # rows=2,
            # columns=9,
        ),
        renderer_kwargs=dict(
            # create_image_format='HWC',
            # output_image_format='CWH',
            output_image_format='CHW',
            # flatten_image=True,
            # normalize_image=False,
        ),
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
            use_fancy_architecture=True,
            decoder_distribution='gaussian_learned_global_scalar_variance',
        ),
        vae_trainer_kwargs=dict(
            vae_lr=1e-3,
            vae_visualization_config=dict(
                num_recons=5,
                num_samples=20,
                # debug_period=50,
                debug_period=20,
                unnormalize_images=True,
                image_format='CHW',
            ),
            beta=1,
            set_loss_weight=0,
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        train_vae_kwargs=dict(
            vae_path=None,
            representation_size=4,
            beta=10.0,
            # beta_schedule_kwargs=dict(
            #     x_values=(0, 500),
            #     y_values=(1  / 128.0, 50  / 128.0),
            # ),
            num_epochs=101,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=False,
                save_trajectories=False,
                enviorment_dataset=False,
                use_cached=False,
                vae_dataset_specific_kwargs=dict(
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
                architecture=dict(
                    conv_args=dict(
                        kernel_sizes=[5, 3, 3],
                        n_channels=[16, 32, 64],
                        strides=[3, 2, 2],
                    ),
                    conv_kwargs=dict(
                        hidden_activation='relu',
                        normalization_type='layer',
                    ),
                    decov_args=dict(

                    ),

                )
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-3,
                # weight_decay=0.01,
            ),
            save_period=5,
        ),
        evaluation_goal_sampling_mode="reset_of_env",
        exploration_goal_sampling_mode="vae_prior",

        launcher_config=dict(
            unpack_variant=True,
        )
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 3
    mode = 'sss'
    exp_prefix = 'exp2-oracle-pygame-latent-reward-1-obj'

    search_space = {
        'vae_algo_kwargs.num_iters': [501],
        # 'algo_kwargs.num_epochs': [1],
        'observation_key': [
            'state_observation',
        ],
        'use_ground_truth_reward': [
            False,
        ],
        'use_onehot_set_embedding': [
            True,
        ],
        'use_dummy_model': [
            False,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = list(sweeper.iterate_hyperparameters())
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(variants):
            if mode == 'local':
                variant['vae_algo_kwargs']['num_iters'] = 0
                variant['algo_kwargs'] = dict(
                    batch_size=5,
                    num_epochs=1,
                    num_eval_steps_per_epoch=2*20,
                    num_expl_steps_per_train_loop=2*20,
                    num_trains_per_train_loop=10,
                    min_num_steps_before_training=10,
                )
                variant['max_path_length'] = 2
            run_experiment(
                rig_experiment,
                exp_name=exp_prefix,
                num_exps_per_instance=2,
                mode=mode,
                variant=variant,
                # slurm_config_name='cpu',
                use_gpu=True,
                # gpu_id=1,
            )
