import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sets.rl_launcher import disco_experiment

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
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            reward_scale='auto_normalize_by_max_magnitude',
        ),
        max_path_length=100,
        # max_path_length=2,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=201,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
            # batch_size=5,
            # num_epochs=1,
            # num_eval_steps_per_epoch=2*8,
            # num_expl_steps_per_train_loop=2*8,
            # num_trains_per_train_loop=10,
            # min_num_steps_before_training=10,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.0,
            fraction_distribution_context=0.8,
            max_size=int(1e6),
        ),
        observation_key='latent_observation',
        # desired_goal_key='latent_desired_goal',
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
            ),
            beta=1,
            # set_loss_weight=1,
            # beta=0.001,
            set_loss_weight=0,
        ),
        data_loader_kwargs=dict(
            batch_size=128,
        ),
        vae_algo_kwargs=dict(
            num_iters=501,
            num_epochs_per_iter=1,
            progress_csv_file_name='vae_progress.csv',
        ),
        generate_set_for_vae_pretraining_kwargs=dict(
            num_sets=6,
            num_samples_per_set=128,
        ),
        generate_set_for_rl_kwargs=dict(
            num_sets=1,
            num_samples_per_set=128,
            # save_to_filename='1sets128samples.pickle',
            saved_filename='1sets128samples.pickle',
        ),
        num_ungrouped_images=12800,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 3
    mode = 'sss'
    exp_prefix = '20-07-24-exp2-single-set-normalized-rewards-take2'

    search_space = {
        # 'vae_algo_kwargs.num_iters': [10],
        'vae_trainer_kwargs.set_loss_weight': [
            0, 1, 100,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = list(sweeper.iterate_hyperparameters())
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(variants):
            variant['exp_id'] = exp_id
            run_experiment(
                disco_experiment,
                exp_name=exp_prefix,
                prepend_date_to_exp_name=False,
                num_exps_per_instance=2,
                mode=mode,
                variant=variant,
                # slurm_config_name='cpu',
                use_gpu=True,
                # gpu_id=1,
            )
