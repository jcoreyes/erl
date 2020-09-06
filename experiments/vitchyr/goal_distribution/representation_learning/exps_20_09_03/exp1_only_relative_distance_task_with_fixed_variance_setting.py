import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sets.rl_launcher import disco_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='OneObject-PickAndPlace-BigBall-RandomInit-2D-v1',
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
            num_epochs=1001,
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
            fraction_future_context=0.0,
            fraction_distribution_context=0.8,
            max_size=int(1e6),
        ),
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=25,
            pad_color=255,
            subpad_length=1,
            pad_length=2,
            num_columns_per_rollout=4,
            num_imgs=8,
            num_example_images_to_show=4,
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
                debug_period=5,
                unnormalize_images=True,
                image_format='CHW',
            ),
            beta=1,
            set_loss_weight=0,
        ),
        data_loader_kwargs=dict(
            batch_size=128,
        ),
        vae_algo_kwargs=dict(
            num_iters=101,
            num_epochs_per_iter=20,
            progress_csv_file_name='vae_progress.csv',
        ),
        generate_set_for_vae_pretraining_kwargs=dict(
            num_samples_per_set=128,
            set_configs=[
                # dict(
                #     version='project_onto_axis',
                #     axis_idx_to_value={
                #         0: 3,
                #         1: 3,
                #     },
                # ),
                # dict(
                #     version='project_onto_axis',
                #     axis_idx_to_value={
                #         0: -2,
                #     },
                # ),
                # dict(
                #     version='project_onto_axis',
                #     axis_idx_to_value={
                #         2: -3,
                #         3: 3,
                #     },
                # ),
                # dict(
                #     version='project_onto_axis',
                #     axis_idx_to_value={
                #         3: -4,
                #     },
                # ),
                dict(
                    version='move_a_to_b',
                    offsets_from_b=(4, 0),
                    a_axis_to_b_axis={
                        0: 2,
                        1: 3,
                    },
                ),
            ],
        ),
        generate_set_for_rl_kwargs=dict(
            num_samples_per_set=128,
            set_configs=[
                # dict(
                #     version='project_onto_axis',
                #     axis_idx_to_value={
                #         0: 3,
                #         1: 3,
                #     },
                # ),
                # dict(
                #     version='project_onto_axis',
                #     axis_idx_to_value={
                #         0: -2,
                #     },
                # ),
                # dict(
                #     version='project_onto_axis',
                #     axis_idx_to_value={
                #         2: -3,
                #         3: 3,
                #     },
                # ),
                # dict(
                #     version='project_onto_axis',
                #     axis_idx_to_value={
                #         3: -4,
                #     },
                # ),
                dict(
                    version='move_a_to_b',
                    offsets_from_b=(4, 0),
                    a_axis_to_b_axis={
                        0: 2,
                        1: 3,
                    },
                ),
            ],
        ),
        num_ungrouped_images=12800,
        reward_fn_kwargs=dict(
            drop_log_det_term=True,
            sqrt_reward=True,
        ),
        rig=False,
        rig_goal_setter_kwargs=dict(
            use_random_goal=True,
        ),
        use_ground_truth_reward=False,
        train_set_vae_kwargs=dict(
            reward_visualization_period=10,
        )
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 3
    mode = 'sss'
    exp_name = __file__.split('/')[-1].split('.')[0].replace('_', '-')
    print('exp_name', exp_name)

    search_space = {
        'vae_algo_kwargs.num_iters': [101],
        'create_vae_kwargs.decoder_distribution': [
            'gaussian_fixed_unit_variance',
        ],
        'create_vae_kwargs.use_fancy_architecture': [
            True,
        ],
        'vae_trainer_kwargs.set_loss_weight': [
            0.,
            0.1,
            1.,
        ],
        'create_vae_kwargs.latent_dim': [
            8,
        ],
        'train_set_vae_kwargs.beta_scale_schedule_kwargs.y_values': [
            [1, 1, 1, 1, 1, 1],
        ],
        'vae_trainer_kwargs.beta': [
            1.,
        ],
        'observation_key': [
            'latent_observation',
        ],
        'use_ground_truth_reward': [
            False,
        ],
        'use_onehot_set_embedding': [
            False,
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
                variant['vae_algo_kwargs']['num_iters'] = 1
                # variant['generate_set_for_rl_kwargs']['saved_filename'] = (
                #     'manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle'
                # )
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
                disco_experiment,
                exp_name=exp_name,
                num_exps_per_instance=2,
                mode=mode,
                variant=variant,
                # slurm_config_name='cpu',
                use_gpu=True,
                # gpu_id=1,
            )
