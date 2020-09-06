import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sets.vae_launcher import train_set_vae

if __name__ == "__main__":
    variant = dict(
        env_id='OneObject-PickAndPlace-BigBall-RandomInit-2D-v1',
        renderer_kwargs=dict(
            output_image_format='CHW',
        ),
        create_vae_kwargs=dict(
            latent_dim=128,
            use_fancy_architecture=True,
            decoder_distribution='gaussian_learned_global_scalar_variance',
        ),
        vae_trainer_kwargs=dict(
            vae_lr=1e-3,
            vae_visualization_config=dict(
                num_recons=10,
                num_samples=20,
                # debug_period=50,
                debug_period=5,
                unnormalize_images=True,
                image_format='CHW',
            ),
            beta=2,
            set_loss_weight=0,
        ),
        beta_scale_schedule_kwargs=dict(
            version='piecewise_linear',
            x_values=[0, 20, 40, 60, 80, 100],
        ),
        data_loader_kwargs=dict(
            batch_size=128,
        ),
        vae_algo_kwargs=dict(
            num_iters=101,
            num_epochs_per_iter=20,
            progress_csv_file_name='vae_progress.csv',
        ),
        include_env_debug=True,
        generate_test_set_kwargs=dict(
            num_samples_per_set=128,
            set_configs=[
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
        generate_train_set_kwargs=dict(
            num_samples_per_set=128,
            set_configs=[
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
        logger_config=dict(
            push_prefix=False,
        ),
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 1
    mode = 'sss'
    exp_name = __file__.split('/')[-1].split('.')[0].replace('_', '-')
    print('exp_name', exp_name)

    search_space = {
        'vae_algo_kwargs.num_iters': [101],
        'create_vae_kwargs.decoder_distribution': [
            'gaussian_learned_global_scalar_variance',
        ],
        'create_vae_kwargs.use_fancy_architecture': [
            True,
        ],
        'vae_trainer_kwargs.set_loss_weight': [
            0.,
        ],
        'create_vae_kwargs.latent_dim': [
            32,
        ],
        'beta_scale_schedule_kwargs.y_values': [
            [0, 1000, 2000, 3000, 4000, 5000],
            [0, 2000, 4000, 6000, 8000, 10000],
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = list(sweeper.iterate_hyperparameters())
    raise NotImplementedError("""
    Make sure to hardcode this in vae_launcher.py
    ungrouped_imgs = generate_images(
        env, renderer, num_images=num_ungrouped_images, set=train_sets[0])
    """)
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(variants):
            variant['vae_trainer_kwargs']['beta'] = (
                    1. / variant['create_vae_kwargs']['latent_dim']
            )
            variant['vae_trainer_kwargs']['debug_bad_recons'] = (
                    variant['create_vae_kwargs']['decoder_distribution'] ==
                    'gaussian_learned_global_scalar_variance'
            )
            if mode == 'local':
                variant['vae_algo_kwargs']['num_iters'] = 1
                variant['vae_algo_kwargs']['num_epochs_per_iter'] = 1
                # variant['generate_train_set_kwargs']['saved_filename'] = (
                #     'manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle'
                # )
            run_experiment(
                train_set_vae,
                exp_name=exp_name,
                prepend_date_to_exp_name=True,
                num_exps_per_instance=2,
                mode=mode,
                variant=variant,
                # slurm_config_name='cpu',
                use_gpu=True,
                # gpu_id=1,
            )
