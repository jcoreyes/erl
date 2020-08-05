import railrl.misc.hyperparameter as hyp

from railrl.launchers.launcher_util import run_experiment
from railrl.torch.sets.launcher import test_offline_set_vae

if __name__ == '__main__':
    variant = dict(
        create_vae_kwargs=dict(
            latent_dim=128,
            encoder_cnn_kwargs=dict(
                # kernel_sizes=[4],
                # n_channels=[128],
                # strides=[2],
                # paddings=[0],
                kernel_sizes=[5, 3, 3],
                n_channels=[16, 32, 64],
                strides=[3, 2, 2],

                # kernel_sizes=[3, 3, 3],
                # n_channels=[32, 64, 128],
                # strides=[1, 1, 1],
                # paddings=[0, 0, 0],

                pool_type='none',
                hidden_activation='relu',
                normalization_type='layer',
            ),
            encoder_mlp_kwargs=dict(
                # hidden_sizes=[128, 128],
                hidden_sizes=[],
            ),
            decoder_mlp_kwargs=dict(
                hidden_sizes=[256, 256],
            ),
            decoder_distribution='gaussian_learned_global_scalar_variance',
            use_mlp_decoder=True,
        ),
        vae_trainer_kwargs=dict(
            vae_lr=1e-3,
            vae_visualization_config=dict(
                num_recons=5,
                num_samples=20,
                # debug_period=50,
                debug_period=10,
                unnormalize_images=True,
            ),
            beta=0.001,
            set_loss_weight=0,
        ),
        data_loader_kwargs=dict(
            batch_size=32,
        ),
        algo_kwargs=dict(
            num_iters=101,
            num_epochs_per_iter=10,
            # num_epochs=101,
            # num_epochs=11,
        ),
        generate_set_kwargs=dict(
            num_sets=16,
            num_samples_per_set=32,
        ),
        env_kwargs=dict(
            num_objects=4,
        ),
        num_ungrouped_images=1024,
        beta_schedule_kwargs=dict(
            version='piecewise_linear',
            x_values=[0, 10, 20, 30],
            y_values=[1, 1, 1, 1],
        ),
    )
    n_seeds = 1

    search_space = {
        'beta_schedule_kwargs.y_values': [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0.001, 0.01, 0.1, 1],
        ],
        'vae_trainer_kwargs.beta': [
            1,
        ],
        'vae_trainer_kwargs.set_loss_weight': [
            0,
        ],
        'create_vae_kwargs.use_mlp_decoder': [
            False,
            # True,
        ],
        'create_vae_kwargs.decoder_mlp_kwargs.hidden_sizes': [
            [300, 400],
            # [256, 256],
            [128, 128],
        ],
        'create_vae_kwargs.latent_dim': [
            128, 64, 32, 16
        ],
        'create_vae_kwargs.encoder_cnn_kwargs': [
            # dict(
            #     kernel_sizes=[4],
            #     n_channels=[128],
            #     strides=[2],
            #     paddings=[0],
            #
            #     pool_type='none',
            #     hidden_activation='relu',
            #     normalization_type='batch',
            # ),
            dict(
                kernel_sizes=[4],
                n_channels=[128],
                strides=[2],
                paddings=[0],

                pool_type='none',
                hidden_activation='relu',
                normalization_type='none',
            ),
        ],
    }

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 1
    mode = 'sss'
    exp_prefix = 'exp4-slowly-increase-beta-steven-architecture'

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        # beta = variant['vae_trainer_kwargs']['beta']
        # slw = variant['vae_trainer_kwargs']['set_loss_weight']
        for _ in range(n_seeds):
            run_experiment(
                test_offline_set_vae,
                variant=variant,
                exp_name=exp_prefix,
                mode=mode,
                # slurm_config_name='gpu_low_pri',
                use_gpu=True,
            )
