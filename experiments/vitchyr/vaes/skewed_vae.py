"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.vae.skewed_vae import (
    train_from_variant,
    uniform_truncated_data,
    four_corners,
    empty_dataset,
)

if __name__ == '__main__':
    variant = dict(
        dataset_generator=uniform_truncated_data,
        n_start_samples=300,
        bs=32,
        n_epochs=1000,
        # n_epochs=2,
        n_samples_to_add_per_epoch=1000,
        skew_config=dict(
            alpha=1,
            mode='biased_encoder',
            n_average=100,
        ),
        skew_sampling=False,
        weight_loss=False,
        z_dim=16,
        hidden_size=32,
        save_period=50,
        # save_period=1,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'skew-vae-four-corners-heatmaps'
    # exp_prefix = 'skew-vae-dynamics-noise-2'
    exp_prefix = 'skew-vae-sweep-many-with-video-4c'

    search_space = {
        'dataset_generator': [
            four_corners,
            # empty_dataset,
        ],
        'skew_config.mode': [
            # 'importance_sampling',
            'recon_mse',
            'exp_recon_mse',
            # 'biased_encoder',
            # 'prior'
            # 'none',
        ],
        # 'skew_config.temperature': [
        # ],
        'skew_config.alpha': [
            1,
        ],
        'append_all_data': [
            False,
        ],
        'skew_sampling': [
            True,
        ],
        'weight_loss': [
            True,
        ],
        'n_start_samples': [
            4,
        ],
        'dynamics_noise': [
            0,
            0.1,
            # 1,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                train_from_variant,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                skip_wait=True,
            )
