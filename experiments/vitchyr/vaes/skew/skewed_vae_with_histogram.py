"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import ConstantSchedule
from railrl.torch.vae.skew.datasets import (
    uniform_truncated_data,
    four_corners,
    empty_dataset,
    zeros_dataset,
    negative_one_dataset,
    gaussian_data,
    small_gaussian_data,
    project_samples_square_np,
    project_samples_ell_np,
    project_square_border_np,
    project_square_cap_np,
    project_square_cap_split_np,
)
from railrl.torch.vae.skew.skewed_vae_with_histogram import train_from_variant

if __name__ == '__main__':
    variant = dict(
        dataset_generator=negative_one_dataset,
        dynamics_noise=0.05,
        # n_epochs=1000,
        # save_period=50,
        save_period=1,
        n_epochs=50,
        n_samples_to_add_per_epoch=10000,
        # n_epochs=5,
        # n_samples_to_add_per_epoch=10,
        n_start_samples=0,
        z_dim=16,
        hidden_size=32,
        append_all_data=False,
        vae_kwargs=dict(
            mode='importance_sampling',
            # mode='biased_encoder',
            # mode='prior',
            min_prob=1e-6,
            n_average=100,
            batch_size=500,
            weight_loss=True,
            skew_sampling=False,
            num_inner_vae_epochs=10,
        ),
        use_dataset_generator_first_epoch=True,
        skew_config=dict(
            weight_type='sqrt_inv_p',
            minimum_prob=1e-6,
        ),
        reset_vae_every_epoch=False,
        decoder_output_var='learned',
        num_bins=60,
        use_perfect_samples=False,
        use_perfect_density=False,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    search_space = {
        'projection': [
            # project_samples_square_np,
            project_square_border_np,
            # project_square_cap_np,
            # project_square_cap_split_np,
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
                # skip_wait=True,
            )
