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
    gaussian_data,
    small_gaussian_data,
    project_samples_square_np,
    project_samples_ell_np,
    project_square_border_np,
)
from railrl.torch.vae.skew.skewed_vae_with_histogram import train_from_variant

if __name__ == '__main__':
    variant = dict(
        dataset_generator=empty_dataset,
        n_start_samples=4,
        bs=32,
        n_epochs=1000,
        save_period=50,
        # n_epochs=500,
        # n_epochs=10,
        n_samples_to_add_per_epoch=1000,
        skew_sampling=False,
        weight_loss=False,
        z_dim=16,
        hidden_size=32,
        append_all_data=False,
        beta_schedule_class=ConstantSchedule,
        # beta_schedule_kwargs=dict(
        #     value=0.1,
        # )
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # exp_prefix = 'skew-vae-biased-beta0.025-skew-weight-sweep'

    search_space = {
        'projection': [
            project_samples_square_np,
        ],
        'append_all_data': [
            False,
        ],
        'skew_sampling': [
            # True,
            False,
        ],
        'weight_loss': [
            # True,
            False,
        ],
        'dynamics_noise': [
            0.2,
        ],
        'beta_schedule_kwargs.value': [
            0.1,
        ],
        'decoder_output_std': [
            'learned'
        ],
        'num_bins': [20],
        'train_vae_from_histogram': [True],
        'weight_type': [
            'inv_p',
            # 'sqrt_inv_p',
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
