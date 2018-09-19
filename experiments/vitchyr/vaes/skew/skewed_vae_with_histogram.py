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
        dataset_generator=uniform_truncated_data,
        n_start_samples=4,
        bs=32,
        n_epochs=1000,
        # n_epochs=500,
        # n_epochs=10,
        n_samples_to_add_per_epoch=1000,
        skew_sampling=False,
        weight_loss=False,
        z_dim=16,
        hidden_size=32,
        save_period=50,
        # save_period=25,
        # save_period=1,
        beta_schedule_class=ConstantSchedule,
        # beta_schedule_kwargs=dict(
        #     value=0.1,
        # )
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # exp_prefix = 'skew-vae-biased-beta0.025-skew-weight-sweep'
    # exp_prefix = 'skew-vae-all-correct-sweep-weight-skew-2'
    # exp_prefix = 'skew-vae-with-histogram-sweep-weight-and-skew'
    # exp_prefix = 'dev-skew-vae-with-histogram-square-border'

    search_space = {
        'dataset_generator': [
            # four_corners,
            empty_dataset,
            # small_gaussian_data,
        ],
        'projection': [
            project_samples_square_np,
            # project_square_border_np,
            # project_samples_ell_np,
        ],
        'append_all_data': [
            False,
        ],
        'skew_sampling': [
            True,
            # False,
        ],
        'n_start_samples': [
            4,
        ],
        'weight_loss': [
            True,
            # False,
        ],
        'dynamics_noise': [
            0.3,
            # 0,
        ],
        'beta_schedule_kwargs.value': [
            # 1,
            0.1,
            # 0.075,
            # 0.05,
            # 0.025,
            # 0.01,
            # 0.0075,
            # 0.005,
            # 0.0025,
            # 0.001,
            # 0.0001,
            # 0,
        ],
        'decoder_output_std': [
            # 1,
            # 0.5,
            # 0.3,
            # 0.2,
            # 0.1,
            # 0.05,
            'learned'
        ],
        'num_bins': [10],
        'train_vae_from_histogram': [True],
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
