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
)
from railrl.torch.vae.skew.histogram import train_from_variant

if __name__ == '__main__':
    variant = dict(
        dataset_generator=uniform_truncated_data,
        n_start_samples=0,
        n_epochs=5,
        n_samples_to_add_per_epoch=1000,
        save_period=1,
        append_all_data=False,
        dynamics_noise=0.2,
        num_bins=20,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # exp_prefix = 'skew-vae-biased-beta0.025-skew-weight-sweep'
    # exp_prefix = 'skew-vae-all-correct-sweep-weight-skew-2'
    # exp_prefix = 'skew-vae-ell'

    search_space = {
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
