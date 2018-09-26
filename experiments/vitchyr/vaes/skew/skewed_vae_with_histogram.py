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
        bs=32,
        # n_epochs=1000,
        # save_period=50,
        save_period=1,
        n_epochs=50,
        n_samples_to_add_per_epoch=10000,
        # n_epochs=5,
        # n_samples_to_add_per_epoch=10,
        n_start_samples=0,
        skew_sampling=False,
        weight_loss=False,
        z_dim=16,
        hidden_size=32,
        append_all_data=False,
        beta_schedule_class=ConstantSchedule,
        # beta_schedule_kwargs=dict(
        #     value=0.1,
        # )
        vae_weight_config=dict(
            alpha=1,
            mode='importance_sampling',
            n_average=100,
        ),
        use_dataset_generator_first_epoch=True,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # exp_prefix = 'sv-square-full-method'
    # exp_prefix = 'sv-square-full-method-big-bs-inv-sqrt'
    # exp_prefix = 'sv-square-weight-loss'
    # exp_prefix = 'sv-zero-init-square-border'
    # exp_prefix = 'sv-zero-init-square-cap'
    # exp_prefix = 'sv-zero-init-square-cap-split'

    search_space = {
        # Optimization hyperparameters
        'bs': [
            500,
        ],
        'num_inner_vae_epochs': [10],
        # Env hyperparameters
        'dynamics_noise': [
            # 0.2,
            # 0.1,
            0.05,
        ],
        'projection': [
            # project_samples_square_np,
            project_square_border_np,
            # project_square_cap_np,
            # project_square_cap_split_np,
        ],
        # Method hyperparameters
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
        'beta_schedule_kwargs.value': [
            1,
        ],
        'decoder_output_var': [
            # 'learned'
            # 0.5,
            5,
        ],
        'reset_vae_every_epoch': [
            # True,
            False,
        ],
        'weight_type': [
            # 'inv_p',
            'sqrt_inv_p',
            # 'nll',
        ],
        # Cheating hyperparameters
        'num_bins': [60],
        'train_vae_from_histogram': [
            True,
            # False,
        ],
        'use_perfect_samples': [
            # True,
            False,
        ],
        'use_perfect_density': [
            True,
            # False,
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
