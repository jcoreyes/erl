import torch

import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.reset_free.pointmass.generate_state_based_vae_dataset import generate_vae_dataset
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_reset import SawyerPushAndReachXYEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.pythonplusplus import identity
from rlkit.torch.vae.vae import VAE, VAETrainer, AutoEncoder
import numpy as np

def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_vae_dataset(
        **variant['generate_vae_dataset_kwargs']
    )
    scale=10
    train_data *= scale
    test_data *= scale
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        kwargs = variant['beta_schedule_kwargs']
        kwargs['y_values'][2] = variant['beta']
        kwargs['x_values'][1] = variant['flat_x']
        kwargs['x_values'][2] = variant['ramp_x'] + variant['flat_x']
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    output_scale=4
    if variant['algo_kwargs']['is_auto_encoder']:
        m = AutoEncoder(representation_size,
                train_data.shape[1],
                output_scale=output_scale,
                **variant['vae_kwargs']
                )
    else:
        m = VAE(representation_size,
                        train_data.shape[1],
                        output_scale=output_scale,
                        **variant['vae_kwargs']
                        )
    t = VAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    for epoch in range(variant['num_epochs']):
        t.train_epoch(epoch)
        t.test_epoch(epoch)


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'sawyer_push_vae_state_based_beta_schedule'

    variant = dict(
        num_epochs=200,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
        ),
        vae_kwargs=dict(
            hidden_sizes=[32, 32]
        ),
        generate_vae_dataset_kwargs=dict(
            N=100,
            oracle_dataset=True,
            use_cached=False,
            env_class=SawyerPushAndReachXYEnv,
            env_kwargs=dict(
                reward_type='puck_distance',
                goal_low=(-0.1, 0.5, 0.02, -0.1, 0.6),
                goal_high=(0.1, 0.7, 0.02, 0.1, 0.7),
            ),
        ),
        representation_size=5,
        beta=0,
    )

    search_space = {
        'algo_kwargs.lr':[1e-3],
        'beta':[1],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
            )
