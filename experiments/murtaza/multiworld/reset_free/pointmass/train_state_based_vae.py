import torch

import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.reset_free.pointmass.generate_state_based_vae_dataset import generate_vae_dataset
from multiworld.envs.pygame.point2d import Point2DWallEnv
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

    #n_seeds = 1
    #mode = 'ec2'
    #exp_prefix = 'pointmass_vae_state_based_beta_schedule'

    variant = dict(
        num_epochs=200,
        algo_kwargs=dict(
            is_auto_encoder=True,
            batch_size=64,
        ),
        vae_kwargs=dict(
            hidden_sizes=[1000, 1000]
        ),
        generate_vae_dataset_kwargs=dict(
            N=5000,
            oracle_dataset=True,
            use_cached=False,
            env_class=Point2DWallEnv,
            env_kwargs=dict(
                ball_radius=0.5,
                render_onscreen=False,
                inner_wall_max_dist=2,
                wall_shape="u",
            ),
        ),
        # beta_schedule_kwargs=dict(
        #     x_values=[0, 50, 100],
        #     y_values=[0, 0, .1],
        # )
    )

    search_space = {
        'representation_size': [2, 16],
        'algo_kwargs.lr':[1e-3, 1e-4],
        'beta':[.01, .05, .1, .25],
        'flat_x': [25, 50, 75],
        'ramp_x': [50, 100],
        # 'vae_kwargs.hidden_sizes': [[32, 32], [10, 10], [100, 100], [1000]],
        'vae_kwargs.output_activation': [torch.tanh, identity],
        'vae_kwargs.layer_norm':[True, False]
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
