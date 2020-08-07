import torch

import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.reset_free.pointmass.generate_state_based_vae_dataset import generate_vae_dataset
from multiworld.envs.pygame.point2d import Point2DWallEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.vae import VAE, VAETrainer, AutoEncoder
import numpy as np
from rlkit.pythonplusplus import identity

def generate_dataset(N=1000, test_p=.9):
    xs = np.linspace(-np.pi*5, 5*np.pi, N).reshape(-1, 1)
    ys = np.sin(xs).reshape(-1, 1)
    info = dict()
    n = int(N * test_p)
    # data = np.hstack((xs,  ys))
    # train_dataset = data[:n, :]
    # test_dataset = data[n:, :]

    train_dataset = ys[:n, :]
    test_dataset = ys[n:, :]
    # plt.plot(xs, ys)
    # plt.show()
    return train_dataset, test_dataset, info

def experiment(variant):
    from rlkit.core import logger
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_dataset(
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
    output_scale=1
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
    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sin_sweep_beta_schedule'

    variant = dict(
        beta=.1,
        flat_x=50,
        ramp_x=50,
        num_epochs=200,
        algo_kwargs=dict(
            batch_size=64,
            is_auto_encoder=False,
        ),
        vae_kwargs=dict(
            hidden_sizes=[100, 100],
            output_activation=identity,
        ),
        representation_size=16,
        beta_schedule_kwargs = dict(
            x_values=[0, 50, 100],
            y_values=[0, 0, .1],
        )
    )

    search_space = {
        'representation_sizes':[4],
        'algo_kwargs.lr': [1e-3],
        'beta': [.01, .05, .1],
        'vae_kwargs.output_activation':[torch.tanh, identity],
        'flat_x':[25, 50, 75],
        'ramp_x':[50, 100],
        'vae_kwargs.hidden_sizes': [[32, 32], [10, 10], [100, 100]],
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
