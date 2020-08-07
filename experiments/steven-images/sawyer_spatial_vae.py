import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.vae.conv_vae import ConvVAE, SpatialAutoEncoder
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.vae.sawyer2d_reach_data import get_data


def experiment(variant):
    num_feat_points=variant['feat_points']
    from rlkit.core import logger
    beta = variant["beta"]
    print('collecting data')
    train_data, test_data, info = get_data(**variant['get_data_kwargs'])
    print('finish collecting data')
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    m = SpatialAutoEncoder(2 * num_feat_points, num_feat_points, input_channels=3)
#    m = ConvVAE(2*num_feat_points, input_channels=3)
    t = ConvVAETrainer(train_data, test_data, m,  lr=variant['lr'], beta=beta)
    for epoch in range(variant['num_epochs']):
        t.train_epoch(epoch)
        t.test_epoch(epoch)
        t.dump_samples(epoch)


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    variant = dict(
        num_epochs=150,
        get_data_kwargs=dict(
            N=10000,
            use_cached=True,
        ),
    )

    search_space = {
        'beta': [1],
        'feat_points': [16],
        'lr': [1e-2]

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                experiment,
                exp_prefix='test-spatial',
                mode=mode,
                variant=variant,
                use_gpu=True,
            )
