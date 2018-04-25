import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.vae.conv_vae import ConvVAE, ConvVAETrainer
from railrl.torch.vae.sawyer2d_reach_data import get_data


def experiment(variant):
    from railrl.core import logger
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = get_data(**variant['get_data_kwargs'])
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    m = ConvVAE(representation_size, input_channels=3)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta, use_cuda=True)
    for epoch in range(variant['num_epochs']):
        t.train_epoch(epoch)
        t.test_epoch(epoch)
        t.dump_samples(epoch)


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    # mode = 'local_docker'
    exp_prefix = 'dev-sawyer-reacher-vae-train-2'
    use_gpu=True

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer-reach-xy-vae-train'
    use_gpu=False

    variant = dict(
        beta=5.0,
        num_epochs=100,
        get_data_kwargs=dict(
            N=10000,
            use_cached=False,
        ),
    )

    search_space = {
        'representation_size': [2, 4, 8, 16],
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
                use_gpu=use_gpu,
            )
