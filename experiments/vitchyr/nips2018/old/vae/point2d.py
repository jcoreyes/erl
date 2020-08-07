import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.vae.point2d_data import get_data


def experiment(variant):
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data = get_data(10000)
    m = ConvVAE(representation_size)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta, use_cuda=False)
    for epoch in range(10):
        t.train_epoch(epoch)
        t.test_epoch(epoch)
        t.dump_samples(epoch)


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'name'

    variant = dict(
        beta=5.0,
    )

    search_space = {
        'representation_size': [2, 4, 8, 16],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
            )
