import numpy as np

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import train_vae


def get_dataset(variant):
    filename = variant['filename']
    data = np.load(filename).item()
    imgs = data['obs']
    split = int(0.8 * len(imgs))
    train_dataset, test_dataset = imgs[:split], imgs[split:]
    return train_dataset, test_dataset, {}


if __name__ == '__main__':
    variant = dict(
        beta=0.5 / 128,
        representation_size=32,
        imsize=84,
        generate_vae_data_fctn=get_dataset,
        save_period=10,
        num_epochs=1000,
        decoder_activation='sigmoid',
        generate_vae_dataset_kwargs=dict(
            filename="/home/vitchyr/git/railrl/data/raw/12"
                     "-29_Image84WheeledCarEnv-v0_N10000__imsize84_oracleTrue.npy",
        ),
        vae_kwargs=dict(
            input_channels=3,
        ),
        algo_kwargs=dict(
            lr=1e-3,
            # use_parallel_dataloading=False,
        ),
    )

    n_seeds = 1
    mode = 'here_no_doodad'
    exp_prefix = 'dev-intro-vae'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'name'

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                train_vae,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                use_gpu=True,
            )
