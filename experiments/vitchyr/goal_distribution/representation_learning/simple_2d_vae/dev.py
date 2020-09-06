import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sets.vae_2d_launcher import train_2d_set_vae

if __name__ == '__main__':
    variant = dict(
        create_vae_kwargs=dict(
            latent_dim=16,
            encoder_kwargs=dict(
                hidden_sizes=[256, 256],
            ),
        ),
        vae_trainer_kwargs=dict(
            vae_lr=1e-3,
            discriminator_lr=1e-3,
            vae_visualization_config=dict(
                debug_period=0,
            ),
            beta=0.1,
            set_loss_weight=1.,
            set_loss_version='kl',
        ),
        discriminator_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        vae_algo_kwargs=dict(
            num_iters=51,
            num_epochs_per_iter=20,
        ),
        debug_kwargs=dict(
            debug_period=10,
            num_samples=128,
        ),
        create_train_dataset_kwargs=dict(
            create_set_kwargs_list=[
                # dict(num_examples=128, version='box', xlim=(0, 2), ylim=(0, 2)),
                # dict(num_examples=128, version='box', xlim=(-2, 0), ylim=(-2, 0)),
                # dict(num_examples=128, version='box', xlim=(0, 2), ylim=(-2, 0)),
                # dict(num_examples=128, version='box', xlim=(-2, 2), ylim=(0, 2)),
                # dict(num_examples=128, version='circle', radius=3),
                dict(num_examples=1024, version='circle', radius=0),
                dict(num_examples=1024, version='circle', radius=3),
                # dict(num_examples=1024, version='circle', radius=0),
            ],
        ),
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 3
    # mode = 'sss'
    # exp_name = __file__.split('/')[-1].split('.')[0].replace('_', '-')
    # print('exp_name', exp_name)
    exp_name = 'dev-set-vae-2d-two-circles'

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                train_2d_set_vae,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )

