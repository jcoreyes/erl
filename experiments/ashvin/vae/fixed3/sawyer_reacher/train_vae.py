import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.vae.sawyer2d_reach_data import generate_vae_dataset


def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_vae_dataset(
        **variant['get_data_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = ConvVAE(representation_size, input_channels=3)
    if ptu.gpu_enabled():
        m.to(ptu.device)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(epoch, save_reconstruction=should_save_imgs,
                     save_scatterplot=should_save_imgs)
        if should_save_imgs:
            t.dump_samples(epoch)


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-sawyer-reacher-vae-train'
    use_gpu = True

    # n_seeds = 1
    # mode = 'ec2'
    exp_prefix = 'sawyer-vae-reacher-recreate-results'
    # use_gpu = False

    variant = dict(
        beta=5.0,
        num_epochs=100,
        get_data_kwargs=dict(
            N=5000,
            # dataset_path="05-10-sawyer-reacher/sawyer_reacher_5000.npy",
        ),
        algo_kwargs=dict(
        ),
        # beta_schedule_kwargs=dict(
        #     x_values=[0, 100, 200, 500],
        #     # y_values=[0, 0, 0.1, 0.5],
        #     y_values=[0, 0, 5, 5],
        # ),
        save_period=1,
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
                trial_dir_suffix='r'+str(variant.get('representation_size', 0)),
            )
