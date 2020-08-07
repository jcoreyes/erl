import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.vae.sawyer2d_multi_push_data import generate_vae_dataset

from rlkit.launchers.arglauncher import run_variants
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
        gpu_id = variant.get("gpu_id", None)
        if gpu_id is not None:
            ptu.set_device(gpu_id)
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
    exp_prefix = 'dev-sawyer-push-new-vae'
    use_gpu = True

    # n_seeds = 1
    # mode = 'ec2'
    exp_prefix = 'vae-sawyer-new-push-easy-3'

    variant = dict(
        beta=5.0,
        num_epochs=500,
        get_data_kwargs=dict(
            N=10000,
        ),
        algo_kwargs=dict(
            do_scatterplot=False,
            lr=1e-3,
        ),
        beta_schedule_kwargs=dict(
            x_values=[0, 100, 200, 500],
            # y_values=[0, 0, 0.1, 0.5],
            y_values=[0, 0, 5, 5],
        ),
        save_period=5,
    )

    search_space = {
        'representation_size': [4, 16],
        # 'beta_schedule_kwargs.y_values': [
        #     [0, 0, 0.1, 0.5],
        #     [0, 0, 0.1, 0.1],
        #     [0, 0, 5, 5],
        # ],
        # 'algo_kwargs.lr': [1e-3, 1e-2],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=7)

