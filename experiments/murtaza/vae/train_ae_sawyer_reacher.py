import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
#from rlkit.torch.vae.sawyer2d_multi_push_data import generate_vae_dataset
from rlkit.torch.vae.sawyer2d_reach_data import generate_vae_dataset
#from rlkit.torch.vae.sawyer2d_push_new_easy_data import generate_vae_dataset

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
    m = ConvVAE(representation_size, is_auto_encoder=variant['is_auto_encoder'], input_channels=3, **variant['conv_vae_kwargs'])
    if ptu.gpu_enabled():
        m.cuda()
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
    exp_prefix = 'sawyer_reacher_ae_sweep_for_test'
    use_gpu = True

    variant = dict(
        beta=0,
        num_epochs=100,
        get_data_kwargs=dict(
            N=10000,
            use_cached=True,
        ),
        algo_kwargs=dict(
            # batch_size=32,
            lr=1e-1,
        ),
        conv_vae_kwargs=dict(
            min_variance=None,
            use_old_architecture=True,
        ),
        is_auto_encoder=True,
        save_period=5,
    )

    search_space = {
        'representation_size': [4, 16],
        'algo_kwargs.lr': [1e-2],
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
                snapshot_mode='gap',
                snapshot_gap=10,
            )
